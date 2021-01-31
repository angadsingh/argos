import logging

from appmetrics import reporter, metrics

from lib import setup_logging

setup_logging()

log = logging.getLogger(__name__)

log.info("package import START")

import importlib
import jsonpickle

from detection.motion_detector import SimpleMotionDetector
from detection.object_detector_base import BaseTFObjectDetector
from detection.object_detector_streaming import StreamingTFObjectDetector
from detection.pattern_detector import PatternDetector
from detection.state_managers.door_state_manager import DoorStateManager
from detection.state_managers.motion_state_manager import MotionStateManager
from lib.framelimiter import FrameLimiter

from base_detector import DetectorView
from broker import Broker
from configs.constants import InputMode
from lib.getch import getch

import argparse
import threading
import time

import cv2
from flask import Flask
from flask import Response
from flask import jsonify
from flask import render_template
from flask import request

from lib.fps import FPS
from lib.task_queue import BlockingTaskSingleton, NonBlockingTaskSingleton, BlockingTaskQueue
from flask_classful import route

log.info("package import END")


class StreamDetector():
    def __init__(self, config, object_detector: BaseTFObjectDetector, pattern_detector: PatternDetector):
        self.output_video_frame_q = NonBlockingTaskSingleton()
        self.active_video_feeds = 0
        self.config = config
        self.od = object_detector
        self.pattern_detector = pattern_detector
        self.door_state_manager = DoorStateManager(pattern_detector, pattern_detector.broker_q)
        self.motion_state_manager = MotionStateManager(pattern_detector, pattern_detector.broker_q)
        self.motion_detector = SimpleMotionDetector(config)
        self.stopped = False

    def start(self):
        log.info("TFObjectDetector init START")
        self.od.start()

        if self.config.input_mode == InputMode.RTMP_STREAM:
            from input.rtmpstream import RTMPVideoStream
            self.vs = RTMPVideoStream(self.config.rtmp_stream_url).start()
        elif self.config.input_mode == InputMode.PI_CAM:
            from input.picamstream import PiVideoStream
            self.vs = PiVideoStream(resolution=(640, 480), framerate=30).start()
        elif self.config.input_mode == InputMode.VIDEO_FILE:
            from input.videofilestream import VideoFileStream
            self.vs = VideoFileStream(self.config.video_file_path, self.config.video_in_sync).start()

        self.od.wait_for_ready()
        log.info("TFObjectDetector init END")

        # start a thread that will perform object detection
        log.info("detect_objects init..")
        self.t = threading.Thread(target=self.detect_objects)
        self.t.daemon = True
        self.t.start()

    def wait_for_completion(self, timeout=None):
        self.t.join(timeout=timeout)
        return self.t.is_alive()

    def stop(self):
        self.stopped = True
        self.vs.stop()
        if self.t.is_alive():
            self.t.join()
        self.od.stop()

    def draw_masks(self, frame):
        if self.config.md_mask:
            xmin, ymin, xmax, ymax = self.config.md_mask
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (128, 0, 128), 1)

    def detect_objects(self):
        total = 0

        fps = FPS(50, 100)

        # loop over frames from the video stream
        limiter = FrameLimiter(self.config.md_frame_rate)
        while limiter.limit() and not self.vs.stopped and not self.stopped:
            frame = self.vs.read()
            if frame is not None:
                output_frame = frame.copy()
                if self.config.tf_apply_md:
                    output_frame, crop, motion_outside = self.motion_detector.detect(output_frame)
                    if self.config.pattern_detection_enabled:
                        door_state = self.config.door_state_detector.detect_door_state(frame)
                        self.door_state_manager.add_state(door_state)
                        self.motion_state_manager.add_state(motion_outside)
                        if self.config.door_state_detector_show_detection:
                            self.config.door_state_detector.show_detection(output_frame, door_state)
                    if crop is not None:
                        minX, minY, maxX, maxY = crop
                        cropped_frame = frame[minY:maxY, minX:maxX]
                        ts = time.time()
                        self.od.add_task((frame, cropped_frame, (minX, minY), ts))
                else:
                    self.od.add_task((frame, frame, (0, 0), None, None))

                self.draw_masks(output_frame)

                fps.count()

                if total % self.config.fps_print_frames == 0:
                    log.info("od=%.2f/md=%.2f/st=%.2f fps" % (self.od.fps.fps, fps.fps, self.vs.fps.fps))
                log.debug("total: %d" % total)
                total += 1

                if self.config.show_fps:
                    cv2.putText(output_frame,
                                "od=%.2f/md=%.2f/st=%.2f fps" % (self.od.fps.fps, fps.fps, self.vs.fps.fps),
                                (10, output_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                self.output_video_frame_q.enqueue(output_frame)

            else:
                log.info("frame is NONE")

            if self.config.debug_mode:
                ch = getch()
                if ch == 'q':
                    break

        fps.stop()

    def generate(self):
        self.active_video_feeds += 1
        current_feed_num = self.active_video_feeds
        # loop over frames from the output stream
        try:
            limiter = FrameLimiter(self.config.video_feed_fps)
            while limiter.limit():
                output_frame = self.output_video_frame_q.read()
                # encode the frame in JPEG format
                (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
                # ensure the frame was successfully encoded
                if not flag:
                    continue
                # yield the output frame in the byte format
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')
        finally:
            self.active_video_feeds -= 1


class StreamDetectorView(DetectorView):
    def __init__(self, streaming_detector: StreamDetector):
        super().__init__()
        self.sd = streaming_detector
        self.config = self.sd.config

    @route("/")
    def index(self):
        return render_template("index.html")

    @route('/status')
    def status(self):
        return jsonify ({
                'active_video_feeds': self.sd.active_video_feeds,
                'od_active_video_feeds': self.sd.od.active_video_feeds,
                'appmetrics': metrics.metrics_by_name_list(metrics.metrics())
            })

    @route('/config')
    def apiconfig(self):
        super().apiconfig()

        self.config.send_mqtt = bool(request.args.get('send_mqtt', self.config.send_mqtt))
        self.config.mqtt_heartbeat_secs = int(
            request.args.get('mqtt_heartbeat_secs', self.config.mqtt_heartbeat_secs))
        self.config.show_fps = bool(request.args.get('show_fps', self.config.show_fps))
        self.config.video_feed_fps = int(request.args.get('video_feed_fps', self.config.video_feed_fps))

        self.config.md_tval = int(request.args.get('md_tval', self.config.md_tval))
        self.config.md_bg_accum_weight = float(request.args.get('md_bg_accum_weight', self.config.md_bg_accum_weight))
        self.config.md_show_all_contours = bool(
            request.args.get('md_show_all_contours', self.config.md_show_all_contours))
        self.config.md_min_cont_area = int(request.args.get('md_min_cont_area', self.config.md_min_cont_area))
        self.config.md_frame_rate = int(request.args.get('md_frame_rate', self.config.md_frame_rate))
        self.config.md_box_threshold_x = int(request.args.get('md_box_threshold_x', self.config.md_box_threshold_x))
        self.config.md_box_threshold_y = int(request.args.get('md_box_threshold_y', self.config.md_box_threshold_y))
        self.config.md_reset_bg_model = bool(request.args.get('md_reset_bg_model', self.config.md_reset_bg_model))

        return Response(jsonpickle.encode(self.config.__dict__, max_depth=2), mimetype='application/json')

    @route("/image")
    def image(self):
        (flag, encodedImage) = cv2.imencode(".jpg", self.sd.output_video_frame_q.read())
        return Response(bytearray(encodedImage),
                        mimetype='image/jpeg')

    @route("/video_feed")
    def video_feed(self):
        return Response(self.sd.generate(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @route("/od_video_feed")
    def od_video_feed(self):
        return Response(self.sd.od.generate_output_frames(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-c", "--config", type=str, required=True,
                    help="path to the python config file")
    args = vars(ap.parse_args())

    m = importlib.import_module(args["config"])
    config = getattr(m, "Config")()
    broker_q = BlockingTaskSingleton(metric_prefix='broker_q')
    notify_q = BlockingTaskQueue(config.notifier_queue_size, metric_prefix='notifier_q')
    pattern_detector = None
    if config.pattern_detection_enabled:
        pattern_detector = PatternDetector(broker_q, config.pattern_detection_pattern_steps,
                                           config.pattern_detection_state_history_length,
                                           config.pattern_detection_state_history_length_partial,
                                           config.pattern_detection_interval)
    od = StreamingTFObjectDetector(config, broker_q)
    sd = StreamDetector(config, od, pattern_detector)
    mb = Broker(sd.config, od, pattern_detector, broker_q, notify_q)

    log.info("flask init..")
    app = Flask(__name__)
    def stdout_report(metrics):
        log.info(metrics)
    reporter.register(stdout_report, reporter.fixed_interval_scheduler(30))
    StreamDetectorView.register(app, init_argument=sd, route_base='/')
    f = threading.Thread(target=app.run, kwargs={'host': args["ip"], 'port': args["port"], 'debug': False,
                                                 'threaded': True, 'use_reloader': False})
    f.daemon = True
    f.start()

    log.info("start reading video file")
    sd.start()
    sd.wait_for_completion()
    sd.stop()
    mb.stop()
    if pattern_detector:
        pattern_detector.stop()
