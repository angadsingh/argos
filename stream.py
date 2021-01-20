import logging
import importlib
import sys

for _ in ("colormath.color_conversions", "colormath.color_objects"):
    logging.getLogger(_).setLevel(logging.CRITICAL)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

from base_detector import DetectorView
from broker import Broker
from configs.constants import InputMode
from detection.detect_streaming import StreamingTFObjectDetector
from detection.door_detect import DoorMovementDetector
from lib.getch import getch

log.info("package import START")
import argparse
import threading
import time

import cv2
import imutils
import numpy as np
from flask import Flask
from flask import Response
from flask import jsonify
from flask import render_template
from flask import request

from lib.fps import FPS
from lib.singleton_q import SingletonBlockingQueue
from flask_classful import route

log.info("package import END")


class StreamDetector():
    def __init__(self, config, broker_q, door_detector):
        self.outputFrame = SingletonBlockingQueue()
        self.active_video_feeds = 0
        self.config = config
        self.broker_q = broker_q
        self.door_detector = door_detector

    def start(self):
        log.info("TFObjectDetector init START")
        self.od = StreamingTFObjectDetector(self.config, self.broker_q).start()

        if self.config.input_mode == InputMode.RTMP_STREAM:
            from input.rtmpstream import RTMPVideoStream
            self.vs = RTMPVideoStream(self.config.rtmp_stream_url).start()
        elif self.config.input_mode == InputMode.PI_CAM:
            from input.picamstream import PiVideoStream
            self.vs = PiVideoStream(resolution=(640, 480), framerate=30).start()
        elif self.config.input_mode == InputMode.VIDEO_FILE:
            from input.videofilestream import VideoFileStream
            self.vs = VideoFileStream(self.config.video_file_path).start()

        self.od.wait_for_ready()
        log.info("TFObjectDetector init END")

        # start a thread that will perform object detection
        log.info("detect_objects init..")
        t = threading.Thread(target=self.detect_objects)
        t.daemon = True
        t.start()
        return t

    def cleanup(self):
        self.vs.stop()

    def apply_md(self, frame, bg):
        num_filtered_cnts = 0
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        # whether part or whole of the motion occured outside the mask
        motion_outside = None

        # narrow the frame to a box with motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if bg is not None:
            delta = cv2.absdiff(bg.astype("uint8"), gray)
            thresh = cv2.threshold(delta, self.config.md_tval, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if len(cnts) > 0:
                filter_pass = False
                for c in cnts:
                    if cv2.contourArea(c) > self.config.md_min_cont_area:
                        filter_pass = True
                        (x, y, w, h) = cv2.boundingRect(c)
                        (minX, minY) = (min(minX, x), min(minY, y))
                        (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))
                        if self.config.md_show_all_contours:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                if filter_pass:
                    cv2.accumulateWeighted(gray, bg, self.config.md_bg_accum_weight)
                    crop = minX, minY, maxX, maxY
                    if self.config.md_mask:
                        crop, is_contained = self.apply_mask_to_crop(minX, minY, maxX, maxY)
                        if crop:
                            minX, minY, maxX, maxY = crop
                            motion_outside = not is_contained
                    if crop:
                        if maxX - minX > self.config.md_box_threshold_x and \
                                maxY - minY > self.config.md_box_threshold_y:
                            cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)
                            return (frame, crop, bg, motion_outside)
        else:
            bg = gray.copy().astype("float")

        cv2.accumulateWeighted(gray, bg, self.config.md_bg_accum_weight)
        return (frame, None, bg, motion_outside)

    def draw_masks(self, frame):
        if self.config.md_mask:
            xmin, ymin, xmax, ymax = self.config.md_mask
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (128, 0, 128), 1)

    def apply_mask_to_crop(self, minX, minY, maxX, maxY):
        is_contained = False

        mask_minx, mask_miny, mask_maxx, mask_maxy = self.config.md_mask
        if minX > mask_maxx or minY > mask_maxy or mask_minx > maxX or mask_miny > maxY:
            return None, is_contained
        else:
            #keep the part of crop, which is overlapped by mask
            minX = max(mask_minx, minX)
            maxX = min(mask_maxx, maxX)
            minY = max(mask_miny, minY)
            maxY = min(mask_maxy, maxY)

            if minX > mask_minx and minY > mask_miny and maxX < mask_maxx and maxY < mask_maxy:
                is_contained = True
        return (minX, minY, maxX, maxY), is_contained

    def detect_objects(self):
        total = 0
        bg = None

        fps = FPS(50, 100)

        # loop over frames from the video stream
        while True:
            frame = self.vs.read()
            if frame is not None:
                output_frame = frame.copy()
                if self.config.tf_apply_md:
                    output_frame, crop, bg, motion_outside = self.apply_md(output_frame, bg)
                    if self.config.door_movement_detection:
                        door_state = self.door_detector.detect_door_state(frame, self.config.door_detect_open_door_contour)
                        self.door_detector.add_door_state(door_state)
                        self.door_detector.add_motion_state(motion_outside)
                        if self.config.door_detect_show_detection:
                            minX, minY, maxX, maxY = self.config.door_detect_open_door_contour
                            cv2.rectangle(output_frame, (minX, minY), (maxX, maxY), (0, 255, 0), 1)
                            cv2.putText(output_frame, door_state.name, (minX, minY - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    if crop is not None:
                        minX, minY, maxX, maxY = crop
                        cropped_frame = frame[minY:maxY, minX:maxX]
                        self.od.add_task((frame, cropped_frame, (minX, minY)))
                else:
                    self.od.add_task((frame, frame, (0, 0), None))

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
                self.outputFrame.enqueue(output_frame)

            else:
                log.info("frame is NONE")

            if self.config.md_frame_rate > 0:
                time.sleep(1 / self.config.md_frame_rate)
            if self.config.debug_mode:
                ch = getch()
                if ch == 'q':
                    break

    def generate(self):
        self.active_video_feeds += 1
        current_feed_num = self.active_video_feeds
        # loop over frames from the output stream
        try:
            while True:
                if self.config.video_feed_fps > 0:
                    time.sleep(1 / self.config.video_feed_fps)
                output_frame = self.outputFrame.read()
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
        return jsonify(
            {
                'active_video_feeds': self.sd.active_video_feeds,
                'od_active_video_feeds': self.sd.od.active_video_feeds,
            }
        )

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

        return jsonify(self.config.__dict__)

    @route("/image")
    def image(self):
        (flag, encodedImage) = cv2.imencode(".jpg", self.sd.outputFrame.read())
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
    ap.add_argument("--log", type=str)
    args = vars(ap.parse_args())

    m = importlib.import_module(args["config"])
    config = getattr(m, "Config")()
    broker_q = SingletonBlockingQueue()
    notify_q = SingletonBlockingQueue()
    door_detector = None
    if config.door_movement_detection:
        door_detector = DoorMovementDetector(broker_q, config.door_detect_state_history_length)
    sd = StreamDetector(config, broker_q, door_detector)
    mb = Broker(sd.config, door_detector, broker_q, notify_q)

    log.info("flask init..")
    app = Flask(__name__)
    StreamDetectorView.register(app, init_argument=sd, route_base='/')
    f = threading.Thread(target=app.run, kwargs={'host': args["ip"], 'port': args["port"], 'debug': False,
                                                 'threaded': True, 'use_reloader': False})
    f.daemon = True
    f.start()

    log.info("start reading video file")
    t = sd.start()

    t.join()
    sd.cleanup()
    mb.stop()
