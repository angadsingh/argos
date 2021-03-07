import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pascal_voc_writer

from detection.object_detector_base import BaseTFObjectDetector
from detection.state_managers.state_manager import CommittedOffset
from lib.fps import FPS
from lib.framelimiter import FrameLimiter
from lib.task_queue import BlockingTaskSingleton, NonBlockingTaskSingleton, BlockingTaskQueue
from notifier import NotificationTypes


class StreamingTFObjectDetector(BaseTFObjectDetector):
    def __init__(self, config, broker_q: BlockingTaskSingleton):
        super().__init__(config)
        self.input_frame_q = BlockingTaskQueue(max_size=self.config.od_task_q_size, metric_prefix='object_detector')
        self.fps = FPS(600, 100)

        self.latest_committed_offset = CommittedOffset.CURRENT
        self.broker_q = broker_q
        self.output_video_frame_q = NonBlockingTaskSingleton()
        self.active_video_feeds = 0

    def start(self):
        self.t = threading.Thread(target=self.detect_continuously, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def stop(self):
        self.input_frame_q.abrupt_stop(-1)
        if self.t.is_alive():
            self.t.join()
        self.fps.stop()

    def is_alive(self):
        return self.t.is_alive()

    def add_task(self, task):
        self.input_frame_q.enqueue(task)

    def detect_continuously(self):
        self.initialize_tf_model()

        limiter = FrameLimiter(self.config.od_frame_rate)
        while True:
            task = self.input_frame_q.dequeue()
            if task == -1:
                break
            (frame, cropped_frame, (cropOffsetX, cropOffsetY), ts) = task
            if not self.task_skipper.skip_task(ts):
                limiter.limit()
                self.fps.count()
                cropped_frame = np.copy(cropped_frame)
                cropped_frame.setflags(write=1)
                self.detect_image_buffered(frame, cropped_frame, cropOffsetX, cropOffsetY, ts)
            self.latest_committed_offset = ts

    def detect_image_buffered(self, frame, cropped_frame, cropOffsetX, cropOffsetY, ts):
        if not ts:
            ts = time.time()
        det_boxes = self.detect_image(cropped_frame)
        if det_boxes is not None and len(det_boxes) > 0:
            for box in det_boxes:
                minx, miny, maxx, maxy, label, accuracy = box
                image_path = "%s/detection_%s_%s.jpg" % (
                    self.config.tf_output_detection_path, label,
                    datetime.fromtimestamp(ts).strftime("%d-%m-%Y-%H-%M-%S-%f"))
                orig_box = minx + cropOffsetX, miny + cropOffsetY, maxx + cropOffsetX, maxy + cropOffsetY, label, accuracy
                self.config.tf_detection_buffer.add_detection((orig_box, image_path))
                self.process_detection_intermediate(frame, orig_box, image_path)

            label, accuracy, image_path = self.config.tf_detection_buffer.get_max_accuracy_label()
            self.process_detection_final(label, accuracy, image_path, ts)
            return label, accuracy

        return None, None

    def process_detection_intermediate(self, frame, orig_box, image_path):
        minx, miny, maxx, maxy, label, accuracy = orig_box
        outputFrame = frame.copy()
        if self.config.od_blur_output_frame:
            outputFrame = cv2.blur(outputFrame, (80, 80))
        outputFrame = self.tf_detector.DisplayDetection(outputFrame, orig_box)
        if self.config.tf_od_frame_write:
            cv2.imwrite(image_path,
                        outputFrame)
        if self.config.tf_od_annotation_write:
            imH, imW, _ = frame.shape
            writer = pascal_voc_writer.Writer(image_path, imW, imH)
            minX, minY, maxX, maxY, klass, confidence = orig_box
            writer.addObject(klass, minX, minY, maxX, maxY)
            writer.save(image_path.replace('.jpg', '.xml'))
        if self.config.show_fps:
            cv2.putText(outputFrame, "%.2f fps" % self.fps.fps, (10, outputFrame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        self.output_video_frame_q.enqueue(outputFrame)

    def process_detection_final(self, label, accuracy, image_path, ts):
        self.broker_q.enqueue((NotificationTypes.OBJECT_DETECTED, ((label, accuracy, image_path), ts)))

    def generate_output_frames(self):
        self.active_video_feeds += 1
        try:
            while True:
                outputFrame = self.output_video_frame_q.read(timeout=2)
                if outputFrame is not None:
                    # encode the frame in JPEG format
                    (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
                    # ensure the frame was successfully encoded
                    if not flag:
                        continue
                    # yield the output frame in the byte format
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                           bytearray(encodedImage) + b'\r\n')
        finally:
            self.active_video_feeds -= 1
