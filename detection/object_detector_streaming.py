from datetime import datetime
import pascal_voc_writer

import cv2

from detection.object_detector_base import BaseTFObjectDetector
from lib.singleton_q import SingletonBlockingQueue
from notifier import NotificationTypes


class StreamingTFObjectDetector(BaseTFObjectDetector):
    def __init__(self, config, output_q):
        super().__init__(config)
        self.output_q = output_q
        self.output_frame = SingletonBlockingQueue()
        self.active_video_feeds = 0

    def process_detection_intermeddiate(self, frame, orig_box, image_path):
        minx, miny, maxx, maxy, label, accuracy = orig_box
        outputFrame = self.tf_detector.DisplayDetection(frame, orig_box)
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
        self.output_frame.enqueue(outputFrame)

    def process_detection_final(self, label, accuracy, image_path):
        self.output_q.enqueue((NotificationTypes.OBJECT_DETECTED, (label, accuracy, image_path)))

    def generate_output_frames(self):
        self.active_video_feeds += 1
        try:
            while True:
                outputFrame = self.output_frame.read(timeout=2)
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
