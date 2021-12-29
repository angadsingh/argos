from threading import Thread

import cv2

from lib.fps import FPS
from lib.task_queue import NonBlockingTaskSingleton

import logging
log = logging.getLogger(__name__)

class RTMPVideoStream:
    def __init__(self, rtmp_url, **kwargs):
        log.info("rtmp capture init START")
        self.vcap = cv2.VideoCapture(rtmp_url)
        log.info("rtmp capture init END")
        self.stopped = False
        self.fps = FPS(50, 100)
        self.output_frame = NonBlockingTaskSingleton(metric_prefix='rtmp_video_frame_q')

    def start(self):
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def update(self):
        total = 0
        debug_frame = 100
        while self.vcap.isOpened() and not self.stopped:
            ret, frame = self.vcap.read()
            if not ret:
                log.error('RTMP Stream ended!')
                break
            else:
                if total % debug_frame == 0:
                    log.debug("rtmp capturing..")
                self.output_frame.enqueue(frame)
                self.fps.count()
                total += 1

        self.stopped = True
        self.output_frame.enqueue(-1)
        self.vcap.release()
        self.fps.stop()

    def read(self):
        # return the frame most recently read
        frame = self.output_frame.dequeue()
        if frame is not -1:
            return frame

    def stop(self):
        self.stopped = True
