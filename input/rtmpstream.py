from threading import Thread

import cv2

from lib.fps import FPS
from lib.singleton_q import SingletonBlockingQueue

import logging
log = logging.getLogger(__name__)

class RTMPVideoStream:
    def __init__(self, rtmp_url, **kwargs):
        log.info("rtmp capture init START")
        self.vcap = cv2.VideoCapture(rtmp_url)
        log.info("rtmp capture init END")
        self.stopped = False
        self.fps = FPS(50, 100)
        self.frame_singleton = SingletonBlockingQueue()

    def start(self):
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def update(self):
        total = 0
        debug_frame = 100
        while (self.vcap.isOpened()):
            ret, frame = self.vcap.read()
            if total % debug_frame == 0:
                log.debug("rtmp capturing..")
            self.frame_singleton.enqueue(frame)
            self.fps.count()
            total += 1

            if self.stopped:
                self.vcap.release()
                return

    def read(self):
        # return the frame most recently read
        return self.frame_singleton.dequeue()

    def stop(self):
        self.stopped = True
