from threading import Thread

import cv2

from lib.fps import FPS
from lib.blocking_q import BlockingQueue
import logging
log = logging.getLogger(__name__)

class VideoFileStream:
    def __init__(self, file, **kwargs):
        self.vcap = cv2.VideoCapture(file)
        self.stopped = False
        self.fps = FPS(50, 100)
        self.frame_singleton = BlockingQueue()

    def start(self):
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def update(self):
        while self.vcap.isOpened() and not self.stopped:
            ret, frame = self.vcap.read()
            if not ret:
                log.info('Reached the end of the video!')
                break
            else:
                self.frame_singleton.enqueue(frame, wait=True)
                self.fps.count()

        self.stopped = True
        self.vcap.release()
        self.fps.stop()

    def read(self):
        # return the frame most recently read
        return self.frame_singleton.dequeue(notify=True)

    def stop(self):
        self.stopped = True
        self.t.join()
