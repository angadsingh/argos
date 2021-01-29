import logging
import time
from threading import Thread

import cv2
from termcolor import colored

from lib.blocking_q import BlockingQueue
from lib.fps import FPS

log = logging.getLogger(__name__)


class VideoFileStream:
    def __init__(self, file, in_sync=True, **kwargs):
        self.vcap = cv2.VideoCapture(file)
        self.video_fps = int(self.vcap.get(cv2.CAP_PROP_FPS))
        self.stopped = False
        self.fps = FPS(50, 100)
        # set to True to process all frames
        # if False will run at real pace of the video
        # irrespective of how fast/slow the consumer is
        self.in_sync = in_sync
        if not self.in_sync:
            log.info("playing at original video fps of file [%s]: %s" % (file, self.video_fps))
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
                log.info(colored('Reached the end of the video! [%s seconds]' % str(round(int(
                    self.vcap.get(cv2.CAP_PROP_FRAME_COUNT)) / self.video_fps)), 'magenta', attrs=['bold']))
                break
            else:
                self.frame_singleton.enqueue(frame, wait=self.in_sync)
                self.fps.count()
                if not self.in_sync:
                    time.sleep(1 / self.video_fps)

        self.stopped = True
        self.vcap.release()
        self.fps.stop()

    def read(self):
        # return the frame most recently read
        return self.frame_singleton.dequeue(notify=True)

    def stop(self):
        self.stopped = True
        self.t.join()
