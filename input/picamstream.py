import logging
import time
from threading import Thread

from picamera import PiCamera
from picamera.array import PiRGBArray

from lib.fps import FPS
from lib.blocking_q import BlockingQueue

log = logging.getLogger(__name__)

class PiVideoStream:
    def __init__(self, resolution=(320, 240), framerate=32, format='rgb', **kwargs):
        # initialize the camera
        self.camera = PiCamera()

        # set camera parameters
        self.camera.resolution = resolution
        self.camera.framerate = framerate

        # set optional camera parameters (refer to PiCamera docs)
        for (arg, value) in kwargs.items():
            setattr(self.camera, arg, value)

        # initialize the stream
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                                                     format=format, use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.stopped = False

        self.fps = FPS(50, 100)
        self.frame_singleton = BlockingQueue()

    def start(self):
        # start the thread to read frames from the video stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        time.sleep(2.0)
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        try:
            for f in self.stream:
                # grab the frame from the stream and clear the stream in
                # preparation for the next frame
                self.frame_singleton.enqueue(f.array)
                self.fps.count()
                self.rawCapture.truncate(0)

                if self.stopped:
                    break
        except Exception as e:
            log.error(e)
            self.stopped = True

        self.frame_singleton.enqueue(-1)
        self.stream.close()
        self.rawCapture.close()
        self.camera.close()
        self.fps.stop()

    def read(self):
        # return the frame most recently read
        frame = self.frame_singleton.dequeue()
        if frame is not -1:
            return frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
