import time
import random

from lib.fps import FPS


class FrameLimiter():
    def __init__(self, fps):
        self.sleep_time = 1.0/fps if fps > 0 else 0
        self.last_run_ts = None

    def limit(self):
        if self.last_run_ts:
            code_time = time.time()-self.last_run_ts
            time.sleep(max (self.sleep_time - code_time, 0))
        self.last_run_ts = time.time()
        return True

if __name__ == '__main__':
    fps = FPS(50, 100)
    limiter = FrameLimiter(1)
    while limiter.limit():
        # do some artificial work which takes time
        time.sleep(random.random()*1)
        print("running with limiter at %f fps" % fps.fps)
        fps.count()

