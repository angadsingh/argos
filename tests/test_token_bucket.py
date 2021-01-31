import time

import token_bucket

from lib.fps import FPS

desired_fps = 0.4323
storage = token_bucket.MemoryStorage()
limiter = token_bucket.Limiter(desired_fps*10, 10, storage)

fps = FPS()
while True:
    if (limiter.consume('global', 10)):
        fps.count()
        print(str(fps.fps))