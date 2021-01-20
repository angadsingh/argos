import time

from lib.timer import RepeatedTimer


class FPS:
    # the smaller the bucket_size_millis the more precise
    # window length is num_buckets * bucket_size_millis
    # e.g. default is 5 seconds
    def __init__(self, num_buckets=50, bucket_size_millis=100):
        self.num_buckets = num_buckets
        self.sliding_total = 0
        self.bucket_frames = []
        for i in range(0, num_buckets):
            self.bucket_frames.append(0)

        self.start_time = None
        self.fps = 0.0
        self.bucket_size_millis = bucket_size_millis
        self.bucket_head = 0

        self.fps_update_timer = RepeatedTimer(1, self._update_fps)
        self.filled_length = 0

    def _update_fps(self):
        if self.filled_length > 0:
            self.fps = (1000 / self.bucket_size_millis) * self.sliding_total / self.filled_length

    def _shift_bucket_frames(self, bucket):
        if bucket - self.bucket_head > self.num_buckets - 1:
            num_shifts = bucket - self.bucket_head - (self.num_buckets - 1)
            for i in range(0, num_shifts):
                self.sliding_total -= self.bucket_frames.pop(0)
                self.bucket_frames.append(0)
                self.bucket_head += 1

    def count(self):
        if self.start_time == None:
            self.start_time = int(round(time.time() * 1000))

        elapsed_millis = int(round(time.time() * 1000)) - self.start_time
        bucket = int(round(elapsed_millis / self.bucket_size_millis))

        self._shift_bucket_frames(bucket)
        self.bucket_frames[bucket - self.bucket_head] += 1
        self.sliding_total += 1

        self.filled_length = bucket - self.bucket_head + 1
