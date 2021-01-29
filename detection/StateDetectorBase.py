import logging
from abc import ABC, abstractmethod

from termcolor import colored

log = logging.getLogger(__name__)

class SkipAheadOptimizer(ABC):
    """
    skip-ahead optimizer for lagging state detectors
    """
    def __init__(self, skip_state_type):
        self.skip_state_type = skip_state_type
        self.total_frames = 0
        self.total_skipped = 0

    def measure_speedup(self, skip_ahead):
        self.total_frames += 1
        if skip_ahead:
            self.total_skipped += 1

        speedup = round((self.total_skipped / self.total_frames) * 100, 1)
        log.info(colored("%s detector speedup: %d%% (%d/%d)" % (self.skip_state_type, speedup, self.total_skipped, self.total_frames), 'white'))

    @abstractmethod
    def skip_task(self, ts):
        pass

class DefaultSkipAheadOptimizer(SkipAheadOptimizer):
    def __init__(self):
        super().__init__(None)

    def skip_task(self, ts):
        return False

class StateDetectorBase(ABC):
    def __init__(self):
        self.task_skipper = DefaultSkipAheadOptimizer()