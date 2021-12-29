import logging
import time
from enum import Enum

from termcolor import colored

from detection.object_detector_streaming import StreamingTFObjectDetector
from detection.pattern_detector_task_skipper import PatternBasedSkipAheadOptimizer
from detection.state_managers.state_manager import StateManager
from detection.states import StateHistoryStep

log = logging.getLogger(__name__)


class ObjectStates(Enum):
    OBJECT_DETECTED = 1


class ObjectStateManager(StateManager):
    def __init__(self, object_detector: StreamingTFObjectDetector, pattern_detector, broker_q):
        super().__init__(pattern_detector, broker_q)
        self.object_detector = object_detector
        self.object_detector.task_skipper = PatternBasedSkipAheadOptimizer(pattern_detector, ObjectStates.OBJECT_DETECTED)

    def add_state(self, state, ts = None):
        (label, accuracy, image_path) = state
        history_step = StateHistoryStep(ObjectStates.OBJECT_DETECTED, state_attrs=(label, accuracy, image_path), ts=ts)
        added = self.pattern_detector.add_to_state_history(
            history_step, avoid_duplicates = True)
        if added:
            log.info(colored("object state changed: %s" % history_step, 'blue', attrs=['bold']))

    def get_latest_committed_offset(self):
        return self.object_detector.latest_committed_offset

    def get_current_lag(self):
        return self.object_detector.input_frame_q.size()