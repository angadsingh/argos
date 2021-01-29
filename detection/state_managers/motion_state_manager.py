import logging
from enum import Enum

from termcolor import colored

from detection.state_managers.state_manager import StateManager
from detection.states import StateHistoryStep
from notifier import NotificationTypes

log = logging.getLogger(__name__)


class MotionStates(Enum):
    MOTION_OUTSIDE_MASK = 2
    MOTION_INSIDE_MASK = 3
    NO_MOTION = 4


class MotionStateManager(StateManager):
    MOTION_STATE_MAP = {
        False: MotionStates.MOTION_OUTSIDE_MASK,
        True: MotionStates.MOTION_INSIDE_MASK,
        None: MotionStates.NO_MOTION
    }

    def __init__(self, pattern_detector, output_q):
        super().__init__(pattern_detector, output_q)
        self.last_motion_state = None

    def add_state(self, state, ts = None):
        motion_state_label = MotionStateManager.MOTION_STATE_MAP[state]
        if motion_state_label != self.last_motion_state:
            history_step = StateHistoryStep(motion_state_label, ts=ts)
            self.pattern_detector.add_to_state_history(history_step)
            self.last_motion_state = motion_state_label
            self.output_q.enqueue((NotificationTypes.MOTION_STATE_CHANGED, (motion_state_label,)), wait=True)
            log.info(colored("motion state changed: %s" % history_step, 'blue', attrs=['bold']))