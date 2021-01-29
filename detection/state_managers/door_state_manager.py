import logging
from enum import Enum

from termcolor import colored

from detection.state_managers.state_manager import StateManager
from detection.states import StateHistoryStep
from notifier import NotificationTypes

log = logging.getLogger(__name__)


class DoorStates(Enum):
    DOOR_CLOSED = 0
    DOOR_OPEN = 1


class DoorStateManager(StateManager):
    def __init__(self, pattern_detector, output_q):
        super().__init__(pattern_detector, output_q)
        self.last_door_state = None

    def add_state(self, door_state, ts = None):
        if door_state != self.last_door_state:
            history_step = StateHistoryStep(door_state, ts=ts)
            self.pattern_detector.add_to_state_history(history_step)
            self.last_door_state = door_state
            self.output_q.enqueue((NotificationTypes.DOOR_STATE_CHANGED, (door_state,)), wait=True)
            log.info(colored("door state changed: %s" % history_step, 'blue', attrs=['bold']))
