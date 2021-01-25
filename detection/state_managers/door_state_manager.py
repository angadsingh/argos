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
    def __init__(self, state_history, output_q):
        super().__init__(state_history, output_q)
        self.last_door_state = None

    def add_state(self, door_state):
        if door_state != self.last_door_state:
            self.state_history.append(
                StateHistoryStep(door_state))
            self.last_door_state = door_state
            log.info(colored("door state changed: %s" % str(door_state), 'blue', attrs=['bold']))
            self.output_q.enqueue((NotificationTypes.DOOR_STATE_CHANGED, (door_state,)))
