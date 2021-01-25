import logging
from enum import Enum

from termcolor import colored

from detection.state_managers.state_manager import StateManager
from detection.states import StateHistoryStep

log = logging.getLogger(__name__)


class ObjectStates(Enum):
    OBJECT_DETECTED = 1


class ObjectStateManager(StateManager):
    def __init__(self, state_history, output_q):
        super().__init__(state_history, output_q)
        self.last_door_state = None

    def add_state(self, state):
        (label, accuracy, image_path) = state
        if len(self.state_history) == 0 or self.state_history[-1].state is not ObjectStates.OBJECT_DETECTED:
            self.state_history.append(
                StateHistoryStep(ObjectStates.OBJECT_DETECTED, state_attrs=(label, accuracy, image_path)))
            log.info(colored("object state changed: %s" % str(ObjectStates.OBJECT_DETECTED), 'blue', attrs=['bold']))
