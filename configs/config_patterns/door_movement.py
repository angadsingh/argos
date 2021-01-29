from enum import Enum

from detection.state_managers.door_state_manager import DoorStates
from detection.state_managers.object_state_manager import ObjectStates
from detection.states import NotState

"""
    pattern config to detect door movement
    this is just an example pattern that this project started out with
    you can define your own patterns using the states available!
    
    see the doc of `PatternDetector.find_mov_ptn_in_state_history()` to see how
    all this works with examples.
"""


class MovementPatterns(Enum):
    PERSON_EXITING_DOOR = 0
    PERSON_ENTERING_DOOR = 1
    PERSON_VISITED_AT_DOOR = 2
    TEST = 3


pattern_steps = [
    (MovementPatterns.PERSON_VISITED_AT_DOOR, [ObjectStates.OBJECT_DETECTED, DoorStates.DOOR_OPEN,
                                               DoorStates.DOOR_CLOSED, ObjectStates.OBJECT_DETECTED]),
    (MovementPatterns.PERSON_EXITING_DOOR, [ObjectStates.OBJECT_DETECTED, DoorStates.DOOR_CLOSED,
                                            NotState(ObjectStates.OBJECT_DETECTED, 5)]),
    (MovementPatterns.PERSON_ENTERING_DOOR, [NotState(ObjectStates.OBJECT_DETECTED, 5),
                                             DoorStates.DOOR_OPEN, ObjectStates.OBJECT_DETECTED])
]
