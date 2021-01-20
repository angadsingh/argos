import unittest

from detection.door_detect import DoorMovementDetector, MotionStates, DoorStates


class TestDoorDetect(unittest.TestCase):

    def test_find_mov_ptn_state_history(self):
        mov_ptn = [MotionStates.MOTION_OUTSIDE_MASK, DoorStates.DOOR_OPEN, DoorStates.DOOR_CLOSED]
        state_history = [MotionStates.MOTION_INSIDE_MASK,
                         MotionStates.MOTION_INSIDE_MASK,
                         MotionStates.MOTION_OUTSIDE_MASK,
                         MotionStates.MOTION_INSIDE_MASK,
                         DoorStates.DOOR_OPEN,
                         MotionStates.MOTION_INSIDE_MASK,
                         DoorStates.DOOR_CLOSED,
                         MotionStates.MOTION_INSIDE_MASK]
        self.assertTrue(DoorMovementDetector().find_mov_ptn_in_state_history(mov_ptn, state_history))