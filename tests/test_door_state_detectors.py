import unittest

from parameterized import parameterized

from detection.state_managers.door_state_manager import DoorStates
from tests.door_state_detect_tool import DoorStateDetectTools


class TestDoorDetectState(unittest.TestCase):
    DOOR_STATE_CONTOUR = (215, 114, 227, 123)
    door_state_detector_door_close_avg_rgb = (118, 80, 26)
    door_state_detector_door_open_avg_rgb = (151, 117, 72)

    @parameterized.expand([
        ["doorclosed_day.jpg", DoorStates.DOOR_CLOSED],
        ["doorclosed_night.jpg", DoorStates.DOOR_CLOSED],
        ["doorclosed_night2.jpg", DoorStates.DOOR_CLOSED],
        ["doorclosed_night3.jpg", DoorStates.DOOR_CLOSED],
        ["dooropen_day.jpg", DoorStates.DOOR_OPEN],
        ["dooropen_night.jpg", DoorStates.DOOR_OPEN],
        ["dooropen_night2.jpg", DoorStates.DOOR_OPEN],
        ["dooropen_night3.jpg", DoorStates.DOOR_OPEN]
    ])
    def test_door_state(self, image, exp_door_state):
        full_image_path = "./door_state_test_images/%s" % image
        actual_result = DoorStateDetectTools()._door_state_from_image(full_image_path,
                                                                    TestDoorDetectState.DOOR_STATE_CONTOUR,
                                                                    TestDoorDetectState.door_state_detector_door_close_avg_rgb,
                                                                    TestDoorDetectState.door_state_detector_door_open_avg_rgb,
                                                                    show_result=False)
        print("image/actual/expected: %s/%s/%s" % (image, actual_result, exp_door_state))
        self.assertEqual(
            exp_door_state,
            actual_result)
