import unittest

from parameterized import parameterized

from detection.door_state_detectors import SingleShotDoorStateDetector, AdaptiveDoorStateDetector, \
    SingleShotFrameDiffDoorStateDetector
from detection.state_managers.door_state_manager import DoorStates
from tests.door_state_detect_tool import DoorStateDetectTools


class TestDoorStateDetectors(unittest.TestCase):
    DOOR_STATE_CONTOUR = (215, 114, 227, 123)
    DOOR_FRAME_CONTOUR = (196, 131, 215, 147)
    door_state_detector_door_close_avg_rgb = (118, 80, 26)
    door_state_detector_door_open_avg_rgb = (151, 117, 72)

    def _get_pruned_results(self, actual_results):
        pruned_actual_results = []
        prev_result = None
        for result in actual_results:
            if not prev_result:
                prev_result = result[1]
                pruned_actual_results.append(result)
            if result[1] != prev_result:
                pruned_actual_results.append(result)
                prev_result = result[1]
        return pruned_actual_results

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
    def test_single_shot_door_state_detector(self, image, exp_door_state):
        full_image_path = "./data/door_state_test_images/%s" % image
        detector = SingleShotDoorStateDetector(TestDoorStateDetectors.DOOR_STATE_CONTOUR,
                                               TestDoorStateDetectors.door_state_detector_door_close_avg_rgb,
                                               TestDoorStateDetectors.door_state_detector_door_open_avg_rgb)
        actual_result = DoorStateDetectTools(detector, show_result=False)._door_state_from_image(full_image_path)

        print("image/actual/expected: %s/%s/%s" % (image, actual_result, exp_door_state))
        self.assertEqual(
            exp_door_state,
            actual_result)

    @parameterized.expand([
        ["doorstatetest.mov", [(0, DoorStates.DOOR_CLOSED),
                               (163, DoorStates.DOOR_OPEN),
                               (247, DoorStates.DOOR_CLOSED)]]
    ])
    def test_adaptive_door_state_detector(self, video_file, exp_door_states):
        full_video_path = "./data/door_state_test_videos/%s" % video_file
        detector = AdaptiveDoorStateDetector(TestDoorStateDetectors.DOOR_STATE_CONTOUR,
                                             (DoorStates.DOOR_CLOSED, DoorStates.DOOR_OPEN))
        actual_results = DoorStateDetectTools(detector, show_result=False)._door_state_from_video(full_video_path)
        pruned_actual_results = self._get_pruned_results(actual_results)

        print("video/actual/expected: %s/%s/%s" % (video_file, pruned_actual_results, exp_door_states))
        self.assertEqual(
            exp_door_states,
            pruned_actual_results)

    @parameterized.expand([
        ["doorstatetest_dawn1.mp4", [(0, DoorStates.DOOR_CLOSED)]],
        ["doorstatetest_dawn2.mp4", [(0, DoorStates.DOOR_CLOSED)]],
        ["doorstatetest.mov", [(0, DoorStates.DOOR_CLOSED), (163, DoorStates.DOOR_OPEN), (247, DoorStates.DOOR_CLOSED)]]
    ])
    def test_single_shot_frame_diff_door_state_detector_video(self, video_file, exp_door_states):
        full_video_path = "./data/door_state_test_videos/%s" % video_file
        detector = SingleShotFrameDiffDoorStateDetector(
            TestDoorStateDetectors.DOOR_STATE_CONTOUR,
            TestDoorStateDetectors.DOOR_FRAME_CONTOUR)
        actual_results = DoorStateDetectTools(detector, show_result=False)._door_state_from_video(full_video_path)
        pruned_actual_results = self._get_pruned_results(actual_results)

        print("video/actual/expected: %s/%s/%s" % (video_file, pruned_actual_results, exp_door_states))
        self.assertEqual(
            exp_door_states,
            pruned_actual_results)

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
    def test_single_shot_frame_diff_door_state_detector_image(self, image, exp_door_state):
        full_image_path = "./data/door_state_test_images/%s" % image
        detector = SingleShotFrameDiffDoorStateDetector(
            TestDoorStateDetectors.DOOR_STATE_CONTOUR,
            TestDoorStateDetectors.DOOR_FRAME_CONTOUR)
        actual_result = DoorStateDetectTools(detector, show_result=False)._door_state_from_image(full_image_path)

        print("image/actual/expected: %s/%s/%s" % (image, actual_result, exp_door_state))
        self.assertEqual(
            exp_door_state,
            actual_result)
