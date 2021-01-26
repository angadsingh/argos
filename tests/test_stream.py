import logging
import time
import unittest

import cv2
from parameterized import parameterized
from termcolor import colored

from broker import Broker
from configs.config_patterns.door_movement import MovementPatterns
from configs.constants import InputMode
from detection.pattern_detector import PatternDetector
from lib.singleton_q import SingletonBlockingQueue
from notifier import Notifier, NotificationTypes
from stream import StreamDetector
from tests import config_test_stream

log = logging.getLogger(__name__)


class MockNotifier(Notifier):
    def __init__(self, config, notify_q, expected_pattern):
        super().__init__(config, notify_q)
        self.expected_pattern = expected_pattern
        self.pattern_found = False

    def listen_notify_q(self):
        while not self.stopped:
            notification_type, notification_payload = self.notify_q.dequeue()
            self.notification_handlers[notification_type](*notification_payload)
            if notification_type == NotificationTypes.PATTERN_DETECTED:
                (ptn, state_attrs) = notification_payload
                if ptn == self.expected_pattern:
                    log.info(colored("expected pattern found: %s" % str(self.expected_pattern), 'magenta', attrs=['bold']))
                    self.pattern_found = True
                    self.stopped = True


class TestArgosStream(unittest.TestCase):
    @parameterized.expand([
        ["doorentering1.mov", MovementPatterns.PERSON_ENTERING_DOOR, 0],
        ["doorentering2.mov", MovementPatterns.PERSON_ENTERING_DOOR, 0],
        ["doorentering3.mov", MovementPatterns.PERSON_ENTERING_DOOR, 0],
        ["doorexiting1.mov", MovementPatterns.PERSON_EXITING_DOOR, 0],
        ["doorexiting2.mov", MovementPatterns.PERSON_EXITING_DOOR, 8],
        ["doorexiting3.mov", MovementPatterns.PERSON_EXITING_DOOR, 0],
        ["doorexiting4.mov", MovementPatterns.PERSON_EXITING_DOOR, 0],
        ["doorvisit.mp4", MovementPatterns.PERSON_VISITED_AT_DOOR, 0]
    ])
    def test_stream_detect(self, video_file, expected_pattern, time_pass):
        config = config_test_stream.Config()
        config.input_mode = InputMode.VIDEO_FILE
        config.video_file_path = './data/door_pattern_videos/%s' % video_file

        broker_q = SingletonBlockingQueue()
        notify_q = SingletonBlockingQueue()
        mock_notifier = MockNotifier(config, notify_q, expected_pattern)

        pattern_detector = PatternDetector(broker_q, config.pattern_detection_pattern_steps,
                                           config.pattern_detection_patter_eval_order,
                                           config.pattern_detection_state_history_length,
                                           config.pattern_detection_state_history_length_partial)
        sd = StreamDetector(config, broker_q, pattern_detector)
        mb = Broker(sd.config, pattern_detector, broker_q, notify_q, notifier=mock_notifier)

        sd_thread = sd.start()

        while not sd.vs.stopped:
            frame = sd.outputFrame.dequeue()
            cv2.imshow("image", frame)
            cv2.waitKey(1)

        sd_thread.join()
        time.sleep(time_pass)
        sd.cleanup()
        self.assertTrue(mock_notifier.pattern_found)
