import logging
import subprocess
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
import numpy as np

log = logging.getLogger(__name__)


class MockNotifier(Notifier):
    def __init__(self, config, notify_q, expected_pattern):
        super().__init__(config, notify_q)
        self.expected_pattern = expected_pattern
        self.pattern_found = False

    def listen_notify_q(self):
        while True:
            task = self.notify_q.dequeue()
            if task == -1:
                break
            notification_type, notification_payload = task
            self.notification_handlers[notification_type](*notification_payload)
            if notification_type == NotificationTypes.PATTERN_DETECTED:
                (ptn, state_attrs) = notification_payload
                if ptn == self.expected_pattern:
                    log.info(colored("expected pattern found: %s" % str(self.expected_pattern), 'magenta', attrs=['bold']))
                    self.pattern_found = True
                    self.stopped = True

FIRST_RUN = True

class TestArgosStream(unittest.TestCase):
    @parameterized.expand([
        ["doorentering1.mov", MovementPatterns.PERSON_ENTERING_DOOR, 0],
        ["doorentering2.mov", MovementPatterns.PERSON_ENTERING_DOOR, 0],
        ["doorentering3.mov", MovementPatterns.PERSON_ENTERING_DOOR, 0],
        ["doorexiting1.mov", MovementPatterns.PERSON_EXITING_DOOR, 2],
        ["doorexiting2.mov", MovementPatterns.PERSON_EXITING_DOOR, 8],
        ["doorexiting3.mov", MovementPatterns.PERSON_EXITING_DOOR, 4],
        ["doorexiting4.mov", MovementPatterns.PERSON_EXITING_DOOR, 2],
        ["doorvisit.mp4", MovementPatterns.PERSON_VISITED_AT_DOOR, 0]
    ])
    def test_stream_detect(self, video_file, expected_pattern, time_pass):
        global FIRST_RUN
        config = config_test_stream.Config()
        config.input_mode = InputMode.VIDEO_FILE
        config.video_file_path = './data/door_pattern_videos/%s' % video_file

        broker_q = SingletonBlockingQueue()
        notify_q = SingletonBlockingQueue()
        mock_notifier = MockNotifier(config, notify_q, expected_pattern)

        pattern_detector = PatternDetector(broker_q, config.pattern_detection_pattern_steps,
                                           config.pattern_detection_patter_eval_order,
                                           config.pattern_detection_state_history_length,
                                           config.pattern_detection_state_history_length_partial,
                                           detection_interval=3)
        sd = StreamDetector(config, broker_q, pattern_detector)
        mb = Broker(sd.config, pattern_detector, broker_q, notify_q, notifier=mock_notifier)

        md_window_name = "motion detector"
        od_window_name = "object detector"

        if FIRST_RUN:
            blank_image = np.zeros(shape=[720, 1280, 3], dtype=np.uint8)
            cv2.namedWindow(md_window_name, cv2.WINDOW_KEEPRATIO)
            cv2.namedWindow(od_window_name, cv2.WINDOW_KEEPRATIO)
            cv2.imshow(md_window_name, blank_image)
            cv2.imshow(od_window_name, blank_image)
            cv2.resizeWindow(md_window_name, 870, 490)
            cv2.resizeWindow(od_window_name, 870, 490)
            cv2.moveWindow(md_window_name, 1048, 12)
            cv2.moveWindow(od_window_name, 1048, 524)
            subprocess.call(["/usr/bin/osascript", "-e", 'tell app "Finder" to set frontmost of process "Python" to true'])
            cv2.waitKey(1)
            FIRST_RUN = False

        sd.start()

        while not sd.vs.stopped:
            frame = sd.outputFrame.dequeue()
            cv2.imshow(md_window_name, frame)
            cv2.waitKey(1)

            od_frame = sd.od.output_frame.read(timeout=0.01)
            if od_frame is not None:
                cv2.imshow(od_window_name, od_frame)

        sd.wait_for_completion()
        # some patterns require the passage of time to process
        if time_pass > 0:
            log.info(colored("waiting a bit more for the pattern detector..", "yellow"))
            time.sleep(time_pass)
        sd.stop()
        mb.stop()
        pattern_detector.stop()
        self.assertTrue(mock_notifier.pattern_found)
