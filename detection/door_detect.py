import logging
import os
import time
from enum import Enum

import cv2
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cmc
from colormath.color_objects import sRGBColor, LabColor
from termcolor import colored

from lib.timer import RepeatedTimer
from notifier import NotificationTypes

log = logging.getLogger(__name__)

"""
    potential improvement with object tracking:
        https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
        https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
        https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
"""


class DoorStates(Enum):
    DOOR_CLOSED = 0
    DOOR_OPEN = 1


class ObjectStates(Enum):
    OBJECT_DETECTED = 1


class MotionStates(Enum):
    MOTION_OUTSIDE_MASK = 2
    MOTION_INSIDE_MASK = 3
    NO_MOTION = 4


class MovementPatterns(Enum):
    PERSON_EXITING_DOOR = 0
    PERSON_ENTERING_DOOR = 1


class StateHistoryStep:
    def __init__(self, state, state_attrs = None):
        self.ts = int(round(time.time()))
        self.state = state
        self.state_attrs = state_attrs

    def __repr__(self):
        return '%s[%s]' % (self.state, int(round(time.time())) - self.ts)


class DoorMovementDetector():
    MOTION_STATE_MAP = {
        False: MotionStates.MOTION_OUTSIDE_MASK,
        True: MotionStates.MOTION_INSIDE_MASK,
        None: MotionStates.NO_MOTION
    }

    MovementPatternSteps = {
        MovementPatterns.PERSON_EXITING_DOOR: [ObjectStates.OBJECT_DETECTED, DoorStates.DOOR_OPEN,
                                               DoorStates.DOOR_CLOSED],
        MovementPatterns.PERSON_ENTERING_DOOR: [DoorStates.DOOR_OPEN, ObjectStates.OBJECT_DETECTED,
                                                DoorStates.DOOR_CLOSED]
    }

    def __init__(self, output_q, state_history_length=20, detection_interval=1):
        self.output_q = output_q
        self.state_history = []
        self.last_door_state = None
        self.last_motion_state = None
        self.state_history_length = state_history_length
        self.pattern_detection_timer = RepeatedTimer(detection_interval, self.detect_door_movement)

    def detect_door_state(self, image, open_door_contour):
        minX, minY, maxX, maxY = open_door_contour
        img = image[minY:maxY, minX:maxX]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        average = img.mean(axis=0).mean(axis=0)
        average = (int(average[0]), int(average[1]), int(average[2]))
        avg_color = convert_color(sRGBColor(*average), LabColor)

        TARGET_COLORS = {DoorStates.DOOR_CLOSED: convert_color(sRGBColor(*(51, 32, 25)), LabColor),
                         DoorStates.DOOR_OPEN: convert_color(sRGBColor(*(151, 117, 72)), LabColor)}

        differences = [[delta_e_cmc(avg_color, target_value, pl=1, pc=1), target_name] for target_name, target_value in
                       TARGET_COLORS.items()]
        differences.sort()  # sorted by the first element of inner lists

        door_state = differences[0][1]
        return door_state

    def find_mov_ptn_in_state_history(self, pattern_steps, state_history):
        i = 0
        ptn_step_idx = 0
        while i < len(state_history):
            state_step = state_history[i]
            if state_step.state == pattern_steps[ptn_step_idx]:
                ptn_step_idx += 1
            if ptn_step_idx > len(pattern_steps) - 1:
                return True
            i += 1

        return False

    def detect_door_movement(self):
        self.prune_state_history()
        log.info(colored("stateHistory: %s" % str(self.state_history), 'white'))
        for (ptn, ptn_steps) in DoorMovementDetector.MovementPatternSteps.items():
            if self.find_mov_ptn_in_state_history(ptn_steps, self.state_history):
                log.info(colored("pattern detected: %s" % ptn.name, 'red', attrs=['bold']))
                state_attrs = None
                for state_step in self.state_history:
                    if state_step.state == ObjectStates.OBJECT_DETECTED:
                        state_attrs = state_step.state_attrs
                self.state_history.clear()
                self.output_q.enqueue((NotificationTypes.MOVEMENT_PATTERN_DETECTED, (ptn, state_attrs)))

    def prune_state_history(self):
        now = int(round(time.time()))
        for state_step in self.state_history:
            if now - state_step.ts  > self.state_history_length:
                self.state_history.remove(state_step)

    def add_door_state(self, door_state):
        if door_state != self.last_door_state:
            self.state_history.append(
                StateHistoryStep(door_state))
            self.last_door_state = door_state
            log.info(colored("door state changed: %s" % str(door_state), 'blue', attrs=['bold']))

    def add_motion_state(self, motion_outside):
        motion_state_label = DoorMovementDetector.MOTION_STATE_MAP[motion_outside]
        if motion_state_label != self.last_motion_state:
            self.state_history.append(
                StateHistoryStep(motion_state_label))
            self.last_motion_state = motion_state_label
            log.info(colored("motion state changed: %s" % str(motion_state_label), 'blue', attrs=['bold']))

    def add_object_state(self, label, accuracy, image_path):
        if len(self.state_history) == 0 or self.state_history[-1] is not ObjectStates.OBJECT_DETECTED:
            self.state_history.append(
                StateHistoryStep(ObjectStates.OBJECT_DETECTED, state_attrs=(label, accuracy, image_path)))
            log.info(colored("object state changed: %s" % str(ObjectStates.OBJECT_DETECTED), 'blue', attrs=['bold']))

    def _test_door_state_from_image_dir(self, images_dir, open_door_contour):
        for file in sorted(os.scandir(images_dir), key=lambda e: e.name, reverse=True):
            if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(images_dir, file.name)
                log.info(image_path)
                self._test_door_state_from_image(image_path, open_door_contour)

    def _test_door_state_from_image(self, image_path, open_door_contour):
        frame = cv2.imread(image_path)
        self._test_door_state_from_frame(frame, open_door_contour)

    def _test_door_state_from_frame(self, frame, open_door_contour):
        inferred_state = self.detect_door_state(frame, open_door_contour)
        log.info(inferred_state)
        cv2.imshow("image", frame)
        cv2.waitKey()

    def _test_door_state_from_video(self, video_file, open_door_contour):
        video = cv2.VideoCapture(video_file)
        while (video.isOpened()):
            ret, frame = video.read()
            if not ret:
                log.info('Reached the end of the video!')
                break
            self._test_door_state_from_frame(frame, open_door_contour)

    def _create_contour_from_video(self, video_file):
        video = cv2.VideoCapture(video_file)
        while (video.isOpened()):
            ret, frame = video.read()
            if not ret:
                break
            r = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            xmin = r[0]
            xmax = int(r[0] + r[2])
            ymin = int(r[1])
            ymax = int(r[1] + r[3])

            img = frame[ymin:ymax, xmin:xmax]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            average = img.mean(axis=0).mean(axis=0)
            average = (int(average[0]), int(average[1]), int(average[2]))
            log.info(str(average))


if __name__ == '__main__':
    file = 'detection/doordetecttestdata/door movement/doorentering4.mov'
    open_door_contour = (215, 114, 227, 123)
    DoorMovementDetector()._test_door_state_from_video(file, open_door_contour)
    # DoorMovementDetector()._create_contour_from_video(file)
