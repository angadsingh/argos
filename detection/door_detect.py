import logging
import os
import time
from enum import Enum

import cv2
import numpy as np
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


class NotState():
    def __init__(self, state, duration=np.inf):
        self.state = state
        self.duration = duration

    def __repr__(self):
        return 'NOT{%s(for %ss)}' % (self.state, self.duration)


class ObjectStates(Enum):
    OBJECT_DETECTED = 1


class MotionStates(Enum):
    MOTION_OUTSIDE_MASK = 2
    MOTION_INSIDE_MASK = 3
    NO_MOTION = 4


class MovementPatterns(Enum):
    PERSON_EXITING_DOOR = 0
    PERSON_ENTERING_DOOR = 1
    PERSON_VISITED_AT_DOOR = 2


class StateHistoryStep:
    def __init__(self, state, state_attrs=None, ts=None):
        if ts == None:
            ts =int(round(time.time()))
        self.ts = ts
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
        MovementPatterns.PERSON_VISITED_AT_DOOR: [ObjectStates.OBJECT_DETECTED, DoorStates.DOOR_OPEN,
                                                  DoorStates.DOOR_CLOSED, ObjectStates.OBJECT_DETECTED],
        MovementPatterns.PERSON_EXITING_DOOR: [ObjectStates.OBJECT_DETECTED, DoorStates.DOOR_OPEN,
                                               DoorStates.DOOR_CLOSED, NotState(ObjectStates.OBJECT_DETECTED, 5)],
        MovementPatterns.PERSON_ENTERING_DOOR: [NotState(ObjectStates.OBJECT_DETECTED),
                                                DoorStates.DOOR_OPEN, ObjectStates.OBJECT_DETECTED]
    }

    def __init__(self, output_q, state_history_length=20, detection_interval=1):
        self.output_q = output_q
        self.state_history = []
        self.last_door_state = None
        self.last_motion_state = None
        self.state_history_length = state_history_length
        if detection_interval:
            self.pattern_detection_timer = RepeatedTimer(detection_interval, self.detect_door_movement)

    def detect_door_state(self, image, open_door_contour, door_closed_avg_rgb, door_open_avg_rgb):
        minX, minY, maxX, maxY = open_door_contour
        img = image[minY:maxY, minX:maxX]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        average = img.mean(axis=0).mean(axis=0)
        average = (int(average[0]), int(average[1]), int(average[2]))
        avg_color = convert_color(sRGBColor(*average), LabColor)

        TARGET_COLORS = {DoorStates.DOOR_CLOSED: convert_color(sRGBColor(*door_closed_avg_rgb), LabColor),
                         DoorStates.DOOR_OPEN: convert_color(sRGBColor(*door_open_avg_rgb), LabColor)}

        differences = [[delta_e_cmc(avg_color, target_value, pl=1, pc=1), target_name] for target_name, target_value in
                       TARGET_COLORS.items()]
        differences.sort()  # sorted by the first element of inner lists

        door_state = differences[0][1]
        return door_state

    def find_mov_ptn_in_state_history(self, pattern_steps, state_history):
        i = 0
        ptn_step_idx = 0
        not_step_ts = None
        last_step_ts = None
        not_state: NotState = None
        while i < len(state_history):
            state_step = state_history[i]

            if not_state:
                if state_step.state == not_state.state:
                    not_step_ts = state_step.ts
                    i += 1

            if type(pattern_steps[ptn_step_idx]) == NotState:
                not_state = pattern_steps[ptn_step_idx]
                if state_step.state == not_state.state:
                    not_state_not_since = state_step.ts - last_step_ts if last_step_ts else np.inf
                    if not_state_not_since < not_state.duration:
                        ptn_step_idx = 0

                    not_step_ts = state_step.ts
                    i += 1
                ptn_step_idx += 1
            else:
                if state_step.state == pattern_steps[ptn_step_idx]:
                    last_step_ts = state_step.ts
                    if not_state:
                        not_state_not_since = state_step.ts - not_step_ts if not_step_ts else np.inf
                        if not_state_not_since < not_state.duration:
                            ptn_step_idx = 0
                            i += 1
                            not_state = None
                            not_step_ts = None
                            continue
                        else:
                            not_state = None
                            not_step_ts = None
                    ptn_step_idx += 1
                i += 1

            if ptn_step_idx > len(pattern_steps) - 1:
                if not_state:
                    ptn_step_idx -= 1
                    break
                else:
                    return True

        if ptn_step_idx == len(pattern_steps) - 1 and type(pattern_steps[ptn_step_idx]) == NotState:
            if last_step_ts:
                current_ts = int(round(time.time()))
                not_state_not_since = current_ts - last_step_ts
                if not_state_not_since >= pattern_steps[ptn_step_idx].duration:
                    return True

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
            if now - state_step.ts > self.state_history_length:
                # print("removing state %s from state_history" % state_step)
                self.state_history.remove(state_step)

    def add_door_state(self, door_state):
        if door_state != self.last_door_state:
            self.state_history.append(
                StateHistoryStep(door_state))
            self.last_door_state = door_state
            log.info(colored("door state changed: %s" % str(door_state), 'blue', attrs=['bold']))
            self.output_q.enqueue((NotificationTypes.DOOR_STATE_CHANGED, (door_state,)))

    def add_motion_state(self, motion_outside):
        motion_state_label = DoorMovementDetector.MOTION_STATE_MAP[motion_outside]
        if motion_state_label != self.last_motion_state:
            self.state_history.append(
                StateHistoryStep(motion_state_label))
            self.last_motion_state = motion_state_label
            log.info(colored("motion state changed: %s" % str(motion_state_label), 'blue', attrs=['bold']))
            self.output_q.enqueue((NotificationTypes.MOTION_STATE_CHANGED, (motion_state_label,)))

    def add_object_state(self, label, accuracy, image_path):
        if len(self.state_history) == 0 or self.state_history[-1] is not ObjectStates.OBJECT_DETECTED:
            self.state_history.append(
                StateHistoryStep(ObjectStates.OBJECT_DETECTED, state_attrs=(label, accuracy, image_path)))
            log.info(colored("object state changed: %s" % str(ObjectStates.OBJECT_DETECTED), 'blue', attrs=['bold']))

    def _test_door_state_from_image_dir(self, images_dir, open_door_contour, door_close_avg_rgb, door_open_avg_rgb):
        for file in sorted(os.scandir(images_dir), key=lambda e: e.name, reverse=True):
            if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(images_dir, file.name)
                print(image_path)
                self._test_door_state_from_image(image_path, open_door_contour, door_close_avg_rgb, door_open_avg_rgb)

    def _test_door_state_from_image(self, image_path, open_door_contour, door_close_avg_rgb, door_open_avg_rgb, show_result = True):
        frame = cv2.imread(image_path)
        return self._test_door_state_from_frame(frame, open_door_contour, door_close_avg_rgb, door_open_avg_rgb, show_result)

    def _test_door_state_from_frame(self, frame, open_door_contour, door_close_avg_rgb, door_open_avg_rgb, show_result = True):
        inferred_state = self.detect_door_state(frame, open_door_contour, door_close_avg_rgb, door_open_avg_rgb, )

        if show_result:
            print(inferred_state)
            minX, minY, maxX, maxY = open_door_contour
            cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 255, 0), 1)
            cv2.putText(frame, inferred_state.name, (minX, minY - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            cv2.imshow("image", frame)
            cv2.waitKey()

        return inferred_state

    def _test_door_state_from_video(self, video_file, open_door_contour, door_close_avg_rgb, door_open_avg_rgb):
        video = cv2.VideoCapture(video_file)
        while (video.isOpened()):
            ret, frame = video.read()
            if not ret:
                print('Reached the end of the video!')
                break
            self._test_door_state_from_frame(frame, open_door_contour, door_close_avg_rgb, door_open_avg_rgb)

    def _create_contour_from_frame(self, frame):
        r = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        xmin = r[0]
        xmax = int(r[0] + r[2])
        ymin = int(r[1])
        ymax = int(r[1] + r[3])

        img = frame[ymin:ymax, xmin:xmax]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        average = img.mean(axis=0).mean(axis=0)
        average = (int(average[0]), int(average[1]), int(average[2]))
        print(str(average))

    def _create_contour_from_video(self, video_file):
        video = cv2.VideoCapture(video_file)
        while (video.isOpened()):
            ret, frame = video.read()
            if not ret:
                break
            self._create_contour_from_frame(frame)

    def _create_contour_from_image(self, image_path):
        frame = cv2.imread(image_path)
        return self._create_contour_from_frame(frame)

if __name__ == '__main__':
    # file = 'detection/doordetecttestdata/door movement/doorentering4.mov'
    file = '../tests/door_state_test_images/doorclosed_night2.jpg'
    # open_door_contour = (215, 114, 227, 123)
    # DoorMovementDetector(None, detection_interval=None)._test_door_state_from_image(file, open_door_contour)
    # DoorMovementDetector(None, detection_interval=None)._test_door_state_from_video(file, open_door_contour)
    # DoorMovementDetector(None, detection_interval=None)._create_contour_from_video(file)

    DoorMovementDetector(None, detection_interval=None)._create_contour_from_image(file)
