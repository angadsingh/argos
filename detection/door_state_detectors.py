from abc import ABC, abstractmethod

import cv2
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cmc
from colormath.color_objects import sRGBColor, LabColor

from detection.state_managers.door_state_manager import DoorStates


class DoorStateDetector(ABC):
    def __init__(self, open_door_contour):
        """
        :param open_door_contour: the box where the door state can be detected based on color similarity
        keep it at the corner of the door where closed state will be the color of
        the wood of the door and open state will be the road/lobby behind the door
        """
        self.open_door_contour = open_door_contour

    def show_detection(self, output_frame, detection):
        minX, minY, maxX, maxY = self.open_door_contour
        cv2.rectangle(output_frame, (minX, minY), (maxX, maxY), (0, 255, 0), 1)
        cv2.putText(output_frame, detection.name, (minX, minY - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 255, 0), 1)

    @abstractmethod
    def detect_door_state(self, frame):
        pass


class SingleShotDoorStateDetector(DoorStateDetector):
    def __init__(self, open_door_contour, door_closed_avg_rgb, door_open_avg_rgb):
        """
        this door state detector uses the color similarity from the below colors to determine door state

        :param door_closed_avg_rgb: the average rgb colors of the closed state of the dour contour
        :param door_open_avg_rgb: the average rgb colors of the open state of the dour contour
        """
        super().__init__(open_door_contour)
        self.door_closed_avg_rgb = door_closed_avg_rgb
        self.door_open_avg_rgb = door_open_avg_rgb

    def detect_door_state(self, frame):
        minX, minY, maxX, maxY = self.open_door_contour
        img = frame[minY:maxY, minX:maxX]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        average = img.mean(axis=0).mean(axis=0)
        average = (int(average[0]), int(average[1]), int(average[2]))
        avg_color = convert_color(sRGBColor(*average), LabColor)

        TARGET_COLORS = {DoorStates.DOOR_CLOSED: convert_color(sRGBColor(*self.door_closed_avg_rgb), LabColor),
                         DoorStates.DOOR_OPEN: convert_color(sRGBColor(*self.door_open_avg_rgb), LabColor)}

        differences = [[delta_e_cmc(avg_color, target_value, pl=1, pc=1), target_name] for target_name, target_value in
                       TARGET_COLORS.items()]
        differences.sort()  # sorted by the first element of inner lists

        door_state = differences[0][1]
        return door_state


class AdaptiveDoorStateDetector(DoorStateDetector):
    def __init__(self, open_door_contour, door_state_order, warmup_frames=100, diff_threshold=500,
                 avg_update_frames=100):
        """
        this door state detector adapts to the changing colors of the contour defined
        it creates an average color of the contour for warmup_frames, and then updates it every
        avg_update_frames frames.

        :param door_state_order: a tuple representing the first and second states. the first state
                                is considered for whatever average color is computed for the warmup frames
        :param warmup_frames: initial frames when the average color of the contour is taken as the first
                              door state (e.g DoorStates.DOOR_CLOSED)
        :param diff_threshold: the Delta E-CMC difference in color between a new average and initial average
                                to consider that the door state has flipped to the second state in door_state_order
        :param avg_update_frames: will refresh the first door state's color average every so frames
                                   useful to adapt to day and night lighting conditions
        """
        super().__init__(open_door_contour)
        self.warmup_frames = warmup_frames
        self.diff_threshold = diff_threshold
        self.door_state_order = door_state_order
        self.avg_update_frames = avg_update_frames
        self.first_door_state_avg = None
        self.total_frames = 0

    def add_to_avg(self, avg_rgb, new_rgb):
        if avg_rgb is None:
            return new_rgb
        else:
            return ((avg_rgb[0] + new_rgb[0]) / 2,
                    (avg_rgb[1] + new_rgb[1]) / 2,
                    (avg_rgb[2] + new_rgb[2]) / 2)

    def get_current_contour_avg(self, frame):
        minX, minY, maxX, maxY = self.open_door_contour
        img = frame[minY:maxY, minX:maxX]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        average = img.mean(axis=0).mean(axis=0)
        average = (int(average[0]), int(average[1]), int(average[2]))
        return average

    def detect_door_state(self, frame):
        self.total_frames += 1
        new_contour_avg = self.get_current_contour_avg(frame)
        if self.total_frames > self.warmup_frames:
            new_contour_avg_clr = convert_color(sRGBColor(*new_contour_avg), LabColor)
            init_cont_avg_clr = convert_color(sRGBColor(*self.first_door_state_avg), LabColor)
            diff = delta_e_cmc(new_contour_avg_clr, init_cont_avg_clr, pl=1, pc=1)
            if diff > self.diff_threshold:
                return self.door_state_order[1]
            if self.total_frames % self.avg_update_frames == 0:
                self.first_door_state_avg = self.add_to_avg(
                    self.first_door_state_avg, new_contour_avg)
        else:
            self.first_door_state_avg = self.add_to_avg(
                self.first_door_state_avg, new_contour_avg)

        return self.door_state_order[0]
