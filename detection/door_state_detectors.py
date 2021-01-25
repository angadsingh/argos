import cv2
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cmc
from colormath.color_objects import sRGBColor, LabColor

from detection.state_managers.door_state_manager import DoorStates


class SingleShotDoorStateDetector():
    @staticmethod
    def detect_door_state(image, open_door_contour, door_closed_avg_rgb, door_open_avg_rgb):
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

class AdaptiveDoorStateDetector():
    pass