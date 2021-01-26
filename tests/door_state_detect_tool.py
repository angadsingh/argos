import os

import cv2

from detection.door_state_detectors import SingleShotDoorStateDetector, DoorStateDetector


class DoorStateDetectTools():
    def __init__(self, door_state_detector: DoorStateDetector, show_result = True):
        self.door_state_detector = door_state_detector
        self.show_result = show_result

    def _door_state_from_image_dir(self, images_dir):
        for file in sorted(os.scandir(images_dir), key=lambda e: e.name, reverse=True):
            if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(images_dir, file.name)
                print(image_path)
                self._door_state_from_image(image_path)

    def _door_state_from_image(self, image_path):
        frame = cv2.imread(image_path)
        return self._door_state_from_frame(frame)

    def _door_state_from_frame(self, frame):
        inferred_state = self.door_state_detector.detect_door_state(frame)

        if self.show_result:
            self.door_state_detector.show_detection(frame, inferred_state)
            cv2.imshow("image", frame)
            cv2.waitKey()

        return inferred_state

    def _door_state_from_video(self, video_file):
        video = cv2.VideoCapture(video_file)
        results = []
        frame_no = 0
        while (video.isOpened()):
            ret, frame = video.read()
            if not ret:
                print('Reached the end of the video!')
                break
            results.append((frame_no, self._door_state_from_frame(frame)))
            frame_no += 1
        return results

    def _create_contour_from_frame(self, frame):
        r = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        xmin = r[0]
        xmax = int(r[0] + r[2])
        ymin = int(r[1])
        ymax = int(r[1] + r[3])
        print(str((xmin, ymin, xmax, ymax)))

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
    # file = 'data/door_state_test_images/doorclosed_night2.jpg'
    # open_door_contour = (215, 114, 227, 123)
    # DoorStateDetectTools()._door_state_from_image(file, open_door_contour)
    # DoorStateDetectTools()._door_state_from_video(file, open_door_contour)
    # DoorStateDetectTools()._create_contour_from_video(file)
    file = 'data/door_state_test_images/doorclosed_night2.jpg'
    DoorStateDetectTools(None)._create_contour_from_image(file)



