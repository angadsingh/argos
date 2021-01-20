import argparse

import cv2

def create_mask_from_image(image_path):
        orig_img = cv2.imread(image_path)
        r = cv2.selectROI("Frame", orig_img, fromCenter=False, showCrosshair=True)
        print(str((int(r[0]), int(r[1]), int(r[0] + r[2]), int(r[1] + r[3]))))


def create_mask_from_video_file(video_file):
    video = cv2.VideoCapture(video_file)
    while (video.isOpened()):
        ret, frame = video.read()
        if not ret:
            print('Reached the end of the video!')
            break

        r = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        print(str((int(r[0]), int(r[1]), int(r[0] + r[2]), int(r[1] + r[3]))))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_path", type=str, required=False)
    ap.add_argument("-v", "--video_file", type=str, required=False)
    args = vars(ap.parse_args())

    if args['image_path']:
        create_mask_from_image(args['image_path'])
    elif args['video_file']:
        create_mask_from_video_file(args['video_file'])
