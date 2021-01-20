import glob
import unittest

import requests


class TestServe(unittest.TestCase):

    def test1(self):
        PATH_TO_IMAGES = "/Users/asingh/workspace/pi object detection" + "/training-data/manjula/done/*.jpg"
        images = glob.glob(PATH_TO_IMAGES)
        content_type = 'image/jpeg'
        for image in images:
            img = open(image, 'rb')
            det_boxes = requests.post('http://192.168.1.99:8080/detect', files={'file': (image, img, content_type)}).json()
            if len(det_boxes) > 0:
                for box in det_boxes:
                    minx, miny, maxx, maxy, label, accuracy = box
                    if label == 'person':
                        print(box)