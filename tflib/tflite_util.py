import importlib

import cv2
import numpy as np

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter


class DetectorTFLite:

    def __init__(self, path_to_checkpoint, path_to_labelmap, filter_labels=None):
        self.filter_labels = filter_labels

        with open(path_to_labelmap, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self.labels[0] == '???':
            del (self.labels[0])

        self.interpreter = Interpreter(model_path=path_to_checkpoint)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.tf_height = self.input_details[0]['shape'][1]
        self.tf_width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

    def ExtractBoxes(self, imH, imW, boxes, classes, scores):
        det_boxes = []
        for i in range(len(scores)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            miny = int(max(1, (boxes[i][0] * imH)))
            minx = int(max(1, (boxes[i][1] * imW)))
            maxy = int(min(imH, (boxes[i][2] * imH)))
            maxx = int(min(imW, (boxes[i][3] * imW)))
            label = self.labels[int(classes[i])]
            det_boxes.append((minx, miny, maxx, maxy, label, float(scores[i])))
        return det_boxes

    def DetectFromImage(self, img):
        imH, imW, _ = img.shape
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.tf_width, self.tf_height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[
            0]  # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]  # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]  # Confidence of detected objects

        return self.ExtractBoxes(imH, imW, boxes, classes, scores)

    def DisplayDetection(self, image, box, det_time=None):
        img = image.copy()

        x_min = box[0]
        y_min = box[1]
        x_max = box[2]
        y_max = box[3]
        cls = str(box[4])
        score = str(np.round(box[-1], 2))

        text = cls + ": " + score
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.rectangle(img, (x_min, y_min - 20), (x_min, y_min), (255, 255, 255), -1)
        cv2.putText(img, text, (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if det_time != None:
            fps = round(1000. / det_time, 1)
            fps_txt = str(fps) + " FPS"
            cv2.putText(img, fps_txt, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return img
