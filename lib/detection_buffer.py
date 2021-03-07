import abc
import time
import logging
import numpy as np

log = logging.getLogger(__name__)


class DetectionBuffer(abc.ABC):
    @abc.abstractmethod
    def get_max_accuracy_label(self):
        pass

    @abc.abstractmethod
    def add_detection(self, detection, current_ts=None):
        pass


class SimpleDetectionBuffer(DetectionBuffer):
    def __init__(self):
        self.detections = []

    def get_max_accuracy_label(self):
        max_accuracy = -np.inf
        max_accuracy_label = None
        max_accuracy_img_path = None
        for (box, image_path) in self.detections:
            (minx, miny, maxx, maxy, label, accuracy) = box
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_accuracy_label = label
                max_accuracy_img_path = image_path
        self.detections.clear()
        return (max_accuracy_label, max_accuracy, max_accuracy_img_path)

    def add_detection(self, detection, current_ts=None):
        self.detections.append(detection)


class SlidingWindowDetectionBuffer(DetectionBuffer):
    def __init__(self, duration=3000, threshold_detections=4):
        self.duration = duration
        self.threshold_detections = threshold_detections
        self.buffer = []

    def _trim_buffer(self, current_ts):
        for det in list(self.buffer):
            if current_ts - det['ts'] > self.duration:
                self.buffer.remove(det)

    def get_max_accuracy_label(self):
        if len(self.buffer) < self.threshold_detections:
            return (None, None)

        cum_lbl_accuracy = {}
        label_list = []
        max_cum_lbl_acc = 0
        max_wt_label = None
        max_wt_img_path = None

        for det in self.buffer:
            ((minx, miny, maxx, maxy, label, accuracy), img_path) = det['dt']
            label_list.append(label)
            if label not in cum_lbl_accuracy:
                cum_lbl_accuracy[label] = accuracy
            else:
                cum_lbl_accuracy[label] += accuracy
            if cum_lbl_accuracy[label] > max_cum_lbl_acc:
                max_cum_lbl_acc = cum_lbl_accuracy[label]
                max_wt_label = label
                max_wt_img_path = img_path

        log.debug(str(label_list))
        log.info('cum_lbl_accuracy: ' + str(cum_lbl_accuracy))

        return (max_wt_label, max_cum_lbl_acc, max_wt_img_path)

    @abc.abstractmethod
    def add_detection(self, detection, current_ts=None):
        if current_ts is None:
            current_ts = int(round(time.time() * 1000))
        self.buffer.append({'ts': current_ts, 'dt': detection})
        self._trim_buffer(current_ts)