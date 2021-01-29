import threading
import time
from datetime import datetime

import numpy as np

from configs.constants import DetectorType
from detection.StateDetectorBase import StateDetectorBase
from lib.blocking_q import BlockingQueue
from lib.detection_buffer import DetectionBuffer
from lib.fps import FPS
from termcolor import colored

import logging
log = logging.getLogger(__name__)

class BaseTFObjectDetector(StateDetectorBase):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_frame = BlockingQueue(max_size=1000)

        if self.config.tf_detection_buffer_enabled:
            self.detection_buffer = DetectionBuffer(config.tf_detection_buffer_duration,
                                                    config.tf_detection_buffer_threshold)
        self.fps = FPS(600, 100)
        self.ready = False
        self.__cv = threading.Condition()

    def start(self):
        self.t = threading.Thread(target=self.detect_continuously, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def stop(self):
        self.input_frame.enqueue(-1, wait=True)
        self.t.join()
        self.fps.stop()

    def initialize_tf_model(self):
        with self.__cv:
            if self.config.tf_detector_type == DetectorType.TF2:
                from tflib.tf2_util import DetectorTF2
                self.tf_detector = DetectorTF2(self.config.tf_model_path,
                                               self.config.tf_path_to_labelmap)
            elif self.config.tf_detector_type == DetectorType.TFLITE:
                from tflib.tflite_util import DetectorTFLite
                self.tf_detector = DetectorTFLite(self.config.tf_model_path,
                                                  self.config.tf_path_to_labelmap)
            self.ready = True
            self.__cv.notifyAll()

    def wait_for_ready(self):
        with self.__cv:
            while not self.ready:
                self.__cv.wait()
            return self.ready

    def add_task(self, task):
        self.input_frame.enqueue(task, wait=True)

    def apply_od_filters(self, det_boxes, accuracy_threshold=None, box_thresholds=None, masks=None, nmasks=None):
        accuracy_threshold = self.config.tf_accuracy_threshold if accuracy_threshold is None else accuracy_threshold
        box_thresholds = self.config.tf_box_thresholds if box_thresholds is None else box_thresholds
        masks = self.config.tf_detection_masks if masks is None else masks
        nmasks = self.config.tf_detection_nmasks if nmasks is None else nmasks
        filter_labels = self.config.tf_detection_labels

        filtered_det_boxes = []

        for box in det_boxes:
            (minx, miny, maxx, maxy, label, accuracy) = box
            if ((accuracy > accuracy_threshold) and (accuracy <= 1.0)):
                if filter_labels is None or label in filter_labels:
                    if box_thresholds and (maxx - minx < box_thresholds[0] or \
                                           maxy - miny < box_thresholds[1]):
                        log.info(colored("detection [%s] smaller than box thresholds [%s]" % (
                            str((minx, miny, maxx, maxy)), str(box_thresholds)), 'grey'))
                    else:
                        detection_is_masked = False
                        if masks:
                            detection_is_masked = True
                            for mask in masks:
                                mask_minx, mask_miny, mask_maxx, mask_maxy = mask
                                if minx > mask_minx and maxx < mask_maxx and miny > mask_miny and maxy < mask_maxy:
                                    log.info(colored("detection [%s] allowed by mask [%s]" % (
                                        str((minx, miny, maxx, maxy)), str(mask)), 'grey'))
                                    detection_is_masked = False
                                    break
                            if detection_is_masked:
                                log.info(colored("detection [%s] NOT allowed by any mask" % str((minx, miny, maxx, maxy)), 'grey'))
                        if not detection_is_masked:
                            if nmasks:
                                for nmask in nmasks:
                                    mask_minx, mask_miny, mask_maxx, mask_maxy = nmask
                                    if minx > mask_minx and maxx < mask_maxx and miny > mask_miny and maxy < mask_maxy:
                                        log.info(colored("detection [%s] not allowed by nmask [%s]" % (
                                            str((minx, miny, maxx, maxy)), str(nmask)), 'grey'))
                                        detection_is_masked = True
                                        break
                        if not detection_is_masked:
                            filtered_det_boxes.append(box)

        return filtered_det_boxes

    def detect_image(self, frame, threshold=None, masks=None, nmasks=None):
        return self.apply_od_filters(
            self.tf_detector.DetectFromImage(frame),
         accuracy_threshold=threshold, masks = masks, nmasks = nmasks
        )

    def detect_image_buffered(self, frame, cropped_frame, cropOffsetX, cropOffsetY, ts):
        if not ts:
            ts = time.time()
        det_boxes = self.apply_od_filters(self.tf_detector.DetectFromImage(cropped_frame))
        if det_boxes is not None and len(det_boxes) > 0:
            detections = True
            max_accuracy = -np.inf
            max_accuracy_label = None
            max_accuracy_img_path = None

            for box in det_boxes:
                minx, miny, maxx, maxy, label, accuracy = box
                image_path = "%s/detection_%s_%s.jpg" % (
                    self.config.tf_output_detection_path, label, datetime.fromtimestamp(ts).strftime("%d-%m-%Y-%H-%M-%S"))
                orig_box = minx + cropOffsetX, miny + cropOffsetY, maxx + cropOffsetX, maxy + cropOffsetY, label, accuracy
                if self.config.tf_detection_buffer_enabled:
                    self.detection_buffer.add_detection((orig_box, image_path))
                else:
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        max_accuracy_label = label
                        max_accuracy_img_path = image_path
                self.process_detection_intermeddiate(frame, orig_box, image_path)

            if detections:
                if self.config.tf_detection_buffer_enabled:
                    label, accuracy, image_path = self.detection_buffer.get_max_cumulative_accuracy_label()
                    self.process_detection_final(label, accuracy, image_path, ts)
                    return label, accuracy
                else:
                    self.process_detection_final(max_accuracy_label, max_accuracy, max_accuracy_img_path, ts)
                    return max_accuracy_label, max_accuracy

        return None, None

    def process_detection_intermeddiate(self, frame, orig_box, image_path):
        pass

    def process_detection_final(self, label, accuracy, image_path, ts):
        pass

    def detect_continuously(self):
        self.initialize_tf_model()

        while True:
            task = self.input_frame.dequeue(notify=True)
            if task == -1:
                break
            (frame, cropped_frame, (cropOffsetX, cropOffsetY), ts) = task
            if not self.task_skipper.skip_task(ts):
                self.fps.count()
                cropped_frame = np.copy(cropped_frame)
                cropped_frame.setflags(write=1)
                self.detect_image_buffered(frame, cropped_frame, cropOffsetX, cropOffsetY, ts)

                if self.config.od_frame_rate > 0:
                    time.sleep(1 / self.config.od_frame_rate)