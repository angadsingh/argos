import logging
import threading

from termcolor import colored

from detection.StateDetectorBase import StateDetectorBase
from lib.constants import DetectorType

log = logging.getLogger(__name__)


class BaseTFObjectDetector(StateDetectorBase):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ready = False
        self.__cv = threading.Condition()

    def wait_for_ready(self):
        with self.__cv:
            while not self.ready:
                self.__cv.wait()
            return self.ready

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
                                log.info(
                                    colored("detection [%s] NOT allowed by any mask" % str((minx, miny, maxx, maxy)),
                                            'grey'))
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
            accuracy_threshold=threshold, masks=masks, nmasks=nmasks
        )
