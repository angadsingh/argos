from configs.config_base import ConfigBase
from lib.constants import DetectorType


class Config(ConfigBase):
    def __init__(self):
        super().__init__()
        self.fps_print_frames = 10
        self.tf_model_path = 'tf_models/tflite/coco_ssd_mobilenet_v1_1.0_quant/detect.tflite'
        self.tf_path_to_labelmap = 'tf_models/tflite/coco_ssd_mobilenet_v1_1.0_quant/labelmap.txt'
        self.tf_accuracy_threshold = 0.5
        self.tf_detection_labels = ['person']
        self.tf_detector_type = DetectorType.TFLITE