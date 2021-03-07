from configs.config_base import ConfigBase
from configs.config_patterns import door_movement
from lib.constants import DetectorType
from detection.door_state_detectors import SingleShotFrameDiffDoorStateDetector
from lib.detection_buffer import SimpleDetectionBuffer


class Config(ConfigBase):
    def __init__(self):
        super().__init__()
        self.show_fps = True
        self.video_feed_fps = -1
        self.send_mqtt = False
        self.send_webhook = False
        self.fps_print_frames = 10
        self.md_min_cont_area = 50
        self.md_tval = 25
        self.md_bg_accum_weight = 0.5
        self.md_show_all_contours = True
        self.md_warmup_frame_count = -1
        self.md_update_bg_model = True
        self.md_reset_bg_model = False
        self.md_enable_erode = False
        self.md_enable_dilate = False
        self.md_erode_iterations = 2
        self.md_dilate_iterations = 2
        self.md_box_threshold_y = 200
        self.md_box_threshold_x = 200
        self.md_mask = (250, 0, 690, 720)

        self.tf_model_path = '../tf_models/tflite/coco_ssd_mobilenet_v1_1.0_quant/detect.tflite'
        self.tf_path_to_labelmap = '../tf_models/tflite/coco_ssd_mobilenet_v1_1.0_quant/labelmap.txt'
        self.tf_accuracy_threshold = 0.5
        self.tf_detection_labels = ['person', 'dog']
        self.tf_detection_masks = None
        self.tf_detection_nmasks = None
        self.tf_box_thresholds = (150, 150)
        self.tf_detection_buffer = SimpleDetectionBuffer()
        self.tf_detector_type = DetectorType.TFLITE
        self.tf_apply_md = True
        self.tf_od_frame_write = True
        self.od_task_q_size = 1000
        self.tf_output_detection_path = './data/detections'
        self.tf_od_annotation_write = False
        self.pattern_detection_enabled = True
        self.pattern_detection_pattern_steps = door_movement.pattern_steps
        self.pattern_detection_state_history_length = 200
        self.pattern_detection_state_history_length_partial = 300
        self.pattern_detection_interval = 1
        self.door_state_detector = SingleShotFrameDiffDoorStateDetector((215, 114, 227, 123), (196, 131, 215, 147))
        self.door_state_detector_show_detection = True

        # emulates real speeds of the raspberry pi 4
        self.md_frame_rate = 15
        self.od_frame_rate = 2

        self.debug_mode = False
        self.md_blur_output_frame = False
        self.od_blur_output_frame = False
        self.test_show_video = False