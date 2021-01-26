import numpy

from configs.constants import InputMode, DetectorType


class ConfigBase:
    def __init__(self):
        self.show_fps = False
        self.video_feed_fps = -1
        self.send_mqtt = False
        self.send_webhook = False
        self.fps_print_frames = -1
        self.md_min_cont_area = 0
        self.md_tval = 25
        self.md_bg_accum_weight = 0.5
        self.md_show_all_contours = False
        self.md_warmup_frame_count = -1
        self.md_update_bg_model = True
        self.md_reset_bg_model = False
        self.md_enable_erode = False
        self.md_enable_dilate = False
        self.md_erode_iterations = 2
        self.md_dilate_iterations = 2
        self.md_frame_rate = 5
        self.md_box_threshold_y = 0
        self.md_box_threshold_x = 0
        self.md_mask = None
        self.md_nmask = None
        self.md_blur_output_frame = False

        self.tf_model_path = None
        self.tf_path_to_labelmap = None
        self.tf_accuracy_threshold = 0
        self.tf_detection_labels = None
        self.tf_detection_masks = None
        self.tf_detection_nmasks = None
        self.tf_box_thresholds = None
        self.tf_detection_buffer_enabled = False
        self.tf_detection_buffer_duration = 3000
        self.tf_detection_buffer_threshold = 4
        self.tf_detector_type = DetectorType.TFLITE
        self.tf_apply_md = False
        self.tf_od_frame_write = False
        self.tf_od_annotation_write = False
        self.tf_output_detection_path = None
        self.pattern_detection_enabled = True
        self.pattern_detection_pattern_steps = None
        self.pattern_detection_patter_eval_order = None
        self.pattern_detection_state_history_length = 20
        self.pattern_detection_state_history_length_partial = 300
        self.door_state_detector = None
        self.door_state_detector_show_detection = False
        self.debug_mode = False

        self.mqtt_heartbeat_secs = 30
        self.mqtt_object_detect_topic = None
        self.mqtt_movement_pattern_detect_topic = None
        self.mqtt_state_detect_topic = None
        self.ha_webhook_object_detect_url = None
        self.ha_webhook_pattern_detect_url = None
        self.ha_webhook_state_detect_url = None
        self.ha_webhook_ssh_host = None
        self.ha_webhook_ssh_username = None
        self.ha_webhook_target_dir = None
        self.mqtt_host = None
        self.mqtt_port = None
        self.mqtt_username = None
        self.mqtt_password = None
        self.input_mode = None
        self.rtmp_stream_url = None
        self.video_file_path = None