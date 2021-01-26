from configs.config_base import ConfigBase
from configs.config_patterns import door_movement
from configs.constants import InputMode, DetectorType
from detection.door_state_detectors import AdaptiveDoorStateDetector, SingleShotDoorStateDetector, \
    SingleShotFrameDiffDoorStateDetector
from detection.state_managers.door_state_manager import DoorStates


class Config(ConfigBase):
    def __init__(self):
        super().__init__() 

        # whether to show fps in the output video feed
        self.show_fps = True
        # the fps to limit the output video feed to.
        # not necessary since it is automatically limited to the speed of the motion detector
        self.video_feed_fps = -1

        # print fps on the console every x frames
        self.fps_print_frames = 10

        ## MOTION DETECTOR CONFIG

        # size of the smallest box or contour of motion (shown by yellow boxes if md_show_all_contours is True)
        self.md_min_cont_area = 50

        # the image thresholding value for the motion detector
        # read about image thresholding in opencv here: https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
        self.md_tval = 25

        # the higher the background accumulation weight the lower the "memory" of the motion detector
        # for new objects. in other words a higher value will show motion detected for shorter periods
        # while lower value will show motion detected longer for any new objects that came into the frame
        # tune this as per the fps of your video stream and speed of objects you are detecting (e.g. a snail vs a person)
        self.md_bg_accum_weight = 0.5

        # if True will show granular motion detection contours
        self.md_show_all_contours = False

        # set to number of frames to "warm up" the motion detector
        # during these frames only the background model is updated
        # and no motion detection is done
        self.md_warmup_frame_count = -1

        # whether to update background model at all
        # motion detection won't happen if this is False
        self.md_update_bg_model = True

        # you can reset the background model dynamically
        # through /config by setting this
        self.md_reset_bg_model = False

        # limits the motion detector frame rate (so that you can save CPU cycles for the object detector)
        self.md_frame_rate = 5

        # minimum size of the box for detected motion (useful for filtering small motion like tiny shadows or curtains moving)
        self.md_box_threshold_y = 200
        self.md_box_threshold_x = 200

        # do motion detection only within this mask
        self.md_mask = (250, 0, 690, 520)

        # don't do motion detection in this mask
        self.md_nmask = None

        # blur the output video wherever there is motion
        # useful to share videos of argos in action or even
        # if you are privacy conscious at home
        self.md_blur_output_frame = False

        # blurs the output frame of the object detector
        self.od_blur_output_frame = False

        ## OBJECT DETECTOR CONFIG

        # path to the tensorflow model. will be the "saved_model" directory for a TF2 model
        # or the tflite file for a tf lite model
        self.tf_model_path = 'tf_models/tflite/coco_ssd_mobilenet_v1_1.0_quant/detect.tflite'

        # path to the label map of the tensor flow model
        self.tf_path_to_labelmap = 'tf_models/tflite/coco_ssd_mobilenet_v1_1.0_quant/labelmap.txt'

        # tensorflow score threshold (i like to call it accuracy)
        self.tf_accuracy_threshold = 0.4

        # list of labels to filter on. use your own labels if you have a custom trained model
        # i just use coco
        self.tf_detection_labels = ['person', 'dog']

        # list of masks to filter the detections by (objects detected outside these masks are ignored)
        self.tf_detection_masks = None

        # list of negative masks to filter the detections by (objects detected inside these masks are ignored)
        self.tf_detection_nmasks = None

        # minimum size of the object detected (lets not detect a person inside a newspaper kept on the table or a photo frame)
        self.tf_box_thresholds = (150, 150)

        # enable the detection buffer. this accumulates detections and the detector only reports one
        # when buffer_threshold number of detections were found in buffer_duration millis
        self.tf_detection_buffer_enabled = False
        self.tf_detection_buffer_duration = 3000
        self.tf_detection_buffer_threshold = 4

        # switch between TF1, TF2 or TFLITE
        self.tf_detector_type = DetectorType.TFLITE

        # you can disable motion detection entirely and run the object detector directly
        # if you have an expensive GPU or a coral TPU
        self.tf_apply_md = True

        # write a jpeg file with the detection shown on it
        self.tf_od_frame_write = True

        # write pascal VOC format xml file with the detection
        # this is useful to then later train a model on these detections
        # do transfer learning and create your own labels
        self.tf_od_annotation_write = True

        # path where above jpegs are stored
        self.tf_output_detection_path = '/home/pi/detections'

        ## PATTERN DETECTOR CONFIG

        # enable the movement pattern detector
        self.pattern_detection_enabled = True

        # this is where you can define your own patterns and do something
        # beyond door movement (the first use case this project was built for)

        # `pattern_detection_pattern_steps` needs to be a map of [pattern: list of states]
        # the list of states can include NotStates for negation
        # see the doc of `PatternDetector.find_mov_ptn_in_state_history()` to see how
        # all this works with examples.
        # argos ships with a pattern out of the box in config_patterns/door_movement.py
        self.pattern_detection_pattern_steps = door_movement.pattern_steps
        self.pattern_detection_patter_eval_order = door_movement.pattern_evaluation_order

        # duration of time in seconds for which the pattern detection state history should
        # to be maintained
        self.pattern_detection_state_history_length = 20

        # duration (in seconds) for which pattern detection state history length
        # should be kept if there are partial pattern matches
        self.pattern_detection_state_history_length_partial = 300

        # see the docs for the respective door state detectors to understand how they work
        # and how their parameters make them behave. you can choose one based on your environment
        # self.door_state_detector = SingleShotDoorStateDetector((215, 114, 227, 123), (118, 80, 26), (151, 117, 72))
        # self.door_state_detector = AdaptiveDoorStateDetector((215, 114, 227, 123),
        #                                  (DoorStates.DOOR_CLOSED, DoorStates.DOOR_OPEN))
        self.door_state_detector = SingleShotFrameDiffDoorStateDetector((215, 114, 227, 123), (196, 131, 215, 147))

        # show the door state in the output video frame
        self.door_state_detector_show_detection = True

        # if True you can run through the output video (on the flask server) step by step
        # by pressing any key on the console where you're running stream.py.
        # press 'q' to quit
        self.debug_mode = False

        ## NOTIFIER CONFIG

        #whether to enable MQTT notifications to HA
        self.send_mqtt = False

        #whether to enable webhook based notifications to HA
        self.send_webhook = True

        # keep sending the last motion state every x seconds (in case HA restarted or just didnt
        # get our message last time
        self.mqtt_heartbeat_secs = 30

        # topic where object detections are sent
        self.mqtt_object_detect_topic = 'home-assistant/pi-object-detection/main_door/object'

        # topic where pattern detections are sent
        self.mqtt_movement_pattern_detect_topic = 'home-assistant/pi-object-detection/main_door/pattern'

        # topic where other state changes are sent
        self.mqtt_state_detect_topic = 'home-assistant/pi-object-detection/main_door/state'

        # webhook url where object detections are sent
        self.ha_webhook_object_detect_url = "https://<your-hass>.duckdns.org:8123/api/webhook/pi_object_detection_main_door?object={}&img={}"

        # webhook url where pattern detections are sent
        self.ha_webhook_pattern_detect_url = "https://<your-hass>.duckdns.org:8123/api/webhook/pi_pattern_detection_main_door?pattern={}&img={}"

        # webhook url where other state detections are sent
        self.ha_webhook_state_detect_url = "https://<your-hass>.duckdns.org:8123/api/webhook/pi_state_detection_main_door?state={}"

        # this is how the detection images are scp'ed to your HA installation
        # so that they can be referenced in the android/iOS notification (served from HA's webserver)!
        self.ha_webhook_ssh_host = '<your-hass-host>'
        self.ha_webhook_ssh_username = 'pi'
        self.ha_webhook_target_dir = '/usr/share/hassio/homeassistant/www/detections/'

        # usual mqtt stuff to connect to HA
        self.mqtt_host = '<your-mqtt-host>'
        self.mqtt_port = 1883
        self.mqtt_username = "mqtt"
        self.mqtt_password = "------"

        ## INPUT CONFIG

        # supports RTMP, picamera and local video file
        self.input_mode = InputMode.RTMP_STREAM
        self.rtmp_stream_url = "rtmp://192.168.1.19:43331/live/main_door"
