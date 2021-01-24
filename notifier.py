import threading
from enum import Enum

from termcolor import colored

from lib.ha_mqtt import HaMQTT

from lib.ha_webhook import HaWebHook
from lib.timer import RepeatedTimer
import logging
log = logging.getLogger(__name__)

class NotificationTypes(Enum):
    OBJECT_DETECTED = 1
    MOVEMENT_PATTERN_DETECTED = 2
    DOOR_STATE_CHANGED = 3
    MOTION_STATE_CHANGED = 4


class Notifier():
    def __init__(self, config, notify_q):
        self.notify_q = notify_q
        self.config = config
        if self.config.send_mqtt:
            log.info("mqtt init")
            self.mqtt = HaMQTT(self.config.mqtt_host, self.config.mqtt_port,
                               self.config.mqtt_username, self.config.mqtt_password)
            self.mqtt_heartbeat_timer = RepeatedTimer(self.config.mqtt_heartbeat_secs, self.mqtt.publish,
                                                      self.config.mqtt_state_topic, "none")
        elif self.config.send_webhook:
            self.ha_webhook_od = HaWebHook(self.config.ha_webhook_object_detect_url, self.config.ha_webhook_ssh_host,
                                        self.config.ha_webhook_ssh_username, self.config.ha_webhook_target_dir)
            self.ha_webhook_pd = HaWebHook(self.config.ha_webhook_pattern_detect_url)
            self.ha_webhook_ot = HaWebHook(self.config.ha_webhook_state_detect_url)

        self.notification_handlers = {
            NotificationTypes.OBJECT_DETECTED: self.notify_object_detected,
            NotificationTypes.MOVEMENT_PATTERN_DETECTED: self.notify_pattern_detected,
            NotificationTypes.MOTION_STATE_CHANGED: self.notify_state_detected,
            NotificationTypes.DOOR_STATE_CHANGED: self.notify_state_detected
        }
        self.stopped = False
        self.t = threading.Thread(target=self.listen_notify_q)
        self.t.daemon = True
        self.t.start()

    def notify_object_detected(self, label, accuracy, img_path):
        if label is not None:
            if self.config.send_mqtt:
                self.mqtt.publish(self.config.mqtt_object_detect_topic, label)
            elif self.config.send_webhook:
                self.ha_webhook_od.send(label, img_path)
            else:
                log.info(colored("unsent object notification: label [%s], accuracy[%s]" % (
                    label, str(accuracy)
                ), attrs=['bold']))

    def notify_pattern_detected(self, pattern, pattern_attrs):
        label, accuracy, img_path = pattern_attrs
        if self.config.send_mqtt:
            self.mqtt.publish(self.config.mqtt_movement_pattern_detect_topic, str(pattern.name))
        elif self.config.send_webhook:
            self.ha_webhook_pd.send(str(pattern.name), img_path)
        else:
            log.info(colored("unsent pattern notification: %s" % pattern, attrs=['bold']))

    def notify_state_detected(self, state):
        if self.config.send_mqtt:
            self.mqtt.publish(self.config.mqtt_state_detect_topic, str(state))
        elif self.config.send_webhook:
            self.ha_webhook_ot.send(str(state))
        else:
            log.info(colored("unsent state detection notification: %s" % state, attrs=['bold']))

    def listen_notify_q(self):
        while not self.stopped:
            notification_type, notification_payload = self.notify_q.dequeue()
            self.notification_handlers[notification_type](*notification_payload)

    def stop(self):
        self.stopped = True
        self.t.join()
        if self.config.send_mqtt:
            self.mqtt_heartbeat_timer.stop()
