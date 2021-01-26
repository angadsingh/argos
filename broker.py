import threading

from detection.state_managers.object_state_manager import ObjectStateManager
from lib.singleton_q import SingletonBlockingQueue
from notifier import Notifier, NotificationTypes


class Broker():
    def __init__(self, config, pattern_detector, broker_q: SingletonBlockingQueue, notify_q, notifier = None):
        self.config = config
        self.pattern_detector = pattern_detector
        self.object_state_manager = ObjectStateManager(pattern_detector.state_history, pattern_detector.output_q)
        self.broker_q = broker_q
        self.notify_q = notify_q
        if notifier:
            self.notifier = notifier
        else:
            self.notifier = Notifier(self.config, notify_q)
        self.stopped = False
        self.t = threading.Thread(target=self.broke)
        self.t.daemon = True
        self.t.start()

    def broke(self):
        while not self.stopped:
            notification_type, notification_payload = self.broker_q.dequeue()
            if notification_type is NotificationTypes.OBJECT_DETECTED:
                if self.config.pattern_detection_enabled:
                    self.object_state_manager.add_state(notification_payload)
            self.notify_q.enqueue((notification_type, notification_payload))

    def stop(self):
        self.stopped = True
        self.t.join()
        self.notifier.stop()
