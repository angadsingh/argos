import threading

from lib.singleton_q import SingletonBlockingQueue
from notifier import Notifier, NotificationTypes


class Broker():
    def __init__(self, config, door_detector, broker_q: SingletonBlockingQueue, notify_q: SingletonBlockingQueue):
        self.config = config
        self.door_detector = door_detector
        self.broker_q = broker_q
        self.notify_q = notify_q
        self.notifier = Notifier(self.config, notify_q)
        self.stopped = False
        self.t = threading.Thread(target=self.broke)
        self.t.daemon = True
        self.t.start()

    def broke(self):
        while not self.stopped:
            notification_type, notification_payload = self.broker_q.dequeue()
            if notification_type is NotificationTypes.OBJECT_DETECTED:
                if self.config.door_movement_detection:
                    self.door_detector.add_object_state(*notification_payload)
            self.notify_q.enqueue((notification_type, notification_payload))

    def stop(self):
        self.stopped = True
        self.t.join()
        self.notifier.stop()
