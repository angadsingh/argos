import threading

from detection.state_managers.object_state_manager import ObjectStateManager
from lib.task_queue import BlockingTaskSingleton, BlockingTaskQueue
from notifier import Notifier, NotificationTypes

import logging
log = logging.getLogger(__name__)

class Broker():
    def __init__(self, config, object_detector, pattern_detector, broker_q: BlockingTaskSingleton, notify_q: BlockingTaskQueue, notifier = None):
        self.config = config
        self.pattern_detector = pattern_detector
        self.object_state_manager = ObjectStateManager(object_detector, pattern_detector, pattern_detector.broker_q)
        self.broker_q = broker_q
        self.notify_q = notify_q
        if notifier:
            self.notifier = notifier
        else:
            self.notifier = Notifier(self.config, notify_q)
        self.t = threading.Thread(target=self.broke)
        self.t.daemon = True
        self.t.start()

    def broke(self):
        while True:
            task = self.broker_q.dequeue()
            if task == -1:
                break
            notification_type, notification_payload = task
            if notification_type is NotificationTypes.OBJECT_DETECTED:
                if self.config.pattern_detection_enabled:
                    (state, ts) = notification_payload
                    self.object_state_manager.add_state(state, ts)
            log.info("enqueuing notification %s [%s]" % (notification_type, notification_payload))
            self.notify_q.enqueue((notification_type, notification_payload))

    def stop(self):
        self.broker_q.enqueue(-1)
        self.t.join()
        self.notifier.stop()
