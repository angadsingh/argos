# Time:  O(n)
# Space: O(1)

import collections
import threading
from abc import ABC

from lib import gauge, get_metric_prefix


class TaskQueue(ABC):
    def __init__(self, max_size, metric_prefix=None):
        self.__cv = threading.Condition()
        self.__q = collections.deque()
        self.max_size = max_size
        self.metric_prefix = get_metric_prefix(self, metric_prefix)

    def abrupt_stop(self, stop_element):
        with self.__cv:
            self.__q.appendleft(stop_element)
            self.__cv.notifyAll()

    def _enqueue(self, element, wait):
        """
        :type element: int
        :rtype: void
        """
        with self.__cv:
            while len(self.__q) == self.max_size:
                if wait:
                    self.__cv.wait()
                else:
                    self.__q.popleft()
            self.__q.append(element)
            self.__cv.notifyAll()
            gauge(self.metric_prefix, "queue_size").notify(len(self.__q))

    def read(self, timeout=None):
        with self.__cv:
            self.__cv.wait(timeout)
            if self.__q:
                return self.__q[-1]

    def _dequeue(self, notify):
        with self.__cv:
            while not self.__q:
                self.__cv.wait()
            element = self.__q.popleft()
            if notify:
                self.__cv.notifyAll()
            gauge(self.metric_prefix, "queue_size").notify(len(self.__q))
            return element

    def wait_for_empty(self, timeout=None):
        with self.__cv:
            if len(self.__q) > 0:
                self.__cv.wait(timeout)
            return len(self.__q) == 0

    def notify(self):
        self.__cv.notifyAll()

    def size(self):
        """
        :rtype: int
        """
        with self.__cv:
            return len(self.__q)


class BlockingTaskQueue(TaskQueue):
    def __init__(self, max_size, metric_prefix=None):
        super().__init__(max_size, metric_prefix)

    def enqueue(self, element):
        self._enqueue(element, wait=True)

    def dequeue(self):
        return self._dequeue(notify=True)


class NonBlockingTaskQueue(TaskQueue):
    def __init__(self, max_size, metric_prefix=None):
        super().__init__(max_size, metric_prefix)

    def enqueue(self, element):
        self._enqueue(element, wait=False)

    def dequeue(self):
        return self._dequeue(notify=False)


class BlockingTaskSingleton(BlockingTaskQueue):
    def __init__(self, metric_prefix=None):
        super().__init__(1, metric_prefix)


class NonBlockingTaskSingleton(NonBlockingTaskQueue):
    def __init__(self, metric_prefix=None):
        super().__init__(1, metric_prefix)
