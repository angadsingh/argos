# Time:  O(n)
# Space: O(1)

import collections
import threading


class BlockingQueue(object):
    def __init__(self, max_size = 1):
        self.__cv = threading.Condition()
        self.__q = collections.deque()
        self.max_size = max_size

    def abrupt_stop(self, stop_element):
        with self.__cv:
            self.__q.appendleft(stop_element)
            self.__cv.notifyAll()

    def enqueue(self, element, wait = False):
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

    def read(self, timeout = None):
        with self.__cv:
            self.__cv.wait(timeout)
            if self.__q:
                return self.__q[-1]

    def dequeue(self, notify = False):
        with self.__cv:
            while not self.__q:
                self.__cv.wait()
            element = self.__q.popleft()
            if notify:
                self.__cv.notifyAll()
            return element

    def wait_for_empty(self, timeout = None):
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
