# Time:  O(n)
# Space: O(1)

import collections
import threading


class SingletonBlockingQueue(object):
    def __init__(self):
        self.__cv = threading.Condition()
        self.__q = collections.deque()

    def enqueue(self, element, wait = False):
        """
        :type element: int
        :rtype: void
        """
        with self.__cv:
            while len(self.__q) == 1:
                if wait:
                    self.__cv.wait()
                else:
                    self.__q.clear()
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

    def size(self):
        """
        :rtype: int
        """
        with self.__cv:
            return len(self.__q)
