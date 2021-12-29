import time

import numpy as np


class StateHistoryStep:
    def __init__(self, state, state_attrs=None, ts=None):
        if ts == None:
            ts = time.time()
        self.ts = ts
        self.now = None
        self.state = state
        self.state_attrs = state_attrs

    def __repr__(self):
        ts_now = time.time()
        if self.now:
            ts_now = self.now
        return '%s[%s]' % (self.state,  round(ts_now - self.ts, 2))


class NotState():
    def __init__(self, state, duration=np.inf):
        self.state = state
        self.duration = duration

    def __repr__(self):
        return 'NOT{%s(for %ss)}' % (self.state, self.duration)

    def __eq__(self, other):
        if not isinstance(other, NotState):
            return NotImplemented

        return self.state == other.state and self.duration == other.duration

    def __key(self):
        return (self.state, self.duration)

    def __hash__(self):
        return hash(self.__key())