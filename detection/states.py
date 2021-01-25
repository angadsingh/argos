import time
from enum import Enum

import numpy as np


class StateHistoryStep:
    def __init__(self, state, state_attrs=None, ts=None):
        if ts == None:
            ts = int(round(time.time()))
        self.ts = ts
        self.state = state
        self.state_attrs = state_attrs

    def __repr__(self):
        return '%s[%s]' % (self.state, int(round(time.time())) - self.ts)


class NotState():
    def __init__(self, state, duration=np.inf):
        self.state = state
        self.duration = duration

    def __repr__(self):
        return 'NOT{%s(for %ss)}' % (self.state, self.duration)