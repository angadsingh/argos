import logging
import time
from enum import Enum

from termcolor import colored

from detection.state_managers.object_state_manager import ObjectStates
from detection.states import NotState, StateHistoryStep
from lib.timer import RepeatedTimer
from notifier import NotificationTypes

log = logging.getLogger(__name__)

"""
    potential improvement with object tracking:
        https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
        https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
        https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
"""


class PatternMatch(Enum):
    MATCHED = 0
    NOT_MATCHED = 1
    PARTIAL_MATCH = 2


class PatternDetector():
    def __init__(self, output_q, pattern_steps, pattern_evaluation_order, state_history_length=20,
                 state_history_length_partial=300, detection_interval=1):
        self.output_q = output_q
        self.pattern_steps = pattern_steps
        self.pattern_evaluation_order = pattern_evaluation_order
        self.state_history = []
        self.state_history_length = state_history_length
        self.state_history_length_partial = state_history_length_partial
        if detection_interval:
            self.pattern_detection_timer = RepeatedTimer(detection_interval, self.detect_patterns)

    def find_not_state_before_step(self, not_state: NotState, state_history, after_step_ts, from_idx, to_idx):
        while from_idx >= to_idx:
            shist_step: StateHistoryStep = state_history[from_idx]
            if not_state.state == shist_step.state and abs(after_step_ts - shist_step.ts) <= not_state.duration:
                return True
            from_idx -= 1

    def find_mov_ptn_in_state_history_at_idx(self, pattern_steps, state_history, shist_idx=0):
        ptn_idx = 0
        prev_match_idx = -1
        prev_match_ts = 0

        while ptn_idx < len(pattern_steps) and shist_idx < len(state_history):
            ptn_step = pattern_steps[ptn_idx]
            shist_step: StateHistoryStep = state_history[shist_idx]

            if ptn_step == shist_step.state:
                if ptn_idx > 0 and shist_idx > 0 and type(pattern_steps[ptn_idx - 1]) is NotState:
                    if self.find_not_state_before_step(pattern_steps[ptn_idx - 1], state_history, shist_step.ts,
                                                       shist_idx - 1, prev_match_idx + 1):
                        ptn_idx = 0
                        break
                ptn_idx += 1
                prev_match_idx = shist_idx
                prev_match_ts = shist_step.ts

            if type(ptn_step) is NotState:
                if ptn_idx == len(pattern_steps) - 1:
                    if self.find_not_state_before_step(ptn_step, state_history, prev_match_ts, len(state_history) - 1,
                                                       shist_idx):
                        ptn_idx = 0
                        break
                ptn_idx += 1
            else:
                shist_idx += 1

        if ptn_idx == 0 or (ptn_idx == 1 and type(pattern_steps[ptn_idx - 1]) is NotState):
            return PatternMatch.NOT_MATCHED
        elif 0 < ptn_idx <= len(pattern_steps) - 1:
            if type(pattern_steps[ptn_idx]) is NotState and ptn_idx == len((pattern_steps)) - 1 \
                    and int(round(time.time())) - prev_match_ts > pattern_steps[ptn_idx].duration:
                return PatternMatch.MATCHED
            else:
                return PatternMatch.PARTIAL_MATCH
        else:
            return PatternMatch.MATCHED

    def find_mov_ptn_in_state_history(self, pattern_steps, state_history):
        """
        this function takes a "pattern" and a "state history" and returns whether that pattern is found in that state history.
        the steps of the pattern can be spread across the state history, just need to be present in the same order
        the catch is that there's also negative states and timing involved. a pattern can have multiple
        "NotState(..some state, ..for some time)" in the beginning, anywhere in between or at the end,
        which means "..some state" should not have been seen in the state history for those many seconds
        at that point in the state history. Keep in mind that a pattern can be found multiple times in a state history
        or can not match in one part and match in the remaining state history as well.

        example::
            take the following pattern:
                [NotState(ObjectStates.OBJECT_DETECTED, 2), DoorStates.DOOR_OPEN, ObjectStates.OBJECT_DETECTED]

            it means:
                NO ObjectStates.OBJECT_DETECTED for 2 seconds, (any other state), DoorStates.DOOR_OPEN, (any other state), ObjectStates.OBJECT_DETECTED

            results:
                state_history: [(ObjectStates.OBJECT_DETECTED, 0), (DoorStates.DOOR_OPEN, 1), (ObjectStates.OBJECT_DETECTED, 1),
                               (DoorStates.DOOR_CLOSED, 1)]
                returns PatternMatch.NOT_MATCHED

                state_history: [(ObjectStates.OBJECT_DETECTED, 0), (DoorStates.DOOR_OPEN, 4), (ObjectStates.OBJECT_DETECTED, 1),
                               (DoorStates.DOOR_CLOSED, 1)]
                returns PatternMatch.MATCHED

                state_history: [(ObjectStates.OBJECT_DETECTED, 0), (DoorStates.DOOR_OPEN, 4), (ObjectStates.OBJECT_DETECTED, 1)]
                returns PatternMatch.PARTIAL_MATCH

        :param pattern_steps: the list of steps in the pattern. a step can be any python object. the only special
                              case is `NotState`, which is a wrapper over any other step object
        :param state_history: list of tuples, where each tuple represents (state, the duration in seconds till that state)
                              where state is of type `StateHistoryStep`
        :return: one of the types of PatternMatch
        """
        shist_idx = 0
        partial_match_found = False
        while shist_idx < len(state_history):
            result_at_idx = self.find_mov_ptn_in_state_history_at_idx(pattern_steps, state_history, shist_idx)
            if result_at_idx is PatternMatch.MATCHED:
                return PatternMatch.MATCHED
            elif result_at_idx is PatternMatch.PARTIAL_MATCH:
                partial_match_found = True
                shist_idx += 1
                continue
            elif result_at_idx is PatternMatch.NOT_MATCHED:
                shist_idx += 1

        if partial_match_found:
            return PatternMatch.PARTIAL_MATCH
        else:
            return PatternMatch.NOT_MATCHED

    def detect_patterns(self):
        any_partial_match = False
        log.info(colored("stateHistory: %s" % str(self.state_history), 'white'))
        for ptn in self.pattern_evaluation_order:
            ptn_steps = self.pattern_steps[ptn]
            ptn_match_result = self.find_mov_ptn_in_state_history(ptn_steps, self.state_history)
            if ptn_match_result is PatternMatch.MATCHED:
                log.info(colored("pattern detected: %s" % ptn.name, 'red', attrs=['bold']))
                state_attrs = None
                for state_step in self.state_history:
                    if state_step.state == ObjectStates.OBJECT_DETECTED:
                        state_attrs = state_step.state_attrs
                self.state_history.clear()
                self.output_q.enqueue((NotificationTypes.PATTERN_DETECTED, (ptn, state_attrs)))
            elif ptn_match_result is PatternMatch.PARTIAL_MATCH:
                log.info(colored("pattern partial match: %s" % ptn.name, 'red'))
                any_partial_match = True

        self.prune_state_history(any_partial_match)

    def prune_state_history(self, any_partial_match):
        now = int(round(time.time()))
        for state_step in self.state_history:
            if not any_partial_match:
                if now - state_step.ts > self.state_history_length:
                    self.state_history.remove(state_step)
            else:
                if now - state_step.ts > self.state_history_length_partial:
                    self.state_history.remove(state_step)
