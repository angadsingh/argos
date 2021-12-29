import copy
import logging
import threading
import time
from enum import Enum

import numpy as np
from termcolor import colored

from detection.state_managers.object_state_manager import ObjectStates
from detection.state_managers.state_manager import StateManager, CommittedOffset
from detection.states import NotState, StateHistoryStep
from lib.task_queue import BlockingTaskSingleton
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
    def __init__(self, broker_q: BlockingTaskSingleton, pattern_steps, state_history_length=20,
                 state_history_length_partial=300, detection_interval=1):
        self.broker_q = broker_q
        self.pattern_steps = pattern_steps
        self.state_history = []
        self.state_history_length = state_history_length
        self.state_history_length_partial = state_history_length_partial
        self.detection_interval = detection_interval
        self.state_history_update_lock = threading.Lock()
        self.state_managers = []
        if self.detection_interval:
            self.pattern_detection_timer = RepeatedTimer(detection_interval, self.detect_patterns)

    def stop(self):
        if self.detection_interval:
            self.pattern_detection_timer.stop()

    def register_state_manager(self, state_manager: StateManager):
        self.state_managers.append(state_manager)

    def add_to_state_history(self, new_state: StateHistoryStep, avoid_duplicates=False):
        inserted = False
        with self.state_history_update_lock:
            i = 0
            while i < len(self.state_history) and new_state.ts >= self.state_history[i].ts:
                i += 1

            if not avoid_duplicates or (
                    avoid_duplicates and (i == 0 or self.state_history[i - 1].state is not new_state.state)):
                self.state_history.insert(i, new_state)
                inserted = True

        # self.detect_patterns()
        return inserted

    def find_not_state_before_step(self, not_state: NotState, state_history, after_step_ts, from_idx, to_idx):
        while from_idx >= to_idx:
            shist_step: StateHistoryStep = state_history[from_idx]
            if not_state.state == shist_step.state and abs(after_step_ts - shist_step.ts) <= not_state.duration:
                return True
            from_idx -= 1

    def find_mov_ptn_in_state_history_at_idx(self, pattern_steps, state_history, shist_idx=0, now=None):
        ptn_idx = 0
        prev_match_idx = -1
        prev_match_ts = 0
        if not now or now == CommittedOffset.CURRENT:
            now = max(time.time(), state_history[-1].ts)

        while ptn_idx < len(pattern_steps) and shist_idx < len(state_history) and state_history[shist_idx].ts <= now:
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
                    if (now - prev_match_ts > pattern_steps[ptn_idx].duration or pattern_steps[
                        ptn_idx].duration == np.inf) \
                            and self.find_not_state_before_step(ptn_step, state_history, prev_match_ts,
                                                                len(state_history) - 1, shist_idx):
                        ptn_idx = 0
                        break
                    break
                ptn_idx += 1
            else:
                shist_idx += 1

        if ptn_idx == 0 or (ptn_idx == 1 and type(pattern_steps[ptn_idx - 1]) is NotState):
            states_to_find = [pattern_steps[ptn_idx]] if type(pattern_steps[ptn_idx ]) is not NotState else []
            if ptn_idx == 1 and type(pattern_steps[ptn_idx - 1]):
                states_to_find = [pattern_steps[ptn_idx], pattern_steps[ptn_idx - 1]]
            return (PatternMatch.NOT_MATCHED, states_to_find, 0)
        elif 0 < ptn_idx <= len(pattern_steps) - 1:
            if type(pattern_steps[ptn_idx]) is NotState and ptn_idx == len((pattern_steps)) - 1 \
                    and now - prev_match_ts > pattern_steps[ptn_idx].duration:
                return (PatternMatch.MATCHED, [], len(pattern_steps))
            else:
                states_to_find = [pattern_steps[ptn_idx], pattern_steps[ptn_idx - 1]] if type(
                    pattern_steps[ptn_idx - 1]) is NotState else [pattern_steps[ptn_idx]]
                return (PatternMatch.PARTIAL_MATCH, states_to_find, ptn_idx)
        else:
            return (PatternMatch.MATCHED, [], len(pattern_steps))

    def find_mov_ptn_in_state_history(self, pattern_steps, state_history, now=None):
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
        all_states_to_find = {PatternMatch.MATCHED: [], PatternMatch.NOT_MATCHED: [], PatternMatch.PARTIAL_MATCH: []}
        if len(state_history) > 0:
            while shist_idx < len(state_history):
                result_at_idx, states_to_find, num_steps_matched = self.find_mov_ptn_in_state_history_at_idx(pattern_steps, state_history, shist_idx, now)
                all_states_to_find[result_at_idx].append((states_to_find, num_steps_matched))
                if result_at_idx is PatternMatch.MATCHED:
                    return (PatternMatch.MATCHED, all_states_to_find[PatternMatch.MATCHED][0][0])
                elif result_at_idx is PatternMatch.PARTIAL_MATCH:
                    partial_match_found = True
                    shist_idx += 1
                    continue
                elif result_at_idx is PatternMatch.NOT_MATCHED:
                    shist_idx += 1

            if partial_match_found:
                return (PatternMatch.PARTIAL_MATCH, sorted(all_states_to_find[PatternMatch.PARTIAL_MATCH], key = lambda x: x[1], reverse=True)[0][0])
            else:
                return (PatternMatch.NOT_MATCHED, all_states_to_find[PatternMatch.NOT_MATCHED][0][0])
        else:
            return (PatternMatch.NOT_MATCHED, [])

    def get_min_committed_offset_ts(self):
        """
        get the latest timestamp uptil which all state managers
        have committed their state (have processed the state history
        till that timestamp)
        :return:
        """
        min_committed_offset = np.inf
        lagging_state_managers = 0
        lagging_mgr_frames = 0
        now = time.time()
        for mg in self.state_managers:
            mgr_offset = mg.get_latest_committed_offset()
            mgr_current_lag = mg.get_current_lag()

            if mgr_offset != CommittedOffset.CURRENT and mgr_current_lag > 0:
                lagging_state_managers += 1
                lagging_mgr_frames += mgr_current_lag
                log.info(colored(
                    "state manager [%s] lagging by [%.2f] seconds and [%d] frames" % (
                        mg.__class__.__name__, now - mgr_offset, mgr_current_lag), 'white', attrs=['bold']))
                if mgr_offset < min_committed_offset:
                    min_committed_offset = mgr_offset

        if min_committed_offset is np.inf:
            min_committed_offset = CommittedOffset.CURRENT
        else:
            log.info(colored(
                "pattern detection lagging by [%.2f] seconds due to %d state managers lagging by [%d] frames" % (
                    now - min_committed_offset, lagging_state_managers, lagging_mgr_frames), 'white', attrs=['bold']))

        return min_committed_offset

    def detect_patterns(self):
        if len(self.state_history) > 0:
            min_committed_offset_ts = self.get_min_committed_offset_ts()
            any_partial_match = False
            state_history_till_ts = self.get_state_history_till(min_committed_offset_ts)
            state_history_after_ts = self.get_state_history_after(min_committed_offset_ts)
            log.info("%s, %s " % (
                colored("state history seen by pattern detector: %s" % state_history_till_ts, 'white', attrs=['bold']),
                colored(state_history_after_ts, 'white')))
            for (ptn, ptn_steps) in self.pattern_steps:
                ptn_match_result, states_to_find = self.find_mov_ptn_in_state_history(ptn_steps, self.state_history, now=min_committed_offset_ts)
                if ptn_match_result is PatternMatch.MATCHED:
                    log.info(colored("pattern detected: %s" % ptn.name, 'red', attrs=['bold']))
                    state_attrs = None
                    for state_step in self.state_history:
                        if state_step.state == ObjectStates.OBJECT_DETECTED:
                            state_attrs = state_step.state_attrs
                    self.clear_state_history_till(min_committed_offset_ts)
                    self.broker_q.enqueue((NotificationTypes.PATTERN_DETECTED, (ptn, state_attrs)))
                elif ptn_match_result is PatternMatch.PARTIAL_MATCH:
                    log.info(colored("pattern partial match: %s" % ptn.name, 'red'))
                    any_partial_match = True

            self.prune_state_history(any_partial_match)

    def states_in_demand(self, ts):
        all_states_to_find = set()
        for (ptn, ptn_steps) in self.pattern_steps:
            ptn_match_result, states_to_find = self.find_mov_ptn_in_state_history(ptn_steps, self.state_history, now=ts)
            log.info("states_in_demand by [%s] pattern: %s" % (ptn, str(states_to_find)))
            all_states_to_find.update(states_to_find)
        return all_states_to_find

    def clear_state_history_till(self, now):
        with self.state_history_update_lock:
            if now != CommittedOffset.CURRENT:
                new_state_history = []
                for i in range(0, len(self.state_history)):
                    if self.state_history[i].ts > now:
                        new_state_history.append(self.state_history[i])
                self.state_history = new_state_history
            else:
                self.state_history.clear()

    def get_state_history_till(self, now):
        if now != CommittedOffset.CURRENT:
            state_history_at_now = copy.deepcopy(self.state_history)
            for state in state_history_at_now:
                state.now = now
            return [state_step for state_step in state_history_at_now if state_step.ts <= now]
        else:
            return self.state_history

    def get_state_history_after(self, now):
        if now != CommittedOffset.CURRENT:
            state_history_at_now = copy.deepcopy(self.state_history)
            for state in state_history_at_now:
                state.now = now
            return [state_step for state_step in state_history_at_now if state_step.ts > now]
        else:
            return []

    def prune_state_history(self, any_partial_match):
        with self.state_history_update_lock:
            now = int(round(time.time()))
            for state_step in self.state_history:
                if not any_partial_match:
                    if now - state_step.ts > self.state_history_length:
                        self.state_history.remove(state_step)
                else:
                    if now - state_step.ts > self.state_history_length_partial:
                        self.state_history.remove(state_step)
