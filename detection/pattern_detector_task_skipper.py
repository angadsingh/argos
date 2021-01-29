import logging

from termcolor import colored

from detection.StateDetectorBase import SkipAheadOptimizer
from detection.states import NotState

log = logging.getLogger(__name__)


class PatternBasedSkipAheadOptimizer(SkipAheadOptimizer):
    def __init__(self, pattern_detector, skip_state_type):
        super().__init__(skip_state_type)
        self.pattern_detector = pattern_detector

    def skip_task(self, ts):
        """
        find out if a state of type skip_state_type is needed to be
        detected by the pattern detector in the state history at time ts
        by any of the patterns
        """
        states_in_demand = self.pattern_detector.states_in_demand(ts)
        log.info("states_in_demand by all patterns: %s" % str(states_in_demand))
        state_history_till_ts = self.pattern_detector.get_state_history_till(ts)
        state_history_after_ts = self.pattern_detector.get_state_history_after(ts)
        log.info("%s, %s " % (
        colored("state history seen by skipper: %s" % state_history_till_ts, 'white', attrs=['bold']),
        colored(state_history_after_ts, 'white')))
        skip_ahead = True
        for state_looked_for in states_in_demand:
            state = state_looked_for
            if type(state_looked_for) is NotState:
                state = state_looked_for.state
            if self.skip_state_type == state:
                skip_ahead = False
                break

        self.measure_speedup(skip_ahead)
        return skip_ahead
