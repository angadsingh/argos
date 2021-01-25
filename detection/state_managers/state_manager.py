from abc import ABC, abstractmethod


class StateManager(ABC):
    def __init__(self, state_history, output_q):
        self.state_history = state_history
        self.output_q = output_q

    @abstractmethod
    def add_state(self, state):
        pass