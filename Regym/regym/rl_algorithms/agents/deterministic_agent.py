from typing import Dict, List

from .agent import Agent


class DeterministicAgent(Agent):

    def __init__(self, action: int = None,
                 action_sequence: List[int] = None,
                 name: str = 'DeterministicAgent'):
        self.name = name
        self.action = action

        self.action_index = 0
        self.action_sequence = action_sequence

    def take_action(self, state):
        if self.action: return action
        else:
            action = self.action_sequence[self.action_index]
            self.action_index = (self.action_index + 1) % len(self.action_sequence)
            return action

    def clone(self, training=None):
        clone = DeterministicAgent(action=self.action,
                                   action_sequence=self.action_sequence,
                                   name=self.name)
        clone.action_index = self.action_index
        return clone

    def handle_experience(self, s, a, r, succ_s, done=False):
        pass

    def __repr__(self):
        return f'DeterministicAgent: action: {self.action}.\t action_sequence: {self.action_sequence}, action index: {self.action_index}'


def build_Deterministic_Agent(task, config: Dict, agent_name: str):
    if task.action_type != 'Discrete':
        raise ValueError('Deterministic agents (currently) only support Discrete action spaces')
    if 'action' in config and 'action_sequence' in config:
        raise ValueError(':param: config (dictionary) should only define ONE of the two keys \'action\' and \'action_sequence\', both were passed')
    return DeterministicAgent(action=config.get('action', None),
                              action_sequence=config.get('action_sequence', None),
                              name=agent_name)
