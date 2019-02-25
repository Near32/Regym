import random
from ..TQL import TabularQLearningAlgorithm


class TabularQLearningAgent():
    def __init__(self, algorithm):
        self.name = 'TabularQLearning'
        self.algorithm = algorithm

    def handle_experience(self, s, a, r, succ_s, done=False):
        if self.algorithm.training:
            self.algorithm.update_q_table(self.hashing_function(s), a, r, self.hashing_function(succ_s))

    def take_action(self, state):
        optimal_moves = self.algorithm.find_optimal_moves(state)
        return random.choice(optimal_moves)

    def clone(self, training=None, path=None):
        from ..agent_hook import AgentHook
        cloned = AgentHook(self, training=training, path=path)
        return cloned


def build_TabularQ_Agent(task, config):
    state_space_size, action_space_size = task.state_space_size, task.action_dim
    hash_state = task.hash_function
    vanilla_q_learning = TabularQLearningAlgorithm(state_space_size, action_space_size, hash_state,
                                                   learning_rate=config['learning_rate'])
    return TabularQLearningAgent(algorithm=vanilla_q_learning)