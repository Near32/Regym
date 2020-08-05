from typing import Dict

import .dqn_agent import DQNAgent
from regym.rl_algorithms.R2D3 import R2D3Algorithm
from regym.rl_algorithms.R2D3 import R2D3Algorithm


class R2D3Agent(DQNAgent):
    def clone(self, training=None):
        '''
        TODO: test
        '''
        cloned_algo = self.algorithm.clone()
        clone = R2D3Agent(name=self.name, algorithm=cloned_algo)
        clone.handled_experiences = self.handled_experiences
        clone.episode_count = self.episode_count
        if training is not None: clone.training = training
        clone.nbr_steps = self.nbr_steps
        return clone


def build_R2D3_Agent(task: 'regym.environments.Task',
                     config: Dict, agent_name: str):
    '''
    TODO: say that config is the same as DQN agent except for
    - expert_demonstrations: ReplayStorage object with expert demonstrations
    - demo_ratio: [0, 1] Probability of sampling from expert_demonstrations
                  instead of sampling from replay buffer of gathered
                  experiences. Should be small (i.e 1/256)
    - sequence_length:  TODO

    :returns: R2D3 agent
    '''
    model = dqn_agent.generate_model(task, config)
    algorithm = R2D3Algorithm(
            config, model=model,
            expert_demonstrations=config['expert_demonstrations'])
    agent = R2D3Agent(name=agent_name, algorithm=algorithm)
    return agent
