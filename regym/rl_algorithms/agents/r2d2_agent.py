from typing import Dict

from functools import partial 

from .dqn_agent import DQNAgent, generate_model
from regym.rl_algorithms.algorithms.R2D2 import R2D2Algorithm
from ..networks import PreprocessFunction, ResizeCNNPreprocessFunction, ResizeCNNInterpolationFunction


class R2D2Agent(DQNAgent):
    def clone(self, training=None, with_replay_buffer=False):
        '''
        TODO: test
        '''
        cloned_algo = self.algorithm.clone(with_replay_buffer=with_replay_buffer)
        clone = R2D2Agent(name=self.name, algorithm=cloned_algo)
        
        clone.handled_experiences = self.handled_experiences
        clone.episode_count = self.episode_count
        if training is not None:    clone.training = training
        clone.nbr_steps = self.nbr_steps
        return clone


def build_R2D2_Agent(task: 'regym.environments.Task',
                     config: Dict,
                     agent_name: str):
    '''
    TODO: say that config is the same as DQN agent except for
    - expert_demonstrations: ReplayStorage object with expert demonstrations
    - demo_ratio: [0, 1] Probability of sampling from expert_demonstrations
                  instead of sampling from replay buffer of gathered
                  experiences. Should be small (i.e 1/256)
    - sequence_length:  TODO

    :returns: R2D2 agent
    '''

    kwargs = config.copy()
    kwargs['discount'] = float(kwargs['discount'])
    kwargs['replay_capacity'] = int(float(kwargs['replay_capacity']))
    kwargs['min_capacity'] = int(float(kwargs['min_capacity']))

    # Default preprocess function:
    kwargs['state_preprocess'] = partial(PreprocessFunction, normalization=False)
    kwargs['goal_preprocess'] = partial(PreprocessFunction, normalization=False)

    if not isinstance(kwargs['observation_resize_dim'], int):  kwargs['observation_resize_dim'] = task.observation_shape[0] if isinstance(task.observation_shape, tuple) else task.observation_shape
    #if 'None' in kwargs['goal_resize_dim']:  kwargs['goal_resize_dim'] = task.goal_shape[0] if isinstance(task.goal_shape, tuple) else task.goal_shape

    model = generate_model(task, kwargs)
    
    algorithm = R2D2Algorithm(
        kwargs=kwargs, 
        model=model,
    )
    
    agent = R2D2Agent(name=agent_name, algorithm=algorithm)
    
    return agent
