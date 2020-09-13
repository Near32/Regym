from typing import Dict
from functools import partial

import torch

from .dqn_agent import DQNAgent, generate_model
from regym.rl_algorithms.algorithms.R2D2 import R2D2Algorithm
from ..networks import PreprocessFunction, ResizeCNNPreprocessFunction, ResizeCNNInterpolationFunction


class R2D2Agent(DQNAgent):

    def __init__(self, name, algorithm, action_space_dim):
        super(R2D2Agent, self).__init__(name, algorithm)

        self.action_space_dim = action_space_dim
        self.previous_reward: torch.Tensor = None
        
    # NOTE: overriding from DQNAgent
    def query_model(self, model, state, goal):
        batch_size = state.shape[0]
        if self.current_prediction:
            # Turn previous action to one-hot
            one_hot = torch.zeros(batch_size, self.action_space_dim)
            for actor_i, action_i in enumerate(self.current_prediction['a']):
                one_hot[actor_i, action_i] = 1.
            previous_action = one_hot
        else:
            dummy_action = torch.zeros(batch_size, self.action_space_dim)
            previous_action = dummy_action
            self.previous_reward = torch.zeros(batch_size,1)

        if self.recurrent:
            self._pre_process_rnn_states()
            current_prediction = model(state, rnn_states=self.rnn_states,
                                       previous_action=previous_action,
                                       previous_reward=self.previous_reward,
                                       goal=goal)
        else:
            current_prediction = model(state, goal=goal)
        return current_prediction

    # NOTE: overriding from DQNAgent
    def handle_experience(self, s, a, r, succ_s, done, goals=None, infos=None):
        super().handle_experience(s, a, r, succ_s, done, goals=None, infos=None)
        _, r, _, _ = self.preprocess_environment_signals(s, r, succ_s, done)
        self.previous_reward = r

    def clone(self, training=None, with_replay_buffer=False):
        '''
        TODO: test
        '''
        cloned_algo = self.algorithm.clone(with_replay_buffer=with_replay_buffer)
        clone = R2D2Agent(name=self.name, algorithm=cloned_algo,
                          action_space_dim=self.action_space_dim)

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

    # Clarify a lil'
    # We need to add extra features to LSTM input, explained in R2D2 appendix
    # (i.e appending one-hot encoded of previous action and previous reward)
    kwargs['lstm_input_dim'] = kwargs['phi_arch_hidden_units'][0] + (task.action_dim + 1)  # +1 represents scalar reward
    kwargs['phi_arch_hidden_units'][-1] += task.action_dim + 1

    model = generate_model(task, kwargs)

    algorithm = R2D2Algorithm(
        kwargs=kwargs,
        model=model,
    )

    agent = R2D2Agent(name=agent_name, algorithm=algorithm,
                      action_space_dim=task.action_dim)

    return agent
