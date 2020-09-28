from typing import Dict
from functools import partial

import copy
import torch

from regym.rl_algorithms.agents.agent import ExtraInputsHandlingAgent
from regym.rl_algorithms.agents.dqn_agent import DQNAgent, generate_model
from regym.rl_algorithms.algorithms.R2D2 import R2D2Algorithm
from regym.rl_algorithms.networks import PreprocessFunction, ResizeCNNPreprocessFunction, ResizeCNNInterpolationFunction


class R2D2Agent(ExtraInputsHandlingAgent, DQNAgent):
    def __init__(self, name, algorithm, extra_inputs_infos):
        # Both init will call the self's reset_rnn_states following self.mro's order, i.e. ExtraInputs's one first.
        ExtraInputsHandlingAgent.__init__(
            self, 
            name=name, 
            algorithm=algorithm,
            extra_inputs_infos=extra_inputs_infos
        )
        DQNAgent.__init__(
            self, 
            name=name, 
            algorithm=algorithm
        )

    def _handle_experience(self, s, a, r, succ_s, done, goals=None, infos=None):
        '''
        Note: the batch size may differ from the nbr_actor as soon as some
        actors' episodes end before the others...

        :param s: numpy tensor of states of shape batch x state_shape.
        :param a: numpy tensor of actions of shape batch x action_shape.
        :param r: numpy tensor of rewards of shape batch x reward_shape.
        :param succ_s: numpy tensor of successive states of shape batch x state_shape.
        :param done: list of boolean (batch=nbr_actor) x state_shape.
        :param goals: Dictionnary of goals 'achieved_goal' and 'desired_goal' for each state 's' and 'succ_s'.
        :param infos: Dictionnary of information from the environment.
        '''
        DQNAgent.handle_experience(
            self,
            s=s,
            a=a,
            r=r,
            succ_s=succ_s,
            done=done,
            goals=goals,
            infos=infos
        )

    def clone(self, training=None, with_replay_buffer=False):
        '''
        TODO: test
        '''
        cloned_algo = self.algorithm.clone(with_replay_buffer=with_replay_buffer)
        clone = R2D2Agent(
            name=self.name, 
            algorithm=cloned_algo,
            extra_inputs_infos=copy.deepcopy(self.extra_inputs_infos))

        clone.actor_learner_shared_dict = self.actor_learner_shared_dict
        clone.handled_experiences = self.handled_experiences
        clone.episode_count = self.episode_count
        if training is not None:    clone.training = training
        clone.nbr_steps = self.nbr_steps
        return clone


    def get_async_actor(self, training=None, with_replay_buffer=False):
        '''
        TODO: test
        '''
        self.async_learner = True
        self.async_actor = False

        cloned_algo = self.algorithm.async_actor()
        clone = R2D2Agent(
            name=self.name, 
            algorithm=cloned_algo,
            extra_inputs_infos=copy.deepcopy(self.extra_inputs_infos)
        )
        clone.async_learner = False
        clone.async_actor = True

        clone.actor_learner_shared_dict = self.actor_learner_shared_dict
        clone.handled_experiences = self.handled_experiences
        clone.episode_count = self.episode_count
        if training is not None:    clone.training = training
        clone.nbr_steps = self.nbr_steps
        return clone


def parse_and_check(kwargs: Dict,
                    task: 'regym.environments.Task'):
    
    # Extra Inputs:
    kwargs['task'] = task 

    extra_inputs = kwargs['extra_inputs_infos']
    for key in extra_inputs:
        shape = extra_inputs[key]['shape']
        for idxdim, dimvalue in enumerate(shape):
            if isinstance(dimvalue, str):
                path = dimvalue.split('.')
                if len(path) > 1:
                    pointer = kwargs
                    for el in path:
                        try:     
                            if hasattr(pointer, el):
                                pointer = getattr(pointer, el)
                            elif el in pointer: 
                                pointer = pointer[el]
                            else:
                                raise RuntimeError
                        except:
                            raise RuntimeError
                else:
                    pointer = path
                
                try: 
                    pointer = int(pointer)
                except Exception as e:
                    print('Exception during parsing and checking:', e)
                    raise e
                shape[idxdim] = pointer

    return kwargs    

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

    kwargs = parse_and_check(kwargs, task)

    model = generate_model(task, kwargs)

    algorithm = R2D2Algorithm(
        kwargs=kwargs,
        model=model,
    )

    agent = R2D2Agent(
        name=agent_name, 
        algorithm=algorithm,
        extra_inputs_infos=kwargs['extra_inputs_infos'],
    )
    
    return agent
