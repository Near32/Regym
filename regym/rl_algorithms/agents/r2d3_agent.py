from typing import Dict, Callable, List
from functools import partial
import pickle
import sys
import os
import copy

import torch
import minerl

import regym
from .dqn_agent import generate_model
from .r2d2_agent import R2D2Agent, build_R2D2_Agent, parse_and_check
from ..algorithms.R2D3 import R2D3Algorithm
from regym.rl_algorithms.networks import PreprocessFunction

from regym.util.minerl import get_action_set, generate_action_parser, MineRLTrajectoryBasedEnv, trajectory_based_rl_loop, get_good_demo_names
from regym.util.wrappers import minerl2020_wrap_env
from regym.environments.vec_env import VecEnv
import numpy as np

class R2D3Agent(R2D2Agent):
    def __init__(self, name, algorithm, extra_inputs_infos):
        R2D2Agent.__init__(
            self,
            name=name,
            algorithm=algorithm,
            extra_inputs_infos=extra_inputs_infos
        )

def clone(self, training=None, with_replay_buffer=False, clone_proxies=False):
        '''
        TODO: test
        '''
        cloned_algo = self.algorithm.clone(
            with_replay_buffer=with_replay_buffer,
            clone_proxies=clone_proxies
        )

        clone = R2D3Agent(
            name=self.name,
            algorithm=cloned_algo,
            extra_inputs_infos=copy.deepcopy(self.extra_inputs_infos)
        )

        clone.actor_learner_shared_dict = self.actor_learner_shared_dict
        clone._handled_experiences = self._handled_experiences
        clone.episode_count = self.episode_count
        if training is not None:    clone.training = training
        clone.nbr_steps = self.nbr_steps

        # Goes through all variables 'Proxy' (dealing with multiprocessing)
        # contained in this class and removes them from clone
        if not(clone_proxies):
            proxy_key_values = [
                (key, value) 
                for key, value in clone.__dict__.items() 
                if ('Proxy' in str(type(value)))
            ]
            for key, value in proxy_key_values:
                setattr(clone, key, None)

        return clone

class MineRLTrajectoryEnvironmentCreator():
    def __init__(self, task_name, trajectory_names: List[str], wrapping_fn=None, action_parser: Callable=lambda x:x):
        self.trajectory_names = trajectory_names
        self.wrapping_fn = wrapping_fn

        self.next_env_pointer = 0  # Next environment index to create

        self.envs = []
        for trajectory_name in self.trajectory_names:
            data_pipeline = minerl.data.make(task_name)
            data_iterator = data_pipeline.load_data(trajectory_name)
            self.envs.append(MineRLTrajectoryBasedEnv(data_iterator, action_parser=action_parser))

    def __call__(self, worker_id=None, seed=0):
        env = self.envs[self.next_env_pointer]
        self.next_env_pointer = (self.next_env_pointer + 1) % len(self.trajectory_names)

        env.seed(seed)
        if self.wrapping_fn is not None: env = self.wrapping_fn(env=env)
        return env

def action_parser(action, action_set):
    from sklearn.metrics import pairwise_distances
    true_action = action['vector'] if isinstance(action, dict) else action
    dis = pairwise_distances(action_set, true_action.reshape(1, -1))
    discrete_action = np.argmin(dis, axis=0)
    # (1,)
    return discrete_action

def load_demonstrations_into_replay_buffer(
      agent,
      action_set,
      task_name: str,
      seed: int,
      wrapping_fn: Callable,
      demo_budget=None,
      debug_mode: bool = False,
      base_path:str='./'):
    
    absolute_path = False
    if len(sys.argv) > 2:
      absolute_path = any(['absolute_path' in arg for arg in sys.argv])
    if absolute_path:
      path = os.path.join(base_path, f'{task_name}_good_demo_names.pickle')
    else:
      path = f'{task_name}_good_demo_names.pickle'

    if debug_mode and os.path.exists(path):
        good_demo_names = pickle.load(open(path, 'rb'))
    else:
        good_demo_names = get_good_demo_names(
            task_name,
            path=None,
            score_percent=0.45
        )
        pickle.dump(good_demo_names, open(path, "wb"))

    # Action set
    #continuous_to_discrete_action_parser = generate_action_parser(action_set)
    continuous_to_discrete_action_parser = partial(action_parser,
      action_set=action_set
    )

    next_batch_trajectory_names = []
    for i, demo_name in enumerate(good_demo_names):
        next_batch_trajectory_names += [demo_name]

        if (len(next_batch_trajectory_names) == agent.nbr_actor) or ((i + 1) == len(good_demo_names)):
            if demo_budget is not None and (i+1) > demo_budget:
              break

            env_creator = MineRLTrajectoryEnvironmentCreator(
                task_name=task_name,
                trajectory_names=copy.deepcopy(next_batch_trajectory_names),
                wrapping_fn=wrapping_fn,
                action_parser=continuous_to_discrete_action_parser
            )
            next_batch_trajectory_names = []

            vec_env = VecEnv(
                env_creator=env_creator,
                nbr_parallel_env=agent.nbr_actor,
                seed=seed,
                gathering=False, #True,
                video_recording_episode_period=None,
                video_recording_dirpath=None,
            )

            # Load demoonstrations to agent's replay buffer
            trajectory_based_rl_loop(
                agent=agent,
                minerl_trajectory_env=vec_env,
                action_parser=continuous_to_discrete_action_parser
            )

def build_R2D3_Agent(task: 'regym.environments.Task',
                     config: Dict, 
                     agent_name: str):
    '''
    TODO: say that config is the same as DQN agent except for
    - expert_demonstrations: ReplayStorage object with expert demonstrations
    - demo_ratio: [0, 1] Probability of sampling from expert_demonstrations
                  instead of sampling from replay buffer of gathered
                  experiences. Should be small (i.e 1/256)
    - sequence_length:  TODO

    :returns: R2D3 agent
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

    kwargs = parse_and_check(kwargs, task)

    model = generate_model(task, kwargs)
    
    if kwargs['minerl']:
        # Action set
        action_set = pickle.load(open('{}_action_set.pickle'.format(task.name), 'rb'))
        # Assume rgb frames
        preloading_wrapping_fn = partial(minerl2020_wrap_env,
            action_set=action_set,
            skip=int(kwargs['demo_skip']),
            stack=int(kwargs['demo_stack']),
            grayscale=kwargs['demo_grayscale'],
            previous_reward_action=True,
            trajectory_wrapping=True
        )
        
        debug_mode = False
        if len(sys.argv) > 2:
            debug_mode = any(['debug' in arg for arg in sys.argv])
        
        dummy_config = copy.deepcopy(config)
        dummy_config['nbr_actor'] = 1
        dummy_r2d2_agent = build_R2D2_Agent(task,dummy_config,'Dummy_Agent')
        
        load_demonstrations_into_replay_buffer(dummy_r2d2_agent,action_set,task_name=task.name,seed=task.env.seed,wrapping_fn=preloading_wrapping_fn,demo_budget=int(kwargs['demo_budget']),debug_mode=debug_mode)
        
        expert_buffer = copy.deepcopy(dummy_r2d2_agent.algorithm.storages[0])

    else:
        if kwargs["expert_buffer_path"] != "":
            expert_agent = torch.load(kwargs['expert_buffer_path'])
            expert_buffer = copy.deepcopy(expert_agent.algorithm.storages[0])
        else:
            expert_buffer = None 
            
    algorithm = R2D3Algorithm(
        kwargs=kwargs, 
        model=model,
        expert_buffer=expert_buffer,
        name=f"{agent_name}_algo",
    )

    agent = R2D3Agent(
        name=agent_name,
        algorithm=algorithm,
        extra_inputs_infos=kwargs['extra_inputs_infos'],
    )

    return agent
