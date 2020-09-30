from functools import partial
import json
import select
import time
import logging
import os
import threading


from typing import Callable

#import aicrowd_helper
import gym
import minerl
import abc
import numpy as np

import coloredlogs
coloredlogs.install(logging.DEBUG)

import pickle
import torch

import regym
from regym.environments.utils import EnvironmentCreator
from regym.environments.vec_env import VecEnv
from regym.util.wrappers import minerl2020_wrap_env

# All the evaluations will be evaluated on MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))

# Parallel testing/inference, **you can override** below value based on compute
# requirements, etc to save OOM in this phase.
#EVALUATION_THREAD_COUNT = int(os.getenv('EPISODES_EVALUATION_THREAD_COUNT', 2))
EVALUATION_THREAD_COUNT = int(os.getenv('EPISODES_EVALUATION_THREAD_COUNT', 1))


class EpisodeDone(Exception):
    pass

class Episode(gym.Env):
    """A class for a single episode.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s,r,d,i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s,r,d,i



# DO NOT CHANGE THIS CLASS, THIS IS THE BASE CLASS FOR YOUR AGENT.
class MineRLAgentBase(abc.ABC):
    """
    To compete in the competition, you are required to implement a
    SUBCLASS to this class.
    
    YOUR SUBMISSION WILL FAIL IF:
        * Rename this class
        * You do not implement a subclass to this class 
    This class enables the evaluator to run your agent in parallel, 
    so you should load your model only once in the 'load_agent' method.
    """

    @abc.abstractmethod
    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_agent_on_episode(self, single_episode_env : Episode):
        """This method runs your agent on a SINGLE episode.
        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs)) 
                ...
        
        NOTE: This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.
        Args:
            env (gym.Env): The env your agent should interact with.
        """
        raise NotImplementedError()


class MockEnvironmentCreator():
    def __init__(self, env: gym.Env, wrapping_fn):
        self.env = env
        self.wrapping_fn = wrapping_fn

    def __call__(self, worker_id, seed):
        ''' Need to have params even if they are ignored '''
        wrapped_env = self.wrapping_fn(env=self.env)
        return wrapped_env


class MineRLRegymAgent(MineRLAgentBase):
    def load_agent(self):
        self.action_set_path = './MineRLObtainDiamondVectorObf-v0_action_set.pickle'
        self.agent_path = './mark_debug_agent.pt'

        self.action_set = pickle.load(open(self.action_set_path, 'rb'))
        if not torch.cuda.is_available():
            self.agent = torch.load(self.agent_path, map_location=torch.device('cpu'))
            self.agent.use_cuda = False
            self.agent.algorithm.kwargs['use_cuda'] = False
        else:
            self.agent = torch.load(self.agent_path)

        self.agent.training = False
        self.agent.set_nbr_actor(nbr_actor=1)

    def run_agent_on_episode(self, single_episode_env : Episode):

        # Wrapping environment with everything we need
        wrapping_fn = partial(minerl2020_wrap_env,
            env=single_episode_env.env,
            action_set=self.action_set,
            skip=4,
            stack=4,
            previous_reward_action=True,
            trajectory_wrapping=False,
            competition_testing=True
        )

        env_creator = MockEnvironmentCreator(
            single_episode_env.env,
            wrapping_fn=wrapping_fn
        )

        single_episode_env.env = VecEnv(
            env_creator=env_creator,
            nbr_parallel_env= self.agent.nbr_actor,
            seed=0,  #  AICrowd people, this doesn't actually set the seed
            gathering=False, #True,
            video_recording_episode_period=None,
            video_recording_dirpath=None,
        )

        obs = single_episode_env.reset()
        done, info = False, None

        while not done:
            action = self.agent.take_action(obs, info)
            succ_obs, reward, done, info = single_episode_env.step(action)
            obs = succ_obs

#####################################################################
# IMPORTANT: SET THIS VARIABLE WITH THE AGENT CLASS YOU ARE USING   # 
######################################################################
AGENT_TO_TEST = MineRLRegymAgent



####################
# EVALUATION CODE  #
####################
def main():
    agent = AGENT_TO_TEST()
    assert isinstance(agent, MineRLAgentBase)
    agent.load_agent()

    assert MINERL_MAX_EVALUATION_EPISODES > 0
    assert EVALUATION_THREAD_COUNT > 0

    action_set_file = 'action_set.pickle'
    action_set = pickle.load(open(action_set_file, 'rb'))

    # Create the parallel envs (sequentially to prevent issues!)
    envs = [gym.make(MINERL_GYM_ENV) for _ in range(EVALUATION_THREAD_COUNT)]
    episodes_per_thread = [MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT for _ in range(EVALUATION_THREAD_COUNT)]
    episodes_per_thread[-1] += MINERL_MAX_EVALUATION_EPISODES - EVALUATION_THREAD_COUNT *(MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT)
    # A simple funciton to evaluate on episodes!
    def evaluate(i, env):
        print("[{}] Starting evaluator.".format(i))
        for i in range(episodes_per_thread[i]):
            try:
                agent.run_agent_on_episode(Episode(env))
            except EpisodeDone:
                print("[{}] Episode complete".format(i))
                pass
    
    evaluator_threads = [threading.Thread(target=evaluate, args=(i, envs[i])) for i in range(EVALUATION_THREAD_COUNT)]
    for thread in evaluator_threads:
        thread.start()

    # wait fo the evaluation to finish
    for thread in evaluator_threads:
        thread.join()

if __name__ == "__main__":
    main()
