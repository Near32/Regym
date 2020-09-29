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

# All the evaluations will be evaluated on MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))

# Parallel testing/inference, **you can override** below value based on compute
# requirements, etc to save OOM in this phase.
EVALUATION_THREAD_COUNT = int(os.getenv('EPISODES_EVALUATION_THREAD_COUNT', 2))

class PreviousRewardActionInfoWrapper(gym.Wrapper):
    """
    Integrates the previous reward and previous action into the info dictionnary.
    Args:
        env (gym.Env): Env to wrap.
    """

    def __init__(self, env, trajectory_wrapping=False):
        super(PreviousRewardActionInfoWrapper, self).__init__(env)
        self.nbr_actions = env.action_space.n
        self.trajectory_wrapping = trajectory_wrapping

    def reset(self):
        self.previous_reward = np.zeros((1, 1))
        self.previous_action = np.zeros((1, self.nbr_actions))
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        info['previous_reward'] = copy.deepcopy(self.previous_reward)
        if self.trajectory_wrapping:
            # Only perform discrete-to-ohe transformation:
            # No need to fetch the previous value, it is already given.
            info['previous_action'] = np.eye(self.nbr_actions, dtype=np.float32)[info['previous_action'][0]].reshape(1, -1)
        else:
            # Fetch the previous value:
            info['previous_action'] = copy.deepcopy(self.previous_action)

        self.previous_reward = np.ones((1, 1), dtype=np.float32)*reward
        if not(self.trajectory_wrapping):
            # Perform the discrete-to-ohe transformation:
            # And the value will be used at the next step.
            self.previous_action = np.eye(self.nbr_actions, dtype=np.float32)[action].reshape(1, -1)

        return observation, reward, done, info

class FrameSkipStack(gym.Wrapper):
    """
    Return a stack of frames composed of every 'skip'-th repeat.
    The rewards are summed over the skipped and stacked frames.
    
    This wrapper assumes:
    - the observation space of the environment to be frames solely.
    - the frames are concatenated on the last axis, i.e. the channel axis.
    """
    def __init__(self, env, skip=8, stack=4, trajectory_wrapping=False):
        gym.Wrapper.__init__(self,env)
        self.skip = skip if skip is not None and not(trajectory_wrapping) else 1
        self.stack = stack if stack is not None else 1
        self.trajectory_wrapping = trajectory_wrapping
        
        self.observations = deque([], maxlen=self.stack)
        
        assert(isinstance(self.env.observation_space, gym.spaces.Box))
        
        low_obs_space = np.repeat(self.env.observation_space.low, self.stack, axis=-1)
        high_obs_space = np.repeat(self.env.observation_space.high, self.stack, axis=-1)
        self.observation_space = gym.spaces.Box(low=low_obs_space, high=high_obs_space, dtype=self.env.observation_space.dtype)
    
    def _get_obs(self):
        assert(len(self.observations) == self.stack)
        return LazyFrames(list(self.observations))
        
    def reset(self, **args):
        obs = self.env.reset()
        for _ in range(self.stack):
            self.observations.append(obs)
        return self._get_obs()
    
    def step(self, action):
        total_reward = 0.0
        infos = []
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            infos.append(info)
            total_reward += reward
            if done:break
        self.observations.append(obs)
        
        # When wrapping  trajectory env,
        # the actual previous action is in the initial info:
        if self.trajectory_wrapping:
            # It could be worth considering sampling from the list of infos
            # for the previous_action the most representative of the current
            # set of actions by weighting proportionaly...
            info['previous_action'] = infos[0]['previous_action']
            info['current_action'] = infos[0]['current_action']

        return self._get_obs(), total_reward, done, info

class DiscreteActionWrapper(gym.ActionWrapper):
    '''
    Given an actions set
    Convert continuous action to nearest discrete action
    '''
    def __init__(self, env, action_set, key_name=None):
        super().__init__(env)
        self.action_set = action_set
        self.key_name = key_name
        self.action_space = gym.spaces.Discrete(len(action_set))
        
    def action(self,action):
        if self.key_name == None:
            return self.action_set[action]
        else:
            return {self.key_name: self.action_set[action]}

class MineRLObservationSplitWrapper(gym.Wrapper):
    """
    Split state dictionary into pov and inventory
    """
    def __init__(self, env, trajectory_wrapping=False):
        gym.Wrapper.__init__(self, env)
        self.trajectory_wrapping = trajectory_wrapping
        self.observation_space = gym.spaces.Box(low=0.0, high=255.0, shape=(64,64,3), dtype=np.float32)
        
    def reset(self,**args):
        obs = self.env.reset()
        return obs['pov']
    
    def step(self,action):
        obs, reward, done, info = self.env.step(action)
        if not(self.trajectory_wrapping):
            info['inventory'] = np.expand_dims(obs['vector'], axis=0)
        return obs['pov'], reward, done, info

class ContinuingTimeLimit(gym.Wrapper):
    """TimeLimit wrapper for continuing environments.

    Adapted from:
    https://github.com/chainer/chainerrl/blob/5d833d6cb3b6e7de0b5bfa7cc8c8534516fbd7ba/chainerrl/wrappers/continuing_time_limit.py
    
    This is similar gym.wrappers.TimeLimit, which sets a time limit for
    each episode, except that done=False is returned and that
    info['real_done'] is set to True when past the limit.
    Code that calls env.step is responsible for checking the info dict, the
    fourth returned value, and resetting the env if it has the 'needs_reset'
    key and its value is True.
    Args:
        env (gym.Env): Env to wrap.
        max_episode_steps (int): Maximum number of timesteps during an episode,
            after which the env needs a reset.
    """

    def __init__(self, env, max_episode_steps):
        super(ContinuingTimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps

        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None,\
            "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._max_episode_steps <= self._elapsed_steps:
            info['real_done'] = True

        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()

def minerl2020_wrap_env(env,
                        action_set,
                        skip=None,
                        stack=None,
                        previous_reward_action=True,
                        trajectory_wrapping=False):
    '''
    Add all wrappers need for minerl 2020
    '''
    if isinstance(env,gym.wrappers.TimeLimit):
        env = env.env
        max_episode_steps = env.spec.max_episode_steps
        env = ContinuingTimeLimit(env,max_episode_steps=max_episode_steps)
    
    # {POV, vector}, continuous action
    env = MineRLObservationSplitWrapper(env=env, trajectory_wrapping=trajectory_wrapping)
    # state=POV, continuous action, 
    # infos={inventory (if traj_wrap: , previous_action(d), current_action(d))}
    env = DiscreteActionWrapper(env, action_set, 'vector')
    # state=POV, input action is discrete, propagated action is continuous, 
    # infos={inventory (if traj_wrap: , previous_action(d), current_action(d))}
    
    if skip is not None or stack is not None:
        env = FrameSkipStack(
            env=env, 
            skip=skip, 
            stack=stack,
            trajectory_wrapping=trajectory_wrapping
        )
        # state=POV, input action is discrete, propagated action is continuous, 
        # infos={inventory (if traj_wrap: , previous_action(d), current_action(d))}
    
    # The agent deals with discrete actions so we want this wrapper to be the last one:
    if previous_reward_action:
        env = PreviousRewardActionInfoWrapper(
            env=env,
            trajectory_wrapping=trajectory_wrapping
        )
        # state=POV, 
        # input action is discrete, propagated action is continuous, 
        # infos={inventory, previous_reward, previous_action(ohe) (if traj_wrap: current_action(d))}
    
    return env

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

class MineRLRegymAgent(MineRLAgentBase):
    def load_agent(self):
        filepath = '/home/mark/Documents/Imitation/MineRL/AI_Crowd/agent.pt'
        self.agent = torch.load(filepath)

    def run_agent_on_episode(self, single_episode_env : Episode):
        obs = single_episode_env.reset()
        done = False
        while not done:
            #act = self.agent.take_action(obs)
            act = 0
            single_episode_env.step(act)

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
