from typing import Dict, Any, Optional, List, Callable, Union
import os
import copy
from functools import partial 
from collections.abc import Iterable
from collections import deque, OrderedDict

import cv2
cv2.setNumThreads(0)
import numpy as np


import gym
from gym.wrappers import TimeLimit

import logging
import coloredlogs

coloredlogs.install(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

verbose = False



def is_leaf(node: Dict):
    return all([ not isinstance(node[key], dict) for key in node.keys()])

def _concatenate_list_hdict(
    lhds: List[Dict],
    concat_fn: Optional[Callable] = partial(np.concatenate, axis=0),
    preprocess_fn: Optional[Callable] = (lambda x: x)
    ):
    out_hd = {key: {} for key in lhds[0]}

    queue = [lhds]
    pointers = None

    out_queue = [out_hd]
    out_pointer = None

    while len(queue):
        pointers = [hds for hds in queue.pop(0)]
        out_pointer = out_queue.pop(0)

        if not is_leaf(pointers[0]):
            #out_pointer = {}
            # previously is taken care of at 145 upon initialization,
            # and then at 165 upon 'recurrence'.
            for k in pointers[0]:
                queue_element = [pointer[k] for pointer in pointers if k in pointer]
                queue.insert(0, queue_element)

                out_pointer[k] = {}
                out_queue.insert(0, out_pointer[k])
        else:
            for k in pointers[0]:
                # Previously assigned as a dictionnary in 145 or 165...
                out_pointer[k] = []
                # Since we are at a leaf then value is
                # either numpy or numpy.float64
                # or list of tensors:
                if isinstance(pointers[0][k], list):
                    for idx in range(len(pointers[0][k])):
                        concat_list = [
                            preprocess_fn(pointer[k][idx])
                            for pointer in pointers if k in pointer
                        ]
                        out_pointer[k].append(
                            concat_fn(concat_list)
                        )
                else:
                    concat_list = [
                        preprocess_fn(pointer[k])
                        for pointer in pointers if k in pointer
                    ]
                    out_pointer[k] = concat_fn(concat_list)
    return out_hd

class VDNVecEnvWrapper(object):
    def __init__(self, env, nbr_players):
        '''
        Value-Decomposition Network-purposed wrapper expects the action argument to
        contain an extra dimension for the number of players.
        It reorganises the action by separating it into a list of batched actions,
        where each element of the list corresponds to the set of batched actions 
        of a given player.
        As far as the inner environment outputs are concerned, they are expected
        to be list of batched elements. This wrapper transforms the list of elements
        into a singleton list whose element contains an extra dimension as the player
        dimension.
        '''
        self.env = env
        self.nbr_players = nbr_players

    def get_nbr_envs(self):
        return self.env.get_nbr_envs()

    def set_nbr_envs(self, nbr_envs):
        self.env.set_nbr_envs(nbr_envs)

    def render(self, render_mode="rgb_array", env_indices=None) :
        return self.env.render(render_mode=render_mode, env_indices=env_indices)

    def close(self):
        return self.env.close()

    def reset(self, **kwargs):
        input_dict = self.env.reset(**kwargs)
        nvdn_next_obs = input_dict["observations"]
        nvdn_next_infos = input_dict["info"]

        vdn_obs = np.concatenate(
            nvdn_next_obs,
            axis=0
        )
        next_obs = [vdn_obs]

        list_infos = []
        for li in nvdn_next_infos:
            for k in range(len(li)):
                list_infos.append(li[k])
        """
        next_infos = [
            _concatenate_list_hdict(
                lhds=list_infos, 
                concat_fn=partial(np.stack, axis=1),   # stack on new player dimension.
                preprocess_fn=lambda x: x,
            )
        ]
        """
        next_infos = [list_infos]

        output_dict = {
            "observations":nvdn_next_obs,
            "info":nvdn_next_infos,

            "vdn_observations":next_obs,
            "vdn_info":next_infos,
        }

        return output_dict

    def step(self, action, **kwargs):
        assert isinstance(action, list) and len(action)==1, "action argument must be a singleton list of dictionnary (SAD) or tensor."
        # We are transforming it into an list of dictionnary with nbr_players
        
        observed_batch_size : int
        if isinstance(action[0], dict): #sad?
            observed_batch_size = action[0]['action'].shape[0]
        else:
            observed_batch_size = action[0].shape[0]
        
        nbr_env = observed_batch_size // self.nbr_players
        
        env_action = []
        if isinstance(action[0], dict): #sad?
            for pidx in range(self.nbr_players):
                ad = {}
                for k, av in action[0].items():
                    ad[k] = av[pidx*nbr_env:(pidx+1)*nbr_env, ...]
                env_action.append(ad)
        else:
            for pidx in range(self.nbr_players):
                a = action[0][pidx*nbr_env:(pidx+1)*nbr_env, ...]
                env_action.append(a)

        nonvdn_action = env_action
        env_output_dict = self.env.step(env_action, **kwargs)
        if "actions" in env_output_dict:
            nonvdn_action = env_output_dict["actions"]
        nvdn_next_obs = env_output_dict["succ_observations"]
        nvdn_reward = env_output_dict["reward"]
        nvdn_done = env_output_dict["done"]
        nvdn_next_infos = env_output_dict["succ_info"]

        next_obs = [
            np.concatenate(
                nvdn_next_obs,
                axis=0
            )
        ]
        # 1 x (batch_size*num_player, ...)

        reward_shape = nvdn_reward[0].shape
        reward = [
            np.concatenate(
                nvdn_reward,
                axis=0
            )
        ]
        # 1 x (batch_size*num_player, ...)
        
        list_infos = []
        for li in nvdn_next_infos:
            for k in range(len(li)):
                list_infos.append(li[k])
        
        next_infos = [list_infos]
        """next_infos = [
            _concatenate_list_hdict(
                lhds=list_infos, 
                concat_fn=partial(np.concatenate, axis=0),   # stack on new player dimension.
                preprocess_fn=lambda x: x,
            )
        ]
        # 1 x key x (batch_size*num_player, ...)
        """

        output_dict = {
            "actions":nonvdn_action,

            "succ_observations":nvdn_next_obs, 
            "reward":nvdn_reward, 
            "done":nvdn_done, 
            "succ_info":nvdn_next_infos,

            "vdn_succ_observations":next_obs, 
            "vdn_reward":reward, 
            "vdn_done":nvdn_done, 
            "vdn_succ_info":next_infos
        }

        return output_dict


# # Wrappers:
# # Observation Wrappers:
'''
Adapted from:
https://github.com/chainer/chainerrl/blob/master/chainerrl/wrappers/atari_wrappers.py
'''
class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
    
    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=-1)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class RandNoOpStartWrapper(gym.Wrapper):
    def __init__(self, env, nbr_max_random_steps=30):
        gym.Wrapper.__init__(self,env)
        self.nbr_max_random_steps = nbr_max_random_steps
        self.total_reward = 0
        
    def reset(self, **args):
        obs = self.env.reset()
        nbr_rand_noop = random.randint(0, self.nbr_max_random_steps)
        self.total_reward = 0
        for _ in range(nbr_rand_noop):
            # Execute No-Op:
            obs, r, d, i = self.env.step(0)
            self.total_reward += r
        return obs 

    def step(self, action):
        obs, r, d, info = self.env.step(action=action)
        if self.total_reward != 0:
            r += self.total_reward
            self.total_reward = 0
        return obs, r, d, info 


class SingleLifeWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.done = False
        self.life_done = True 

        AtariEnv = env
        while True:
            env = getattr(AtariEnv, 'env', None)
            if env is not None:
                AtariEnv = env
            else:
                break
        self.AtariEnv = AtariEnv
        self.lives = self.AtariEnv.ale.lives()

    def reset(self, **args):
        self.done = False
        self.lives = self.env.env.ale.lives()
        return self.env.reset(**args)

    def step(self, action):
        if self.done:
            self.reset()
        obs, reward, done, info = self.env.step(action)

        force_done = done
        if self.life_done:
            if self.lives > info['ale.lives']:
                force_done = True
                self.lives = info['ale.lives']
        
        if force_done:
            reward = -1
    
        self.done = done
        info['real_done'] = done
        
        return obs, reward, force_done, info 
        
class SingleRewardWrapper(gym.Wrapper):
    def __init__(self, env, penalizing=False):
        gym.Wrapper.__init__(self, env)
        self.penalizing = penalizing

    def reset(self, **args):
        return self.env.reset(**args)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if reward > 0:
            done = True 
        elif self.penalizing:
            reward = -0.001

        return obs, reward, done, info 


class ProgressivelyMultiRewardWrapper(gym.Wrapper):
    def __init__(self, env, penalizing=False, start_count=0.0, end_count=100.0, nbr_episode=1e3):
        gym.Wrapper.__init__(self, env)
        self.penalizing = penalizing
        self.start_count = start_count
        self.end_count = end_count
        self.nbr_episode = nbr_episode
        self.episode_count = 0

        self.per_episode_increase = (self.end_count-self.start_count)/self.nbr_episode  
        self.current_threshold = self.start_count
        self.cum_reward = 0

    def reset(self, **args):
        self.cum_reward = 0
        self.current_threshold += self.per_episode_increase
        return self.env.reset(**args)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.cum_reward += reward
        
        if self.cum_reward > self.current_threshold:
            done = True 
        
        if reward<=0 and self.penalizing:
            reward = -0.001

        return obs, reward, done, info 


class FrameSkipStackAtari(gym.Wrapper):
    """
    Return a stack of framed composed of every 'skip'-th repeat.
    The rewards are summed over the skipped and stackedd frames.
    
    This wrapper assumes:
    - the observation space of the environment to be frames solely.
    - the frames are concatenated on the last axis, i.e. the channel axis.
    """
    def __init__(self, env, skip=4, act_rand_repeat=False, stack=4, single_life_episode=False):
        gym.Wrapper.__init__(self,env)
        self.skip = skip if skip is not None else 1
        self.stack = stack if stack is not None else 1
        self.act_rand_repeat = act_rand_repeat
        self.single_life_episode = single_life_episode

        self.observations = deque([], maxlen=self.stack)
        
        assert(isinstance(self.env.observation_space, gym.spaces.Box))
        
        low_obs_space = np.repeat(self.env.observation_space.low, self.stack, axis=-1)
        high_obs_space = np.repeat(self.env.observation_space.high, self.stack, axis=-1)
        self.observation_space = gym.spaces.Box(low=low_obs_space, high=high_obs_space, dtype=self.env.observation_space.dtype)

        self.done = False

        if self.single_life_episode: 
            self.life_done = True 

            AtariEnv = env
            while True:
                env = getattr(AtariEnv, 'env', None)
                if env is not None:
                    AtariEnv = env
                else:
                    break
            self.AtariEnv = AtariEnv
            self.lives = self.AtariEnv.ale.lives()
        
    def _get_obs(self):
        assert(len(self.observations) == self.stack)
        return LazyFrames(list(self.observations))
        
    def reset(self, **args):
        obs = self.env.reset()
        
        self.done = False
        
        if self.single_life_episode:
            self.lives = self.AtariEnv.ale.lives()
        
        for _ in range(self.stack):
            self.observations.append(obs)
        return self._get_obs()
    
    def step(self, action):
        if self.done:
            self.reset()
        
        total_reward = 0.0
        nbr_it = self.skip
        if self.act_rand_repeat:
            nbr_it = random.randint(1, nbr_it)

        for i in range(nbr_it):
            obs, reward, done, info = self.env.step(action)

            force_done = done
            if self.single_life_episode:
                if self.life_done:
                    if self.lives > info['ale.lives'] and info['ale.lives'] > 0:
                        force_done = True
                        self.lives = info['ale.lives']
                
                if reward < 0:
                    force_done = True
                elif force_done:
                    reward = -1
            
                info['real_done'] = done

            total_reward += reward

            if self.act_rand_repeat:
                self.observations.append(obs)

            self.done = done
            if done or force_done:
                break
            
        if not(self.act_rand_repeat):
            self.observations.append(obs)
        
        return self._get_obs(), total_reward, force_done, info


def atari_pixelwrap(env, 
                    size, 
                    skip=None, 
                    act_rand_repeat=False, 
                    stack=None, 
                    grayscale=False, 
                    nbr_max_random_steps=0, 
                    single_life_episode=True):
    # Observations:
    if grayscale:
        env = GrayScaleObservation(env=env) 
    if nbr_max_random_steps > 0:
        env = RandNoOpStartWrapper(env=env, nbr_max_random_steps=nbr_max_random_steps)
    #env = PixelObservationWrapper(env=env)
    env = FrameResizeWrapper(env, size=size) 
    if skip is not None or stack is not None:
        env = FrameSkipStackAtari(env=env, skip=skip, act_rand_repeat=act_rand_repeat, stack=stack, single_life_episode=single_life_episode)
    #if single_life:
    #    env = SingleLifeWrapper(env=env)
    return env

import sys
import gc
import pdb
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
#forkedPdb = ForkedPdb()

class GrayScaleObservation(gym.ObservationWrapper):
    r"""Convert the image observation from RGB to gray scale. """
    def __init__(self, env, keep_dim=True):
        self._env = env
        if isinstance(env, gym.wrappers.time_limit.TimeLimit):
            self._env = env.env
        self._env._get_image = self._env.ale.getScreenGrayscale
        self._env._get_obs = self._env.ale.getScreenGrayscale

        super(GrayScaleObservation, self).__init__(env)
        
        assert len(env.observation_space.shape) == 3 and env.observation_space.shape[-1] == 3
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8)
        
    def observation(self, observation):
        if self._env._get_image != self._env.ale.getScreenGrayscale:
            self._env._get_image = self._env.ale.getScreenGrayscale
        if self._env._get_obs != self._env.ale.getScreenGrayscale:
            self._env._get_obs = self._env.ale.getScreenGrayscale
        return observation

class GrayScaleObservationCV(gym.ObservationWrapper):
    r"""Convert the image observation from RGB to gray scale. """
    def __init__(self, env, keep_dim=True):
        super(GrayScaleObservationCV, self).__init__(env)
        self.keep_dim = keep_dim

        assert len(env.observation_space.shape) == 3 and env.observation_space.shape[-1] == 3
        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        return observation


class FrameResizeWrapper(gym.ObservationWrapper):
    """
    """
    def __init__(self, env, size=(64, 64)):
        gym.ObservationWrapper.__init__(self, env=env)
        
        self.size = size
        if isinstance(self.size, int):
            self.size = (self.size, self.size)

        low = np.zeros((*self.size, self.env.observation_space.shape[-1]))
        high  = 255*np.ones((*self.size, self.env.observation_space.shape[-1]))
        
        self.observation_space = gym.spaces.Box(low=low, high=high)
    
    def observation(self, observation):
        obs = cv2.resize(observation, tuple(self.size))
        obs = obs.reshape(self.observation_space.shape)
        return obs


# https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/atari_wrappers.py#L275
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, stack=4,):
        gym.Wrapper.__init__(self,env)
        self.stack = stack if stack is not None else 1
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
        obs, reward, done, info = self.env.step(action)
        self.observations.append(obs)        
        return self._get_obs(), reward, done, info


# https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/atari_wrappers.py#L12
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


# https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/atari_wrappers.py#L97
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/atari_wrappers.py#L125
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)



def baseline_atari_pixelwrap(env, 
                             size=None, 
                             skip=4, 
                             stack=4, 
                             grayscale=True,  
                             single_life_episode=True, 
                             nbr_max_random_steps=30, 
                             clip_reward=True,
                             previous_reward_action=False):
    if grayscale:
        #env = GrayScaleObservation(env=env) 
        env = GrayScaleObservationCV(env=env) 
    
    if nbr_max_random_steps > 0:
        env = NoopResetEnv(env, noop_max=nbr_max_random_steps)
    
    if skip > 0:
        env = MaxAndSkipEnv(env, skip=skip)
    
    if size is not None and isinstance(size, int):
        env = FrameResizeWrapper(env, size=size) 
    
    if single_life_episode:
        env = EpisodicLifeEnv(env)
    
    if stack > 1:
        env = FrameStack(env, stack=stack)
    
    if clip_reward:
        env = ClipRewardEnv(env)

    if previous_reward_action:
        env = PreviousRewardActionInfoWrapper(env=env)
    
    return env



def hanabi_wrap(
    env,
    sad=False, 
    clip_reward=True,
    previous_reward_action=True
    ):
    if clip_reward:
        env = ClipRewardEnv(env)

    if previous_reward_action:
        env = PreviousRewardActionInfoMultiAgentWrapper(env=env)
    
    if sad:
        env = SADEnvWrapper(env=env)

    return env



#---------------------------------------------------------#


# MineRL:

'''
From: https://github.com/minerllabs/baselines/blob/2f1ddc5b049decfa7b20969ac319552032f9a315/general/chainerrl/baselines/env_wrappers.py#L173

MIT License
'''
class ObtainPOVWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""
    def __init__(self, env, size=84, grayscale=False, scaling=False):
        super().__init__(env)
        self.size = size
        if isinstance(self.size, int):
            self.size = (self.size, self.size)
        self.grayscale = grayscale
        self.scaling = scaling

        pov_space = self.env.observation_space.spaces['pov']
        low = 0.0
        high = 255.0
        if self.scaling: high =1.0
        if self.grayscale:
            assert len(pov_space.shape) == 3 and pov_space.shape[-1] == 3
            obs_shape = pov_space.shape[:2]
            self.pov_space = gym.spaces.Box(low=low, high=high, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.float32)
        else:
            self.pov_space = pov_space 
        # Resize:
        if self.size != self.pov_space.shape[:2]:
            self.pov_space = gym.spaces.Box(low=low, high=high, shape=(*self.size, self.pov_space.shape[-1]), dtype=np.float32)
        
        self.observation_space = self.pov_space

    def observation(self, observation):
        obs = observation['pov'].astype(np.float32)
        if self.scaling:
            obs /= 255.0 #this line is scaling between 0 and 1...
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            # (*obs_shape)
            obs = np.expand_dims(obs, -1)
            # (*obs_shape, 1)
        if self.size != obs.shape[:2]:
            obs = cv2.resize(obs, self.size)
            obs = obs.reshape(self.pov_space.shape)
        
        return obs

# Unified Observation Wrapper:

'''
Adapted from:
https://github.com/minerllabs/baselines/blob/master/general/chainerrl/baselines/env_wrappers.py

MIT License

Copyright (c) Kevin Denamganaï.

Modifications:
Adding equipped_items to the set of observations to take into account.
Adding the possibility of yielding grayscaled frames.
Adding the possibility of re-sizing the output frames.
'''
from enum import Enum
class UnifiedObservationWrapper(gym.ObservationWrapper):
    """
    Returns a frame/gym.space.Box with multiple channels that account for:
    - 'pov' (3 channels)
    - 'compassAngle', if any (1 channel)
    - 'inventory', if any (1 channel)
    - 'equipped_items', if any (1 channel)
    
    The parameter region_size is used to build squares of information that each corresponds
    to a different element in the 'inventory', or in the 'equipped_items'.
    """
    def __init__(self, 
                 env, 
                 size=84,
                 grayscale=True,
                 region_size=8, 
                 scaling=True):
        gym.ObservationWrapper.__init__(self, env=env)
        
        self.size = size
        if isinstance(self.size, int):
            self.size = (self.size, self.size)
        self.grayscale = grayscale
        self.region_size = region_size
        self.scaling = scaling
        
        self.compass_angle_scale = 180.0 / 255.0
        
        pov_space = self.env.observation_space.spaces['pov']
        self.scaler_dict = {'pov': 255.0}
        
        # POV:
        # Grayscale:
        if self.grayscale:
            assert len(pov_space.shape) == 3 and pov_space.shape[-1] == 3
            obs_shape = pov_space.shape[:2]
            self.pov_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.float32)
        else:
            self.pov_space = pov_space 
        # Resize:
        if self.size != self.pov_space.shape[:2]:
            low = np.zeros((*self.size, self.pov_space.shape[-1]))
            high  = 255*np.ones((*self.size, self.pov_space.shape[-1]))
            self.pov_space = gym.spaces.Box(low=low, high=high)
        
        low_dict = {'pov':pov_space.low}
        high_dict = {'pov':pov_space.high}
        
        # Others:
        if 'compassAngle' in self.env.observation_space:
            compass_angle_space = self.env.observation_space.spaces['compassAngle']
            low_dict['compassAngle'] = compass_angle_space.low
            high_dict['compassAngle'] = compass_angle_space.high
            self.scaler_dict['compassAngle'] = (high_dict['compassAngle']-low_dict['compassAngle']) / 255.0
        
        if 'inventory' in self.env.observation_space.spaces:
            inventory_space = self.env.observation_space.spaces['inventory']
            low_dict['inventory'] = {}
            high_dict['inventory'] = {}
            self.scaler_dict['inventory'] = {}
            for key in inventory_space.spaces.keys():
                low_dict['inventory'][key] = inventory_space.spaces[key].low
                high_dict['inventory'][key] = inventory_space.spaces[key].high
                self.scaler_dict['inventory'][key] = (high_dict['inventory'][key]-low_dict['inventory'][key]) / 255.0
        
        if 'equipped_items' in self.env.observation_space.spaces:
            self.items_str2value = {'none':0,
                                    'air':1,
                                    'wooden_axe':2,
                                    'wooden_pickaxe':3,
                                    'stone_axe':4,
                                    'stone_pickaxe':5,
                                    'iron_axe':6,
                                    'iron_pickaxe':7,
                                    'other':8}
            eq_item_space = self.env.observation_space.spaces['equipped_items'].spaces['mainhand']
            low_dict['equipped_items'] = {'mainhand':{}}
            high_dict['equipped_items'] = {'mainhand':{}}
            self.scaler_dict['equipped_items'] = {'mainhand':{}}
            for key in eq_item_space.spaces.keys():
                if key != 'type':
                    low_dict['equipped_items']['mainhand'][key] = eq_item_space.spaces[key].low
                    high_dict['equipped_items']['mainhand'][key] = eq_item_space.spaces[key].high
                else:
                    '''
                    enumtypes = list(eq_item_space.spaces[key])
                    enumvalues = [ (e, e.value) for e in enumtypes]
                    enumvalues.sort(key=lambda x: x[1])
                    
                    enum_max = enumvalues[0][0]
                    enum_min = enumvalues[-1][0]
                    '''
                    low_dict['equipped_items']['mainhand'][key] = 0
                    high_dict['equipped_items']['mainhand'][key] = len(eq_item_space.spaces[key])
                self.scaler_dict['equipped_items']['mainhand'][key] = (high_dict['equipped_items']['mainhand'][key]-low_dict['equipped_items']['mainhand'][key]) / 255.0
        
        low = self.observation(low_dict)
        high = self.observation(high_dict)
        
        self.observation_space = gym.spaces.Box(low=low, high=high)
    
    def observation(self, observation):
        obs = observation['pov']
        obs = obs.astype(np.float32)
        #obs /= self.scaler_dict['pov'] #this line is scaling between 0 and 1...
        pov_dtype = obs.dtype

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            # (*obs_shape)
            obs = np.expand_dims(obs, -1)
            # (*obs_shape, 1)
        if self.size != obs.shape[:2]:
            obs = cv2.resize(obs, self.size)
            obs = obs.reshape(self.pov_space.shape)
        
        
        if 'compassAngle' in observation:
            compass_scaled = observation['compassAngle'] / self._compass_angle_scale
            print(f"Compass scaledangle: {compass_scaled}.")
            compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=pov_dtype) * compass_scaled
            obs = np.concatenate([obs, compass_channel], axis=-1)
        if 'inventory' in observation:
            assert len(obs.shape[:-1]) == 2
            region_max_height = obs.shape[0]
            region_max_width = obs.shape[1]
            rs = self.region_size
            if min(region_max_height, region_max_width) < rs:
                raise ValueError("'region_size' is too large.")
            num_element_width = region_max_width // rs
            inventory_channel = np.zeros(shape=list(obs.shape[:-1]) + [1], dtype=pov_dtype)
            for idx, key in enumerate(observation['inventory']):
                item_value = observation['inventory'][key]
                
                if verbose: logger.info(f"Inventory :: {key} :: {item_value}.")
                
                # Scaling between 0 and 255:
                if self.scaling:
                    item_value = item_value / self.scaler_dict['inventory'][key] 
                
                item_scaled = np.clip(255 - 255 / (item_value + 1),  # Inversed
                                      0, 255)
                
                if verbose: logger.info(f"Scaled {key} :: {item_value}.")
                
                item_channel = np.ones(shape=[rs, rs, 1], dtype=pov_dtype) * item_scaled
                width_low = (idx % num_element_width) * rs
                height_low = (idx // num_element_width) * rs
                if height_low + rs > region_max_height:
                    raise ValueError("Too many elements on 'inventory'. Please decrease 'region_size' of each component")
                inventory_channel[height_low:(height_low + rs), width_low:(width_low + rs), :] = item_channel
            obs = np.concatenate([obs, inventory_channel], axis=-1)
        if 'equipped_items' in observation:
            #assert len(obs.shape[:-1]) == 2
            region_max_height = obs.shape[0]
            region_max_width = obs.shape[1]
            rs = self.region_size
            if min(region_max_height, region_max_width) < rs:
                raise ValueError("'region_size' is too large.")
            num_element_width = region_max_width // rs
            eq_item_channel = np.zeros(shape=list(obs.shape[:-1]) + [1], dtype=pov_dtype)
            for idx, key in enumerate(observation['equipped_items']['mainhand']):
                item_value = observation['equipped_items']['mainhand'][key]
            
                if verbose: logger.info(f"Equipped Item :: {key} :: {item_value}.")
                
                #if key == 'type':
                #    item_value = item_value.value
                if isinstance(item_value, str):
                    item_value = self.items_str2value[item_value]

                # Scaling between 0 and 255:
                if self.scaling:
                    item_value = item_value / self.scaler_dict['equipped_items']['mainhand'][key]
                #item_scaled = np.clip(255 - 255 / (item_value + 1),  # Inversed
                #                      0, 255)
                item_scaled = np.clip(item_value, 0, 255)
                if verbose: logger.info(f"Scaled {key} :: {item_value}.")
                
                item_channel = np.ones(shape=[rs, rs, 1], dtype=pov_dtype) * item_scaled
                width_low = (idx % num_element_width) * rs
                height_low = (idx // num_element_width) * rs
                if height_low + rs > region_max_height:
                    raise ValueError("Too many elements on 'inventory'. Please decrease 'region_size' of each component")
                eq_item_channel[height_low:(height_low + rs), width_low:(width_low + rs), :] = item_channel
            obs = np.concatenate([obs, eq_item_channel], axis=-1)
            
            # obs is scaled between 0 and 255 if scaling==True...

        return obs




# Action Wrapper: (actions from agent (discrete) format to dict environment format)

"""
Adapted from:
https://github.com/minerllabs/baselines/blob/master/general/chainerrl/baselines/env_wrappers.py

MIT License

Copyright (c) Kevin Denamganaï

Modifications:
From the viewpoint of the agent, the action_space is independant of the actual environment.
The action_space is viewed as if the agent was interacting with "ObtainDiamond-v0".
From the viewpoint of the wrapped environment, the action is adapted accordingly.

"""
class SerialDiscreteInterfaceActionWrapper(gym.ActionWrapper):
    """Convert MineRL env's `Dict` action space as a serial discrete action space.

    The term "serial" means that this wrapper can only push one key at each step.
    "attack" action will be alwarys triggered.

    Parameters
    ----------
    env
        Wrapping gym environment.
    always_keys
        List of action keys, which should be always pressed throughout interaction with environment.
        If specified, the "noop" action is also affected.
    reverse_keys
        List of action keys, which should be always pressed but can be turn off via action.
        If specified, the "noop" action is also affected.
    exclude_keys
        List of action keys, which should be ignored for discretizing action space.
    exclude_noop
        The "noop" will be excluded from discrete action list.
    num_camera_discretize
        Number of discretization of yaw control (must be odd).
    allow_pitch
        If specified, this wrapper appends commands to control pitch.
    max_camera_range
        Maximum value of yaw control.
    """

    BINARY_KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack']

    def __init__(self, env, always_keys=None, reverse_keys=None, exclude_keys=None, exclude_noop=False,
                 num_camera_discretize=3, allow_pitch=False,
                 max_camera_range=10):
        super().__init__(env)

        self.always_keys = [] if always_keys is None else always_keys
        self.reverse_keys = [] if reverse_keys is None else reverse_keys
        self.exclude_keys = [] if exclude_keys is None else exclude_keys
        if len(set(self.always_keys) | set(self.reverse_keys) | set(self.exclude_keys)) != len(self.always_keys) + len(self.reverse_keys) + len(self.exclude_keys):
            raise ValueError('always_keys ({}) or reverse_keys ({}) or exclude_keys ({}) intersect each other.'.format(
                self.always_keys, self.reverse_keys, self.exclude_keys))
        self.exclude_noop = exclude_noop

        self.wrapping_action_space = self.env.action_space
        self.num_camera_discretize = num_camera_discretize
        self._noop_template = OrderedDict([
            ('forward', 0),
            ('back', 0),
            ('left', 0),
            ('right', 0),
            ('jump', 0),
            ('sneak', 0),
            ('sprint', 0),
            ('attack', 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            # 'none', 'dirt' (Obtain*:)+ 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'
            ('place', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'
            ('equip', 0),
            # (Obtain* tasks only) 'none', 'torch', 'stick', 'planks', 'crafting_table'
            ('craft', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'
            ('nearbyCraft', 0),
            # (Obtain* tasks only) 'none', 'iron_ingot', 'coal'
            ('nearbySmelt', 0),
        ])
        self._noop_template_ambiguous_op = {}
        self._noop_template_ambiguous_op['place'] = ['none', 'dirt', 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch']
        self._noop_template_ambiguous_op['equip'] = ['none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe']
        self._noop_template_ambiguous_op['craft'] = ['none', 'torch', 'stick', 'planks', 'crafting_table']
        self._noop_template_ambiguous_op['nearbyCraft'] = ['none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace']
        self._noop_template_ambiguous_op['nearbySmelt'] = ['none', 'iron_ingot', 'coal']

        for key, space in self.wrapping_action_space.spaces.items():
            if key not in self._noop_template:
                raise ValueError('Unknown action name: {}'.format(key))

        # get noop
        # according to the actual environment:
        self.noop = copy.deepcopy(self._noop_template)
        
        # check&set always_keys
        for key in self.always_keys:
            if key not in self.BINARY_KEYS:
                raise ValueError('{} is not allowed for `always_keys`.'.format(key))
            self.noop[key] = 1
        if verbose: logger.info('always pressing keys: {}'.format(self.always_keys))
        # check&set reverse_keys
        for key in self.reverse_keys:
            if key not in self.BINARY_KEYS:
                raise ValueError('{} is not allowed for `reverse_keys`.'.format(key))
            self.noop[key] = 1
        if verbose: logger.info('reversed pressing keys: {}'.format(self.reverse_keys))
        # check exclude_keys
        for key in self.exclude_keys:
            if key not in self.noop:
                raise ValueError('unknown exclude_keys: {}'.format(key))
        if verbose: logger.info('always ignored keys: {}'.format(self.exclude_keys))
        
        # tailor noop to the actual environment:
        self.tailored_noop = copy.deepcopy(self.noop)
        for key in list(self.tailored_noop.keys()):
            if key not in self.wrapping_action_space.spaces:
                del self.tailored_noop[key]

        # get each discrete action, independantly of the actual environment:
        self._actions = [self.tailored_noop]
        for key in self.noop:
            if key in self.always_keys or key in self.exclude_keys:
                continue
            if key in self.BINARY_KEYS:
                # action candidate : {1}  (0 is ignored because it is for noop), or {0} when `reverse_keys`.
                op = copy.deepcopy(self.tailored_noop)
                if key in self.tailored_noop:
                    if key in self.reverse_keys:
                        op[key] = 0
                    else:
                        op[key] = 1
                self._actions.append(op)
            elif key == 'camera':
                # action candidate : {[0, -max_camera_range], [0, -max_camera_range + delta_range], ..., [0, max_camera_range]}
                # ([0, 0] is excluded)
                delta_range = max_camera_range * 2 / (self.num_camera_discretize - 1)
                if self.num_camera_discretize % 2 == 0:
                    raise ValueError('Number of camera discretization must be odd.')
                for i in range(self.num_camera_discretize):
                    op = copy.deepcopy(self.tailored_noop)
                    if i < self.num_camera_discretize // 2:
                        op[key] = np.array([0, -max_camera_range + delta_range * i], dtype=np.float32)
                    elif i > self.num_camera_discretize // 2:
                        op[key] = np.array([0, -max_camera_range + delta_range * (i - 1)], dtype=np.float32)
                    else:
                        continue
                    self._actions.append(op)

                if allow_pitch:
                    for i in range(self.num_camera_discretize):
                        op = copy.deepcopy(self.tailored_noop)
                        if i < self.num_camera_discretize // 2:
                            op[key] = np.array([-max_camera_range + delta_range * i, 0], dtype=np.float32)
                        elif i > self.num_camera_discretize // 2:
                            op[key] = np.array([-max_camera_range + delta_range * (i - 1), 0], dtype=np.float32)
                        else:
                            continue
                        self._actions.append(op)

            elif key in {'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt'}:
                # action candidate : {1, 2, ..., len(space)-1}  (0 is ignored because it is for noop)
                if key in self.tailored_noop:
                    for a in range(1, self.wrapping_action_space.spaces[key].n):
                        op = copy.deepcopy(self.tailored_noop)
                        if key in self.tailored_noop: op[key] = a
                        self._actions.append(op)
                else:
                    # If the key is not accessible in this environment,
                    # then we just do a Noop operation:
                    for a in range(1, len(self._noop_template_ambiguous_op[key])):
                        op = copy.deepcopy(self.tailored_noop)
                        self._actions.append(op)

        if self.exclude_noop:
            del self._actions[0]

        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)
        if verbose: logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = self._actions[action]
        if verbose: logger.debug('discrete action {} -> original action {}'.format(action, original_space_action))
        return original_space_action



default_always_keys = ["attack"]
default_reverse_keys = ["forward"]
default_exclude_keys =["back", "left", "right", "sneak", "sprint"]
wrap_env_serial_discrete_interface = partial(SerialDiscreteInterfaceActionWrapper, 
                                             always_keys=default_always_keys, 
                                             reverse_keys=default_reverse_keys, 
                                             exclude_keys=default_exclude_keys, 
                                             exclude_noop=False,
                                             num_camera_discretize=3, 
                                             allow_pitch=False,
                                             max_camera_range=10)


class SerialDiscreteActionWrapper(gym.ActionWrapper):
    """Convert MineRL env's `Dict` action space as a serial discrete action space.

    The term "serial" means that this wrapper can only push one key at each step.
    "attack" action will be alwarys triggered.

    Parameters
    ----------
    env
        Wrapping gym environment.
    always_keys
        List of action keys, which should be always pressed throughout interaction with environment.
        If specified, the "noop" action is also affected.
    reverse_keys
        List of action keys, which should be always pressed but can be turn off via action.
        If specified, the "noop" action is also affected.
    exclude_keys
        List of action keys, which should be ignored for discretizing action space.
    exclude_noop
        The "noop" will be excluded from discrete action list.
    num_camera_discretize
        Number of discretization of yaw control (must be odd).
    allow_pitch
        If specified, this wrapper appends commands to control pitch.
    max_camera_range
        Maximum value of yaw control.
    """

    BINARY_KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack']

    def __init__(self, env, always_keys=None, reverse_keys=None, exclude_keys=None, exclude_noop=False,
                 num_camera_discretize=3, allow_pitch=False,
                 max_camera_range=10):
        super().__init__(env)

        self.always_keys = [] if always_keys is None else always_keys
        self.reverse_keys = [] if reverse_keys is None else reverse_keys
        self.exclude_keys = [] if exclude_keys is None else exclude_keys
        if len(set(self.always_keys) | set(self.reverse_keys) | set(self.exclude_keys)) != \
                len(self.always_keys) + len(self.reverse_keys) + len(self.exclude_keys):
            raise ValueError('always_keys ({}) or reverse_keys ({}) or exclude_keys ({}) intersect each other.'.format(
                self.always_keys, self.reverse_keys, self.exclude_keys))
        self.exclude_noop = exclude_noop

        self.wrapping_action_space = self.env.action_space
        self.num_camera_discretize = num_camera_discretize
        self._noop_template = OrderedDict([
            ('forward', 0),
            ('back', 0),
            ('left', 0),
            ('right', 0),
            ('jump', 0),
            ('sneak', 0),
            ('sprint', 0),
            ('attack', 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            # 'none', 'dirt' (Obtain*:)+ 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'
            ('place', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'
            ('equip', 0),
            # (Obtain* tasks only) 'none', 'torch', 'stick', 'planks', 'crafting_table'
            ('craft', 0),
            # (Obtain* tasks only) 'none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'
            ('nearbyCraft', 0),
            # (Obtain* tasks only) 'none', 'iron_ingot', 'coal'
            ('nearbySmelt', 0),
        ])
        for key, space in self.wrapping_action_space.spaces.items():
            if key not in self._noop_template:
                raise ValueError('Unknown action name: {}'.format(key))

        # get noop
        self.noop = copy.deepcopy(self._noop_template)
        for key in self._noop_template:
            if key not in self.wrapping_action_space.spaces:
                del self.noop[key]

        # check&set always_keys
        for key in self.always_keys:
            if key not in self.BINARY_KEYS:
                raise ValueError('{} is not allowed for `always_keys`.'.format(key))
            self.noop[key] = 1
        if verbose: logger.info('always pressing keys: {}'.format(self.always_keys))
        # check&set reverse_keys
        for key in self.reverse_keys:
            if key not in self.BINARY_KEYS:
                raise ValueError('{} is not allowed for `reverse_keys`.'.format(key))
            self.noop[key] = 1
        if verbose: logger.info('reversed pressing keys: {}'.format(self.reverse_keys))
        # check exclude_keys
        for key in self.exclude_keys:
            if key not in self.noop:
                raise ValueError('unknown exclude_keys: {}'.format(key))
        if verbose: logger.info('always ignored keys: {}'.format(self.exclude_keys))

        # get each discrete action
        self._actions = [self.noop]
        for key in self.noop:
            if key in self.always_keys or key in self.exclude_keys:
                continue
            if key in self.BINARY_KEYS:
                # action candidate : {1}  (0 is ignored because it is for noop), or {0} when `reverse_keys`.
                op = copy.deepcopy(self.noop)
                if key in self.reverse_keys:
                    op[key] = 0
                else:
                    op[key] = 1
                self._actions.append(op)
            elif key == 'camera':
                # action candidate : {[0, -max_camera_range], [0, -max_camera_range + delta_range], ..., [0, max_camera_range]}
                # ([0, 0] is excluded)
                delta_range = max_camera_range * 2 / (self.num_camera_discretize - 1)
                if self.num_camera_discretize % 2 == 0:
                    raise ValueError('Number of camera discretization must be odd.')
                for i in range(self.num_camera_discretize):
                    op = copy.deepcopy(self.noop)
                    if i < self.num_camera_discretize // 2:
                        op[key] = np.array([0, -max_camera_range + delta_range * i], dtype=np.float32)
                    elif i > self.num_camera_discretize // 2:
                        op[key] = np.array([0, -max_camera_range + delta_range * (i - 1)], dtype=np.float32)
                    else:
                        continue
                    self._actions.append(op)

                if allow_pitch:
                    for i in range(self.num_camera_discretize):
                        op = copy.deepcopy(self.noop)
                        if i < self.num_camera_discretize // 2:
                            op[key] = np.array([-max_camera_range + delta_range * i, 0], dtype=np.float32)
                        elif i > self.num_camera_discretize // 2:
                            op[key] = np.array([-max_camera_range + delta_range * (i - 1), 0], dtype=np.float32)
                        else:
                            continue
                        self._actions.append(op)

            elif key in {'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt'}:
                # action candidate : {1, 2, ..., len(space)-1}  (0 is ignored because it is for noop)
                for a in range(1, self.wrapping_action_space.spaces[key].n):
                    op = copy.deepcopy(self.noop)
                    op[key] = a
                    self._actions.append(op)
        if self.exclude_noop:
            del self._actions[0]

        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)
        if verbose: logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = self._actions[action]
        if verbose: logger.debug('discrete action {} -> original action {}'.format(action, original_space_action))
        return original_space_action


wrap_env_serial_discrete = partial(SerialDiscreteActionWrapper, 
                                   always_keys=default_always_keys, 
                                   reverse_keys=default_reverse_keys, 
                                   exclude_keys=default_exclude_keys, 
                                   exclude_noop=False,
                                   num_camera_discretize=3, 
                                   allow_pitch=False,
                                   max_camera_range=10)


"""
Adapted from:
https://github.com/minerllabs/baselines/blob/master/general/chainerrl/baselines/env_wrappers.py

MIT License

Copyright (c) Kevin Denamganaï

Modifications:
From the viewpoint of the agent, the action_space is independant of the actual environment.
The action_space is viewed as if the agent was interacting with "ObtainDiamond-v0".
From the viewpoint of the wrapped environment, the action is adapted accordingly.

"""
class CombineActionWrapper(gym.ActionWrapper):
    """Combine MineRL env's "exclusive" actions.

    "exclusive" actions will be combined as:
        - "forward", "back" -> noop/forward/back (Discrete(3))
        - "left", "right" -> noop/left/right (Discrete(3))
        - "sneak", "sprint" -> noop/sneak/sprint (Discrete(3))
        - "attack", "place", "equip", "craft", "nearbyCraft", "nearbySmelt"
            -> noop/attack/place/equip/craft/nearbyCraft/nearbySmelt (Discrete(n))
    The combined action's names will be concatenation of originals, i.e.,
    "forward_back", "left_right", "snaek_sprint", "attack_place_equip_craft_nearbyCraft_nearbySmelt".
    """
    def __init__(self, env):
        super().__init__(env)

        self.wrapping_action_space = self.env.action_space

        def combine_exclusive_actions(keys):
            """
            Dict({'forward': Discrete(2), 'back': Discrete(2)})
            =>
            new_actions: [{'forward':0, 'back':0}, {'forward':1, 'back':0}, {'forward':0, 'back':1}]
            """
            new_key = '_'.join(keys)
            valid_action_keys = [k for k in keys]# if k in self.wrapping_action_space.spaces]
            tailored_valid_action_keys = [k for k in keys if k in self.wrapping_action_space.spaces]
            noop = {a: 0 for a in valid_action_keys}
            tailored_noop = {a: 0 for a in tailored_valid_action_keys}
            new_actions = [tailored_noop]

            for key in valid_action_keys:
                if key in tailored_valid_action_keys:
                    space = self.wrapping_action_space.spaces[key]
                    for i in range(1, space.n):
                        op = copy.deepcopy(tailored_noop)
                        op[key] = i
                        new_actions.append(op)
                else:
                    new_actions.append(tailored_noop)
            return new_key, new_actions

        self._maps = {}
        for keys in (
                ('forward', 'back'), ('left', 'right'), ('sneak', 'sprint'),
                ('attack', 'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt')):
            new_key, new_actions = combine_exclusive_actions(keys)
            self._maps[new_key] = new_actions

        self.noop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])

        self.action_space = gym.spaces.Dict({
            'forward_back':
                gym.spaces.Discrete(len(self._maps['forward_back'])),
            'left_right':
                gym.spaces.Discrete(len(self._maps['left_right'])),
            'jump':
                self.wrapping_action_space.spaces['jump'],
            'sneak_sprint':
                gym.spaces.Discrete(len(self._maps['sneak_sprint'])),
            'camera':
                self.wrapping_action_space.spaces['camera'],
            'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                gym.spaces.Discrete(len(self._maps['attack_place_equip_craft_nearbyCraft_nearbySmelt']))
        })

        if verbose: 
            logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))
            for k, v in self._maps.items():
                logger.info('{} -> {}'.format(k, v))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = OrderedDict()
        for k, v in action.items():
            if k in self._maps:
                a = self._maps[k][v]
                original_space_action.update(a)
            else:
                original_space_action[k] = v

        if verbose: logger.debug('action {} -> original action {}'.format(action, original_space_action))
        return original_space_action


class SerialDiscreteCombineActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Assumes CombineActionWrapper-ed env:
        self.wrapping_action_space = self.env.action_space

        self.noop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])

        # get each discrete action
        self._actions = [self.noop]
        for key in self.noop:
            if key == 'camera':
                # action candidate : {[0, -10], [0, 10]}
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, -10], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, 10], dtype=np.float32)
                self._actions.append(op)
            else:
                for a in range(1, self.wrapping_action_space.spaces[key].n):
                    op = copy.deepcopy(self.noop)
                    op[key] = a
                    self._actions.append(op)

        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)
        if verbose: logger.info('{} is converted to {}.'.format(self.wrapping_action_space, self.action_space))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = self._actions[action]
        if verbose: logger.debug('discrete action {} -> original action {}'.format(action, original_space_action))
        return original_space_action



def wrap_env_serial_discrete_combine(env):
    wrapped_env = CombineActionWrapper(env)
    wrapped_env = SerialDiscreteCombineActionWrapper(wrapped_env)
    return wrapped_env


# Action and Observation Wrapping:


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
        
        info['real_done'] = done
        if self._max_episode_steps <= self._elapsed_steps:
            info['real_done'] = True

        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()



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
        infos = {}
        infos['previous_reward'] = copy.deepcopy(self.previous_reward)
        infos['previous_action'] = copy.deepcopy(self.previous_action)
    
        return self.env.reset(), infos

    def step(self, action):
        next_observation, reward, done, next_infos = self.env.step(action)
        
        self.previous_reward = np.ones((1, 1), dtype=np.float32)*reward
        self.previous_action = np.eye(self.nbr_actions, dtype=np.float32)[action].reshape(1, -1)

        pa = copy.deepcopy(self.previous_action) 
        if self.trajectory_wrapping:
            pa = np.eye(self.nbr_actions, dtype=np.float32)[next_infos['previous_action'][0]].reshape(1, -1)
        
        next_infos['previous_reward'] = copy.deepcopy(self.previous_reward)
        next_infos['previous_action'] = copy.deepcopy(pa)
        
        return next_observation, reward, done, next_infos

class PreviousRewardActionInfoMultiAgentWrapper(gym.Wrapper):
    """
    Integrates the previous reward and previous action into the info dictionnary for multi-agent environments.
    Args:
        env (gym.Env): Env to wrap.
    """

    def __init__(self, env, trajectory_wrapping=False):
        super(PreviousRewardActionInfoMultiAgentWrapper, self).__init__(env)
        self.nbr_actions = env.action_space.n
        self.trajectory_wrapping = trajectory_wrapping

    def reset(self):
        obs, infos = self.env.reset()
        nbr_agent = len(infos)
        self.previous_reward = [np.zeros((1, 1)) for _ in range(nbr_agent)]
        self.previous_action = [np.zeros((1, self.nbr_actions)) for _ in range(nbr_agent)]
        
        for info_idx in range(len(infos)):
            infos[info_idx]['previous_reward'] = copy.deepcopy(self.previous_reward[info_idx])
            infos[info_idx]['previous_action'] = copy.deepcopy(self.previous_action[info_idx])
        return obs, infos 
    
    def step(self, action):
        next_observation, reward, done, next_infos = self.env.step(action)
        nbr_agent = len(next_infos)
        
        self.previous_reward = [np.ones((1, 1), dtype=np.float32)*reward[agent_idx] for agent_idx in range(nbr_agent)]
        self.previous_action = [
            np.eye(self.nbr_actions, dtype=np.float32)[action[agent_idx]].reshape(1, -1)
            for agent_idx in range(nbr_agent)
        ]

        pa = copy.deepcopy(self.previous_action) 
        if self.trajectory_wrapping:
            pa = [
                np.eye(self.nbr_actions, dtype=np.float32)[next_infos[agent_idx]['previous_action'][0]].reshape(1, -1)
                for agent_idx in range(nbr_agent)
            ]
        
        for info_idx in range(len(next_infos)):
            next_infos[info_idx]['previous_reward'] = copy.deepcopy(self.previous_reward[info_idx])
            next_infos[info_idx]['previous_action'] = copy.deepcopy(pa[info_idx])
        
        return next_observation, reward, done, next_infos


class SADEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Simplified Action Decoder wrapper expects the action argument to
        the step method to be a list of dictionnary containing the following keys:
        - "action": the actual action to execute in the environment.
        - "greedy_action": the greedy action that the agent would have used at test time.
        It passes the action to the wrapped environment and writes the greedy action
        of each player into the next_info dictionnary of the other player.
        """
        gym.Wrapper.__init__(self, env)
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        assert isinstance(action, list), "action argument must be a list of dictionnary."
        assert len(action)==2, "not implemented yet for more than 2 players..."
        
        env_action = []
        if isinstance(action[0], dict):
            for a in action:
                env_action.append(a["action"])
        else:
            env_action = action

        next_obs, reward, done, next_infos = self.env.step(env_action)
        
        for info_idx in range(2):
            other_idx = (info_idx+1)%2
            next_infos[info_idx]["greedy_action"] = action[other_idx]["greedy_action"]

        import ipdb; ipdb.set_trace()
        return next_obs, reward, done, next_infos

class SADVecEnvWrapper_depr(object):
    def __init__(self, env, nbr_actions):
        """
        Simplified Action Decoder wrapper expects the action argument for
        the step method to be a list of dictionnary containing the following keys:
        - "action": the actual action to execute in the environment.
        - "greedy_action": the greedy action that the agent would have used at test time.
        It passes the action to the wrapped environment and writes the greedy action
        of each player into the next_info dictionnary of the other player.
        """
        self.env = env
        self.nbr_actions = nbr_actions

    def get_nbr_envs(self):
        return self.env.get_nbr_envs()

    def set_nbr_envs(self, nbr_envs):
        self.env.set_nbr_envs(nbr_envs)

    def render(self, render_mode="rgb_array", env_indices=None):
        return self.env.render(render_mode=render_mode, env_indices=env_indices)
    
    def close(self):
        return self.env.close() 

    def reset(self, **kwargs):
        next_obs, next_infos = self.env.reset(**kwargs)
        
        for player_idx in range(2):
            other_idx = (player_idx+1)%2
            for env_idx in range(len(next_infos[player_idx])):
                ohe_ga = np.zeros((1,self.nbr_actions), dtype=np.float32)
                next_infos[player_idx][env_idx]["greedy_action"] = ohe_ga

        return next_obs, next_infos

    def step(self, action, **kwargs):
        assert isinstance(action, list), "action argument must be a list of dictionnary (or tensors if test-time...)."
        assert len(action)==2, "not implemented yet for more than 2 players..."
        
        env_action = []
        if isinstance(action[0], dict):
            for a in action:
                env_action.append(a["action"])
        else:
            env_action = action

        next_obs, reward, done, next_infos = self.env.step(env_action, **kwargs)
        
        for player_idx in range(2):
            other_idx = (player_idx+1)%2
            for env_idx in range(len(next_infos[player_idx])):
                if isinstance(action[0], dict):
                    ga = action[other_idx]["greedy_action"][env_idx]
                else:
                    ga = action[other_idx][env_idx]
                ohe_ga = np.zeros((1,self.nbr_actions), dtype=np.float32)
                ohe_ga[0,ga] = 1
                next_infos[player_idx][env_idx]["greedy_action"] = ohe_ga

        return next_obs, reward, done, next_infos

class SADVecEnvWrapper(object):
    def __init__(self, env, nbr_actions, otherplay=False):
        """
        Simplified Action Decoder wrapper expects the action argument for
        the step method to be a list of dictionnary containing the following keys:
        - "action": the actual action to execute in the environment.
        - "greedy_action": the greedy action that the agent would have used at test time.
        It also expects the info dictionnary to contain a "current_player" key with an integer
        as the value of the active player.
        It passes the action to the wrapped environment and writes the greedy action
        of the CURRENT PLAYER into the next_info dictionnary of ALL players with
        an extra player_offset tensor.
        """
        self.env = env
        self.otherplay=otherplay
        self.nbr_actions = nbr_actions
        self.nbr_players = None
        self.current_player_idx = None

    def get_nbr_envs(self):
        return self.env.get_nbr_envs()

    def set_nbr_envs(self, nbr_envs):
        self.env.set_nbr_envs(nbr_envs)

    def render(self, render_mode="rgb_array", env_indices=None):
        return self.env.render(render_mode=render_mode, env_indices=env_indices)
    
    def close(self):
        return self.env.close()
        
    def reset(self, **kwargs):
        input_dict = self.env.reset(**kwargs)
        next_obs = input_dict["observations"]
        next_infos = input_dict["info"]

        self.nbr_players = len(next_obs)
        self.current_player_idx = [i["current_player"].item() for i in next_infos[0]]
        # (nbr_env, )

        for player_idx in range(2):
            for env_idx in range(len(next_infos[player_idx])):
                ohe_ga = np.zeros((1,self.nbr_actions), dtype=np.float32)
                ohe_ap = np.zeros((1,self.nbr_players), dtype=np.float32)
                next_infos[player_idx][env_idx]["greedy_action"] = np.concatenate(
                    [ohe_ga, ohe_ap],
                    axis=-1,
                )
        
        output_dict = {
            "observations":next_obs, 
            "info":next_infos,
        }

        return output_dict

    def step(self, action, **kwargs):
        assert isinstance(action, list), "action argument must be a list of dictionnary (or tensors if test-time...)."
        
        env_action = []
        if isinstance(action[0], dict):
            for a in action:
                env_action.append(a["action"])
        else:
            env_action = action

        #next_obs, reward, done, next_infos = self.env.step(env_action, **kwargs)
        env_output_dict = self.env.step(env_action, **kwargs)
        next_obs = env_output_dict["succ_observations"]
        reward = env_output_dict["reward"]
        done = env_output_dict["done"]
        next_infos = env_output_dict["succ_info"]

        for player_idx in range(self.nbr_players):
            for env_idx in range(len(next_infos[player_idx])):
                current_player = self.current_player_idx[env_idx] 
                relative_current_player_idx = (current_player-player_idx) % self.nbr_players
                if isinstance(action[0], dict):
                    #ga = action[other_idx]["greedy_action"][env_idx]
                    ga = action[current_player]["greedy_action"][env_idx]
                    if self.otherplay:
                        # expects env to be wrapped with DiscreteCombinedActionWrapper:
                        dcaw_env = self.env.env_processes[env_idx]
                        while not hasattr(dcaw_env, "_decode_action"):
                            dcaw_env = dcaw_env.env 
                        # expects other play wrapper:
                        ow_env = dcaw_env.env
                        # decode current player's action in the original env:
                        decoded_ga = ow_env._decode_action( 
                            action=dcaw_env._decode_action(ga),
                            player_id=current_player,
                        )
                        # encode current player's action into other player's env:
                        otherplayer_encoded_ga = dcaw_env._encode_action(
                            action_dict=ow_env._encode_action(
                                action=decoded_ga,
                                player_id=player_idx, # other player's view point
                            )
                        )
                        import ipdb; ipdb.set_trace()
                        ga = otherplayer_encoded_ga
                else:
                    #ga = action[other_idx][env_idx]
                    ga = action[current_player][env_idx]
                ohe_ga = np.zeros((1,self.nbr_actions), dtype=np.float32)
                ohe_ga[0,ga] = 1
                ohe_ap = np.zeros((1,self.nbr_players), dtype=np.float32)
                ohe_ap[0,relative_current_player_idx] = 1
                next_infos[player_idx][env_idx]["greedy_action"] = np.concatenate(
                    [ohe_ga, ohe_ap],
                    axis=-1,
                )

        # update:
        self.current_player_idx = [i["current_player"].item() for i in next_infos[0]]
        # (nbr_env, )
        
        output_dict = {
            "actions":env_action, #non-sad actions

            "succ_observations":next_obs, 
            "reward":reward, 
            "done":done, 
            "succ_info":next_infos,
        }

        return output_dict



class FailureEndingTimeLimit(gym.Wrapper):
    """TimeLimit wrapper for failure-ending environments.

    Args:
        env (gym.Env): Env to wrap.
    """

    def __init__(self, env):
        super(FailureEndingTimeLimit, self).__init__(env)
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None,\
            "Cannot call env.step() before calling reset()."
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self.env._max_episode_steps:
            done = False
            self.reset()

        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()


'''
Adapted from:
https://github.com/minerllabs/baselines/blob/master/general/chainerrl/baselines/env_wrappers.py

MIT License

Copyright (c) Kevin Denamganaï

Modifications:
Combination of the two wrappers into one.
'''
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
        self.skip = skip if skip is not None else 1
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


def minerl_wrap_env(env, 
                    size=84,
                    skip=None, 
                    stack=None, 
                    scaling=True, 
                    region_size=8, 
                    observation_wrapper='ObtainPOV',
                    action_wrapper='SerialDiscrete', #'SerialDiscreteCombine'
                    grayscale=False,
                    reward_scheme='None'):
    if isinstance(env, gym.wrappers.TimeLimit):
        #logger.info('Detected `gym.wrappers.TimeLimit`! Unwrap it and re-wrap our own time limit.')
        env = env.env
        max_episode_steps = env.spec.max_episode_steps
        #max_episode_steps = env.env.spec.max_episode_steps
        assert( max_episode_steps == 8e3)
        env = ContinuingTimeLimit(env, max_episode_steps=max_episode_steps)
        
    # Observations:
    if observation_wrapper == 'ObtainPOV':
        env = ObtainPOVWrapper(env=env,
                               size=size,
                               grayscale=grayscale,
                               scaling=scaling)
    elif observation_wrapper == 'UnifiedObservation':
        env = UnifiedObservationWrapper(env=env, 
                                        size=size,
                                        grayscale=grayscale,
                                        region_size=region_size, 
                                        scaling=scaling)
    else:
        raise NotImplementedError

    penalizing = ('penalizing' in reward_scheme)
    if penalizing: reward_scheme = reward_scheme.replace("penalizing", "")
    if reward_scheme == 'single_reward_episode':
        env = SingleRewardWrapper(env=env, penalizing=penalizing)
    elif 'progressive' in reward_scheme:
        reward_scheme = reward_scheme.replace("progressive", "")
        nbr_episode = 1e4
        try:
            reward_scheme = reward_scheme.replace("_", "")
            nbr_episode = float(reward_scheme)
            print(f"Reward Scheme :: Progressive :: nbr_episode = {nbr_episode}")
        except Exception as e:
            print(f'Reward Scheme :: number of episode not understood... ({reward_scheme})')
        env = ProgressivelyMultiRewardWrapper(env=env, penalizing=penalizing, nbr_episode=nbr_episode) 
    
    if skip is not None or stack is not None:
        env = FrameSkipStack(
            env=env, 
            skip=skip, 
            stack=stack
        )
    # Actions:
    if action_wrapper == 'SerialDiscrete':
        env = wrap_env_serial_discrete(env=env)
    elif action_wrapper == 'SerialDiscreteCombine':
        env = wrap_env_serial_discrete_combine(env=env)
    elif action_wrapper == 'SerialDiscreteInterface':
        env = wrap_env_serial_discrete_interface(env=env)
    
    return env



class TextualGoal2IdxWrapper(gym.ObservationWrapper):
    """
    """
    def __init__(self, 
                 env, 
                 max_sentence_length=32, 
                 vocabulary=None, 
                 observation_keys_mapping={'mission':'desired_goal'}):
        gym.ObservationWrapper.__init__(self, env)
        self.max_sentence_length = max_sentence_length
        self.observation_keys_mapping = observation_keys_mapping

        if vocabulary is None:
            vocabulary = set('key ball red green blue purple \
            yellow grey verydark dark neutral light verylight \
            tiny small medium large giant get go fetch go get \
            a fetch a you must fetch a'.split(' '))
        self.vocabulary = set([w.lower() for w in vocabulary])

        # Make padding_idx=0:
        self.vocabulary = ['PAD', 'SoS', 'EoS'] + list(self.vocabulary)

        self.w2idx = {}
        self.idx2w = {}
        for idx, w in enumerate(self.vocabulary):
            self.w2idx[w] = idx
            self.idx2w[idx] = w 
        
        self.observation_space = env.observation_space
        
        for obs_key, map_key in self.observation_keys_mapping.items():
            self.observation_space.spaces[map_key] = gym.spaces.MultiDiscrete([len(self.vocabulary)]*self.max_sentence_length)
        
    def observation(self, observation):
        for obs_key, map_key in self.observation_keys_mapping.items():
            t_goal = [w.lower() for w in observation[obs_key].split(' ')]
            for w in t_goal:
                if w not in self.vocabulary:
                    raise NotImplementedError
                    self.vocabulary.append(w)
                    self.w2idx[w] = len(self.vocabulary)-1
                    self.idx2w[len(self.vocabulary)-1] = w 
            
            idx_goal = self.w2idx['PAD']*np.ones(shape=(self.max_sentence_length), dtype=np.long)
            final_idx = min(self.max_sentence_length, len(t_goal))
            for idx in range(final_idx):
                idx_goal[idx] = self.w2idx[t_goal[idx]]
            # Add 'EoS' token:
            idx_goal[final_idx] = self.w2idx['EoS']
            #padded_idx_goal = nn.utils.rnn.pad_sequence(idx_goal, padding_value=self.w2idx["PAD"])
            #observation[map_key] = padded_idx_goal
            
            observation[map_key] = idx_goal
            
        return observation

class DictObservationSpaceReMapping(gym.ObservationWrapper):
    def __init__(self, env, remapping={'image':'observation'}):
        gym.ObservationWrapper.__init__(self, env)
        self.remapping = remapping

        for obs_key, map_key in self.remapping.items():
            self.observation_space.spaces[map_key] = self.observation_space.spaces[obs_key]
            del self.observation_space.spaces[obs_key]

    def observation(self, observation):
        for obs_key, map_key in self.remapping.items():
            observation[map_key] = observation[obs_key]
            del observation[obs_key]
        return observation


class DictFrameStack(gym.Wrapper):
    def __init__(self, env, stack=4, keys=[]):
        gym.Wrapper.__init__(self,env)
        self.stack = stack if stack is not None else 1
        
        self.keys = keys
        self.observations = {}
        for k in self.keys:
            self.observations[k] = deque([], maxlen=self.stack)
            assert(isinstance(self.env.observation_space.spaces[k], gym.spaces.Box))
        
            low_obs_space = np.repeat(self.env.observation_space.spaces[k].low, self.stack, axis=-1)
            high_obs_space = np.repeat(self.env.observation_space.spaces[k].high, self.stack, axis=-1)
            self.observation_space.spaces[k] = gym.spaces.Box(low=low_obs_space, high=high_obs_space, dtype=self.env.observation_space.spaces[k].dtype)

    def _get_obs(self, observation):
        for k in self.keys:
            observation[k] = LazyFrames(list(self.observations[k]))
        return observation
    
    def reset(self, **args):
        obs = self.env.reset()
        for k in self.keys:
            for _ in range(self.stack):
                self.observations[k].append(obs[k])
        return self._get_obs(obs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for k in self.keys:
            self.observations[k].append(obs[k])        
        return self._get_obs(obs), reward, done, info


from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers.monitoring.video_recorder import logger as video_recorder_logger

class ConfigVideoRecorder(VideoRecorder):
    def __init__(self, env, render_mode, path=None, metadata=None, enabled=True, base_path=None):
        super(ConfigVideoRecorder, self).__init__(
            env=env,
            path=path,
            metadata=metadata,
            enabled=enabled,
            base_path=base_path
        )
        
        modes = env.metadata.get('render.modes', [])
        
        if render_mode not in modes:
            video_recorder_logger.warn('Render mode required is not available with this Env. Disabling further rendering.')
            self.broken = True 

        self.render_mode = render_mode

    def capture_frame(self):
        if not self.functional: return 
        logger.debug('Capturing video frame: path=%s', self.path)

        render_mode = self.render_mode
        frame = self.env.render(mode=render_mode)

        if frame is None:
            if self._async:
                return 
            else:
                video_recorder_logger.warn('Env returned none on render(). Disabling further rendering for video recorder by marking as disabled: path=%s metadat_path=%s',
                    self.path,
                    self.metadat_path
                )
                self.broken=True
        else:
            self.last_frame = frame
            if self.ansi_mode:
                self._encode_ansi_frame(frame)
            else:
                self._encode_image_frame(frame)


class PeriodicVideoRecorderWrapper(gym.Wrapper):
    def __init__(self, env, base_dirpath, video_recording_episode_period=1, render_mode='rgb_array',):
        gym.Wrapper.__init__(self, env)

        self.render_mode = render_mode
        self.episode_idx = 0
        self.base_dirpath = base_dirpath
        os.makedirs(self.base_dirpath, exist_ok=True)
        self.video_recording_episode_period = video_recording_episode_period
        
        self.is_video_enabled = True
        self._init_video_recorder(env=env, path=os.path.join(self.base_dirpath, 'video_0.mp4'))

    def _init_video_recorder(self, env, path):
        #self.video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env=env, path=path, enabled=True)
        self.video_recorder = ConfigVideoRecorder(env=env, render_mode=self.render_mode, path=path, enabled=True)

    def reset(self, **args):
        self.episode_idx += 1

        env_output = super(PeriodicVideoRecorderWrapper, self).reset()

        if self.episode_idx % self.video_recording_episode_period == 0:
            path = os.path.join(self.base_dirpath, f'video_{self.episode_idx}.mp4')
            self._init_video_recorder(env=self.env, path=path) 
            self.is_video_enabled = True
        else:
            if self.is_video_enabled:
                self.video_recorder.capture_frame()
                self.video_recorder.close()
                del self.video_recorder
                self.is_video_enabled = False

        return env_output

    def step(self, action):
        if self.is_video_enabled:
            self.video_recorder.capture_frame()

        return super(PeriodicVideoRecorderWrapper, self).step(action)


def baseline_ther_wrapper(env, 
                          size=None, 
                          skip=0, 
                          stack=4, 
                          single_life_episode=False, 
                          nbr_max_random_steps=0, 
                          clip_reward=False,
                          max_sentence_length=32,
                          vocabulary=None,
                          time_limit=40):
    
    env = TimeLimit(env, max_episode_steps=time_limit)

    if nbr_max_random_steps > 0:
        env = NoopResetEnv(env, noop_max=nbr_max_random_steps)
    
    if skip > 0:
        env = MaxAndSkipEnv(env, skip=skip)
    
    if size is not None and 'None' not in size:
        env = FrameResizeWrapper(env, size=size) 
    
    if single_life_episode:
        env = EpisodicLifeEnv(env)
    
    if stack > 1:
        env = DictFrameStack(env, stack=stack, keys=['image'])
    
    if clip_reward:
        env = ClipRewardEnv(env)

    env = TextualGoal2IdxWrapper(env=env,
                                 max_sentence_length=max_sentence_length,
                                 vocabulary=vocabulary)

    env = DictObservationSpaceReMapping(env=env, remapping={'image':'observation'})

    return env

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

class MineRLObservationSplitFrameSkipWrapper(gym.Wrapper):
    """
    Split state dictionary into pov and inventory
    Repeat action for n steps
    """
    def __init__(self,env,skip=4):
        gym.Wrapper.__init__(self,env)
        self.observation_space = gym.spaces.Box(low=0.0, high=255.0, shape=(64,64,3), dtype=np.float32)
        self.skip = skip
    
    def reset(self,**args):
        obs = self.env.reset()
        return obs['pov']
    
    def step(self,action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:break
        info['inventory'] = obs['vector']
        return obs['pov'],total_reward,done,info

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

def minerl2020_wrap_env(env,
                        action_set,
                        skip=None,
                        stack=None,
                        grayscale: bool = False,
                        previous_reward_action=True,
                        trajectory_wrapping=False,
                        competition_testing: bool = False):
    '''
    Add all wrappers need for minerl 2020
    '''
    if isinstance(env,gym.wrappers.TimeLimit):
        env = env.env
        max_episode_steps = env.spec.max_episode_steps
        if not(competition_testing):
            max_episode_steps = 5000
        env = ContinuingTimeLimit(env,max_episode_steps=max_episode_steps)
    
    # {POV, vector}, continuous action
    env = MineRLObservationSplitWrapper(env=env, trajectory_wrapping=trajectory_wrapping)
    
    if grayscale:
        env = gym.wrappers.GrayScaleObservation(env,keep_dim=True)
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
    
    # Clip reward to (-1,0,+1)
    env = ClipRewardEnv(env)
    
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
