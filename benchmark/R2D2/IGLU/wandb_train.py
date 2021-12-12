from typing import Dict, Any, Optional, List, Callable

import torch
import sklearn 
import gym
import iglu

from perceiver_lm.perceiver_io import LMCapsule
from collections import OrderedDict
from copy import deepcopy

import logging
import yaml
import os
import sys

import torch.multiprocessing
import ray

from tensorboardX import SummaryWriter
from tqdm import tqdm
from functools import partial

import argparse
import numpy as np
import random

import regym
from regym.environments import generate_task, EnvType
from regym.rl_loops.multiagent_loops import marl_loop
from regym.util.experiment_parsing import initialize_agents
from regym.modules import EnvironmentModule, CurrentAgentsModule
from regym.modules import MARLEnvironmentModule, RLAgentModule

#from regym.modules import ReconstructionFromHiddenStateModule, MultiReconstructionFromHiddenStateModule
from iglu_task_curriculum_module import build_IGLUTaskCurriculumModule, IGLUTaskCurriculumModule
from iglu_task_curriculum_module import IGLUTaskCurriculumWrapper

from regym.pubsub_manager import PubSubManager

import wandb


LanguageModelCapsule = None

# TODO: figure out the actual air.id:
# eventhough the doc says air.id=-1, the actual data are showing 0...
air_block_offset = 0
reward_divider = 10 #60 #600 when using inverted...

BUILD_ZONE_SIZE_X = 11
BUILD_ZONE_SIZE_Z = 11

class MaxIntersectionComputation:
    def __init__(self, target_grid):
        self.target_grid = target_grid
    
        self.init()

    def init(self):
        self.target_size = (self.target_grid != 0).sum().item()
        self.target_grids = [self.target_grid]
        self.admissible = [[] for _ in range(4)]
        # fill self.target_grids with four rotations of the original grid around the vertical axis
        for _ in range(3):
            self.target_grids.append(np.zeros(self.target_grid.shape, dtype=np.int32))
            for x in range(BUILD_ZONE_SIZE_X):
                for z in range(BUILD_ZONE_SIZE_Z):
                    self.target_grids[-1][:, z, BUILD_ZONE_SIZE_X - x - 1] \
                        = self.target_grids[-2][:, x, z]
        # (dx, dz) is admissible iff the translation of target grid by (dx, dz) preserve (== doesn't cut)
        # target structure within original (unshifted) target grid
        for i in range(4):
            for dx in range(-BUILD_ZONE_SIZE_X + 1, BUILD_ZONE_SIZE_X):
                for dz in range(-BUILD_ZONE_SIZE_Z + 1, BUILD_ZONE_SIZE_Z):
                    sls_target = self.target_grids[i][:, max(dx, 0):BUILD_ZONE_SIZE_X + min(dx, 0),
                                                         max(dz, 0):BUILD_ZONE_SIZE_Z + min(dz, 0):]
                    if (sls_target != 0).sum().item() == self.target_size:
                        self.admissible[i].append((dx, dz))
    
    def set_target_grid(self, target_grid):
        target_grid = target_grid.reshape((9,11,11))
        if not isinstance(target_grid, np.ndarray): target_grid = target_grid.numpy()
        if not isinstance(self.target_grid, np.ndarray): self.target_grid = self.target_grid.numpy()

        if (target_grid-self.target_grid).sum().item() != 0:
            self.target_grid = target_grid
            self.init()

    def maximal_intersection(self, grid):
        grid = grid.reshape((9,11,11))
        max_int = 0
        for i, admissible in enumerate(self.admissible):
            for dx, dz in admissible:
                x_sls = slice(max(dx, 0), BUILD_ZONE_SIZE_X + min(dx, 0))
                z_sls = slice(max(dz, 0), BUILD_ZONE_SIZE_Z + min(dz, 0))
                sls_target = self.target_grids[i][:, x_sls, z_sls]

                x_sls = slice(max(-dx, 0), BUILD_ZONE_SIZE_X + min(-dx, 0))
                z_sls = slice(max(-dz, 0), BUILD_ZONE_SIZE_Z + min(-dz, 0))
                sls_grid = grid[:, x_sls, z_sls]
                intersection = ((sls_target == sls_grid) & (sls_target != 0)).sum().item()
                if intersection > max_int:
                    max_int = intersection
        return max_int

MaxIntComputer = MaxIntersectionComputation( np.zeros((9,11,11)))

def IGLU_maxint_goal_predicated_reward_fn(
    achieved_exp,
    target_exp,
    _extract_goal_from_info_fn,
    goal_key,
    latent_goal_key=None,
    epsilon=1e-3,
    ):
    """
    This reward function is always negative,
    and it is maximized when the target goal 
    is achieved.

    HYP1: target goals that consist of empty
    grids are also able to maximize this
    reward function, without any actions
    from the agent.
    It might incentivise the agent to do
    strictly nothing...
    """
    achieved_goal = _extract_goal_from_info_fn(
        achieved_exp['succ_info'],
        goal_key=goal_key,
    )
    target_goal = _extract_goal_from_info_fn(
        target_exp['succ_info'],
        goal_key=goal_key,
    )
    
    achieved_latent_goal = None
    target_latent_goal = None
    if latent_goal_key is not None:
        achieved_latent_goal = _extract_goal_from_info_fn(
            achieved_exp['succ_info'],
            goal_key=latent_goal_key,
        )
        target_latent_goal = _extract_goal_from_info_fn(
            target_exp['succ_info'],
            goal_key=latent_goal_key,
        )
    
    # Preprocessing: air.id=-1...
    # assuming grids:
    global air_block_offset
    target_goal += air_block_offset
    achieved_goal += air_block_offset 

    global MaxIntComputer
    MaxIntComputer.set_target_grid(target_goal)
    max_int = MaxIntComputer.maximal_intersection(achieved_goal)
    
    reward = max_int-MaxIntComputer.target_size
    
    global reward_divider
    reward /= reward_divider

    # sparsification:
    if reward < -epsilon:
        reward = -1 
    if reward > 0:
        raise NotImplementedError
    
    return reward*torch.ones(1,1), target_goal, target_latent_goal

def IGLU_goal_predicated_reward_fn(
    achieved_exp,
    target_exp,
    _extract_goal_from_info_fn,
    goal_key,
    latent_goal_key=None,
    epsilon=1e-3,
    ):
    """
    This reward function is always negative,
    and it is maximized when the target goal 
    is achieved.

    HYP1: target goals that consist of empty
    grids are also able to maximize this
    reward function, without any actions
    from the agent.
    It might incentivise the agent to do
    strictly nothing...
    """
    achieved_goal = _extract_goal_from_info_fn(
        achieved_exp['succ_info'],
        goal_key=goal_key,
    )
    target_goal = _extract_goal_from_info_fn(
        target_exp['succ_info'],
        goal_key=goal_key,
    )
    
    achieved_latent_goal = None
    target_latent_goal = None
    if latent_goal_key is not None:
        achieved_latent_goal = _extract_goal_from_info_fn(
            achieved_exp['succ_info'],
            goal_key=latent_goal_key,
        )
        target_latent_goal = _extract_goal_from_info_fn(
            target_exp['succ_info'],
            goal_key=latent_goal_key,
        )
    

    # Preprocessing: air.id=-1...
    # assuming grids:
    global air_block_offset
    target_goal += air_block_offset
    achieved_goal += air_block_offset 

    abs_diff = np.abs(target_goal-achieved_goal).sum()  
    reward = -abs_diff

    global reward_divider
    reward /= reward_divider

    # sparsification:
    if reward < -epsilon:
        reward = -1 
    if reward > 0:
        raise NotImplementedError
    
    return reward*torch.ones(1,1), target_goal, target_latent_goal

# TODO: update as above with latent goals...
def IGLU_inverted_goal_predicated_reward_fn(
    achieved_exp,
    target_exp,
    _extract_goal_from_info_fn,
    goal_key,
    epsilon=1e-3,
    ):
    """
    This reward function is always positive,
    and it is maximized when the target goal 
    is achieved and that target goal contains
    a lot of blocks.

    HYP1: target goals that consist of empty
    grids are minimizing this reward function.
    """
    raise NotImplementedError

    achieved_goal = _extract_goal_from_info_fn(
        achieved_exp['info'],
        goal_key=goal_key,
    )
    target_goal = _extract_goal_from_info_fn(
        target_exp['info'],
        goal_key=goal_key,
    )

    # assuming grids:
    # mapping air.id=>0
    global air_block_offset
    target_goal += air_block_offset
    achieved_goal += air_block_offset

    target_goal_size = target_goal.sum()
    abs_diff = np.abs(target_goal-achieved_goal).sum()
    reward = target_goal_size - abs_diff
    
    global reward_divider
    reward /= reward_divider
    
    # sparsification?
    # if reward > -epsilon:
    #    reward = 0
        
    return reward*torch.ones(1,1), target_goal


class IGLUGoalOHEWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **args):
        obs, info = self.env.reset(**args)
        
        info['target_grid_ohe'] = self._ohe_encoding(info['target_grid'].reshape((1,-1)))
        info['grid_ohe'] = self._ohe_encoding(info['grid'].reshape((1,-1)))   
        
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        info['target_grid_ohe'] = self._ohe_encoding(info['target_grid'].reshape((1,-1)))
        info['grid_ohe'] = self._ohe_encoding(info['grid'].reshape((1,-1)))   
        
        return obs, reward, done, info

    def _ohe_encoding(
        self,
        grid,
        ):
        grid = np.expand_dims(grid, axis=-1)
        # (1, 1089, 1)
        grid_ohe = np.zeros((1,1089,7))
        np.put_along_axis(grid_ohe,indices=grid,values=1,axis=-1)
        # Removing the air-related encoding:
        grid_ohe = grid_ohe[...,1:].reshape((1,-1))
        return grid_ohe


class IGLUHERGoalPredicatedRewardWrapper(gym.Wrapper):
    def __init__(self, env, inverted=False, epsilon=1e-3):
        super().__init__(env)
        self.inverted = inverted
        self.epsilon = epsilon
    
    def reset(self, **args):
        obs, info = self.env.reset(**args)
        
        # There is not 'target_grid' available on the reset...
        info['target_grid'] = info['grid'].reshape((1,-1))
        info['grid'] = info['grid'].reshape((1,-1))
        
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        global air_block_offset
        target_goal = info['target_grid']+air_block_offset
        achieved_goal = info['grid']+air_block_offset
    
        #reward = self._reward_fn1(target_goal, achieved_goal)
        reward = self._maxint_reward_fn(target_goal, achieved_goal)

        info['target_grid'] = info['target_grid'].reshape((1,-1))
        info['grid'] = info['grid'].reshape((1,-1))   
        
        """
        # TODO: Testing was done using:
        if (info['grid']>1).sum().item()>0:
            reward = 0
        """

        return obs, reward, done, info

    def _maxint_reward_fn(
        self,
        target_goal,
        achieved_goal,
        ):
        target_goal_size = target_goal.sum()
        abs_diff = np.abs(target_goal-achieved_goal).sum()
        
        global MaxIntComputer
        MaxIntComputer.set_target_grid(target_goal)
        max_int = MaxIntComputer.maximal_intersection(achieved_goal)
    
        reward = max_int-MaxIntComputer.target_size
    
        global reward_divider
        reward /= reward_divider

        # sparsification:
        if reward < -self.epsilon:
            reward = -1 
        if reward > 0:
            raise NotImplementedError
    
        return reward

    def _reward_fn1(
        self,
        target_goal,
        achieved_goal,
        ):
        target_goal_size = target_goal.sum()
        abs_diff = np.abs(target_goal-achieved_goal).sum()
        
        # the following sends a different message than the 
        # HER predicate fn above which expects exact
        # correspondance between grid and target grid, 
        # rather than max int...
        if False: #reward>=2:
            # if the max int. is increased, then it is a win:
            # HYP: when target contains many blocks,
            # then a modular reward is provided for any increase towards goal.
            #TODO: implement max int situtation...
            reward = 0
        else:
            if self.inverted:
                reward = target_goal_size - abs_diff
            else:
                reward = (-1)*abs_diff

            global reward_divider
            reward /= reward_divider 
        
            if not self.inverted:
                # sparsification:
                if reward < -self.epsilon:
                    reward = -1
                if reward > 0:
                    raise NotImplementedError
        return reward


class IgluActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.noop_dict = OrderedDict({
            'attack': np.array(0),
            'back': np.array(0),
            'camera': np.array([0, 0]),
            'forward': np.array(0),
            'hotbar': np.array(0),
            'jump': np.array(0),
            'left': np.array(0),
            'right': np.array(0),
            'use': np.array(0),
        })

        """
        self.actions_list = [
            {'attack': np.array(1)},
            {'back': np.array(1)},

            {'camera': np.array([5.0, 0.0])},
            {'camera': np.array([-5.0, 0.0])},
            {'camera': np.array([0.0, 5.0])},
            {'camera': np.array([0.0, -5.0])},
            {'camera': np.array([0.0, 0.0])}, # noop

            {'forward': np.array(1)},
            # 'hotbar', np.array(0),
            # ('hotbar', np.array(1)),
            # 'hotbar', np.array(2),
            # 'hotbar', np.array(3),
            # 'hotbar', np.array(4),
            # 'hotbar', np.array(5),
            {'jump': np.array(1)},
            {'left': np.array(1)},
            {'right': np.array(1)},
            {'use': np.array(1)},
        ]
        """
        self.actions_list = [
            {'attack': np.array(1)},
            {'back': np.array(1)},

            {'camera': np.array([5.0, 0.0])},
            {'camera': np.array([-5.0, 0.0])},
            {'camera': np.array([0.0, 5.0])},
            {'camera': np.array([0.0, -5.0])},
            #{'camera': np.array([0.0, 0.0])}, # noop

            {'forward': np.array(1)},

            {'hotbar': np.array(0)},
            {'hotbar': np.array(1)},
            {'hotbar': np.array(2)},
            {'hotbar': np.array(3)},
            {'hotbar': np.array(4)},
            {'hotbar': np.array(5)},
            
            {'jump': np.array(1)},
            {'left': np.array(1)},
            {'right': np.array(1)},
            
            {'use': np.array(1)},
        ]

        self.action_space = gym.spaces.Discrete(len(self.actions_list))

    def action(self, action):
        result = deepcopy(self.noop_dict)
        result.update(self.actions_list[action])
        return result

    def reverse_action(self, action):
        pass


class IgluBlockDenseActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.noop_dict = OrderedDict({
            'attack': np.array(0),
            'back': np.array(0),
            'camera': np.array([0, 0]),
            'forward': np.array(0),
            'hotbar': np.array(0),
            'jump': np.array(0),
            'left': np.array(0),
            'right': np.array(0),
            'use': np.array(0),
        })
        
        """
            {'hotbar': np.array(0), 'use': np.array(1), 'camera': np.array([1.0, 0.0])},
            {'hotbar': np.array(1), 'use': np.array(1), 'camera': np.array([1.0, 0.0])},
            {'hotbar': np.array(2), 'use': np.array(1), 'camera': np.array([1.0, 0.0])},
            {'hotbar': np.array(3), 'use': np.array(1), 'camera': np.array([1.0, 0.0])},
            {'hotbar': np.array(4), 'use': np.array(1), 'camera': np.array([1.0, 0.0])},
            {'hotbar': np.array(5), 'use': np.array(1), 'camera': np.array([1.0, 0.0])},
        """            
        self.actions_list = [
            {'attack': np.array(1)},
            {'back': np.array(1)},

            {'camera': np.array([5.0, 0.0])},
            {'camera': np.array([-5.0, 0.0])},
            {'camera': np.array([0.0, 5.0])},
            {'camera': np.array([0.0, -5.0])},
            #{'camera': np.array([0.0, 0.0])}, # noop

            {'forward': np.array(1)},
            
            {'hotbar': np.array(0), 'use': np.array(1)},
            {'hotbar': np.array(1), 'use': np.array(1)},
            {'hotbar': np.array(2), 'use': np.array(1)},
            {'hotbar': np.array(3), 'use': np.array(1)},
            {'hotbar': np.array(4), 'use': np.array(1)},
            {'hotbar': np.array(5), 'use': np.array(1)},
              
            {'jump': np.array(1)},
            {'left': np.array(1)},
            {'right': np.array(1)},
            
            {'use': np.array(1)},
        ]

        self.action_space = gym.spaces.Discrete(len(self.actions_list))

    def action(self, action):
        result = deepcopy(self.noop_dict)
        result.update(self.actions_list[action])
        return result

    def reverse_action(self, action):
        pass


class FakeResetWrapper(gym.Wrapper):
    def __init__(
        self, 
        env, 
        max_episode_length=None,
        fake=True,
        ):
        super().__init__(env)
        self.fake = fake 

        self.fake_reset = False
        self.fake_reset_obs = None 
        self.fake_reset_info = None
        self.reset_outputs_info = False 
        self.max_episode_length = max_episode_length

        self.obs_counter = 0

    def reset(self, **args):
        """
        WARNING: This wrapper assumes that the reset
        method delivers both an observation
        and a dictionnary info.
        """
        self.obs_counter = 1

        if self.fake\
        and self.fake_reset:
            self.fake_reset = False
            if self.reset_outputs_info:
                return self.fake_reset_obs, self.fake_reset_info
            else:
                return self.fake_reset_obs
        else:
            reset_output = self.env.reset(**args)
            self.reset_outputs_info = isinstance(reset_output, tuple)
            return reset_output

    def step(self, action):
        self.obs_counter += 1
        obs, reward, done, info = self.env.step(action)
        
        if self.max_episode_length is not None\
        and not done\
        and self.obs_counter >= self.max_episode_length:
            if self.fake:
                self.fake_reset = True
                self.fake_reset_obs = obs
                self.fake_reset_info = info
            done = True

        return obs, reward, done, info

class IGLUCurriculumFakeResetWrapper(gym.Wrapper):
    def __init__(
        self, 
        env, 
        max_episode_length=None,
        sparse_positive_reward=False,
        ):
        super().__init__(env)
        
        self.fake_reset = False
        self.fake_reset_obs = None 
        self.fake_reset_info = None
        self.reset_outputs_info = False 
        self.max_episode_length = max_episode_length
        self.sparse_positive_reward = sparse_positive_reward

        self.obs_counter = 0

    def reset(self, **args):
        """
        WARNING: This wrapper assumes that the reset
        method delivers both an observation
        and a dictionnary info.
        """
        self.obs_counter = 1

        if self.fake_reset:
            self.fake_reset = False
            if self.reset_outputs_info:
                return self.fake_reset_obs, self.fake_reset_info
            else:
                return self.fake_reset_obs
        else:
            reset_output = self.env.reset(**args)
            self.reset_outputs_info = isinstance(reset_output, tuple)
            return reset_output

    def step(self, action):
        self.obs_counter += 1
        obs, reward, done, info = self.env.step(action)
        
        if reward>2 or reward<-2:
            #TODO: debug this situation,
            # which has its roots in the reward handler...
            reward = 0
            #import ipdb; ipdb.set_trace()
        
        if reward == -1:
            # we assume that this reward prevents the agent
            # from trying to lay down blocks:
            reward = 0
        
        if self.sparse_positive_reward\
        and reward < 1:
            reward = 0
        if self.sparse_positive_reward\
        and reward > 1:
            reward = 2
       
        if reward>1 or reward<-1:
            self.fake_reset=True
            self.fake_reset_obs = obs
            self.fake_reset_info = info
            done = True
        
        if self.max_episode_length is not None\
        and not done\
        and self.obs_counter >= self.max_episode_length:
            done = True
            reward = 0

        return obs, reward, done, info


class PovOnlyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)

    """
    def observation(self, observation):
        return observation['pov']
    """
    
    def reset(self, **args):
        obs = self.env.reset(**args)
        info = {}
        for key, value in obs.items():
            if key=="pov":  continue
            info[key] = value
        obs = obs["pov"]
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for key, value in obs.items():
            if key=="pov":  continue
            info[key] = value
        obs = obs["pov"]
        return obs, reward, done, info

class ChatEmbeddingWrapper(gym.Wrapper):
    def __init__(self, env, use_cuda=True):
        super().__init__(env)
        global LanguageModelCapsule
        if LanguageModelCapsule is None:
            LanguageModelCapsule = LMCapsule(use_cuda=use_cuda)
        self.lm = LanguageModelCapsule
        self.chat = None

    def reset(self, **args):
        obs = self.env.reset(**args)
        self.chat = None
        if "chat" in obs:
            obs["chat"] = self.lm(obs["chat"]).detach().numpy()
            self.chat = obs["chat"]
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "chat" in obs:
            if self.chat is None:
                self.chat = self.lm(obs["chat"]).detach().numpy()
            else:
                obs["chat"] = self.chat

        return obs, reward, done, info
    

class ErrorCatchingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_valid_exp = None
    def reset(self, **args):
        obs = self.env.reset(**args)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info.get("error", False):
            print("//////////////////////////////////////////////////////")
            print("WARNING: IGLU ENV ERROR CAUGHT.")
            print("//////////////////////////////////////////////////////")
            if self.previous_valid_exp is not None:
                return self.previous_valid_exp
            else:
                import ipdb; ipdb.set_trace()
        self.previous_valid_exp = (obs, reward, True, info)
        return obs, reward, done, info

class Vocabulary(object):
    def __init__(self):
        self.vocabulary = set('key ball red green blue purple \
            yellow grey verydark dark neutral light verylight \
            tiny small medium large giant get go fetch go get \
            a fetch a you must fetch a'.split(' '))
        self.vocabulary = set([w.lower() for w in self.vocabulary])
        
        # Make padding_idx=0:
        self.vocabulary = ['PAD', 'SoS', 'EoS'] + list(self.vocabulary)

        self.w2idx = {}
        self.idx2w = {}
        for idx, w in enumerate(self.vocabulary):
            self.w2idx[w] = idx
            self.idx2w[idx] = w 
        
    def __len__(self):
        return len(self.vocabulary)
    
    def contains(self, w):
        return (w in self.vocabulary)
    
    def add(self, w):
        self.vocabulary.append(w)
        self.w2idx[w] = len(self.vocabulary)-1
        self.idx2w[len(self.vocabulary)-1] = w 
            
 
global_vocabulary = Vocabulary()

class TextualGoal2IdxWrapper(gym.ObservationWrapper):
    """
    """
    def __init__(
        self, 
        env, 
        max_sentence_length=256, 
        vocabulary=None, 
        observation_keys_mapping={'chat':'chat'},
        ):
        gym.ObservationWrapper.__init__(self, env)
        self.max_sentence_length = max_sentence_length
        self.observation_keys_mapping = observation_keys_mapping

        global global_vocabulary
        self.vocabulary = global_vocabulary
        
        self.observation_space = env.observation_space
        
        for obs_key, map_key in self.observation_keys_mapping.items():
            self.observation_space.spaces[map_key] = gym.spaces.MultiDiscrete([len(self.vocabulary)]*self.max_sentence_length)
    
    @property
    def w2idx(self, w):
        return self.vocabulary.w2idx[w]

    @property
    def idx2w(self, idx):
        return self.vocabulary.idx2w[idx]

    def observation(self, observation):
        import ipdb; ipdb.set_trace()
        for obs_key, map_key in self.observation_keys_mapping.items():
            t_goal = [w.lower() for w in observation[obs_key].split(' ')]
            
            for w in t_goal:
                if not self.vocabulary.contains(w):
                    # Increase the vocabulary as we train:
                    self.vocabulary.add(w)

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


from regym.util.wrappers import (
    ClipRewardEnv, 
    PreviousRewardActionInfoWrapper,
    FrameStackWrapper,
)

def wrap_iglu(
    env,
    stack=1,
    clip_reward=False,
    previous_reward_action=True,
    trajectory_wrapping=False,
    curriculum_fake_reset=False,
    max_episode_length=None,
    sparse_positive_reward=False,
    block_dense_actions=False,
    use_HER=False,
    task_curriculum=False,
    inverted_goal_predicated_reward=False,
    use_THER=False,
    use_OHE=False,
    ):
    env = gym.make('IGLUSilentBuilder-v0', max_steps=1000, )
    env = ErrorCatchingWrapper(env)
    if task_curriculum:
        env = IGLUTaskCurriculumWrapper(env)

    #env.update_taskset(TaskSet(preset=[task]))
    if use_THER:
        #TODO: set hyperparameters...
        env = TextualGoal2IdxWrapper(
            env=env,
            max_sentence_length=256,
            vocabulary=None,
        )
    else:
        env = ChatEmbeddingWrapper(env)
    env = PovOnlyWrapper(env)
    if block_dense_actions:
        env = IgluBlockDenseActionWrapper(env)
    else:
        env = IgluActionWrapper(env)
    
    # Clip reward to (-1,0,+1)
    if clip_reward:
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
    
    if curriculum_fake_reset:
        env = IGLUCurriculumFakeResetWrapper(
            env,
            max_episode_length=max_episode_length,
            sparse_positive_reward=sparse_positive_reward,
        )
    elif max_episode_length is not None:
        env = FakeResetWrapper(
            env=env,
            max_episode_length=max_episode_length,
            fake=False, #only used for max episode length here...
        )

    if use_HER:
        env = IGLUHERGoalPredicatedRewardWrapper(
            env,
            inverted_goal_predicated_reward,
        )
    
    if use_OHE:
        env = IGLUGoalOHEWrapper(env)
    
    if stack > 1:
        env = FrameStackWrapper(env, stack=stack)
    

    return env


def check_path_for_agent(filepath):
    #filepath = os.path.join(path,filename)
    agent = None
    offset_episode_count = 0
    if os.path.isfile(filepath):
        print('==> loading checkpoint {}'.format(filepath))
        agent = torch.load(filepath).clone(with_replay_buffer=True)
        offset_episode_count = agent.episode_count
        #setattr(agent, 'episode_count', offset_episode_count)
        print('==> loaded checkpoint {}'.format(filepath))
    return agent, offset_episode_count


def make_rl_pubsubmanager(
    agents,
    config, 
    logger=None,
    load_path=None,
    save_path=None,
    node_id_to_extract="hidden",
    ):
    """
    Create a PubSubManager.
    :param agents: List of Agents to use in the rl loop.
    :param config: Dict that specifies all the important hyperparameters of the network.
        - "task"
        - "sad"
        - "vdn"
        - "otherplay"
        - "max_obs_count"
        - "sum_writer": str where to save the summary...

    """
    modules = config.pop("modules")

    cam_id = "current_agents"
    modules[cam_id] = CurrentAgentsModule(
        id=cam_id,
        agents=agents
    )

    envm_id = "MARLEnvironmentModule_0"
    envm_input_stream_ids = {
        "logs_dict":"logs_dict",
        "iteration":"signals:iteration",
        "current_agents":f"modules:{cam_id}:ref",
    }
    
    rlam_ids = [
      f"rl_agent_{rlaidx}"
      for rlaidx in range(len(agents))
    ]
    for aidx, (rlam_id, agent) in enumerate(zip(rlam_ids, agents)):
      rlam_config = {
        'agent': agent,
        'actions_stream_id':f"modules:{envm_id}:player_{aidx}:actions",
      }

      envm_input_stream_ids[f'player_{aidx}'] = f"modules:{rlam_id}:ref"

      rlam_input_stream_ids = {
        "logs_dict":"logs_dict",
        "losses_dict":"losses_dict",
        "epoch":"signals:epoch",
        "mode":"signals:mode",

        "reset_actors":f"modules:{envm_id}:reset_actors",
        
        "observations":f"modules:{envm_id}:ref:player_{aidx}:observations",
        "infos":f"modules:{envm_id}:ref:player_{aidx}:infos",
        "actions":f"modules:{envm_id}:ref:player_{aidx}:actions",
        "succ_observations":f"modules:{envm_id}:ref:player_{aidx}:succ_observations",
        "succ_infos":f"modules:{envm_id}:ref:player_{aidx}:succ_infos",
        "rewards":f"modules:{envm_id}:ref:player_{aidx}:rewards",
        "dones":f"modules:{envm_id}:ref:player_{aidx}:dones",
      }
      modules[rlam_id] = RLAgentModule(
          id=rlam_id,
          config=rlam_config,
          input_stream_ids=rlam_input_stream_ids,
      )

    modules[envm_id] = MARLEnvironmentModule(
        id=envm_id,
        config=config,
        input_stream_ids=envm_input_stream_ids
    )
    
    if config["use_task_curriculum"]:
        tcm_id = "IGLUTaskCurriculumModule_0"
        tcm_config = {
            "task":config["task"],
            "max_episode_length":config["max_episode_length"],
        }
        tcm_input_stream_ids = None
        modules[tcm_id] = build_IGLUTaskCurriculumModule(
            id=tcm_id,
            config=tcm_config,
            input_stream_ids=tcm_input_stream_ids,
        )
        
    pipelines = config.pop("pipelines")
    
    pipelines["rl_loop_0"] = [
        envm_id,
    ]
    for rlam_id in rlam_ids:
      pipelines['rl_loop_0'].append(rlam_id)
    
    if config["use_task_curriculum"]:
        pipelines['rl_loop_0'].append(tcm_id)

    optim_id = "global_optim"
    optim_config = {
      "modules":modules,
      "learning_rate":3e-4,
      "optimizer_type":'adam',
      "with_gradient_clip":False,
      "adam_eps":1e-16,
    }

    optim_module = regym.modules.build_OptimizationModule(
      id=optim_id,
      config=optim_config,
    )
    modules[optim_id] = optim_module

    logger_id = "per_epoch_logger"
    logger_module = regym.modules.build_PerEpochLoggerModule(id=logger_id)
    modules[logger_id] = logger_module
    
    pipelines[optim_id] = []
    pipelines[optim_id].append(optim_id)
    pipelines[optim_id].append(logger_id)

    pbm = PubSubManager(
        config=config,
        modules=modules,
        pipelines=pipelines,
        logger=logger,
        load_path=load_path,
        save_path=save_path,
    )
    
    return pbm



def train_and_evaluate(
    agent_config: Dict[str,Any],
    task_config: Dict[str,Any],
    agent: object, 
    task: object, 
    sum_writer: object, 
    base_path: str, 
    offset_episode_count: int = 0, 
    nbr_pretraining_steps: int = 0, 
    nbr_max_observations: int = 1e7,
    test_obs_interval: int = 1e4,
    test_nbr_episode: int = 10,
    benchmarking_record_episode_interval: int = None,
    step_hooks=[],
    render_mode="human",
    ):
    
    config = {
      "modules": {},
      "pipelines": {},
    }

    config['training'] = True
    config['env_configs'] = None
    config['task'] = task 
    
    sum_writer_path = os.path.join(sum_writer, 'actor.log')
    sum_writer = config['sum_writer'] = SummaryWriter(sum_writer_path, flush_secs=1)

    config['base_path'] = base_path 
    config['offset_episode_count'] = offset_episode_count
    config['nbr_pretraining_steps'] = nbr_pretraining_steps 
    config['max_obs_count'] = nbr_max_observations
    config['test_obs_interval'] = test_obs_interval
    config['test_nbr_episode'] = test_nbr_episode
    config['benchmarking_record_episode_interval'] = benchmarking_record_episode_interval
    config['render_mode'] = render_mode
    config['save_traj_length_divider'] = 5
    config['sad'] = False
    config['vdn'] = False
    config['otherplay'] = False
    config['nbr_players'] = 1
    config['step_hooks'] = [] 
    
    # Task Curriculum:
    config["use_task_curriculum"] = task_config["task_curriculum"]
    config["max_episode_length"] = task_config["max_fake_episode_length"]
    if not isinstance(config["max_episode_length"], int)\
    and config["use_task_curriculum"]:
        raise NotImplementedError
        config["max_episode_length"] = 1000

    agents = [agent]

    pubsubmanager = make_rl_pubsubmanager(
      agents=agents,
      config=config,
      logger=sum_writer,
    )

    pubsubmanager.train() 

    save_replay_buffer = False
    if len(sys.argv) > 2:
      save_replay_buffer = any(['save_replay_buffer' in arg for arg in sys.argv])

    """
    try:
        for agent in agents:
            agent.save(with_replay_buffer=save_replay_buffer)
            print(f"Agent saved at: {agent.save_path}")
    except Exception as e:
        print(e)
    """

    task.env.close()
    task.test_env.close()

    return agents


def IGLU_HER_filtering_fn(
    d2store,
    episode_buffer,
    achieved_goal_from_target_exp,
    **kwargs,
    ):
    global air_block_offset
    goal_size = (achieved_goal_from_target_exp+air_block_offset).sum().item()
    return goal_size>0


def training_process(
    agent_config: Dict, 
    task_config: Dict,
    benchmarking_interval: int = 1e4,
    benchmarking_episodes: int = 10, 
    benchmarking_record_episode_interval: int = None,
    train_observation_budget: int = 1e7,
    base_path: str = './',
    video_recording_episode_period: int = None,
    seed: int = 0,
    ):
    
    test_only = task_config.get('test_only', False)
    path_suffix = task_config.get('path_suffix', None)
    if path_suffix=='None':  path_suffix=None
    
    if len(sys.argv) > 2:
      override_seed_argv_idx = [
        idx for idx, arg in enumerate(sys.argv) 
        if '--new_seed' in arg
      ]
      if len(override_seed_argv_idx):
        seed = int(sys.argv[override_seed_argv_idx[0]+1])
        print(f"NEW RANDOM SEED: {seed}")

      override_reload_argv = [
        idx for idx, arg in enumerate(sys.argv) 
        if '--reload_path' in arg
      ]
      if len(override_reload_argv):
        task_config["reload"] = sys.argv[override_reload_argv[0]+1]
        print(f"NEW RELOAD PATH: {task_config['reload']}")

      path_suffix_argv = [
        idx for idx, arg in enumerate(sys.argv) 
        if '--path_suffix' in arg
      ]
      if len(path_suffix_argv):
        path_suffix = sys.argv[path_suffix_argv[0]+1]
        print(f"ADDITIONAL PATH SUFFIX: {path_suffix}")

      obs_budget_argv = [
        idx for idx, arg in enumerate(sys.argv) 
        if '--obs_budget' in arg
      ]
      if len(obs_budget_argv):
        train_observation_budget = int(sys.argv[obs_budget_argv[0]+1])
        print(f"TRAINING OBSERVATION BUDGET: {train_observation_budget}")

    if test_only:
      base_path = os.path.join(base_path,"TESTING")
    else:
      base_path = os.path.join(base_path,"TRAINING")
    
    base_path = os.path.join(base_path,f"SEED{seed}")
    
    if path_suffix is not None:
      base_path = os.path.join(base_path, path_suffix)

    print(f"Final Path: -- {base_path} --")

    if not os.path.exists(base_path): os.makedirs(base_path)
    
    task_config['final_path'] = base_path
    task_config['command_line'] = ' '.join(sys.argv)
    print(task_config['command_line'])
    yaml.dump(
      task_config, 
      open(
        os.path.join(base_path, "task_config.yaml"), 'w',
        encoding='utf8',
      ),
    )
    yaml.dump(
      agent_config, 
      open(
        os.path.join(base_path, "agent_config.yaml"), 'w',
        encoding='utf8',
      ),
    )
    
    #//////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if hasattr(torch.backends, "cudnn"):
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

    max_episode_length = None
    if task_config["max_fake_episode_length"]!='None':
        print("WARNING: Using M(F)SL.")
        max_episode_length = int(task_config["max_fake_episode_length"])
        #assert(task_config["curriculum_fake_reset"])
    
    sparse_positive_reward = task_config["sparse_positive_reward"]
    if sparse_positive_reward:
        print("WARNING: Using SparsePosReward.")
        assert(task_config["curriculum_fake_reset"]) 

    inverted_goal_predicated_reward = task_config["inverted_goal_predicated_reward"]
    if inverted_goal_predicated_reward:
        print("WARNING: Using InvertedGoalPredRewardFN.")
        assert(agent_config["use_HER"]) 
        agent_config['HER_goal_predicated_reward_fn'] = IGLU_inverted_goal_predicated_reward_fn
    else:
        print("WARNING: Using NegativeGoalPredRewardFN.")
        #agent_config['HER_goal_predicated_reward_fn'] = IGLU_goal_predicated_reward_fn
        agent_config['HER_goal_predicated_reward_fn'] = IGLU_maxint_goal_predicated_reward_fn
            
    pixel_wrapping_fn = partial(
        wrap_iglu,
        #size=task_config['observation_resize_dim'], 
        stack=task_config['framestacking'],
        clip_reward=task_config['clip_reward'],
        previous_reward_action=task_config["previous_reward_action"],
        curriculum_fake_reset=task_config["curriculum_fake_reset"],
        max_episode_length=max_episode_length,
        sparse_positive_reward=sparse_positive_reward,
        block_dense_actions=task_config["block_dense_actions"],
        use_HER=task_config.get('use_HER_reward', False),
        task_curriculum=task_config['task_curriculum'],
        inverted_goal_predicated_reward=inverted_goal_predicated_reward,
        use_THER=agent_config.get('use_THER', False),
        use_OHE=task_config.get('use_OHE', False),
    )


    test_pixel_wrapping_fn =partial(
        wrap_iglu,
        #size=task_config['observation_resize_dim'], 
        stack=task_config['framestacking'],
        clip_reward=False,
        previous_reward_action=task_config["previous_reward_action"],
        curriculum_fake_reset=False,
        max_episode_length=None,
        sparse_positive_reward=False,
        block_dense_actions=task_config["block_dense_actions"],
        use_HER=agent_config['use_HER'],
        task_curriculum=task_config['task_curriculum'],
        inverted_goal_predicated_reward=inverted_goal_predicated_reward,
        use_THER=agent_config.get('use_THER', False),
        use_OHE=task_config.get('use_OHE', False),
    )
    
    """
    pixel_wrapping_fn = partial(
      baseline_ther_wrapper,
      size=task_config['observation_resize_dim'], 
      skip=task_config['nbr_frame_skipping'], 
      stack=task_config['nbr_frame_stacking'],
      single_life_episode=task_config['single_life_episode'],
      nbr_max_random_steps=task_config['nbr_max_random_steps'],
      clip_reward=task_config['clip_reward'],
      max_sentence_length=agent_config['THER_max_sentence_length'],
      vocabulary=agent_config['THER_vocabulary'],
    )

    test_pixel_wrapping_fn = partial(
      baseline_ther_wrapper,
      size=task_config['observation_resize_dim'], 
      skip=task_config['nbr_frame_skipping'], 
      stack=task_config['nbr_frame_stacking'],
      single_life_episode=False,
      nbr_max_random_steps=task_config['nbr_max_random_steps'],
      clip_reward=False,
      max_sentence_length=agent_config['THER_max_sentence_length'],
      vocabulary=agent_config['THER_vocabulary'],
    )
    """

    video_recording_dirpath = os.path.join(base_path,'videos')
    video_recording_render_mode = 'human'
    task = generate_task(
      task_config['env-id'],
      env_type=EnvType.SINGLE_AGENT,
      nbr_parallel_env=task_config['nbr_actor'],
      wrapping_fn=pixel_wrapping_fn,
      test_wrapping_fn=test_pixel_wrapping_fn,
      env_config=task_config.get('env-config', {}),
      test_env_config=task_config.get('env-config', {}),
      seed=seed,
      test_seed=100+seed,
      gathering=True,
      train_video_recording_episode_period=benchmarking_record_episode_interval,
      train_video_recording_dirpath=video_recording_dirpath,
      train_video_recording_render_mode=video_recording_render_mode,
    )

    """
    task = generate_task(
        task_config['env-id'],
        nbr_parallel_env=task_config['nbr_actor'],
        wrapping_fn=pixel_wrapping_fn,
        test_wrapping_fn=test_pixel_wrapping_fn,
        seed=seed,
        test_seed=100+seed,
        train_video_recording_episode_period=video_recording_episode_period,
        train_video_recording_dirpath=os.path.join(base_path, 'recordings/train/'),
        #test_video_recording_episode_period=video_recording_episode_period,
        #test_video_recording_dirpath=os.path.join(base_path, 'recordings/test/'),
        gathering=True,
    )
    """
    
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////

    agent_config['nbr_actor'] = task_config['nbr_actor']
    
    if agent_config.get('use_THER', False):
        global global_vocabulary
        agent_config['vocabulary_instance'] = global_vocabulary

    regym.RegymSummaryWriterPath = base_path 
    #regym.RegymSummaryWriter = GlobalSummaryWriter(base_path)
    sum_writer = base_path
    
    save_path1 = os.path.join(base_path,f"./{task_config['agent-id']}.agent")
    if task_config.get("reload", 'None')!='None':
      agent, offset_episode_count = check_path_for_agent(task_config["reload"])
    else:
      agent, offset_episode_count = check_path_for_agent(save_path1)
    
    if agent is None: 
        if agent_config['use_HER']:
            print("WARNING: Using HER.")
            agent_config['HER_achieved_goal_key_from_info'] = "grid"
            agent_config['HER_target_goal_key_from_info'] = "target_grid"
            if task_config.get('use_OHE', False):
                agent_config['HER_achieved_latent_goal_key_from_info'] = "grid_ohe"
                agent_config['HER_target_latent_goal_key_from_info'] = "target_grid_ohe"
            agent_config['HER_filtering_fn'] = IGLU_HER_filtering_fn
        agent = initialize_agents(
          task=task,
          agent_configurations={task_config['agent-id']: agent_config}
        )[0]
        # Reload model only:
        if task_config['reload_model']!='None':
            print("RELOADING MODEL ONLY.")
            prev_agent, offset_episode_count = check_path_for_agent(task_config['reload_model'])
            agent.algorithm.set_models(prev_agent.algorithm.get_models())
            agent.algorithm.set_optimizer(prev_agent.algorithm.get_optimizer())
            print(f"MODEL ONLY RELOADED : from : {task_config['reload_model']}")
    
    agent.save_path = save_path1
    
    if test_only:
      print(save_path1)
      agent.training = False
    
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////
    
    config = {
        'task':task_config, 
        'agent': agent_config,
        'seed': seed,
    }
    project_name = task_config['project']
    wandb.init(project=project_name, config=config, settings=wandb.Settings(start_method='fork'))
    #wandb.watch(agents[-1].algorithm.model, log='all', log_freq=100, idx=None, log_graph=True)
    
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////

    trained_agent = train_and_evaluate(
        agent_config=agent_config,
        task_config=task_config,
        agent=agent,
        task=task,
        sum_writer=sum_writer,
        base_path=base_path,
        offset_episode_count=offset_episode_count,
        nbr_pretraining_steps=int(float(agent_config["nbr_pretraining_steps"])) if "nbr_pretraining_steps" in agent_config else 0,
        nbr_max_observations=train_observation_budget,
        test_obs_interval=benchmarking_interval,
        test_nbr_episode=benchmarking_episodes,
        benchmarking_record_episode_interval=benchmarking_record_episode_interval,
        render_mode=video_recording_render_mode
    )

    return trained_agent, task 


def load_configs(config_file_path: str):
    all_configs = yaml.safe_load(open(config_file_path))

    agents_config = all_configs['agents']
    experiment_config = all_configs['experiment']
    envs_config = experiment_config['tasks']

    return experiment_config, agents_config, envs_config


def main():
    #logging.basicConfig(level=logging.INFO)
    logging.getLogger('minerl_patched').setLevel(level=logging.ERROR)
    logger = logging.getLogger('IGLU R2D2 Benchmark')
    logger.setLevel(level=logging.CRITICAL)

    parser = argparse.ArgumentParser(description="IGLU- R2D2 - Test.")
    parser.add_argument("--config", 
        type=str, 
        default="./iglu_r2d2_config.yaml",
    )
    
    parser.add_argument("--seed", 
        type=int, 
        default=10,
    )
 
    parser.add_argument("--project", 
        type=str, 
        default="IGLU",
    )

    parser.add_argument("--path_suffix", 
        type=str, 
        default="",
    )
    #parser.add_argument("--simplified_DNC", 
    #    type=str, 
    #    default="False",
    #)
    parser.add_argument("--learning_rate", 
        type=float, 
        help="learning rate",
        default=3e-4,
    )
    parser.add_argument("--weights_decay_lambda", 
        type=float, 
        default=0.0,
    )
    parser.add_argument("--weights_entropy_lambda", 
        type=float, 
        default=0.0, #0.001,
    )
    #parser.add_argument("--DNC_sparse_K", 
    #    type=int, 
    #    default=0,
    #)
    parser.add_argument("--sequence_replay_unroll_length", 
        type=int, 
        default=5,
    )
    parser.add_argument("--sequence_replay_overlap_length", 
        type=int, 
        default=0,
    )
    parser.add_argument("--sequence_replay_burn_in_ratio", 
        type=float, 
        default=0.0,
    )
    parser.add_argument("--listener_rec_period", 
        type=int, 
        default=10,
    )
    parser.add_argument("--n_step", 
        type=int, 
        default=3,
    )
    parser.add_argument("--tau", 
        type=float, 
        default=4e-4,
    )
    parser.add_argument("--nbr_actor", 
        type=int, 
        default=1,
    )
    parser.add_argument("--batch_size", 
        type=int, 
        default=128,
    )
    #parser.add_argument("--critic_arch_feature_dim", 
    #    type=int, 
    #    default=32,
    #)
    parser.add_argument("--train_observation_budget", 
        type=float, 
        default=1e4,
    )
    parser.add_argument("--nbr_training_iteration_per_cycle", 
        type=int, 
        default=40,
    )
    parser.add_argument("--nbr_episode_per_cycle", 
        type=int, 
        default=16,
    )
    parser.add_argument("--observation_resize_dim", 
        type=int, 
        default=32,
    )
 

    args = parser.parse_args()
    
    args.sequence_replay_overlap_length = min(
        args.sequence_replay_overlap_length,
        args.sequence_replay_unroll_length-5,
    )

    #args.simplified_DNC = True if "Tr" in args.simplified_DNC else False
    dargs = vars(args)
    
    if args.sequence_replay_burn_in_ratio != 0.0:
        dargs['sequence_replay_burn_in_length'] = int(args.sequence_replay_burn_in_ratio*args.sequence_replay_unroll_length)
        dargs['burn_in'] = True 
    
    dargs['seed'] = int(dargs['seed'])
    
    print(dargs)

    from gpuutils import GpuUtils
    GpuUtils.allocate(required_memory=6000, framework="torch")
    
    config_file_path = args.config #sys.argv[1] #'./atari_10M_benchmark_config.yaml'
    experiment_config, agents_config, tasks_configs = load_configs(config_file_path)
    
    for k,v in dargs.items():
        experiment_config[k] = v
    
    print("Experiment config:")
    print(experiment_config)

    # Generate path for experiment
    base_path = experiment_config['experiment_id']
    if not os.path.exists(base_path): os.makedirs(base_path)

    for task_config in tasks_configs:
        agent_name = task_config['agent-id']
        env_name = task_config['env-id']
        run_name = task_config['run-id']
        path = f'{base_path}/{env_name}/{run_name}/{agent_name}'
        print(f"Tentative Path: -- {path} --")
        agent_config =agents_config[task_config['agent-id']] 
        for k,v in dargs.items():
            task_config[k] = v
            agent_config[k] = v
        
        print("Task config:")
        print(task_config)

        training_process(
            agent_config, 
            task_config,
            benchmarking_interval=int(
                float(
                    experiment_config['benchmarking_interval']
                )
            ),
            benchmarking_episodes=int(
                float(
                    experiment_config['benchmarking_episodes']
                )
            ),
            benchmarking_record_episode_interval=int(
                float(
                    experiment_config['benchmarking_record_episode_interval']
                )
            ) if experiment_config['benchmarking_record_episode_interval']!='None' else None,
            train_observation_budget=int(
                float(
                    experiment_config['train_observation_budget']
                )
            ),
            base_path=path,
            seed=experiment_config['seed'],
        )

def s2b_r2d2_wrap(
    env, 
    clip_reward=False,
    previous_reward_action=True,
    otherplay=False
    ):
    env = s2b_wrap(
      env, 
      combined_actions=False,
      dict_obs_space=False,
    )

    if clip_reward:
        env = ClipRewardEnv(env)

    if previous_reward_action:
        env = PreviousRewardActionInfoMultiAgentWrapper(env=env)
    
    return env

if __name__ == '__main__':
    asynch = False 
    __spec__ = None
    if len(sys.argv) > 2:
        asynch = any(['async' in arg for arg in sys.argv])
    if asynch:
        torch.multiprocessing.freeze_support()
        torch.multiprocessing.set_start_method("forkserver", force=True)
        #torch.multiprocessing.set_start_method("spawn", force=True)
        ray.init() #local_mode=True)
      
        from regym import CustomManager as Manager
        from multiprocessing.managers import SyncManager, MakeProxyType, public_methods

        regym.RegymManager = Manager()
        regym.RegymManager.start()

    main()
