import gym
from gym.spaces import Discrete, Box, Dict
from .n_bits_swap_env import NBitsSwapEnv

import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms 

import os 
import numpy as np

import logging
logger = logging.getLogger(__name__)

class NBitsSwapMNISTEnv(NBitsSwapEnv):
    def __init__(self, n=10, simple=True, fixed_goal=False, train=True):
        '''
        :param n: Integer representing the number of digit to use in the game.
        :param simple: Boolean stating whether to use the simplified game where 
                        ones and zeros can only be represented through one and 
                        only one image for each.
        :param fixed_goal: Boolean stating whether to not sample a new goal at
                            each call of the reset method.
        :param train: Boolean stating from which split of MNIST to take the images. 
        '''

        super(NBitsSwapMNISTEnv, self).__init__(n=n, fixed_goal=fixed_goal)
        self.simple = simple
        self.train = train
        self.obs_shape = [32,32]
        self.transform = transforms.Compose([
            transforms.Resize(size=self.obs_shape),
            ])
        dir_path = os.path.dirname(os.path.realpath(__file__))
        root = os.path.join(dir_path,'mnist')
        self.mnist =  MNIST(root=root, 
                            train=self.train,
                            transform=self.transform,
                            download=True)

        self.observation_space = Dict({"observation": Box(low=0, high=255, shape=(*self.obs_shape, self.n), dtype=np.float32), 
                                       "achieved_goal": Box(low=0, high=255, shape=(*self.obs_shape, self.n), dtype=np.float32), 
                                       "desired_goal": Box(low=0, high=255, shape=(*self.obs_shape, self.n), dtype=np.float32)})

        self.zeros_indices = [idx for idx, target in enumerate(self.mnist.targets) if target==0]
        self.ones_indices = [idx for idx, target in enumerate(self.mnist.targets) if target==1]
        
        if self.simple:
            self.zeros_indices = self.zeros_indices[:1]
            self.ones_indices = self.ones_indices[:1]

        self.obs_as_indices = None 
        self.goal_as_indices = None 

    def _calc_reward(self):
        return int(all(self.state == self.goal))

    def _indices2mnist(self, indices):
        mnist = []
        for idx in indices:
            img, target = self.mnist[idx]
            mnist.append(255*np.array(img).astype(np.float32).reshape((*self.obs_shape, 1)))
        mnist = np.concatenate(mnist, axis=-1)
        return mnist 

    def _get_obs(self):
        ret = {}
        ret["observation"] = self._indices2mnist(self.obs_as_indices)
        ret["achieved_goal"] = ret["observation"].copy()
        ret["desired_goal"] = self._indices2mnist(self.goal_as_indices)
        return ret 

    def reset(self):
        self.nbr_steps = 0
        self.state = self.np_random.randint(2, size=self.n)
        self.obs_as_indices = [self.np_random.choice(self.zeros_indices if s==0 else self.ones_indices) for s in self.state]
        if not self.fixed_goal:
            self.goal = self.np_random.randint(2, size=self.n)
        self.goal_as_indices = [self.np_random.choice(self.zeros_indices if s==0 else self.ones_indices) for s in self.goal]
        return self._get_obs()

    def step(self, action):
        assert(action < self.n)
        if isinstance(action, np.ndarray): action=action[0]

        init_state = self.state.copy()
        self.state[action] = not self.state[action]
        self.obs_as_indices[action] = self.np_random.choice(self.zeros_indices if self.state[action]==0 else self.ones_indices)

        obs = self._get_obs()
        reward = 0 if self._calc_reward() else -1
        self.nbr_steps += 1
        terminal = True if reward >= -0.5 or self.nbr_steps >= self.max_episode_steps else False

        info = {'latents':
                    {   's': init_state.copy(), 
                        'succ_s': self.state.copy(),
                        'achieved_goal': self.state.copy(),
                        'desired_goal': self.goal.copy()
                    }
                }
        return obs, reward, terminal, info

    def render(self, mode='human', close=False):
        logger.info(f"State: {self.state} \n Goal: {self.goal}")
        

