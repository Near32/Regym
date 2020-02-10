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
    def __init__(self, n=10, fixed_goal=False, train=True):
        super(NBitsSwapMNISTEnv, self).__init__(n=n, fixed_goal=fixed_goal)
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

        self.zeros_indices = [idx for idx, target in enumerate(self.mnist.targets) if target==0][:1]
        self.ones_indices = [idx for idx, target in enumerate(self.mnist.targets) if target==1][:1]
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

        self.state[action] = not self.state[action]
        self.obs_as_indices[action] = self.np_random.choice(self.zeros_indices if self.state[action]==0 else self.ones_indices)

        obs = self._get_obs()
        reward = self._calc_reward()
        self.nbr_steps += 1
        terminal = True if reward == 1 or self.nbr_steps >= self.max_episode_steps else False
        return obs, reward, terminal, {} 

    def render(self, mode='human', close=False):
        logger.info(f"State: {self.state} \n Goal: {self.goal}")
        

