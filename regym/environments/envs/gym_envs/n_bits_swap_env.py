import gym
from gym import spaces
from gym.spaces import Discrete, MultiBinary, Dict
from gym.utils import seeding
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Adapted from:
# https://github.com/JunkyByte/HER_DQN/blob/master/src/Custom_Env/BitSwap.py
class NBitsSwapEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, n=10, fixed_goal=False, max_steps=None):
        super(NBitsSwapEnv, self).__init__()
        self.n = n 
        self.fixed_goal = fixed_goal
        self.max_episode_steps = max_steps if max_steps is not None else n

        self.action_space = Discrete(self.n)
        self.observation_space = Dict({
            "observation": MultiBinary(self.n), 
            #"achieved_goal": MultiBinary(self.n), 
            "achieved_goal": spaces.Space[str](), 
            #"desired_goal": MultiBinary(self.n),
            "desired_goal": spaces.Space[str](),
        })


        self.nbr_steps = 0
        self.state = None
        self.goal = np.random.randint(2, size=self.n)
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed 

    def _calc_reward(self):
        return int(all(self.state == self.goal))

    def _get_obs(self):
        ret = {}
        ret["observation"] = self.state.copy()
        #ret["achieved_goal"] = self.goal.copy()
        ret["achieved_goal"] = ' '.join(['one' if s==1 else 'zero' for s in self.state])
        #ret["desired_goal"] = self.goal.copy()
        ret["desired_goal"] = ' '.join(['one' if s==1 else 'zero' for s in self.goal])
        return ret 

    def reset(self, **kwargs):
        self.nbr_steps = 0
        self.state = self.np_random.randint(2, size=self.n)
        if not self.fixed_goal:
            self.goal = self.np_random.randint(2, size=self.n)
            while all(self.state==self.goal):
                self.goal = self.np_random.randint(2, size=self.n)

        info = {'latents':
                    {   's': [self.state.copy()], 
                        'a': [0*np.ones(1)],
                        'step_idx': [self.nbr_steps*np.ones(1)],
                        'succ_s': [self.state.copy()],
                        'achieved_goal': [self.state.copy()],
                        'desired_goal': [self.goal.copy()],
                    }
                }

        return self._get_obs(), info

    def step(self, action):
        assert(action < self.n)
        init_state = self.state.copy()
        
        self.state[action] = 1-self.state[action]
        obs = self._get_obs()
        reward = 1 if self._calc_reward() else 0
        self.nbr_steps += 1
        terminal = True if reward >= 0.5 or self.nbr_steps >= self.max_episode_steps else False
        
        info = {'latents':
                    {   's': [init_state.copy()],
                        'a': [action*np.ones(1)],
                        'step_idx': [self.nbr_steps*np.ones(1)],
                        'succ_s': [self.state.copy()],
                        'achieved_goal': [self.state.copy()],
                        'desired_goal': [self.goal.copy()],
                    }
                }

        return obs, reward, terminal, info 

    def render(self, mode='human', close=False):
        logger.info(f"State: {self.state} \n Goal: {self.goal}")


