from typing import Dict, Optional, List
import random

import gym
import numpy as np

from regym.environments import EnvType
from .agent import Agent


class RandomAgent(Agent):
    def __init__(self, name: str, action_space: gym.spaces.Space, action_space_dim: int):
        self.name = name
        self.action_space = action_space
        self.action_space_dim = action_space_dim
        self.recurrent = False

    def set_nbr_actor(self, n):
        pass

    def reset_actors(self, indices:Optional[List]=[], init:Optional[bool]=False):
        pass

    def preprocess_environment_signals(self, state, reward, succ_state, done):
        pass

    def take_action(self, state, infos: List[Dict]):
        legal_actions = [info['legal_actions'] for info in infos]
        if legal_actions:
            # Hope that legal actions is defined as a list of lists!
            actions = [
                random.choice(
                    np.argwhere(legal_actions[i].squeeze() == 1)
                )
               for i in range(len(legal_actions))
            ]
        else:
            actions = [self.action_space.sample()
                       for _ in range(len(observations))]
        return actions

    def handle_experience(self, s, a, r, succ_s, done, goals=None, infos=None):
        pass

    def get_async_actor(self, training=None, with_replay_buffer=False):
        pass

    def clone(self):
        return RandomAgent(name=self.name, action_space=self.action_space, action_space_dim=self.action_space_dim)

    def __repr__(self):
        return f'{self.name}. Action space: {self.action_space}'


def build_Random_Agent(task, config, agent_name: str) -> RandomAgent:
    '''
    Builds an agent that is able to randomly act in a task

    :param task: Task in which the agent will be able to act
    :param config: Ignored, left here to keep `build_X_Agent` interface consistent
    :param name: String identifier
    '''
    # TODO: 
    if task.env_type == EnvType.SINGLE_AGENT: action_space = task.env.action_space
    # Assumes all agents share same action space
    else: action_space = task.env.action_space
    return RandomAgent(name=agent_name, action_space=action_space, action_space_dim=task.action_dim)
