from typing import Dict, List, Union, Any, Optional, Callable

from functools import partial

import torch
import copy

from regym.rl_algorithms.utils import is_leaf

class Algorithm(object):
    def __init__(self, name="algo"):
        self.name = name
        self.unwrapped = self

    def get_models(self):
        raise NotImplementedError

    def get_epsilon(self, nbr_steps, strategy='exponential'):
        raise NotImplementedError

    def get_nbr_actor(self):
        raise NotImplementedError

    def get_update_count(self):
        raise NotImplementedError
    
    def get_obs_count(self):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError
        
    def clone(self, with_replay_buffer=False):
        raise NotImplementedError
