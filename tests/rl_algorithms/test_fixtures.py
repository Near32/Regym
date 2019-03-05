import os
import sys
sys.path.append(os.path.abspath('../../'))

import pytest
from environments.gym_parser import parse_gym_environment


@pytest.fixture
def ppo_config_dict():
    config = dict()
    config['discount'] = 0.99
    config['use_gae'] = False
    config['use_cuda'] = False
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 10
    config['mini_batch_size'] = 256
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5
    config['horizon'] = 1024
    return config


@pytest.fixture
def dqn_config_dict():
    config = dict()
    config['learning_rate'] = 1.0e-3
    config['epsstart'] = 0.8
    config['epsend'] = 0.05
    config['epsdecay'] = 1.0e3
    config['double'] = False
    config['dueling'] = False
    config['use_cuda'] = True
    config['use_PER'] = True
    config['PER_alpha'] = 0.07
    config['min_memory'] = 5.0e1
    config['memoryCapacity'] = 25.0e3
    config['nbrTrainIteration'] = 8
    config['batch_size'] = 128
    config['gamma'] = 0.99
    config['tau'] = 1.0e-3
    return config


@pytest.fixture
def tabular_q_learning_config_dict():
    config = dict()
    config['learning_rate'] = 0.9
    return config


@pytest.fixture
def RPSenv():
    import gym
    import gym_rock_paper_scissors
    return gym.make('RockPaperScissors-v0')


@pytest.fixture
def RPSTask(RPSenv):
    return parse_gym_environment(RPSenv)
