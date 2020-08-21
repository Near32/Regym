from functools import partial

import pytest
from tensorboardX import SummaryWriter
import numpy as np
import torch

import regym
from regym.environments import generate_task
from regym.rl_algorithms.agents import build_R2D3_Agent, build_DQN_Agent
from regym.rl_loops.singleagent_loops import rl_loop
from regym.rl_algorithms.replay_buffers import PrioritizedReplayBuffer
from regym.rl_algorithms.replay_buffers import EXP
from regym.rl_algorithms.networks.utils import ResizeCNNInterpolationFunction


'''
This file assumes that gym_miniworld has been installed:
https://github.com/maximecb/gym-miniworld#installation
'''

@pytest.fixture
def HallwayTask():
    import gym_miniworld
    return generate_task('MiniWorld-Hallway-v0')


@pytest.fixture
def r2d3_config_dict():
    config = dict()
    # Miscellaneous
    config['batch_size'] = 256
    config['discount'] = 0.99
    config['tau'] = 1.0e-2   # soft-update for updating (DQN) target network.
    # R2D3 specific
    config['expert_demonstrations'] = None  # NOTE: Demonstrations are set below
    config['sequence_length'] = 8  # TODO: implement
    config['demo_ratio'] = 1/256  # TODO: implement
    # Multiagent hyperparams
    config['lr_account_for_nbr_actor'] = False
    config['nbr_actor'] = 12  # NOTE: you can use multiprocessing.cpu_count() here if you want to max out on processes.
    config['epsstart'] = 1
    config['epsend'] = 0.05
    config['epsdecay'] = 5.0e4
    config['use_cuda'] = False
    # DQN variants activation flags
    config['double'] = True
    config['dueling'] = True
    config['use_PER'] = False
    config['PER_alpha'] = 0.6
    config['PER_beta'] = 0.4
    # Replay Buffer hyperparams
    config['min_capacity'] = 5.e02
    config['replay_capacity'] = 5.e4
    config['nbrTrainIteration'] = 8
    # NN hyperparams
    config['learning_rate'] = 1.0e-3
    config['adam_eps'] = 1.0e-8
    config['gradient_clip'] = 1e03
    config['weights_decay_lambda'] = 0.
    # NN architecture
    config['noisy'] = False
    config['phi_arch'] = 'CNN'
    config['observation_resize_dim'] = [60, 80]
    config['phi_arch_feature_dim'] = 64
    config['phi_arch_channels'] = [10, 10]
    config['phi_arch_kernels'] = [3, 3]
    config['phi_arch_paddings'] = [1, 1]
    config['phi_arch_strides'] = [1, 1]
    config['critic_arch'] = 'MLP'
    config['critic_arch_hidden_units'] = [64, 32]
    # Non-relevant (but necessary to define)
    config['use_HER'] = False
    config['HER_target_clamping'] = False
    config['goal_oriented'] = False
    return config


@pytest.fixture
def HallwayExpertDemonstrations(HallwayTask):
    # Load hand-coded action sequences of expert demonstrations
    hallway_seed_0 = [2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2]
    hallway_seed_1 = [0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,2,2,2,2,2,2,2,1,2,2,2,2]
    hallway_seed_2 = [2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    hallway_seed_3 = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    hallway_seed_4 = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    hallway_seed_5 = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    hallway_seed_6 = [0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2]
    hallway_seed_7 = [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,2,2,2,2,2,2,2,2,2,2,2,2,2]
    hallway_seed_8 = [0,0,2,2,2,2,2,2,2,2,2,2,2,2,2]
    hallway_seed_9 = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    all_demonstrations = [hallway_seed_0,hallway_seed_1,hallway_seed_2,
                          hallway_seed_3,hallway_seed_4,hallway_seed_5,
                          hallway_seed_6,hallway_seed_7,hallway_seed_8,
                          hallway_seed_9]
    return [HallwayTask.run_episode_from_action_sequence(demonstration, seed)
            for seed, demonstration in enumerate(all_demonstrations)]

@pytest.fixture
def HallwayExpertDemonstrationReplayBuffer(HallwayExpertDemonstrations):
    '''
    Generates a replay buffer from expert demonstrations
    '''
    preprocessing_fn = partial(ResizeCNNInterpolationFunction, size=60)
    demonstrations_buffer = PrioritizedReplayBuffer(capacity=sum(map(lambda t: len(t), trajectories))) # keys=['s', 'a', 'r', 'non_terminal'])
    # Neat double loop to add all demonstrations to buffer.
    [demonstrations_buffer.add(EXP(preprocessing_fn(torch.tensor(s).unsqueeze(0)), a,
                                   preprocessing_fn(torch.tensor(succ_s).unsqueeze(0)),
                                   torch.FloatTensor([[r]]),
                                   torch.from_numpy(1 - np.array([done])).reshape(-1,1).type(torch.FloatTensor)),
                               priority=1.0)  # NOTE: hardcoding initial priority here
    for t in trajectories for (s, a, r, succ_s, done) in t]
    return demonstrations_buffer


def test_dqn_agent_can_act_in_miniworld(HallwayTask, r2d3_config_dict):
    ''' Required, as R2D3 is based on DQN '''
    r2d3_config_dict['goal_resize_dim'] = None
    agent = build_DQN_Agent(HallwayTask, r2d3_config_dict, 'Test')
    HallwayTask.run_episode(agent, training=False)


def test_r2d3_agent_can_act_in_miniworld(HallwayTask, r2d3_config_dict):
    ''' If this test doesn't break, it means that a non-training R2D3 agent
        can act in the environment '''
    r2d3_config_dict['goal_resize_dim'] = None
    agent = build_R2D3_Agent(HallwayTask, r2d3_config_dict, 'Test')
    HallwayTask.run_episode(agent, training=False)


def test_dqn_can_solve_hallway_v0_miniworld(HallwayTask, r2d3_config_dict):
    base_path = './DQN_Hallway'
    sum_writer = SummaryWriter(base_path)
    regym.rl_algorithms.algorithms.DQN.dqn.summary_writer = sum_writer

    agent = build_DQN_Agent(HallwayTask, r2d3_config_dict, 'Test')
    HallwayTask.run_episode(agent, training=False)

    trained_agent = rl_loop.gather_experience_parallel(HallwayTask,
                                                       agent,
                                                       training=True,
                                                       max_obs_count=5e6,
                                                       env_configs=None,
                                                       sum_writer=sum_writer,
                                                       base_path=base_path,
                                                       test_obs_interval=200,
                                                       test_nbr_episode=10,
                                                       benchmarking_record_episode_interval=50000,
                                                       step_hooks=[])


def test_r2d3_can_solve_hallway_v0_miniworld_using_expert_demonstrations(HallwayTask, HallwayExpertDemonstrations, r2d3_config_dict):
    base_path = './R2D3_Hallway'
    sum_writer = SummaryWriter(base_path)
    regym.rl_algorithms.algorithms.R2D3.r2d3.summary_writer = sum_writer

    r2d3_config_dict['goal_resize_dim'] = None
    r2d3_config_dict['expert_demonstrations'] = HallwayExpertDemonstrations

    agent = build_R2D3_Agent(HallwayTask, r2d3_config_dict, agent_name='R2D3-Hallway')
    trained_agent = rl_loop.gather_experience_parallel(HallwayTask,
                                                       agent, training=True,
                                                       max_obs_count=100,  # Total number of observations to be processed during training
                                                       benchmarking_record_episode_interval=50000,  # Record an episode from the agent's perspective every X episodes
                                                       test_obs_interval=200,
                                                       test_nbr_episode=10,
                                                       sum_writer=sum_writer,
                                                       base_path=base_path,
                                                       env_configs=None,
                                                       step_hooks=[])
