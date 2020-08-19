import pytest

import regym
from tensorboardX import SummaryWriter
from regym.environments import generate_task
from regym.rl_algorithms.agents import build_R2D3_Agent, build_DQN_Agent
from regym.rl_loops.singleagent_loops import rl_loop

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
    config['learning_rate'] = 1.0e-3
    config['nbr_actor'] = 12
    config['epsstart'] = 1
    config['epsend'] = 0.05
    config['epsdecay'] = 5.0e4
    config['double'] = True
    config['dueling'] = True
    config['use_cuda'] = False
    config['use_PER'] = False
    config['use_HER'] = False
    config['HER_target_clamping'] = False
    config['goal_oriented'] = False
    config['PER_alpha'] = 0.6
    config['PER_beta'] = 0.4
    config['discount'] = 0.99
    config['min_capacity'] = 5.e02
    config['replay_capacity'] = 5.e4
    config['nbrTrainIteration'] = 8
    config['batch_size'] = 256
    config['gamma'] = 0.99
    config['tau'] = 1.0e-2
    config['adam_eps'] = 1.0e-8
    config['weights_decay_lambda'] = 0.
    config['lr_account_for_nbr_actor'] = False
    config['gradient_clip'] = 1e03
    # NN architecture
    config['noisy'] = False
    config['phi_arch'] = 'CNN'
    config['phi_arch_feature_dim'] = 64
    config['phi_arch_channels'] = [10, 10]
    config['phi_arch_kernels'] = [3, 3]
    config['phi_arch_paddings'] = [1, 1]
    config['phi_arch_strides'] = [1, 1]
    config['observation_resize_dim'] = [60, 80]
    config['critic_arch'] = 'MLP'
    config['critic_arch_hidden_units'] = [64, 32]
    # R2D3 specific
    config['expert_demonstrations'] = None
    config['sequence_length'] = 8
    config['demo_ratio'] = 1/256
    return config


def test_dqn_agent_can_act_in_miniworld(HallwayTask, r2d3_config_dict):
    ''' Required, as R2D3 is based on DQN '''
    r2d3_config_dict['goal_resize_dim'] = None
    agent = build_DQN_Agent(HallwayTask, r2d3_config_dict, 'Test')
    HallwayTask.run_episode(agent, training=False)


def test_r2d3_agent_can_act_in_miniworld(HallwayTask, r2d3_config_dict):
    ''' If this test doesn't break, it means that a non-training R2D3 agent
        can act in the environment '''
    agent = build_R2D3_Agent(HallwayTask, r2d3_config_dict, 'Test')
    HallwayTask.run_episode(agent, training=False)


def test_dqn_can_solve_hallway_v0_miniworld(HallwayTask, r2d3_config_dict):
    base_path = './DQN_Hallway'
    sum_writer = SummaryWriter(base_path)
    regym.rl_algorithms.algorithms.DQN.dqn.summary_writer = sum_writer

    r2d3_config_dict['goal_resize_dim'] = None
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
