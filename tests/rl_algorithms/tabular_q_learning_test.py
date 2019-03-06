import os
import sys
sys.path.append(os.path.abspath('../../'))

from rl_algorithms import rockAgent
from rl_algorithms.agents import build_TabularQ_Agent

from test_fixtures import tabular_q_learning_config_dict, RPSenv, RPSTask


def test_creation_tabular_q_learning_algorithm_from_task_and_config(RPSTask, tabular_q_learning_config_dict):
    expected_training = True
    agent = build_TabularQ_Agent(RPSTask, tabular_q_learning_config_dict)
    assert agent.algorithm.Q_table.shape == (RPSTask.state_space_size, RPSTask.action_space_size)
    assert agent.algorithm.learning_rate == tabular_q_learning_config_dict['learning_rate']
    assert agent.algorithm.epsilon_greedy == tabular_q_learning_config_dict['epsilon_greedy']
    assert agent.algorithm.discount_factor == tabular_q_learning_config_dict['discount_factor']
    assert agent.algorithm.hashing_function == RPSTask.hash_function
    assert agent.training == expected_training

def test_creation_repeated_update_q_learning_algorithm_from_task_and_config(RPSTask, tabular_q_learning_config_dict):
    tabular_q_learning_config_dict['use_repeated_update_q_learning'] = True
    expected_training = True
    agent = build_TabularQ_Agent(RPSTask, tabular_q_learning_config_dict)
    assert agent.algorithm.Q_table.shape == (RPSTask.state_space_size, RPSTask.action_space_size)
    assert agent.algorithm.learning_rate == tabular_q_learning_config_dict['learning_rate']
    assert agent.algorithm.temperature == tabular_q_learning_config_dict['temperature']
    assert agent.algorithm.discount_factor == tabular_q_learning_config_dict['discount_factor']
    assert agent.algorithm.hashing_function == RPSTask.hash_function
    assert agent.training == expected_training


def test_tabular_q_learning_can_take_actions(RPSenv, RPSTask, tabular_q_learning_config_dict):
    agent = build_TabularQ_Agent(RPSTask, tabular_q_learning_config_dict)
    number_of_actions = 30
    for i in range(number_of_actions):
        # asumming that first observation corresponds to observation space of this agent
        random_observation = RPSenv.observation_space.sample()[0]
        a = agent.take_action(random_observation)
        assert RPSenv.action_space.contains([a, a])


def test_learns_to_beat_rock_in_RPS(RPSenv, RPSTask, tabular_q_learning_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    # tabular_q_learning_config_dict['use_repeated_update_q_learning'] = True
    from rps_test import learns_against_fixed_opponent_RPS

    agent = build_TabularQ_Agent(RPSTask, tabular_q_learning_config_dict)
    assert agent.training
    learns_against_fixed_opponent_RPS(agent, fixed_opponent=rockAgent,
                                      total_episodes=100000, training_percentage=0.97,
                                      reward_threshold=0.1)
