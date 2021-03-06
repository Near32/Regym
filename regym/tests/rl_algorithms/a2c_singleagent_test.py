from test_fixtures import a2c_mlp_config_dict, a2c_mlp_rnn_config_dict 
from test_fixtures import a2c_cnn_config_dict, a2c_cnn_rnn_config_dict 
from test_fixtures import BreakoutTask, CartPoleTask
from test_fixtures import BreakoutTask_ma, CartPoleTask_ma

from regym.rl_algorithms.agents import build_A2C_Agent
from utils import can_act_in_environment, learns_to_solve_task

nbr_episode_basis = 100

def test_a2c_can_take_actions_discrete_observation_box(BreakoutTask, a2c_cnn_config_dict):
    can_act_in_environment(BreakoutTask, build_A2C_Agent, a2c_cnn_config_dict, name=__name__)

def test_a2c_can_take_actions_continuous_obvservation_discrete(CartPoleTask, a2c_mlp_config_dict):
    can_act_in_environment(CartPoleTask, build_A2C_Agent, a2c_mlp_config_dict, name=__name__)


def test_a2c_learns_to_solve_cartpole(CartPoleTask, a2c_mlp_config_dict):
    nbr_episode = nbr_episode_basis
    learns_to_solve_task(nbr_episode, CartPoleTask, build_A2C_Agent, a2c_mlp_config_dict, name=__name__)

def test_a2c_learns_to_solve_cartpole_ma(CartPoleTask_ma, a2c_mlp_config_dict):
    a2c_mlp_config_dict['nbr_actor'] = CartPoleTask_ma.env.get_nbr_envs()
    nbr_episode = nbr_episode_basis // CartPoleTask_ma.env.get_nbr_envs()
    learns_to_solve_task(nbr_episode, CartPoleTask_ma, build_A2C_Agent, a2c_mlp_config_dict, name=__name__)


def test_learns_to_beat_Breakout(BreakoutTask, a2c_cnn_config_dict):
    nbr_episode = nbr_episode_basis
    learns_to_solve_task(nbr_episode, BreakoutTask, build_A2C_Agent, a2c_cnn_config_dict, name=__name__)

def test_learns_to_beat_Breakout_ma(BreakoutTask_ma, a2c_cnn_config_dict):
    a2c_cnn_config_dict['nbr_actor'] = BreakoutTask_ma.env.get_nbr_envs()
    nbr_episode = nbr_episode_basis // BreakoutTask_ma.env.get_nbr_envs()
    learns_to_solve_task(nbr_episode, BreakoutTask_ma, build_A2C_Agent, a2c_cnn_config_dict, name=__name__)


def test_a2c_rnn_can_take_actions_discrete_observation_box(BreakoutTask, a2c_cnn_rnn_config_dict):
    can_act_in_environment(BreakoutTask, build_A2C_Agent, a2c_cnn_rnn_config_dict, name=__name__)

def test_a2c_rnn_can_take_actions_continuous_obvservation_discrete(CartPoleTask, a2c_mlp_rnn_config_dict):
    can_act_in_environment(CartPoleTask, build_A2C_Agent, a2c_mlp_rnn_config_dict, name=__name__)


def test_a2c_rnn_learns_to_solve_cartpole(CartPoleTask, a2c_mlp_rnn_config_dict):
    nbr_episode = nbr_episode_basis
    learns_to_solve_task(nbr_episode, CartPoleTask, build_A2C_Agent, a2c_mlp_rnn_config_dict, name=__name__)

def test_a2c_rnn_learns_to_solve_cartpole_ma(CartPoleTask_ma, a2c_mlp_rnn_config_dict):
    a2c_mlp_rnn_config_dict['nbr_actor'] = CartPoleTask_ma.env.get_nbr_envs()
    nbr_episode = nbr_episode_basis // CartPoleTask_ma.env.get_nbr_envs()
    learns_to_solve_task(nbr_episode, CartPoleTask_ma, build_A2C_Agent, a2c_mlp_rnn_config_dict, name=__name__)


def test_learns_to_beat_Breakout(BreakoutTask, a2c_cnn_rnn_config_dict):
    nbr_episode = nbr_episode_basis
    learns_to_solve_task(nbr_episode,BreakoutTask, build_A2C_Agent, a2c_cnn_rnn_config_dict, name=__name__)

def test_learns_to_beat_Breakout_ma(BreakoutTask_ma, a2c_cnn_rnn_config_dict):
    a2c_cnn_rnn_config_dict['nbr_actor'] = BreakoutTask_ma.env.get_nbr_envs()
    nbr_episode = nbr_episode_basis // BreakoutTask_ma.env.get_nbr_envs()
    learns_to_solve_task(nbr_episode, BreakoutTask_ma, build_A2C_Agent, a2c_cnn_rnn_config_dict, name=__name__)
