from test_fixtures import ppo_mlp_config_dict, ppo_mlp_rnn_config_dict 
from test_fixtures import ppo_cnn_config_dict, ppo_cnn_rnn_config_dict 
from test_fixtures import BreakoutTask, CartPoleTask
from test_fixtures import BreakoutTask_ma, CartPoleTask_ma

from regym.rl_algorithms.agents import build_PPO_Agent
from utils import can_act_in_environment, learns_to_solve_task

nbr_episode_basis = 100

def test_ppo_can_take_actions_discrete_observation_box(BreakoutTask, ppo_cnn_config_dict):
    can_act_in_environment(BreakoutTask, build_PPO_Agent, ppo_cnn_config_dict, name=__name__)

def test_ppo_can_take_actions_continuous_obvservation_discrete(CartPoleTask, ppo_mlp_config_dict):
    can_act_in_environment(CartPoleTask, build_PPO_Agent, ppo_mlp_config_dict, name=__name__)


def test_ppo_learns_to_solve_cartpole(CartPoleTask, ppo_mlp_config_dict):
    nbr_episode = nbr_episode_basis
    learns_to_solve_task(nbr_episode, CartPoleTask, build_PPO_Agent, ppo_mlp_config_dict, name=__name__)

def test_ppo_learns_to_solve_cartpole_ma(CartPoleTask_ma, ppo_mlp_config_dict):
    ppo_mlp_config_dict['nbr_actor'] = CartPoleTask_ma.env.get_nbr_envs()
    nbr_episode = nbr_episode_basis // CartPoleTask_ma.env.get_nbr_envs()
    learns_to_solve_task(nbr_episode, CartPoleTask_ma, build_PPO_Agent, ppo_mlp_config_dict, name=__name__)


def test_learns_to_beat_Breakout(BreakoutTask, ppo_cnn_config_dict):
    nbr_episode = nbr_episode_basis
    learns_to_solve_task(nbr_episode, BreakoutTask, build_PPO_Agent, ppo_cnn_config_dict, name=__name__)

def test_learns_to_beat_Breakout_ma(BreakoutTask_ma, ppo_cnn_config_dict):
    ppo_cnn_config_dict['nbr_actor'] = BreakoutTask_ma.env.get_nbr_envs()
    nbr_episode = nbr_episode_basis // BreakoutTask_ma.env.get_nbr_envs()
    learns_to_solve_task(nbr_episode, BreakoutTask_ma, build_PPO_Agent, ppo_cnn_config_dict, name=__name__)


def test_ppo_rnn_can_take_actions_discrete_observation_box(BreakoutTask, ppo_cnn_rnn_config_dict):
    can_act_in_environment(BreakoutTask, build_PPO_Agent, ppo_cnn_rnn_config_dict, name=__name__)

def test_ppo_rnn_can_take_actions_continuous_obvservation_discrete(CartPoleTask, ppo_mlp_rnn_config_dict):
    can_act_in_environment(CartPoleTask, build_PPO_Agent, ppo_mlp_rnn_config_dict, name=__name__)


def test_ppo_rnn_learns_to_solve_cartpole(CartPoleTask, ppo_mlp_rnn_config_dict):
    nbr_episode = nbr_episode_basis
    learns_to_solve_task(nbr_episode, CartPoleTask, build_PPO_Agent, ppo_mlp_rnn_config_dict, name=__name__)

def test_ppo_rnn_learns_to_solve_cartpole_ma(CartPoleTask_ma, ppo_mlp_rnn_config_dict):
    ppo_mlp_rnn_config_dict['nbr_actor'] = CartPoleTask_ma.env.get_nbr_envs()
    nbr_episode = nbr_episode_basis // CartPoleTask_ma.env.get_nbr_envs()
    learns_to_solve_task(nbr_episode, CartPoleTask_ma, build_PPO_Agent, ppo_mlp_rnn_config_dict, name=__name__)


def test_learns_to_beat_Breakout(BreakoutTask, ppo_cnn_rnn_config_dict):
    nbr_episode = nbr_episode_basis
    learns_to_solve_task(nbr_episode,BreakoutTask, build_PPO_Agent, ppo_cnn_rnn_config_dict, name=__name__)

def test_learns_to_beat_Breakout_ma(BreakoutTask_ma, ppo_cnn_rnn_config_dict):
    ppo_cnn_rnn_config_dict['nbr_actor'] = BreakoutTask_ma.env.get_nbr_envs()
    nbr_episode = nbr_episode_basis // BreakoutTask_ma.env.get_nbr_envs()
    learns_to_solve_task(nbr_episode, BreakoutTask_ma, build_PPO_Agent, ppo_cnn_rnn_config_dict, name=__name__)
