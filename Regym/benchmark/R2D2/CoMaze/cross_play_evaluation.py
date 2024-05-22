from typing import List, Dict, Tuple, Union, Optional
import pickle
from functools import partial
from itertools import product
import os
import argparse
import logging
import yaml
import random

from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
import gym

import torch 

import comaze_gym
from comaze_gym.utils.wrappers import comaze_wrap

import regym
from regym.environments import generate_task, EnvType
from regym.util.wrappers import ClipRewardEnv, PreviousRewardActionInfoMultiAgentWrapper
from regym.rl_algorithms import build_Random_Agent


DESCRIPTION = \
'''
This script performs "cross-play", as specified in the paper
"'Other-Play' for Zero-Shot Coordination.", by Hengyuan et al.

Cross-play is a "cheap proxy to evaluate whether a training method has
potential for zero-shot coordination with human players".
'''

from regym.rl_loops.multiagent_loops.marl_loop import test_agent


def cross_play(population: List['Agent'],
               task: 'Task',
               num_games_per_matchup: int,
               num_matrices: List[int],
               save_path: str=None,
               show_progress: bool=True,
               render_mode: str='rgb_array') \
        -> Tuple[np.ndarray, np.ndarray, float, float]:
    '''
    Cross-play is a "cheap proxy to evaluate whether a training method has
    potential for zero-shot coordination with human players". it plays every
    agent in :param: population against each other (including themselves),
    computing a matrix of average joint/pairwise performance for each pair of agents.
    The average value in this cross-play matrix is known as the cross-play value.

    A high average cross-play value, indicates that agents in the population are
    perform well when matched against other agents which are not themselves
    (i.e, they can coordinate with other agents).

    For each pair of agents, pairwise performance is computed over :param:
    num_games_per_matchup. A total of :param: num_matrices cross-play evaluation
    matrices are computed and averaged over.

    :param population: List of agents which will be benchmarked
    :param task: Task whose underlying environment will be used to test
                 :param: population on
    :param num_games_per_matchup: Number of episodes to be played by each pair
                                  of agents in :param: population to obtain
                                  their joint performance value.
    :param num_matrices: Number of cross-play matrices to compute, useful to
                         obtain variance estimations on cross-play values
    :param save_path: If present, the mean resulting cross-play matrix will be
                      pickled and dumped in a file under this path
    :param show_progress: Whether to output progression bars to stdout to indicate
                          cross-play computation progress
    :returns: This function returns 4 values:
          - A mean cross play matrix, containing pairwise
            agent performances computed over :param: num_games_per_matchup and
            averaged over :param: num_matrices.
          - A matrix containing the element-wise standard deviation of each
            pairwise performance over :param: num_matrices
          - Average cross-play value of mean cross-play matrix
          - Average standard deviation over mean cross-play matrix
    '''
    cross_play_matrices = compute_cross_play_matrices(
        num_matrices, population, task, num_games_per_matchup,
        show_progress,
        render_mode=render_mode,)
    
    if save_path: pickle.dump(cross_play_matrices, open(save_path, 'wb'))
    
    for idx in range(len(cross_play_matrices)):
        csm = cross_play_matrices[idx]
        for i in range(csm.shape[0]):
            for j in range(csm.shape[1]):
                csm[i,j] = csm[i,j]['mean_total_pos_return']
        cross_play_matrices[idx] = csm.astype(float)

    mean_cross_play_value = np.mean(cross_play_matrices)
    std_cross_play_value = np.std(cross_play_matrices)
    mean_cross_play_matrix = np.mean(cross_play_matrices, axis=0)
    std_cross_play_matrix = np.std(cross_play_matrices, axis=0)

    return (mean_cross_play_matrix, std_cross_play_matrix,
            mean_cross_play_value, std_cross_play_value)


def compute_cross_play_matrices(num_matrices: int,
                                population:Dict[str,regym.rl_algorithms.agents.agent.Agent],
                                task: 'Task',
                                num_games_per_matchup: int,
                                show_progress: bool,
                                render_mode: str='rgb_array') -> List[np.ndarray]:
    '''
    Computes a list, of length :param: num_matrices, of cross-play matrices
    on :param: task.  Each cross-play matrix is of shape NxN,
    where `n = len(population)`. For each cross-play matrix, each entry
    is computed using :param: num_games_per_matchup number of episodes.
    '''
    cross_play_matrices: List[np.ndarray] = []
    iterator = range(num_matrices)
    if show_progress:
        description = (f'Computing {num_matrices} cross-play matrices '
                       f'for {len(population)} agents on task {task.name} '
                       f'with {num_games_per_matchup} games per matchup.')
        iterator = tqdm(iterator, desc=description)
        iterator.set_description(description)
    for s in iterator:
        cross_play_matrix = compute_cross_play_evaluation_matrix(
            population=population,
            task=task,
            num_games_per_matchup=num_games_per_matchup,
            show_progress=show_progress,
            render_mode=render_mode,
        )
        cross_play_matrices.append(cross_play_matrix)
    return cross_play_matrices


def compute_cross_play_evaluation_matrix(population:Dict[str,regym.rl_algorithms.agents.agent.Agent],
                                         task: 'Task',
                                         num_games_per_matchup: int,
                                         show_progress: bool=True,
                                         render_mode: str='rgb_array') -> np.ndarray:
    '''
    Computes a cross-play matrix of shape NxN, where `n = len(population)`.
    Entry (i,j) represents the average performance of agents
    (population[i], population[j]) on :param: task over :param: num_games_per_matchup.
    '''
    cross_play_matrix = np.zeros((len(population), len(population)), dtype=dict)
    agentIndices2Name = dict(zip(range(len(population)), population.keys()))
    matchups_agent_indices = list(product(range(len(population)), range(len(population))))
    if show_progress:
        description = ('Computing cross play matrix with '
                       f'{len(matchups_agent_indices)} pairwise combinations. '
                       f'with {num_games_per_matchup} num games per matchup')
        matchups_agent_indices = tqdm(matchups_agent_indices, desc=description)
    for i, j in matchups_agent_indices:
        i_name = agentIndices2Name[i]
        j_name = agentIndices2Name[j]
        p1_agent = population[i_name].clone(training=False)
        p2_agent = population[j_name].clone(training=False)
        if hasattr(p1_agent, 'player_idx'):
            p1_agent.player_idx = 0
        if hasattr(p2_agent, 'player_idx'):
            p2_agent.player_idx = 1
        p1_agent.set_nbr_actor(num_games_per_matchup, vdn=False, training=False)
        p2_agent.set_nbr_actor(num_games_per_matchup, vdn=False, training=False)
        pairwise_performance = compute_pairwise_performance(
            agent_vector=[p1_agent, p2_agent],
            task=task,
            num_episodes=num_games_per_matchup,
            render_mode=render_mode,
        )
        cross_play_matrix[i, j] = pairwise_performance
    return cross_play_matrix


def compute_pairwise_performance(agent_vector: List[regym.rl_algorithms.agents.agent.Agent],
                                 task: 'Task',  # TODO: change upstream
                                 num_episodes: int,
                                 render_mode: str='rgb_array') -> float:
    '''
    Computes the average episode reward obtained by :param: agent_vector on
    :param: task over :param: num_episodes
    '''
    trajectory_metrics = test_agent(
        env=task.env, 
        agents=agent_vector, 
        nbr_episode=num_episodes,
        update_count=None, 
        sum_writer=None, 
        iteration=None, 
        base_path='./',
        requested_metrics=['mean_total_return', 'mean_total_pos_return'],
        #save_traj=True,
        #nbr_save_traj=1,
        render_mode=render_mode,
    )
    return trajectory_metrics


def check_input_validity(num_games_per_matchup, num_matrices):
    if int(num_games_per_matchup) <= 0:
        raise ValueError(f'CLI Argument "num_games_per_matchup" must be strictly positive (Given: {num_games_per_matchup})')
    if int(num_matrices) <= 0:
        raise ValueError(f'CLI Argument "num_games_per_matchup" must be strictly positive (Given: {num_matrices})')


def plot_cross_play_matrix(
        population:Dict[str,regym.rl_algorithms.agents.agent.Agent], 
        cross_play_matrix: Union[List, np.ndarray],
        cross_play_value_variance: Optional[float]=None,
        show_annotations: bool=True,
        cbar: bool=True,
        ax: Optional[plt.Axes] = None,
        )-> plt.Axes:
    '''
    Plots the :param: cross_play_matrix on a heatmap.

    Red values mean < 50% winrates
    Positive values are shown in blue.
    If :param: ax is not present
    We'll create one for you <3

    :param cross_play_matrix: Winrate matrix to plot. Values must be within [0, 1]
    :param ax: Ax where the plot should be plotted. Optional
    :show annotations: Flag determining whether values inside of the heatmap should be written
    :returns: ax where the cross_play_matrix has been plotted
    '''
    if not ax: ax = plt.subplot(111)

    #sns.set(font_scale=2.5)
    sns.heatmap(cross_play_matrix, annot=show_annotations, ax=ax, square=True,
                cmap=sns.color_palette('viridis', 16),
                cbar=cbar, cbar_kws={'label': 'Pairwise performance'})
    #ax.set_xlabel('Agent ID', size=20)
    #ax.set_ylabel('Agent ID', size=20)
    ax.set_ylim(len(cross_play_matrix) + 0.2, -0.2)  # Required seaborn hack
    
    plt.xticks(np.arange(len(population)), list(population.keys()), rotation=45)
    plt.yticks(np.arange(len(population)), list(population.keys()), rotation=45)

    title = 'Mean Cross-play {:.2}'.format(np.mean(cross_play_matrix))
    if cross_play_value_variance: title = '{} +- {:.2}'.format(title, cross_play_value_variance)

    ax.set_title(title)
    return ax


def create_task_for_r2d2(task_config):
    def comaze_r2d2_wrap(
        env,
        clip_reward=False,
        previous_reward_action=True
        ):
        env = comaze_wrap(env)

        if clip_reward:
            env = ClipRewardEnv(env)

        if previous_reward_action:
            env = PreviousRewardActionInfoMultiAgentWrapper(env=env)
        return env

    pixel_wrapping_fn = partial(
      comaze_r2d2_wrap,
      clip_reward=task_config['clip_reward'],
      previous_reward_action=task_config.get('previous_reward_action', False)
    )
    test_pixel_wrapping_fn = pixel_wrapping_fn
    #video_recording_dirpath = './videos'
    #video_recording_render_mode = 'human_comm'
    task = generate_task(task_config['env-id'],
      env_type=EnvType.MULTIAGENT_SIMULTANEOUS_ACTION,
      nbr_parallel_env=task_config['nbr_actor'],
      wrapping_fn=pixel_wrapping_fn,
      test_wrapping_fn=test_pixel_wrapping_fn,
      gathering=False,
      #train_video_recording_episode_period=1,
      #train_video_recording_dirpath='./',
      #train_video_recording_render_mode=video_recording_render_mode,
    )
    return task

def load_agents(agents_dict:Dict[str,str])->Dict[str,regym.rl_algorithms.agents.agent.Agent]:
    '''
    For rule-based agents, the paths are replaced by int to be used as seeds.
    Player indices have to be set again upon matchup pairings.
    '''
    for agent_name in agents_dict:
        if 'RB' in agent_name:
            import importlib  
            comaze_gym = importlib.import_module("regym.environments.envs.CoMaze.comaze-gym.comaze_gym")
            from comaze_gym import build_WrappedActionOnlyRuleBasedAgent, build_WrappedCommunicatingRuleBasedAgent 
            build_fn = build_WrappedActionOnlyRuleBasedAgent
            if 'comm' in agent_name:
                build_fn = build_WrappedCommunicatingRuleBasedAgent
            agents_dict[agent_name] = build_fn(
                player_idx=1,
                action_space_dim=task.action_dim,
                seed=int(agents_dict[agent_name]),
            )
        else:
            agents_dict[agent_name] = torch.load(agents_dict[agent_name])
            agents_dict[agent_name].training = False 
            agents_dict[agent_name].kwargs['vdn'] = False
    return agents_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', required=True, help='Path to file containing cross-play parameters')
    args = parser.parse_args()

    cross_play_config = yaml.load(open(args.config))
    task_config = cross_play_config['task']

    '''
    if not os.path.isdir(cross_play_config['population_path']):
        raise ValueError(f"CLI Argument 'population_path' does not point to an existing directory (Given: {cross_play_config['population_path']})")
    '''
    seed = cross_play_config['seed']
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    if hasattr(torch.backends, "cudnn"):
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False


    task = create_task_for_r2d2(task_config)
    
    loaded_population = load_agents(cross_play_config['agents'])

    # Making sure that parameters for evaluation are sound
    check_input_validity(
        int(cross_play_config['num_games_per_matchup']),
        int(cross_play_config['num_matrices'])
    )

    (mean_cross_play_matrix, std_cross_play_matrix,
     mean_cross_play_value, std_cross_play_value) = cross_play(
         loaded_population,
         task,
         cross_play_config['num_games_per_matchup'],
         cross_play_config['num_matrices'],
         save_path=cross_play_config['save_path'],
         show_progress=True,
         render_mode = 'human_comm',

     )

    matplotlib.use('TkAgg')
    plot_cross_play_matrix(
        population=loaded_population,
        cross_play_matrix=mean_cross_play_matrix, 
        cross_play_value_variance=std_cross_play_value
    )
    plt.show()

    import ipdb; ipdb.set_trace()
