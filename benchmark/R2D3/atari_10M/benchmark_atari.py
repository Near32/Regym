import logging
import yaml
import os
import sys
from typing import Dict

import torch.multiprocessing
from tqdm import tqdm
from functools import partial

import torch
import numpy as np
import random

import regym
from regym.environments import generate_task
from regym.rl_loops.singleagent_loops import rl_loop
from regym.util.experiment_parsing import initialize_agents
from regym.util.wrappers import baseline_atari_pixelwrap

import ray

def check_path_for_agent(filepath):
    #filepath = os.path.join(path,filename)
    agent = None
    offset_episode_count = 0
    if os.path.isfile(filepath):
        print('==> loading checkpoint {}'.format(filepath))
        agent = torch.load(filepath)
        offset_episode_count = agent.episode_count
        #setattr(agent, 'episode_count', offset_episode_count)
        print('==> loaded checkpoint {}'.format(filepath))
    return agent, offset_episode_count


def train_and_evaluate(agent: object, 
                       task: object, 
                       sum_writer: object, 
                       base_path: str, 
                       offset_episode_count: int = 0, 
                       nbr_max_observations: int = 1e7,
                       test_obs_interval: int = 1e4,
                       test_nbr_episode: int = 10,
                       benchmarking_record_episode_interval: int = None,
                       step_hooks=[]):
    
    async = False
    if len(sys.argv) > 2:
      async = any(['async' in arg for arg in sys.argv])

    if async:
      trained_agent = rl_loop.async_gather_experience_parallel1(
        task,
        agent,
        training=True,
        max_obs_count=nbr_max_observations,
        env_configs=None,
        sum_writer=sum_writer,
        base_path=base_path,
        test_obs_interval=test_obs_interval,
        test_nbr_episode=test_nbr_episode,
        benchmarking_record_episode_interval=benchmarking_record_episode_interval,
        step_hooks=step_hooks
      )
    else: 
      trained_agent = rl_loop.gather_experience_parallel(
        task,
        agent,
        training=True,
        max_obs_count=nbr_max_observations,
        env_configs=None,
        sum_writer=sum_writer,
        base_path=base_path,
        test_obs_interval=test_obs_interval,
        test_nbr_episode=test_nbr_episode,
        benchmarking_record_episode_interval=benchmarking_record_episode_interval,
        step_hooks=step_hooks
      )

    save_replay_buffer = False
    if len(sys.argv) > 2:
      save_replay_buffer = any(['save_replay_buffer' in arg for arg in sys.argv])

    trained_agent.save(with_replay_buffer=save_replay_buffer)
    print(f"Agent saved at: {trained_agent.save_path}")
    
    task.env.close()
    task.test_env.close()

    return trained_agent


def training_process(agent_config: Dict, 
                     task_config: Dict,
                     benchmarking_interval: int = 1e4,
                     benchmarking_episodes: int = 10, 
                     benchmarking_record_episode_interval: int = None,
                     train_observation_budget: int = 1e7,
                     base_path: str = './', 
                     seed: int = 0):
    if not os.path.exists(base_path): os.makedirs(base_path)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if hasattr(torch.backends, "cudnn"):
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

    pixel_wrapping_fn = partial(
      baseline_atari_pixelwrap,
      size=task_config['observation_resize_dim'], 
      skip=task_config['nbr_frame_skipping'], 
      stack=task_config['nbr_frame_stacking'],
      grayscale=task_config['grayscale'],
      single_life_episode=task_config['single_life_episode'],
      nbr_max_random_steps=task_config['nbr_max_random_steps'],
      clip_reward=task_config['clip_reward'],
      previous_reward_action=task_config.get('previous_reward_action', False)
    )

    test_pixel_wrapping_fn = partial(
      baseline_atari_pixelwrap,
      size=task_config['observation_resize_dim'], 
      skip=task_config['nbr_frame_skipping'], 
      stack=task_config['nbr_frame_stacking'],
      grayscale=task_config['grayscale'],
      single_life_episode=False,
      nbr_max_random_steps=task_config['nbr_max_random_steps'],
      clip_reward=False,
      previous_reward_action=task_config.get('previous_reward_action', False)
    )
    
    task = generate_task(task_config['env-id'],
                         nbr_parallel_env=task_config['nbr_actor'],
                         wrapping_fn=pixel_wrapping_fn,
                         test_wrapping_fn=test_pixel_wrapping_fn,
                         seed=seed,
                         test_seed=100+seed,
                         gathering=True)

    agent_config['nbr_actor'] = task_config['nbr_actor']

    
    regym.RegymSummaryWriterPath = base_path #regym.RegymSummaryWriter = GlobalSummaryWriter(base_path)
    sum_writer =  base_path
    
    save_path = os.path.join(base_path,f"./{task_config['agent-id']}.agent")
    agent, offset_episode_count = check_path_for_agent(save_path)
    if agent is None: 
        agent = initialize_agents(task=task,
                                  agent_configurations={task_config['agent-id']: agent_config})[0]
    agent.save_path = save_path
    
    trained_agent = train_and_evaluate(agent=agent,
                       task=task,
                       sum_writer=sum_writer,
                       base_path=base_path,
                       offset_episode_count=offset_episode_count,
                       nbr_max_observations=train_observation_budget,
                       test_obs_interval=benchmarking_interval,
                       test_nbr_episode=benchmarking_episodes,
                       benchmarking_record_episode_interval=benchmarking_record_episode_interval,
                       )

    return trained_agent, task 


def load_configs(config_file_path: str):
    all_configs = yaml.load(open(config_file_path))

    agents_config = all_configs['agents']
    experiment_config = all_configs['experiment']
    envs_config = experiment_config['tasks']

    return experiment_config, agents_config, envs_config


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Atari 10 Millions Frames Benchmark')

    config_file_path = sys.argv[1] #'./atari_10M_benchmark_config.yaml'
    experiment_config, agents_config, tasks_configs = load_configs(config_file_path)

    # Generate path for experiment
    base_path = experiment_config['experiment_id']
    if not os.path.exists(base_path): os.mkdir(base_path)

    for task_config in tasks_configs:
        agent_name = task_config['agent-id']
        env_name = task_config['env-id']
        run_name = task_config['run-id']
        path = f'{base_path}/{env_name}/{run_name}/{agent_name}'
        print(f"Path: -- {path} --")
        training_process(agents_config[task_config['agent-id']], task_config,
                         benchmarking_interval=int(float(experiment_config['benchmarking_interval'])),
                         benchmarking_episodes=int(float(experiment_config['benchmarking_episodes'])),
                         benchmarking_record_episode_interval=int(float(experiment_config['benchmarking_record_episode_interval'])),
                         train_observation_budget=int(float(experiment_config['train_observation_budget'])),
                         base_path=path,
                         seed=experiment_config['seed'])

if __name__ == '__main__':
  async = False 
  __spec__ = None
  if len(sys.argv) > 2:
      async = any(['async' in arg for arg in sys.argv])
  if async:
      torch.multiprocessing.freeze_support()
      torch.multiprocessing.set_start_method("forkserver", force=True)
      #torch.multiprocessing.set_start_method("spawn", force=True)
      ray.init() #local_mode=True)
      
      from regym import CustomManager as Manager
      from multiprocessing.managers import SyncManager, MakeProxyType, public_methods
      
      # from regym.rl_algorithms.replay_buffers import SharedPrioritizedReplayStorage
      # #SharedPrioritizedReplayStorageProxy = MakeProxyType("SharedPrioritizedReplayStorage", public_methods(SharedPrioritizedReplayStorage))
      # Manager.register("SharedPrioritizedReplayStorage", 
      #   SharedPrioritizedReplayStorage,# SharedPrioritizedReplayStorageProxy) 
      #   exposed=[
      #       "get_beta",
      #       "get_tree_indices",
      #       "cat",
      #       "reset",
      #       "add_key",
      #       "total",
      #       "__len__",
      #       "priority",
      #       "sequence_priority",
      #       "update",
      #       "add",
      #       "sample",
      #       ]
      # )
      # print("WARNING: SharedPrioritizedReplayStorage class has been registered with the RegymManager.")

      regym.RegymManager = Manager()
      regym.RegymManager.start()

  main()
