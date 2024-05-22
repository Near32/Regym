from typing import Dict, Any, Optional, List, Callable
import logging
import yaml
import os
import sys
from typing import Dict

import torch.multiprocessing
from tqdm import tqdm
from functools import partial

import regym.environments.envs.hanabi

import torch
import numpy as np
import random

import regym
from regym.environments import generate_task, EnvType
from regym.rl_loops.multiagent_loops import marl_loop
from regym.util.experiment_parsing import initialize_agents
from regym.util.wrappers import hanabi_wrap

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


def train_and_evaluate(agents: List[object], 
                       task: object, 
                       sum_writer: object, 
                       base_path: str, 
                       offset_episode_count: int = 0,
                       nbr_pretraining_steps: int = 0, 
                       nbr_max_observations: int = 1e7,
                       test_obs_interval: int = 1e4,
                       test_nbr_episode: int = 10,
                       benchmarking_record_episode_interval: int = None,
                       step_hooks=[],
                       sad=False,
                       vdn=False):
    
    async = False
    if len(sys.argv) > 2:
      async = any(['async' in arg for arg in sys.argv])

    if async:
      trained_agent = marl_loop.async_gather_experience_parallel1(
      #trained_agents = marl_loop.async_gather_experience_parallel(
        task,
        agents,
        training=True,
        #nbr_pretraining_steps=nbr_pretraining_steps,
        max_obs_count=nbr_max_observations,
        env_configs=None,
        sum_writer=sum_writer,
        base_path=base_path,
        test_obs_interval=test_obs_interval,
        test_nbr_episode=test_nbr_episode,
        benchmarking_record_episode_interval=benchmarking_record_episode_interval,
        step_hooks=step_hooks,
        sad=sad,
        vdn=vdn,
      )
    else: 
      trained_agents = marl_loop.gather_experience_parallel(
        task,
        agents,
        training=True,
        #nbr_pretraining_steps=nbr_pretraining_steps,
        max_obs_count=nbr_max_observations,
        env_configs=None,
        sum_writer=sum_writer,
        base_path=base_path,
        test_obs_interval=test_obs_interval,
        test_nbr_episode=test_nbr_episode,
        benchmarking_record_episode_interval=benchmarking_record_episode_interval,
        step_hooks=step_hooks,
        sad=sad,
        vdn=vdn,
      )

    save_replay_buffer = False
    if len(sys.argv) > 2:
      save_replay_buffer = any(['save_replay_buffer' in arg for arg in sys.argv])

    for agent in trained_agents:
      agent.save(with_replay_buffer=save_replay_buffer)
      print(f"Agent saved at: {agent.save_path}")
    
    task.env.close()
    task.test_env.close()

    return trained_agents


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

    # SAD is dealt with at the level of VecEnv!
    # Wrapping happens in the marl loop.
    pixel_wrapping_fn = partial(
      hanabi_wrap,
      sad=False,
      clip_reward=task_config.get('clip_reward', False),
      previous_reward_action=task_config.get('previous_reward_action', False)
    )
    
    test_pixel_wrapping_fn = partial(
      hanabi_wrap,
      sad=False,
      clip_reward=False,
      previous_reward_action=task_config.get('previous_reward_action', False)
    )

    task = generate_task(task_config['env-id'],
      env_type=EnvType.MULTIAGENT_SIMULTANEOUS_ACTION,
      nbr_parallel_env=task_config['nbr_actor'],
      wrapping_fn=pixel_wrapping_fn,
      test_wrapping_fn=test_pixel_wrapping_fn,
      seed=seed,
      test_seed=100+seed,
      gathering=True
    )

    agent_config['nbr_actor'] = task_config['nbr_actor']

    regym.RegymSummaryWriterPath = base_path #regym.RegymSummaryWriter = GlobalSummaryWriter(base_path)
    sum_writer =  base_path
    
    #base_path1 = os.path.join(base_path,"1")
    #save_path1 = os.path.join(base_path1,f"./{task_config['agent-id']}.agent")
    save_path1 = os.path.join(base_path,f"./{task_config['agent-id']}.agent")
    
    agent, offset_episode_count = check_path_for_agent(save_path1)
    if agent is None: 
        agent = initialize_agents(
          task=task,
          agent_configurations={task_config['agent-id']: agent_config}
        )[0]
    agent.save_path = save_path1
    
    """
    base_path2 = os.path.join(base_path,"2")
    save_path2 = os.path.join(base_path2,f"./{task_config['agent-id']}.agent")
    
    agent2, offset_episode_count = check_path_for_agent(save_path2)
    if agent2 is None: 
        agent2 = initialize_agents(
          task=task,
          agent_configurations={task_config['agent-id']: agent_config}
        )[0]
    agent2.save_path = save_path2
    """

    #agents = [agent, agent2]

    if "vdn" in agent_config \
    and agent_config["vdn"]:
      import ipdb; ipdb.set_trace()
      agents = [agent]
    else:
      player2_harvest = False
      if len(sys.argv) > 2:
        player2_harvest = any(['player2_harvest' in arg for arg in sys.argv])

      assert player2_harvest, "selfplay with player2harvesting makes more sense."
      import ipdb; ipdb.set_trace()
      agents = [agent, agent.get_async_actor(training=player2_harvest)]
      # We can create non-training or training async actors.
      # If traininging, then their experience is added to the replay buffer
      # of the main agent, which might have some advantanges
      # -given that it proposes decorrelated data-, but it may
      # also have unknown disadvantages. Needs proper investigation.

    trained_agents = train_and_evaluate(
      agents=agents,
      task=task,
      sum_writer=sum_writer,
      base_path=base_path,
      offset_episode_count=offset_episode_count,
      nbr_pretraining_steps=int(float(agent_config["nbr_pretraining_steps"])) if "nbr_pretraining_steps" in agent_config else 0,
      nbr_max_observations=train_observation_budget,
      test_obs_interval=benchmarking_interval,
      test_nbr_episode=benchmarking_episodes,
      benchmarking_record_episode_interval=benchmarking_record_episode_interval,
      sad=task_config["sad"],
      vdn=task_config["vdn"],
    )

    return trained_agents, task 


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
    if not os.path.exists(base_path): os.makedirs(base_path)

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
