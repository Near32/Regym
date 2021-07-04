from typing import Dict, Any, Optional, List, Callable
import logging
import yaml
import os
import sys
from typing import Dict

import torch.multiprocessing

from tensorboardX import SummaryWriter
from tqdm import tqdm
from functools import partial


import torch
import numpy as np
import random
import gym 

import regym
from regym.environments import generate_task, EnvType
from regym.rl_loops.singleagent_loops import rl_loop
from regym.rl_loops.multiagent_loops import marl_loop

from regym.util.experiment_parsing import initialize_agents

from regym.util.wrappers import ClipRewardEnv, PreviousRewardActionInfoMultiAgentWrapper

import ray

from regym.modules import EnvironmentModule, CurrentAgentsModule
from regym.pubsub_manager import PubSubManager


from regym.util.wrappers import baseline_atari_pixelwrap


def make_rl_pubsubmanager(
    agents,
    config, 
    logger=None,
    load_path=None,
    save_path=None):
    """
    Create a PubSubManager.
    :param agents: List of Agents to use in the rl loop.
    :param config: Dict that specifies all the important hyperparameters of the network.
        - "task"
        - "sad"
        - "vdn"
        - "max_obs_count"
        - "sum_writer": str where to save the summary...

    """
    modules = config.pop("modules")

    cam_id = "current_agents"
    modules[cam_id] = CurrentAgentsModule(
        id=cam_id,
        agents=agents
    )

    envm_id = "EnvironmentModule_0"
    envm_input_stream_ids = {
        #"logger":"modules:logger:ref",
        #"logs_dict":"logs_dict",
        
        "iteration":"signals:iteration",

        "current_agents":f"modules:{cam_id}:ref",
    }
    modules[envm_id] = EnvironmentModule(
        id=envm_id,
        config=config,
        input_stream_ids=envm_input_stream_ids
    )

    pipelines = config.pop("pipelines")
    
    pipelines["rl_loop_0"] = [
        envm_id,
    ]
    
    optim_id = "global_optim"
    optim_config = {
      "modules":modules,
      "learning_rate":3e-4,
      "optimizer_type":'adam',
      "with_gradient_clip":False,
      "adam_eps":1e-16,
    }

    optim_module = regym.modules.build_OptimizationModule(
      id=optim_id,
      config=optim_config,
    )
    modules[optim_id] = optim_module

    logger_id = "per_epoch_logger"
    logger_module = regym.modules.build_PerEpochLoggerModule(id=logger_id)
    modules[logger_id] = logger_module
    
    pipelines[optim_id] = []
    pipelines[optim_id].append(optim_id)
    pipelines[optim_id].append(logger_id)

    pbm = PubSubManager(
        config=config,
        modules=modules,
        pipelines=pipelines,
        logger=logger,
        load_path=load_path,
        save_path=save_path,
    )
    
    return pbm

class SingleObservationWrapper(gym.Wrapper):
    """
    Assumes the :arg env: environment to have a Dict observation space,
    that contains the key :arg observation_key:.
    This wrapper makes the observation space consisting of solely the :arg observation_key: entry,
    while the other entries are put in the infos dictionnary.
    Args:
        env (gym.Env): Env to wrap.
        observation_key (str): key to the actual observation
    """

    def __init__(self, env, observation_key):
        super(SingleObservationWrapper, self).__init__(env)
        self.observation_key = observation_key
        self.observation_space = env.observation_space.spaces[self.observation_key]

        self.action_space = env.action_space 

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        
        new_observations = observations[self.observation_key]

        for k,v in observations.items():
            if k==self.observation_key:  continue
            infos[k] = np.expand_dims(np.array(v), axis=0)

        return new_observations, infos 
    
    def step(self, action):
        next_observations, reward, done, next_infos = self.env.step(action)        
        
        new_next_observations = next_observations[self.observation_key]

        for k,v in next_observations.items():
            if k==self.observation_key:  continue
            next_infos[k] = np.expand_dims(np.array(v), axis=0)
    
        return new_next_observations, reward, done, next_infos

    def render(self, mode='human', **kwargs):
        env = self.unwrapped
        return env.render(
            mode=mode,
            **kwargs,
        )
        

def env_r2d2_wrap(
    env, 
    env_wrapper,
    clip_reward=False,
    previous_reward_action=True,
    ):
    env = env_wrapper(env)

    if clip_reward:
        env = ClipRewardEnv(env)

    if previous_reward_action:
        env = PreviousRewardActionInfoMultiAgentWrapper(env=env)
    
    return SingleObservationWrapper(env=env, observation_key="observation")


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
                       nbr_pretraining_steps: int = 0, 
                       nbr_max_observations: int = 1e7,
                       test_obs_interval: int = 1e4,
                       test_nbr_episode: int = 10,
                       benchmarking_record_episode_interval: int = None,
                       render_mode="rgb_array",
                       step_hooks=[]):
    pubsub = False
    if len(sys.argv) > 2:
      pubsub = any(['pubsub' in arg for arg in sys.argv])

    if pubsub:
      import ipdb; ipdb.set_trace()
      config = {
        "modules": {},
        "pipelines": {},
      }

      config['training'] = True
      config['env_configs'] = None
      config['task'] = task 
      
      sum_writer_path = os.path.join(sum_writer, 'actor.log')
      sum_writer = config['sum_writer'] = SummaryWriter(sum_writer_path, flush_secs=1)

      config['base_path'] = base_path 
      config['offset_episode_count'] = offset_episode_count
      config['nbr_pretraining_steps'] = nbr_pretraining_steps 
      config['max_obs_count'] = nbr_max_observations
      config['test_obs_interval'] = test_obs_interval
      config['test_nbr_episode'] = test_nbr_episode
      config['benchmarking_record_episode_interval'] = benchmarking_record_episode_interval
      config['render_mode'] = render_mode
      config['step_hooks'] = step_hooks
      config['save_traj_length_divider'] = 1
      config['nbr_players'] = 1
      pubsubmanager = make_rl_pubsubmanager(
        agents=[agent],
        config=config,
        logger=sum_writer,
      )

      pubsubmanager.train() 

      trained_agent = agent
    else:
      async = False
      if len(sys.argv) > 2:
        async = any(['async' in arg for arg in sys.argv])

      if async:
        trained_agent = rl_loop.async_gather_experience_parallel1(
          task,
          agent,
          training=True,
          #nbr_pretraining_steps=nbr_pretraining_steps,
          max_obs_count=nbr_max_observations,
          env_configs=None,
          sum_writer=sum_writer,
          base_path=base_path,
          test_obs_interval=test_obs_interval,
          test_nbr_episode=test_nbr_episode,
          benchmarking_record_episode_interval=benchmarking_record_episode_interval,
          save_traj_length_divider=1,
          render_mode=render_mode,
          step_hooks=step_hooks,
        )
      else: 
        trained_agent = rl_loop.gather_experience_parallel(
          task=task,
          agent=agent,
          training=True,
          #nbr_pretraining_steps=nbr_pretraining_steps,
          max_obs_count=nbr_max_observations,
          env_configs=None,
          sum_writer=sum_writer,
          base_path=base_path,
          test_obs_interval=test_obs_interval,
          test_nbr_episode=test_nbr_episode,
          benchmarking_record_episode_interval=benchmarking_record_episode_interval,
          save_traj_length_divider=1,
          render_mode=render_mode,
          step_hooks=step_hooks,
        )

    save_replay_buffer = False
    if len(sys.argv) > 2:
      save_replay_buffer = any(['save_replay_buffer' in arg for arg in sys.argv])

    for agent in trained_agents:
      agent.save(with_replay_buffer=save_replay_buffer)
      print(f"Agent saved at: {agent.save_path}")
    
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
    pubsub = False
    test_only = False
    if len(sys.argv) > 2:
      pubsub = any(['pubsub' in arg for arg in sys.argv])
      test_only = any(['test_only' in arg for arg in sys.argv])
      
    if test_only:
      base_path = os.path.join(base_path,"TESTING")
    else:
      base_path = os.path.join(base_path,"TRAINING")
    
    if pubsub:
      base_path = os.path.join(base_path,"PUBSUB")
    else:
      base_path = os.path.join(base_path,"NOPUBSUB")
      
    print(f"Final Path: -- {base_path} --")
    
    if not os.path.exists(base_path): os.makedirs(base_path)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if hasattr(torch.backends, "cudnn"):
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

    pixel_wrapping_fn = partial(baseline_atari_pixelwrap,
                                size=task_config['observation_resize_dim'], 
                                skip=task_config['nbr_frame_skipping'], 
                                stack=task_config['nbr_frame_stacking'],
                                grayscale=task_config['grayscale'],
                                single_life_episode=task_config['single_life_episode'],
                                nbr_max_random_steps=task_config['nbr_max_random_steps'],
                                clip_reward=task_config['clip_reward'])
    pixel_wrapping_fn = partial(env_r2d2_wrap,
      env_wrapper=pixel_wrapping_fn,
      previous_reward_action=task_config.get('previous_reward_action', False)
    )

    test_pixel_wrapping_fn = partial(baseline_atari_pixelwrap,
                                    size=task_config['observation_resize_dim'], 
                                    skip=task_config['nbr_frame_skipping'], 
                                    stack=task_config['nbr_frame_stacking'],
                                    grayscale=task_config['grayscale'],
                                    single_life_episode=False,
                                    nbr_max_random_steps=task_config['nbr_max_random_steps'],
                                    clip_reward=False)
    test_pixel_wrapping_fn = partial(env_r2d2_wrap,
      env_wrapper=test_pixel_wrapping_fn,
      previous_reward_action=task_config.get('previous_reward_action', False)
    )
    
    video_recording_dirpath = os.path.join(base_path,'videos')
    video_recording_render_mode = 'human_comm'
    task = generate_task(
      task_config['env-id'],
      env_type=EnvType.SINGLE_AGENT,
      nbr_parallel_env=task_config['nbr_actor'],
      wrapping_fn=pixel_wrapping_fn,
      test_wrapping_fn=test_pixel_wrapping_fn,
      seed=seed,
      test_seed=100+seed,
      gathering=True,
      train_video_recording_episode_period=benchmarking_record_episode_interval,
      train_video_recording_dirpath=video_recording_dirpath,
      train_video_recording_render_mode=video_recording_render_mode,
    )

    agent_config['nbr_actor'] = task_config['nbr_actor']

    regym.RegymSummaryWriterPath = base_path #regym.RegymSummaryWriter = GlobalSummaryWriter(base_path)
    sum_writer = base_path
    
    save_path1 = os.path.join(base_path,f"./{task_config['agent-id']}.agent")
    if task_config.get("reload", 'None')!='None':
      import ipdb; ipdb.set_trace()
      agent, offset_episode_count = check_path_for_agent(task_config["reload"])
    else:
      agent, offset_episode_count = check_path_for_agent(save_path1)
    
    if agent is None: 
        agent = initialize_agents(
          task=task,
          agent_configurations={task_config['agent-id']: agent_config}
        )[0]
    agent.save_path = save_path1
    #regym.rl_algorithms.algorithms.DQN.dqn.summary_writer = sum_writer
    
    if test_only:
      print(save_path1)
      import ipdb; ipdb.set_trace()
      agent.training = False
    else:
      import ipdb; ipdb.set_trace()

    trained_agent = train_and_evaluate(
      agent=agent,
      task=task,
      sum_writer=sum_writer,
      base_path=base_path,
      offset_episode_count=offset_episode_count,
      nbr_pretraining_steps=int(float(agent_config["nbr_pretraining_steps"])) if "nbr_pretraining_steps" in agent_config else 0,
      nbr_max_observations=train_observation_budget,
      test_obs_interval=benchmarking_interval,
      test_nbr_episode=benchmarking_episodes,
      benchmarking_record_episode_interval=benchmarking_record_episode_interval,
      #render_mode="human_comm",
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
    logger = logging.getLogger('NBitsSwap(MNIST) Benchmark')

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
        print(f"Tentative Path: -- {path} --")
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
