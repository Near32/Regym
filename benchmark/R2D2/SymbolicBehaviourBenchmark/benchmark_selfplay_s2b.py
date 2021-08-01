from typing import Dict, Any, Optional, List, Callable

import torch
import sklearn 

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

import regym
from regym.environments import generate_task, EnvType
from regym.rl_loops.multiagent_loops import marl_loop
from regym.util.experiment_parsing import initialize_agents

import symbolic_behaviour_benchmark
from symbolic_behaviour_benchmark.utils.wrappers import s2b_wrap
from regym.util.wrappers import ClipRewardEnv, PreviousRewardActionInfoMultiAgentWrapper

import ray

from regym.modules import EnvironmentModule, CurrentAgentsModule
from regym.modules import MARLEnvironmentModule, RLAgentModule

#from regym.modules import MultiStepCICMetricModule
#from rl_action_policy import RLActionPolicy
#from comaze_gym.metrics import MultiStepCIC, RuleBasedActionPolicy

#from regym.modules import MessageTrajectoryMutualInformationMetricModule
#from rl_message_policy import RLMessagePolicy
#from comaze_gym.metrics import MessageTrajectoryMutualInformationMetric, RuleBasedMessagePolicy

#from regym.modules import CoMazeGoalOrderingPredictionModule
#from rl_hiddenstate_policy import RLHiddenStatePolicy
#from comaze_gym.metrics import GoalOrderingPredictionMetric, RuleBasedHiddenStatePolicy

from regym.pubsub_manager import PubSubManager

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
        - "otherplay"
        - "max_obs_count"
        - "sum_writer": str where to save the summary...

    """
    pipelined = False
    if len(sys.argv) > 2:
      pipelined = any(['pipelined' in arg for arg in sys.argv])

    modules = config.pop("modules")

    cam_id = "current_agents"
    modules[cam_id] = CurrentAgentsModule(
        id=cam_id,
        agents=agents
    )

    if pipelined:
      envm_id = "MARLEnvironmentModule_0"
      
      rlam_ids = [
        f"rl_agent_{rlaidx}"
        for rlaidx in range(len(agents))
      ]
      for aidx, (rlam_id, agent) in enumerate(zip(rlam_ids, agents)):
        rlam_config = {
          'agent': agent
        }
        rlam_input_stream_ids = {
          "logs_dict":"logs_dict",
          "losses_dict":"losses_dict",
          "epoch":"signals:epoch",
          "mode":"signals:mode",

          "reset_actors":f"modules:{envm_id}:reset_actors",
          
          "observations":f"modules:{envm_id}:player_0:observations",
          "infos":f"modules:{envm_id}:player_0:infos",
          "actions":f"modules:{envm_id}:player_0:actions",
          "succ_observations":f"modules:{envm_id}:player_0:succ_observations",
          "succ_infos":f"modules:{envm_id}:player_0:succ_infos",
          "rewards":f"modules:{envm_id}:player_0:rewards",
          "dones":f"modules:{envm_id}:dones",
        }
        modules[rlam_id] = RLAgentModule(
            id=rlam_id,
            config=rlam_config,
            input_stream_ids=rlam_input_stream_ids,
        )

      envm_input_stream_ids = {
          "iteration":"signals:iteration",
          "current_agents":f"modules:{cam_id}:ref",
      }

      modules[envm_id] = MARLEnvironmentModule(
          id=envm_id,
          config=config,
          input_stream_ids=envm_input_stream_ids
      )
    else:
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
    if pipelined:
      for rlam_id in rlam_ids:
        pipelines['rl_loop_0'].append(rlam_id)

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


def s2b_r2d2_wrap(
    env, 
    clip_reward=False,
    previous_reward_action=True,
    otherplay=False
    ):
    env = s2b_wrap(
      env, 
      combined_actions=True,
      dict_obs_space=False,
    )

    if clip_reward:
        env = ClipRewardEnv(env)

    if previous_reward_action:
        env = PreviousRewardActionInfoMultiAgentWrapper(env=env)
    
    return env


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
                       render_mode="rgb_array",
                       step_hooks=[],
                       sad=False,
                       vdn=False,
                       otherplay=False,
                       ):
    pubsub = False
    if len(sys.argv) > 2:
      pubsub = any(['pubsub' in arg for arg in sys.argv])

    if pubsub:
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
      config['save_traj_length_divider'] =1
      config['sad'] = sad 
      config['vdn'] = vdn
      config['otherplay'] = otherplay
      config['nbr_players'] = 2      
      pubsubmanager = make_rl_pubsubmanager(
        agents=agents,
        config=config,
        logger=sum_writer,
      )

      pubsubmanager.train() 

      trained_agents = agents 
    else:
      asynch = False
      if len(sys.argv) > 2:
        asynch = any(['async' in arg for arg in sys.argv])

      if asynch:
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
          save_traj_length_divider=1,
          render_mode=render_mode,
          step_hooks=step_hooks,
          sad=sad,
          vdn=vdn,
          otherplay=otherplay,
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
          save_traj_length_divider=1,
          render_mode=render_mode,
          step_hooks=step_hooks,
          sad=sad,
          vdn=vdn,
          otherplay=otherplay
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
    
    test_only = False
    path_suffix = None
    pubsub = False
    if len(sys.argv) > 2:
      pubsub = any(['pubsub' in arg for arg in sys.argv])
      test_only = any(['test_only' in arg for arg in sys.argv])
    
      override_seed_argv_idx = [idx for idx, arg in enumerate(sys.argv) if '--seed' in arg]
      if len(override_seed_argv_idx):
        seed = int(sys.argv[override_seed_argv_idx[0]+1])
        print(f"NEW RANDOM SEED: {seed}")

      override_reload_argv = [idx for idx, arg in enumerate(sys.argv) if '--reload_path' in arg]
      if len(override_reload_argv):
        task_config["reload"] = sys.argv[override_reload_argv[0]+1]
        print(f"NEW RELOAD PATH: {task_config['reload']}")

      path_suffix_argv = [idx for idx, arg in enumerate(sys.argv) if '--path_suffix' in arg]
      if len(path_suffix_argv):
        path_suffix = sys.argv[path_suffix_argv[0]+1]
        print(f"ADDITIONAL PATH SUFFIX: {path_suffix}")

      obs_budget_argv = [idx for idx, arg in enumerate(sys.argv) if '--obs_budget' in arg]
      if len(obs_budget_argv):
        train_observation_budget = int(sys.argv[obs_budget_argv[0]+1])
        print(f"TRAINING OBSERVATION BUDGET: {train_observation_budget}")


      task_config["otherplay"] = any(['--otherplay' in arg for arg in sys.argv[2:]])
      
    if test_only:
      base_path = os.path.join(base_path,"TESTING")
    else:
      base_path = os.path.join(base_path,"TRAINING")
    
    if pubsub:
      base_path = os.path.join(base_path,"PUBSUB")
    else:
      base_path = os.path.join(base_path,"NOPUBSUB")
      
    if task_config["otherplay"]:
      base_path = os.path.join(base_path,"OtherPlay")
    
    base_path = os.path.join(base_path,f"SEED{seed}")

    if path_suffix is not None:
      base_path = os.path.join(base_path, path_suffix)

    print(f"Final Path: -- {base_path} --")
    import ipdb; ipdb.set_trace() 

    if not os.path.exists(base_path): os.makedirs(base_path)

    task_config['final_path'] = base_path
    yaml.dump(
      task_config, 
      open(
        os.path.join(base_path, "task_config.yaml"), 'w',
        encoding='utf8',
      ),
    )
    yaml.dump(
      agent_config, 
      open(
        os.path.join(base_path, "agent_config.yaml"), 'w',
        encoding='utf8',
      ),
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if hasattr(torch.backends, "cudnn"):
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

    pixel_wrapping_fn = partial(
      s2b_r2d2_wrap,
      clip_reward=task_config['clip_reward'],
      previous_reward_action=task_config.get('previous_reward_action', False),
      otherplay=task_config.get("otherplay", False),
    )
    
    test_pixel_wrapping_fn = pixel_wrapping_fn
    """
     partial(
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
    """
    video_recording_dirpath = os.path.join(base_path,'videos')
    video_recording_render_mode = 'human_comm'
    task = generate_task(task_config['env-id'],
      env_type=EnvType.MULTIAGENT_SIMULTANEOUS_ACTION,
      nbr_parallel_env=task_config['nbr_actor'],
      wrapping_fn=pixel_wrapping_fn,
      test_wrapping_fn=test_pixel_wrapping_fn,
      env_config=task_config['env-config'],
      test_env_config=task_config['env-config'],
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
      agent, offset_episode_count = check_path_for_agent(task_config["reload"])
    else:
      agent, offset_episode_count = check_path_for_agent(save_path1)
    
    if agent is None: 
        agent = initialize_agents(
          task=task,
          agent_configurations={task_config['agent-id']: agent_config}
        )[0]
    agent.save_path = save_path1
    
    if test_only:
      print(save_path1)
      agent.training = False
    
    if "vdn" in agent_config \
    and agent_config["vdn"]:
      agents = [agent]
    else:
      player2_harvest = False

      if len(sys.argv) > 2:
        player2_harvest = any(['player2_harvest' in arg for arg in sys.argv])

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
      #benchmarking_record_episode_interval=None, 
      benchmarking_record_episode_interval=benchmarking_record_episode_interval,
      render_mode="human_comm",
      sad=task_config["sad"],
      vdn=task_config["vdn"],
      otherplay=task_config.get("otherplay", False),
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
    logger = logging.getLogger('Symbolic Behaviour Benchmark')

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
        print(f"Tentative Path: -- {path} --")
        training_process(agents_config[task_config['agent-id']], task_config,
                         benchmarking_interval=int(float(experiment_config['benchmarking_interval'])),
                         benchmarking_episodes=int(float(experiment_config['benchmarking_episodes'])),
                         benchmarking_record_episode_interval=int(float(experiment_config['benchmarking_record_episode_interval'])) if experiment_config['benchmarking_record_episode_interval']!='None' else None,
                         train_observation_budget=int(float(experiment_config['train_observation_budget'])),
                         base_path=path,
                         seed=experiment_config['seed'])

if __name__ == '__main__':
  asynch = False 
  __spec__ = None
  if len(sys.argv) > 2:
      asynch = any(['async' in arg for arg in sys.argv])
  if asynch:
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
