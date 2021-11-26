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

import argparse
import numpy as np
import random

import regym
from regym.environments import generate_task, EnvType
from regym.rl_loops.multiagent_loops import marl_loop
from regym.util.experiment_parsing import initialize_agents

import ray

from regym.modules import EnvironmentModule, CurrentAgentsModule
from regym.modules import MARLEnvironmentModule, RLAgentModule

from regym.modules import ReconstructionFromHiddenStateModule, MultiReconstructionFromHiddenStateModule

from regym.pubsub_manager import PubSubManager

import wandb
import gym
import iglu

from perceiver_io import LMCapsule
from collections import OrderedDict
from copy import deepcopy


LanguageModelCapsule = None

class IgluActionWrapper(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.noop_dict = OrderedDict({
            'attack': np.array(0),
            'back': np.array(0),
            'camera': np.array([0, 0]),
            'forward': np.array(0),
            'hotbar': np.array(0),
            'jump': np.array(0),
            'left': np.array(0),
            'right': np.array(0),
            'use': np.array(0),
        })

        self.actions_list = [
            ('attack', np.array(1)),
            ('back', np.array(1)),

            ('camera', np.array([5.0, 0.0])),
            ('camera', np.array([-5.0, 0.0])),
            ('camera', np.array([0.0, 5.0])),
            ('camera', np.array([0.0, -5.0])),
            ('camera', np.array([0.0, 0.0])), # noop

            ('forward', np.array(1)),
            # 'hotbar', np.array(0),
            # ('hotbar', np.array(1)),
            # 'hotbar', np.array(2),
            # 'hotbar', np.array(3),
            # 'hotbar', np.array(4),
            # 'hotbar', np.array(5),
            ('jump', np.array(1)),
            ('left', np.array(1)),
            ('right', np.array(1)),
            ('use', np.array(1)),
        ]

        self.action_space = gym.spaces.Discrete(len(self.actions_list))

    def action(self, action):
        result = deepcopy(self.noop_dict)
        result.update({self.actions_list[action][0]: self.actions_list[action][1]})
        return result

    def reverse_action(self, action):
        pass


class PovOnlyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)

    """
    def observation(self, observation):
        return observation['pov']
    """
    
    def reset(self, **args):
        obs = self.env.reset(**args)
        info = {}
        for key, value in obs.items():
            if key=="pov":  continue
            info[key] = value
        obs = obs["pov"]
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for key, value in obs.items():
            if key=="pov":  continue
            info[key] = value
        obs = obs["pov"]
        return obs, reward, done, info

class ChatEmbeddingWrapper(gym.Wrapper):
    def __init__(self, env, use_cuda=True):
        super().__init__(env)
        global LanguageModelCapsule
        if LanguageModelCapsule is None:
            LanguageModelCapsule = LMCapsule(use_cuda=use_cuda)
        self.lm = LanguageModelCapsule
        self.chat = None

    def reset(self, **args):
        obs = self.env.reset(**args)
        self.chat = None
        if "chat" in obs:
            obs["chat"] = self.lm(obs["chat"]).detach().numpy()
            self.chat = obs["chat"]
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "chat" in obs:
            if self.chat is None:
                self.chat = self.lm(obs["chat"]).detach().numpy()
            else:
                obs["chat"] = self.chat

        return obs, reward, done, info
    

from regym.util.wrappers import ClipRewardEnv, PreviousRewardActionInfoWrapper
def wrap_iglu(
    env,
    clip_reward=False,
    previous_reward_action=True,
    trajectory_wrapping=False,
    ):
    env = gym.make('IGLUSilentBuilder-v0', max_steps=1000, )
    #env.update_taskset(TaskSet(preset=[task]))
    env = ChatEmbeddingWrapper(env)
    env = PovOnlyWrapper(env)
    env = IgluActionWrapper(env)
    
    # Clip reward to (-1,0,+1)
    env = ClipRewardEnv(env)
    
    # The agent deals with discrete actions so we want this wrapper to be the last one:
    if previous_reward_action:
        env = PreviousRewardActionInfoWrapper(
            env=env,
            trajectory_wrapping=trajectory_wrapping
        )
        # state=POV, 
        # input action is discrete, propagated action is continuous, 
        # infos={inventory, previous_reward, previous_action(ohe) (if traj_wrap: current_action(d))}
    
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


def make_rl_pubsubmanager(
    agents,
    config, 
    logger=None,
    load_path=None,
    save_path=None,
    node_id_to_extract="hidden",
    ):
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
    modules = config.pop("modules")

    cam_id = "current_agents"
    modules[cam_id] = CurrentAgentsModule(
        id=cam_id,
        agents=agents
    )

    envm_id = "MARLEnvironmentModule_0"
    envm_input_stream_ids = {
        "logs_dict":"logs_dict",
        "iteration":"signals:iteration",
        "current_agents":f"modules:{cam_id}:ref",
    }
    
    rlam_ids = [
      f"rl_agent_{rlaidx}"
      for rlaidx in range(len(agents))
    ]
    for aidx, (rlam_id, agent) in enumerate(zip(rlam_ids, agents)):
      rlam_config = {
        'agent': agent,
        'actions_stream_id':f"modules:{envm_id}:player_{aidx}:actions",
      }

      envm_input_stream_ids[f'player_{aidx}'] = f"modules:{rlam_id}:ref"

      rlam_input_stream_ids = {
        "logs_dict":"logs_dict",
        "losses_dict":"losses_dict",
        "epoch":"signals:epoch",
        "mode":"signals:mode",

        "reset_actors":f"modules:{envm_id}:reset_actors",
        
        "observations":f"modules:{envm_id}:ref:player_{aidx}:observations",
        "infos":f"modules:{envm_id}:ref:player_{aidx}:infos",
        "actions":f"modules:{envm_id}:ref:player_{aidx}:actions",
        "succ_observations":f"modules:{envm_id}:ref:player_{aidx}:succ_observations",
        "succ_infos":f"modules:{envm_id}:ref:player_{aidx}:succ_infos",
        "rewards":f"modules:{envm_id}:ref:player_{aidx}:rewards",
        "dones":f"modules:{envm_id}:ref:player_{aidx}:dones",
      }
      modules[rlam_id] = RLAgentModule(
          id=rlam_id,
          config=rlam_config,
          input_stream_ids=rlam_input_stream_ids,
      )

    modules[envm_id] = MARLEnvironmentModule(
        id=envm_id,
        config=config,
        input_stream_ids=envm_input_stream_ids
    )
    
    pipelines = config.pop("pipelines")
    
    pipelines["rl_loop_0"] = [
        envm_id,
    ]
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



def train_and_evaluate(
    agent: object, 
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
    render_mode="rgb_array",
    ):
    
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
    config['save_traj_length_divider'] =1
    config['sad'] = False
    config['vdn'] = False
    config['otherplay'] = False
    config['nbr_players'] = 1
    config['step_hooks'] = [] 

    agents = [agent]

    pubsubmanager = make_rl_pubsubmanager(
      agents=agents,
      config=config,
      logger=sum_writer,
    )

    pubsubmanager.train() 

    save_replay_buffer = False
    if len(sys.argv) > 2:
      save_replay_buffer = any(['save_replay_buffer' in arg for arg in sys.argv])

    try:
        for agent in trained_agents:
            agent.save(with_replay_buffer=save_replay_buffer)
            print(f"Agent saved at: {agent.save_path}")
    except Exception as e:
        print(e)

    task.env.close()
    task.test_env.close()

    return trained_agents

def training_process(
    agent_config: Dict, 
    task_config: Dict,
    benchmarking_interval: int = 1e4,
    benchmarking_episodes: int = 10, 
    benchmarking_record_episode_interval: int = None,
    train_observation_budget: int = 1e7,
    base_path: str = './',
    video_recording_episode_period: int = None,
    seed: int = 0,
    ):
    
    test_only = task_config.get('test_only', False)
    path_suffix = task_config.get('path_suffix', None)
    if path_suffix=='None':  path_suffix=None
    
    if len(sys.argv) > 2:
      override_seed_argv_idx = [
        idx for idx, arg in enumerate(sys.argv) 
        if '--new_seed' in arg
      ]
      if len(override_seed_argv_idx):
        seed = int(sys.argv[override_seed_argv_idx[0]+1])
        print(f"NEW RANDOM SEED: {seed}")

      override_reload_argv = [
        idx for idx, arg in enumerate(sys.argv) 
        if '--reload_path' in arg
      ]
      if len(override_reload_argv):
        task_config["reload"] = sys.argv[override_reload_argv[0]+1]
        print(f"NEW RELOAD PATH: {task_config['reload']}")

      path_suffix_argv = [
        idx for idx, arg in enumerate(sys.argv) 
        if '--path_suffix' in arg
      ]
      if len(path_suffix_argv):
        path_suffix = sys.argv[path_suffix_argv[0]+1]
        print(f"ADDITIONAL PATH SUFFIX: {path_suffix}")

      obs_budget_argv = [
        idx for idx, arg in enumerate(sys.argv) 
        if '--obs_budget' in arg
      ]
      if len(obs_budget_argv):
        train_observation_budget = int(sys.argv[obs_budget_argv[0]+1])
        print(f"TRAINING OBSERVATION BUDGET: {train_observation_budget}")

    if test_only:
      base_path = os.path.join(base_path,"TESTING")
    else:
      base_path = os.path.join(base_path,"TRAINING")
    
    base_path = os.path.join(base_path,f"SEED{seed}")
    
    if path_suffix is not None:
      base_path = os.path.join(base_path, path_suffix)

    print(f"Final Path: -- {base_path} --")

    if not os.path.exists(base_path): os.makedirs(base_path)
    
    task_config['final_path'] = base_path
    task_config['command_line'] = ' '.join(sys.argv)
    print(task_config['command_line'])
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
    
    #//////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if hasattr(torch.backends, "cudnn"):
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

    pixel_wrapping_fn = partial(
        wrap_iglu,
        #size=task_config['observation_resize_dim'], 
        clip_reward=task_config['clip_reward'],
        previous_reward_action=task_config["previous_reward_action"],
    )
    test_pixel_wrapping_fn = pixel_wrapping_fn
    
    """
    pixel_wrapping_fn = partial(
      baseline_ther_wrapper,
      size=task_config['observation_resize_dim'], 
      skip=task_config['nbr_frame_skipping'], 
      stack=task_config['nbr_frame_stacking'],
      single_life_episode=task_config['single_life_episode'],
      nbr_max_random_steps=task_config['nbr_max_random_steps'],
      clip_reward=task_config['clip_reward'],
      max_sentence_length=agent_config['THER_max_sentence_length'],
      vocabulary=agent_config['THER_vocabulary'],
    )

    test_pixel_wrapping_fn = partial(
      baseline_ther_wrapper,
      size=task_config['observation_resize_dim'], 
      skip=task_config['nbr_frame_skipping'], 
      stack=task_config['nbr_frame_stacking'],
      single_life_episode=False,
      nbr_max_random_steps=task_config['nbr_max_random_steps'],
      clip_reward=False,
      max_sentence_length=agent_config['THER_max_sentence_length'],
      vocabulary=agent_config['THER_vocabulary'],
    )
    """

    video_recording_dirpath = os.path.join(base_path,'videos')
    video_recording_render_mode = 'rgb_array'
    task = generate_task(
      task_config['env-id'],
      env_type=EnvType.SINGLE_AGENT,
      nbr_parallel_env=task_config['nbr_actor'],
      wrapping_fn=pixel_wrapping_fn,
      test_wrapping_fn=test_pixel_wrapping_fn,
      env_config=task_config.get('env-config', {}),
      test_env_config=task_config.get('env-config', {}),
      seed=seed,
      test_seed=100+seed,
      gathering=True,
      train_video_recording_episode_period=benchmarking_record_episode_interval,
      train_video_recording_dirpath=video_recording_dirpath,
      train_video_recording_render_mode=video_recording_render_mode,
    )

    """
    task = generate_task(
        task_config['env-id'],
        nbr_parallel_env=task_config['nbr_actor'],
        wrapping_fn=pixel_wrapping_fn,
        test_wrapping_fn=test_pixel_wrapping_fn,
        seed=seed,
        test_seed=100+seed,
        train_video_recording_episode_period=video_recording_episode_period,
        train_video_recording_dirpath=os.path.join(base_path, 'recordings/train/'),
        #test_video_recording_episode_period=video_recording_episode_period,
        #test_video_recording_dirpath=os.path.join(base_path, 'recordings/test/'),
        gathering=True,
    )
    """
    
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////

    agent_config['nbr_actor'] = task_config['nbr_actor']

    regym.RegymSummaryWriterPath = base_path 
    #regym.RegymSummaryWriter = GlobalSummaryWriter(base_path)
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
    
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////
    
    config = {
        'task':task_config, 
        'agent': agent_config,
        'seed': seed,
    }
    project_name = task_config['project']
    wandb.init(project=project_name, config=config)
    #wandb.watch(agents[-1].algorithm.model, log='all', log_freq=100, idx=None, log_graph=True)
    
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////

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
        render_mode=video_recording_render_mode
    )

    return trained_agent, task 


def load_configs(config_file_path: str):
    all_configs = yaml.safe_load(open(config_file_path))

    agents_config = all_configs['agents']
    experiment_config = all_configs['experiment']
    envs_config = experiment_config['tasks']

    return experiment_config, agents_config, envs_config


def main():
    #logging.basicConfig(level=logging.INFO)
    logging.getLogger('minerl_patched').setLevel(level=logging.ERROR)
    logger = logging.getLogger('IGLU R2D2 Benchmark')
    logger.setLevel(level=logging.CRITICAL)

    parser = argparse.ArgumentParser(description="IGLU- R2D2 - Test.")
    parser.add_argument("--config", 
        type=str, 
        default="./iglu_r2d2_config.yaml",
    )
    
    parser.add_argument("--seed", 
        type=int, 
        default=10,
    )
 
    parser.add_argument("--project", 
        type=str, 
        default="IGLU",
    )

    parser.add_argument("--path_suffix", 
        type=str, 
        default="",
    )
    #parser.add_argument("--simplified_DNC", 
    #    type=str, 
    #    default="False",
    #)
    parser.add_argument("--learning_rate", 
        type=float, 
        help="learning rate",
        default=3e-4,
    )
    parser.add_argument("--weights_decay_lambda", 
        type=float, 
        default=0.0,
    )
    parser.add_argument("--weights_entropy_lambda", 
        type=float, 
        default=0.0, #0.001,
    )
    #parser.add_argument("--DNC_sparse_K", 
    #    type=int, 
    #    default=0,
    #)
    parser.add_argument("--sequence_replay_unroll_length", 
        type=int, 
        default=20,
    )
    parser.add_argument("--sequence_replay_overlap_length", 
        type=int, 
        default=10,
    )
    parser.add_argument("--sequence_replay_burn_in_ratio", 
        type=float, 
        default=0.0,
    )
    parser.add_argument("--listener_rec_period", 
        type=int, 
        default=10,
    )
    parser.add_argument("--n_step", 
        type=int, 
        default=3,
    )
    parser.add_argument("--tau", 
        type=float, 
        default=4e-4,
    )
    parser.add_argument("--nbr_actor", 
        type=int, 
        default=1,
    )
    parser.add_argument("--batch_size", 
        type=int, 
        default=128,
    )
    #parser.add_argument("--critic_arch_feature_dim", 
    #    type=int, 
    #    default=32,
    #)
    parser.add_argument("--train_observation_budget", 
        type=float, 
        default=2e6,
    )


    args = parser.parse_args()
    
    args.sequence_replay_overlap_length = min(
        args.sequence_replay_overlap_length,
        args.sequence_replay_unroll_length-5,
    )

    #args.simplified_DNC = True if "Tr" in args.simplified_DNC else False
    dargs = vars(args)
    
    if args.sequence_replay_burn_in_ratio != 0.0:
        dargs['sequence_replay_burn_in_length'] = int(args.sequence_replay_burn_in_ratio*args.sequence_replay_unroll_length)
        dargs['burn_in'] = True 
    
    dargs['seed'] = int(dargs['seed'])
    
    print(dargs)

    from gpuutils import GpuUtils
    GpuUtils.allocate(required_memory=6000, framework="torch")
    
    config_file_path = args.config #sys.argv[1] #'./atari_10M_benchmark_config.yaml'
    experiment_config, agents_config, tasks_configs = load_configs(config_file_path)
    
    for k,v in dargs.items():
        experiment_config[k] = v
    
    print("Experiment config:")
    print(experiment_config)

    # Generate path for experiment
    base_path = experiment_config['experiment_id']
    if not os.path.exists(base_path): os.makedirs(base_path)

    for task_config in tasks_configs:
        agent_name = task_config['agent-id']
        env_name = task_config['env-id']
        run_name = task_config['run-id']
        path = f'{base_path}/{env_name}/{run_name}/{agent_name}'
        print(f"Tentative Path: -- {path} --")
        agent_config =agents_config[task_config['agent-id']] 
        for k,v in dargs.items():
            task_config[k] = v
            agent_config[k] = v
        
        print("Task config:")
        print(task_config)

        training_process(
            agent_config, 
            task_config,
            benchmarking_interval=int(
                float(
                    experiment_config['benchmarking_interval']
                )
            ),
            benchmarking_episodes=int(
                float(
                    experiment_config['benchmarking_episodes']
                )
            ),
            benchmarking_record_episode_interval=int(
                float(
                    experiment_config['benchmarking_record_episode_interval']
                )
            ) if experiment_config['benchmarking_record_episode_interval']!='None' else None,
            train_observation_budget=int(
                float(
                    experiment_config['train_observation_budget']
                )
            ),
            base_path=path,
            seed=experiment_config['seed'],
        )

def s2b_r2d2_wrap(
    env, 
    clip_reward=False,
    previous_reward_action=True,
    otherplay=False
    ):
    env = s2b_wrap(
      env, 
      combined_actions=False,
      dict_obs_space=False,
    )

    if clip_reward:
        env = ClipRewardEnv(env)

    if previous_reward_action:
        env = PreviousRewardActionInfoMultiAgentWrapper(env=env)
    
    return env

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

        regym.RegymManager = Manager()
        regym.RegymManager.start()

    main()
