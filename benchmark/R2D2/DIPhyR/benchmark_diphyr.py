from typing import Dict, Any, Optional, List, Callable

import logging
import yaml
import os
import sys
import time 

from tensorboardX import SummaryWriter
from tqdm import tqdm
from functools import partial

import torch
import numpy as np

import regym
from regym.environments import generate_task, EnvType
from regym.util.experiment_parsing import initialize_agents
from regym.util.wrappers import ClipRewardEnv, PreviousRewardActionInfoMultiAgentWrapper

import diphyrgym
from diphyr_hook import DIPhyRHook

from regym.modules import EnvironmentModule, CurrentAgentsModule
from regym.modules import MARLEnvironmentModule, RLAgentModule

from regym.pubsub_manager import PubSubManager

import wandb
import argparse
import random


def diphyr_r2d2_wrap(
    env, 
    clip_reward=False,
    previous_reward_action=True,
    ):
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


def check_wandb_path_for_agent(file_path, run_path):
    if os.path.exists('./'+file_path):
        print(f"WARNING:CHECKPOINT PATH DUPLICATE EXISTS: ./{file_path}")
        os.remove('./'+file_path)
        print(f"WARNING: DUPLICATE PATH DELETED: ./{file_path}")
    try:
        agent_ref = wandb.restore(name=file_path, run_path=run_path)
    except Exception as e:
        agent_ref = None
        raise e
    agent = None
    offset_episode_count = 0
    if agent_ref is not None:
        print(f"==> loading checkpoint {run_path}/{file_path}")
        agent = torch.load(agent_ref.name)
        os.remove('./'+file_path)
        offset_episode_count = agent.episode_count
        #setattr(agent, 'episode_count', offset_episode_count)
        print(f"==> loaded checkpoint {run_path}/{file_path}")
    return agent, offset_episode_count


def make_rl_pubsubmanager(
    agents,
    config, 
    task_config=None,
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
        'player_idx': aidx,
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
    
    config['success_threshold'] = task_config['success_threshold'] # 0.0
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
    task_config: Dict[str, object],
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

    config['with_early_stopping'] = task_config['with_early_stopping']
    config['publish_trajectories'] = False 
    config['training'] = True
    config['seed'] = task_config['seed'] 
    config['static_envs'] = task_config.get('static_envs', False)
    config['env_configs'] = {'return_info': True} #None
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

    # Hooks:
    ## DIPhyR accuracy hook:
    diphyr_hook = DIPhyRHook(average_window_length=task_config['DIPhyR_average_window_length'])
    config['step_hooks'].append(diphyr_hook.acc_hook)

    agents = [agent]

    pubsubmanager = make_rl_pubsubmanager(
      agents=agents,
      config=config,
      task_config=task_config,
      logger=sum_writer,
    )

    pubsubmanager.train() 

    save_replay_buffer = False
    if len(sys.argv) > 2:
      save_replay_buffer = any(['save_replay_buffer' in arg for arg in sys.argv])

    """
    try:
        for agent in agents:
            agent.save(with_replay_buffer=save_replay_buffer)
            print(f"Agent saved at: {agent.save_path}")
    except Exception as e:
        print(e)
    """

    task.env.close()
    task.test_env.close()

    return agents

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
    env_seed: int = 0,
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
      diphyr_r2d2_wrap,
      clip_reward=task_config['clip_reward'],
      previous_reward_action=task_config.get('previous_reward_action', False),
    )
    
    test_pixel_wrapping_fn = pixel_wrapping_fn

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
      seed=env_seed,
      test_seed=env_seed if task_config['static_envs'] else 100+env_seed,
      static=task_config.get('static_envs', False),
      gathering=True,
      train_video_recording_episode_period=benchmarking_record_episode_interval,
      train_video_recording_dirpath=video_recording_dirpath,
      train_video_recording_render_mode=video_recording_render_mode,
    )

    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////

    agent_config['task_config'] = task_config
    agent_config['nbr_actor'] = task_config['nbr_actor']

    regym.RegymSummaryWriterPath = base_path 
    #regym.RegymSummaryWriter = GlobalSummaryWriter(base_path)
    sum_writer = base_path
    
    save_path1 = os.path.join(base_path,f"./{task_config['agent-id']}.agent")
    if task_config.get("reload", 'None')!='None':
      agent, offset_episode_count = check_path_for_agent(task_config["reload"])
    elif task_config.get("reload_wandb_run_path", 'None') != 'None':
      agent, offset_episode_count = check_wandb_path_for_agent(
        file_path=task_config["reload_wandb_file_path"],
        run_path=task_config["reload_wandb_run_path"],
      ) 
    else:
      agent = None
      offset_episode_count = 0
      #agent, offset_episode_count = check_path_for_agent(save_path1)
    
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
    '''
    wandb.tensorboard.patch(
        save=True, 
        tensorboard_x=True,
    )
    '''
    agent.save_path = os.path.join(wandb.run.dir, "agent_checkpoints")
    os.makedirs(agent.save_path, exist_ok=True)
    agent.save_path += "/checkpoint.agent"
    '''
    wandb.watch(
        agent.algorithm.unwrapped.model, 
        log='all', 
        log_freq=100, 
        idx=None, 
        log_graph=True,
    )
    '''

    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////
    #/////////////////////////////////////////////////////////////////

    trained_agent = train_and_evaluate(
        agent=agent,
        task=task,
        task_config=task_config,
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


def parse_and_update(config_file_path: str, kwargs: Dict[str, Any]):
    config_file = open(config_file_path, 'r')
    lines = config_file.readlines()
    config = ""
    for line in lines:
        if '&' in line:
            key, value = line.split(': ')[:2]
            key = key.strip()
            value = value.strip()
            if key in kwargs:
                if '&' in value:
                    value = value.split('&')[1].split(' ')[1]
                value = value.strip()
                line = line.replace(value, str(kwargs[key]))
        config += line 
    return config


def load_configs(config_file_path: str, kwargs: Dict[str, Any]):
    yaml_str = parse_and_update(config_file_path, kwargs)
    all_configs = yaml.safe_load(yaml_str)

    agents_config = all_configs['agents']
    experiment_config = all_configs['experiment']
    envs_config = experiment_config['tasks']

    return experiment_config, agents_config, envs_config


def str2bool(instr):
    if isinstance(instr, bool):
        return instr
    if isinstance(instr, str):
        instr = instr.lower()
        if 'true' in instr:
            return True
        elif 'false' in instr:
            return False
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def intOrNone(instr):
    if instr is None:
        return None
    return int(instr)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('DIPhyR Benchmark')

    parser = argparse.ArgumentParser(description="DIPhyR - Test.")
    parser.add_argument("--yaml_config", 
        type=str, 
        default="./diphyr_benchmark_R2D2_config.yaml",
    )
    
    parser.add_argument("--seed", 
        type=int, 
        default=10,
    )
    parser.add_argument("--env_seed", 
        type=int, 
        default=20,
    )
    parser.add_argument("--with_early_stopping", type=str2bool, default=False) 
    parser.add_argument("--static_envs", type=str2bool, default=False) 
    parser.add_argument("--use_cuda", type=str2bool, default=False) 
    parser.add_argument("--benchmarking_interval", type=float, default=5.0e4)
    parser.add_argument("--benchmarking_record_episode_interval", type=int, default=1e10)
    parser.add_argument("--success_threshold", 
        type=float, 
        default=0.0,
    )
    parser.add_argument("--use_grammar", 
        type=str2bool, 
        default=False,
    )
 
    parser.add_argument("--notrace", 
        type=str2bool, 
        default=False,
    )
 
    parser.add_argument("--DIPhyR_average_window_length", 
        type=int, 
        default=128,
    )
 
    parser.add_argument("--project", 
        type=str, 
        default="DIPhyR",
    )

    parser.add_argument("--test_only", type=str2bool, default="False")
    parser.add_argument("--reload_wandb_run_path", 
        type=str, 
        default="None",
    )

    parser.add_argument("--reload_wandb_file_path", 
        type=str, 
        default="None",
    )

    parser.add_argument("--path_suffix", 
        type=str, 
        default="",
    )
    parser.add_argument("--r2d2_use_value_function_rescaling", type=str2bool, default="False",)
    
    parser.add_argument("--PER_use_rewards_in_priority", type=str2bool, default="False")
    parser.add_argument("--PER_alpha", type=float, default=0.9)
    parser.add_argument("--PER_beta", type=float, default=0.6)
    parser.add_argument("--sequence_replay_PER_eta", type=float, default=0.9)
    parser.add_argument("--PER_compute_initial_priority", type=str2bool, default="False")
    parser.add_argument("--saving_interval", 
        type=float, 
        default=1e15,
    )
    parser.add_argument("--learning_rate", 
        type=float, 
        help="learning rate",
        default=3e-4,
    )
    parser.add_argument("--adam_weight_decay", 
        type=float, 
        default=0.0,
    )
    parser.add_argument("--weights_decay_lambda", 
        type=float, 
        default=0.0,
    )
    parser.add_argument("--weights_entropy_lambda", 
        type=float, 
        default=0.0, #0.0,
    )
    parser.add_argument("--sequence_replay_use_online_states", type=str2bool, default="True")
    parser.add_argument("--sequence_replay_use_zero_initial_states", type=str2bool, default="False")
    parser.add_argument("--sequence_replay_store_on_terminal", type=str2bool, default="False")
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
    parser.add_argument("--eps_greedy_alpha", type=float, default=7.0)
    parser.add_argument("--epsstart", type=float, default=1.0)
    parser.add_argument("--epsend", type=float, default=0.1)
    parser.add_argument("--epsdecay", type=int, default=10000)
    parser.add_argument("--n_step", 
        type=int, 
        default=3,
    )
    parser.add_argument("--inverted_tau", 
        type=str, 
        default="None",
    )
    parser.add_argument("--tau", 
        type=str, 
        default="None",
    )
    parser.add_argument("--nbr_actor", 
        type=int, 
        default=4,
    )
    parser.add_argument("--nbr_minibatches", 
        type=int, 
        default=8,
    )
    parser.add_argument("--batch_size", 
        type=int, 
        default=256,
    )
    parser.add_argument("--min_capacity", 
        type=float, 
        default=1e3,
    )
    parser.add_argument("--min_handled_experiences", 
        type=float, 
        default=1,
    )
    parser.add_argument("--replay_capacity", 
        type=float, 
        default=2e4,
    )
    parser.add_argument("--HER_target_clamping",
        type=str2bool, 
        default="False", 
    )
    parser.add_argument("--nbr_training_iteration_per_cycle", type=int, default=10)
    parser.add_argument("--nbr_episode_per_cycle", type=int, default=16)
    
    parser.add_argument("--time_limit", type=int, default=400,) 
    parser.add_argument("--train_observation_budget", 
        type=float, 
        default=2e6,
    )

    parser.add_argument("--use_ORG", type=str2bool, default="False",)
    parser.add_argument("--ORG_rg_tau0", type=float, default=0.2,)
    parser.add_argument("--ORG_rg_init_agent_states_with_online_states", type=str2bool, default="False",)
    parser.add_argument("--ORG_rg_reset_listener_each_training", type=str2bool, default="False",)
    parser.add_argument("--ORG_use_model_in_speaker", type=str2bool, default="True",)
    parser.add_argument("--ORG_trainable_speaker", type=str2bool, default="False",)
    parser.add_argument("--ORG_use_model_in_speaker_pipeline", type=str, default="listener",)
    parser.add_argument("--ORG_use_model_in_speaker_generator", type=str, default="LMModule",)
    parser.add_argument("--ORG_use_model_in_listener", type=str2bool, default="True",)
    parser.add_argument("--ORG_trainable_listener", type=str2bool, default="True",)
    parser.add_argument("--ORG_use_model_in_listener_pipeline", type=str, default="listener",)
    parser.add_argument("--ORG_use_model_in_listener_generator", type=str, default="LMModule",)
    # TODO: setup the postprocessing fn for speaker
    parser.add_argument("--ORG_with_Oracle_speaker", type=str2bool, default="False",)
    #parser.add_argument("--ORG_with_Oracle_type", type=str, default="visible-entities",)
    parser.add_argument("--ORG_with_Oracle_listener", type=str2bool, default="False",)
    parser.add_argument("--ORG_use_ORG", type=str2bool, default="True",)
    parser.add_argument("--ORG_with_S2B", type=str2bool, default="False",)
    parser.add_argument("--ORG_use_supervised_training", type=str2bool, default="True",)
    parser.add_argument("--ORG_use_continuous_feedback", type=str2bool, default=False,)
    #parser.add_argument("--ORG_listener_based_predicated_reward_fn", type=str2bool, default=False,)
    parser.add_argument("--ORG_with_compactness_ambiguity_metric", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_sanity_check_compactness_ambiguity_metric", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_training_period", type=int, default=4)
    parser.add_argument("--ORG_rg_accuracy_threshold", type=float, default=75)
    parser.add_argument("--ORG_rg_verbose", type=str2bool, default="True",)
    parser.add_argument("--ORG_rg_use_cuda", type=str2bool, default="True",)
    parser.add_argument("--ORG_exp_key", type=str, default="s",)
    #parser.add_argument("--semantic_embedding_init", type=str, default="none",)
    #parser.add_argument("--semantic_prior_mixing", type=str, default="multiplicative",)
    #parser.add_argument("--semantic_prior_mixing_with_detach", type=str2bool, default=True)
    parser.add_argument("--ORG_rg_with_semantic_grounding_metric", type=str2bool, default="False",)
    parser.add_argument("--ORG_rg_use_semantic_cooccurrence_grounding", type=str2bool, default="False",)
    parser.add_argument("--ORG_grounding_signal_key", type=str, default="info:desired_goal",)
    parser.add_argument("--ORG_rg_semantic_cooccurrence_grounding_lambda", type=float, default=1.0)
    parser.add_argument("--ORG_rg_semantic_cooccurrence_grounding_noise_magnitude", type=float, default=0.0)
    parser.add_argument("--ORG_rg_semantic_cooccurrence_grounding_semantic_level", type=str2bool, default="False",)
    parser.add_argument("--ORG_rg_semantic_cooccurrence_grounding_semantic_level_ungrounding", type=str2bool, default="False",)
    parser.add_argument("--ORG_rg_semantic_cooccurrence_grounding_sentence_level", type=str2bool, default="True",)
    parser.add_argument("--ORG_rg_semantic_cooccurrence_grounding_sentence_level_ungrounding", type=str2bool, default="False",)
    parser.add_argument("--ORG_rg_semantic_cooccurrence_grounding_sentence_level_lambda", type=float, default=1.0)
    parser.add_argument("--ORG_split_strategy", type=str, default="divider-1-offset-0",)
    parser.add_argument("--ORG_replay_capacity", type=int, default=16)
    parser.add_argument("--ORG_rg_filter_out_non_unique", type=str2bool, default=False)
    # WARNING: very important to lock the test replay in order to ensure that 
    # observed variations are not due to variations of the test data.
    # If the test set is large enough, then it does not matter.
    parser.add_argument("--ORG_lock_test_storage", type=str2bool, default=True)
    parser.add_argument("--ORG_test_replay_capacity", type=int, default=4)
    parser.add_argument("--ORG_test_train_split_interval",type=int, default=5)
    parser.add_argument("--ORG_train_dataset_length", type=intOrNone, default=None)
    parser.add_argument("--ORG_test_dataset_length", type=intOrNone, default=None)
    parser.add_argument("--ORG_rg_object_centric_version", type=int, default=1)
    parser.add_argument("--ORG_rg_distractor_sampling_scheme_version", type=int, default=2)
    parser.add_argument("--ORG_rg_descriptive_version", type=int, default=2)
    parser.add_argument("--ORG_rg_with_color_jitter_augmentation", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_color_jitter_prob", type=float, default=0)
    parser.add_argument("--ORG_rg_with_gaussian_blur_augmentation", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_gaussian_blur_prob", type=float, default=0)
    parser.add_argument("--ORG_rg_egocentric_tr_degrees", type=float, default=30)
    parser.add_argument("--ORG_rg_egocentric_tr_xy", type=float, default=10)
    parser.add_argument("--ORG_rg_egocentric", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_egocentric_prob", type=float, default=0)
    parser.add_argument("--ORG_rg_nbr_train_distractors", type=int, default=0)
    parser.add_argument("--ORG_rg_nbr_test_distractors", type=int, default=0)
    parser.add_argument("--ORG_rg_descriptive", type=str2bool, default=True)
    parser.add_argument("--ORG_rg_descriptive_ratio", type=float, default=0.0)
    parser.add_argument("--ORG_rg_observability", type=str, default='partial')
    parser.add_argument("--ORG_rg_max_sentence_length", type=int, default=10)
    parser.add_argument("--ORG_rg_distractor_sampling", type=str, default='uniform')
    parser.add_argument("--ORG_rg_object_centric", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_object_centric_type", type=str, default="hard")
    parser.add_argument("--ORG_rg_graphtype", type=str, default='obverter')
    parser.add_argument("--ORG_rg_vocab_size", type=int, default=32)
    # TODO : integrate this feature in ArchiPredictorSpeaker ...
    parser.add_argument("--ORG_rg_force_eos", type=str2bool, default=True)
    parser.add_argument("--ORG_rg_symbol_embedding_size", type=int, default=64)
    parser.add_argument("--ORG_rg_arch", type=str, default='BN+MLP')#'BN+7x4x3xCNN')
    parser.add_argument("--ORG_rg_shared_architecture", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_normalize_features", type=str2bool, default=False, 
        #help="Will be toggled on automatically if using (listener) continuous feedback without descriptive RG.",
    )
    parser.add_argument("--ORG_rg_agent_loss_type", type=str, default='Hinge')
    parser.add_argument("--ORG_rg_use_aita_sampling", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_aita_update_epoch_period", type=int, default=32)
    parser.add_argument("--ORG_rg_aita_levenshtein_comprange", type=float, default=1.0)

    parser.add_argument("--ORG_rg_with_logits_mdl_principle", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_logits_mdl_principle_factor", type=float, default=0.0)
    parser.add_argument("--ORG_rg_logits_mdl_principle_accuracy_threshold", type=float, help='in percent.', default=10.0)
    
    parser.add_argument("--ORG_rg_cultural_pressure_it_period", type=int, default=0)
    parser.add_argument("--ORG_rg_cultural_speaker_substrate_size", type=int, default=1)
    parser.add_argument("--ORG_rg_cultural_listener_substrate_size", type=int, default=1)
    parser.add_argument("--ORG_rg_cultural_reset_strategy", type=str, default='uniformSL')
    #"oldestL", # "uniformSL" #"meta-oldestL-SGD"
    parser.add_argument("--ORG_rg_cultural_pressure_meta_learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--ORG_rg_iterated_learning_scheme", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_iterated_learning_period", type=int, default=5)
    parser.add_argument("--ORG_rg_iterated_learning_rehearse_MDL", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_iterated_learning_rehearse_MDL_factor", type=float, default=1.0)
    
    parser.add_argument("--ORG_rg_obverter_threshold_to_stop_message_generation", type=float, default=0.9)
    parser.add_argument("--ORG_rg_obverter_nbr_games_per_round", type=int, default=20)
    parser.add_argument("--ORG_rg_use_obverter_sampling", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_obverter_sampling_round_alternation_only", type=str2bool, default=False)
    
    parser.add_argument("--ORG_rg_batch_size", type=int, default=2)
    parser.add_argument("--ORG_rg_dataloader_num_worker", type=int, default=1)
    parser.add_argument("--ORG_rg_learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--ORG_rg_weight_decay", type=float, default=0.0)
    parser.add_argument("--ORG_rg_l1_weight_decay", type=float, default=0.0)
    parser.add_argument("--ORG_rg_l2_weight_decay", type=float, default=0.0)
    parser.add_argument("--ORG_rg_dropout_prob", type=float, default=0.0)
    parser.add_argument("--ORG_rg_emb_dropout_prob", type=float, default=0.0)
    parser.add_argument("--ORG_rg_homoscedastic_multitasks_loss", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_use_feat_converter", type=str2bool, default=True)
    parser.add_argument("--ORG_rg_distractor_sampling_with_replacement", type=str2bool, default=True)
    parser.add_argument("--ORG_rg_use_curriculum_nbr_distractors", type=str2bool, default=False)
    parser.add_argument("--ORG_rg_init_curriculum_nbr_distractors", type=int, default=1)
    parser.add_argument("--ORG_rg_nbr_experience_repetition", type=int, default=1)
    parser.add_argument("--ORG_rg_agent_nbr_latent_dim", type=int, default=32)
    parser.add_argument("--ORG_rg_symbol_processing_nbr_hidden_units", type=int, default=512)
    
    parser.add_argument("--ORG_rg_mini_batch_size", type=int, default=32)
    parser.add_argument("--ORG_rg_optimizer_type", type=str, default='adam')
    parser.add_argument("--ORG_rg_nbr_epoch_per_update", type=int, default=3)

    parser.add_argument("--ORG_rg_metric_epoch_period", type=int, default=10024)
    parser.add_argument("--ORG_rg_dis_metric_epoch_period", type=int, default=10024)
    parser.add_argument("--ORG_rg_metric_batch_size", type=int, default=16)
    parser.add_argument("--ORG_rg_metric_fast", type=str2bool, default=True)
    parser.add_argument("--ORG_rg_parallel_TS_worker", type=int, default=8)
    parser.add_argument("--ORG_rg_nbr_train_points", type=int, default=1024)
    parser.add_argument("--ORG_rg_nbr_eval_points", type=int, default=512)
    parser.add_argument("--ORG_rg_metric_resampling", type=str2bool, default=True)
    parser.add_argument("--ORG_rg_dis_metric_resampling", type=str2bool, default=True)
    parser.add_argument("--ORG_rg_seed", type=int, default=1)
    parser.add_argument("--ORG_rg_metric_active_factors_only", type=str2bool, default=True)
    parser.add_argument("--ORG_rg_with_ortho_metric", type=str2bool, default=False)
    
    args = parser.parse_args()
    
    args.sequence_replay_overlap_length = min(
        args.sequence_replay_overlap_length,
        args.sequence_replay_unroll_length-5,
    )

    dargs = vars(args)
    
    if args.sequence_replay_burn_in_ratio != 0.0:
        dargs['sequence_replay_burn_in_length'] = int(args.sequence_replay_burn_in_ratio*args.sequence_replay_unroll_length)
        dargs['burn_in'] = True 
    
    dargs['seed'] = int(dargs['seed'])
    
    if args.use_ORG:
        dargs['preprocessed_observation_shape'] = [32] #dummy value [args.nbr_latents]
        from regym.rl_algorithms.algorithms.wrappers.org_wrapper import (
            DIPhyR_preprocess_utter_oracle_fn,
            DIPhyR_preprocess_reason_detach_fn,
            DIPhyR_postprocess_utter_oracle_fn,
            DIPhyR_postprocess_reason_fn,
        ) 
        dargs['ORG_speaker_preprocess_utter_fn'] = DIPhyR_preprocess_utter_oracle_fn 
        dargs['ORG_speaker_preprocess_reason_fn'] = DIPhyR_preprocess_reason_detach_fn
        dargs['ORG_speaker_postprocess_utter_fn'] = DIPhyR_postprocess_utter_oracle_fn
        dargs['ORG_speaker_postprocess_reason_fn'] = DIPhyR_postprocess_reason_fn
    
        dargs['ORG_listener_preprocess_utter_fn'] = DIPhyR_preprocess_utter_oracle_fn 
        dargs['ORG_listener_preprocess_reason_fn'] = DIPhyR_preprocess_reason_detach_fn
        dargs['ORG_listener_postprocess_utter_fn'] = DIPhyR_postprocess_utter_oracle_fn
        dargs['ORG_listener_postprocess_reason_fn'] = DIPhyR_postprocess_reason_fn
        
        dargs['ORG_use_model_in_speaker_pipelines'] = {'utter':'speaker', 'reason':'listener'}
        dargs['ORG_use_model_in_speaker_generators'] = {'utter':'LMModule', 'reason':'LMModule'}
        dargs['ORG_use_model_in_listener_pipelines'] = {'utter':'speaker', 'reason':'listener'} 
        dargs['ORG_use_model_in_listener_generators'] = {'utter':'LMModule', 'reason':'LMModule'}

        dargs['ORG_rg_demonstration_dataset_extra_keys'] = {
            "experiences":"infos:prompt",
        }

    print(dargs)

    #from gpuutils import GpuUtils
    #GpuUtils.allocate(required_memory=6000, framework="torch")
    
    config_file_path = args.yaml_config #sys.argv[1] #'./atari_10M_benchmark_config.yaml'
    experiment_config, agents_config, tasks_configs = load_configs(
        config_file_path,
        kwargs=dargs,
    )
    
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
            
            if k in task_config.get('env-config', {}):
                task_config['env-config'][k] = v
 	
        ac_pointer = None
        ac_queue = [agent_config]
        while len(ac_queue):
            ac_pointer = ac_queue.pop(0)
            if isinstance(ac_pointer, dict):
                for k in ac_pointer.keys():
                    if isinstance(ac_pointer[k], dict):
                        ac_queue.append(ac_pointer[k])
                    else:
                        overriden = False
                        for karg in dargs.keys():
                            #if k in karg:
                            if k == karg:
                                print(f"WARNING: overriding {k} \n = {ac_pointer[k]} \n --> {dargs[karg]}")
                                ac_pointer[k] = dargs[karg]
                                overriden = True 
                                break
                        if not overriden:
                            print(f"WARNING: {k} was NOT OVERRIDEN")
                
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
            env_seed=experiment_config['env_seed'],
        )

if __name__ == '__main__':
    if False: #True:
      #torch.multiprocessing.freeze_support()
      torch.multiprocessing.set_start_method("forkserver")#, force=True)
      #torch.multiprocessing.set_start_method("spawn", force=True)
      #ray.init() #local_mode=True)
      #ray.init(local_mode=True)
      
      #from regym import CustomManager as Manager
      #from multiprocessing.managers import SyncManager, MakeProxyType, public_methods
      #regym.RegymManager = Manager()
      regym.AlgoManager = mp.Manager()
      #regym.AlgoManager.start()
    
    main()

