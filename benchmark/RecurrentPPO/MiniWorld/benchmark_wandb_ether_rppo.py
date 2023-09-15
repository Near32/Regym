from typing import Dict, Any, Optional, List, Callable

import logging
import yaml
import os
import sys

from tensorboardX import SummaryWriter
from tqdm import tqdm
from functools import partial

import torch
import numpy as np

import regym
from regym.environments import generate_task, EnvType
from regym.util.experiment_parsing import initialize_agents
from regym.util.wrappers import baseline_ther_wrapper

#import babyai
import minigrid
#import miniworld 

from regym.modules import EnvironmentModule, CurrentAgentsModule
from regym.modules import MARLEnvironmentModule, RLAgentModule

from regym.pubsub_manager import PubSubManager

import wandb
import argparse
import random


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

    if task_config["BabyAI_Bot_action_override"]:
        from babyai_bot_module import BabyAIBotModule
        from babyai.bot import Bot
        babyai_bot_id = f"babyai_bot"
        babyai_bot_config = {
            'agent': Bot,
            'player_idx': 0,
            'actions_stream_id':f"modules:rl_agent_0:ref:actions",
        }

        #envm_input_stream_ids[f'player_{aidx}'] = f"modules:{rlam_id}:ref"
        aidx = 0
        babyai_bot_input_stream_ids = {
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
        
        modules[babyai_bot_id] = BabyAIBotModule(
            id=babyai_bot_id,
            config=babyai_bot_config,
            input_stream_ids=babyai_bot_input_stream_ids,
        )
        
        pipelines['rl_loop_0'].append(babyai_bot_id)

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

    config['training'] = True
    config['publish_trajectories'] = False 
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
      baseline_ther_wrapper,
      size=task_config['observation_resize_dim'], 
      skip=task_config['nbr_frame_skipping'], 
      stack=task_config['nbr_frame_stacking'],
      single_life_episode=task_config['single_life_episode'],
      nbr_max_random_steps=task_config['nbr_max_random_steps'],
      clip_reward=task_config['clip_reward'],
      time_limit=task_config['time_limit'],
      max_sentence_length=agent_config['THER_max_sentence_length'],
      vocabulary=agent_config['THER_vocabulary'],
      vocab_size=agent_config['THER_vocab_size'],
      previous_reward_action=task_config['previous_reward_action'],
      observation_key=task_config['observation_key'],
      concatenate_keys_with_obs=task_config['concatenate_keys_with_obs'],
      add_rgb_wrapper=task_config['add_rgb_wrapper'],
      full_obs=task_config['full_obs'],
      single_pick_episode=task_config['single_pick_episode'],
      observe_achieved_goal=task_config['THER_observe_achieved_goal'],
      babyai_mission=task_config['BabyAI_Bot_action_override'],
      miniworld_entity_visibility_oracle=task_config['MiniWorld_entity_visibility_oracle'],
      miniworld_entity_visibility_oracle_top_view=task_config['MiniWorld_entity_visibility_oracle_top_view'],
      language_guided_curiosity=task_config['language_guided_curiosity'],
      coverage_metric=task_config['coverage_metric'],
    )

    test_pixel_wrapping_fn = partial(
      baseline_ther_wrapper,
      size=task_config['observation_resize_dim'], 
      skip=task_config['nbr_frame_skipping'], 
      stack=task_config['nbr_frame_stacking'],
      single_life_episode=False,
      nbr_max_random_steps=task_config['nbr_max_random_steps'],
      clip_reward=False,
      time_limit=task_config['time_limit'],
      max_sentence_length=agent_config['THER_max_sentence_length'],
      vocabulary=agent_config['THER_vocabulary'],
      vocab_size=agent_config['THER_vocab_size'],
      previous_reward_action=task_config['previous_reward_action'],
      observation_key=task_config['observation_key'],
      concatenate_keys_with_obs=task_config['concatenate_keys_with_obs'],
      add_rgb_wrapper=task_config['add_rgb_wrapper'],
      full_obs=task_config['full_obs'],
      single_pick_episode=task_config['single_pick_episode'],
      observe_achieved_goal=task_config['THER_observe_achieved_goal'],
      babyai_mission=task_config['BabyAI_Bot_action_override'],
      miniworld_entity_visibility_oracle=task_config['MiniWorld_entity_visibility_oracle'],
      miniworld_entity_visibility_oracle_top_view=task_config['MiniWorld_entity_visibility_oracle_top_view'],
      language_guided_curiosity=task_config['language_guided_curiosity'],
      coverage_metric=task_config['coverage_metric'],
    )
    
    video_recording_dirpath = os.path.join(base_path,'videos')
    video_recording_render_mode = 'rgb_array'
    if "MiniWorld" in task_config['env-id']:
        import miniworld
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

    if agent_config['THER_use_THER']==False \
    and agent_config['THER_contrastive_training_nbr_neg_examples'] != 0:
        raise NotImplementedError


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


def load_configs(config_file_path: str):
    all_configs = yaml.safe_load(open(config_file_path))

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
    logger = logging.getLogger('Emergent Textual HER Benchmark')

    parser = argparse.ArgumentParser(description="ETHER - Test.")
    parser.add_argument("--config", 
        type=str, 
        default="./babyAI_wandb_benchmark_ETHER_config.yaml",
    )
    
    parser.add_argument("--seed", 
        type=int, 
        default=10,
    )
 
    parser.add_argument("--success_threshold", 
        type=float, 
        default=0.0,
    )
 
    parser.add_argument("--project", 
        type=str, 
        default="ETHER",
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
    #parser.add_argument("--simplified_DNC", 
    #    type=str, 
    #    default="False",
    #)
    parser.add_argument("--r2d2_use_value_function_rescaling", type=str2bool, default="False",)
    
    parser.add_argument("--mini_batch_size", type=int, default=256)
    parser.add_argument("--standardized_adv", type=str2bool, default=True)
    parser.add_argument("--optimization_epochs", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--adam_eps", type=float, default=1e-12)
    parser.add_argument("--gradient_clip", type=float, default=0.5)
    parser.add_argument("--ppo_ratio_clip", type=float, default=0.1)
    parser.add_argument("--discount", type=float, default=0.999)
    parser.add_argument("--intrinsic_discount", type=float, default=0.99)
    parser.add_argument("--value_weight", type=float, default=0.5)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    #parser.add_argument("--single_life_episode", type=str2bool, default="True",)
    #parser.add_argument("--grayscale", type=str2bool, default="False",)
    
    parser.add_argument("--learning_rate", 
        type=float, 
        help="learning rate",
        default=3e-4,
    )
    parser.add_argument("--ther_adam_weight_decay", 
        type=float, 
        default=0.0,
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
    #parser.add_argument("--DNC_sparse_K", 
    #    type=int, 
    #    default=0,
    #)
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
    parser.add_argument("--eps_greedy_alpha", 
        type=float, 
        default=2.0,
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
    parser.add_argument("--replay_capacity", 
        type=float, 
        default=2e4,
    )
    parser.add_argument("--HER_target_clamping",
        type=str2bool, 
        default="False", 
    )
    parser.add_argument("--RP_replay_period", # in episodes
        type=int, 
        default=40, #10 #1
    )
    parser.add_argument("--RP_nbr_training_iteration_per_update", 
        type=int, 
        default=2, 
    )
    parser.add_argument("--RP_replay_capacity", 
        type=float, 
        default=500, #250 #5000
    )
    parser.add_argument("--RP_lock_test_storage", type=str2bool, default=False)
    parser.add_argument("--RP_test_replay_capacity", 
        type=float, 
        default=50, #25 #1000
    )
    parser.add_argument("--RP_min_capacity", 
        type=float, 
        default=32, #1e4
    )
    parser.add_argument("--RP_test_min_capacity", 
        type=float, 
        default=12, #1e4
    )
    parser.add_argument("--RP_predictor_nbr_minibatches", 
        type=int, 
        default=8,
    )
    parser.add_argument("--RP_predictor_batch_size", type=int, default=256)
    parser.add_argument("--RP_predictor_learning_rate", type=float, default=6.25e-5)
    parser.add_argument("--RP_gradient_clip", type=float, default=10.0)
    parser.add_argument("--RP_predictor_accuracy_threshold", 
        type=float, 
        default=0.75,
    )
    parser.add_argument("--RP_predictor_test_train_split_interval",
        type=int,
        default=10,#3 #10 #5
    )

    parser.add_argument("--use_RP", type=str2bool, default="True",)
    parser.add_argument("--RP_use_RP", type=str2bool, default="True",)
    parser.add_argument("--RP_use_PER", type=str2bool, default="False",)
    
    parser.add_argument("--THER_replay_period", # in episodes
        type=int, 
        default=40, #10 #1
    )
    parser.add_argument("--THER_nbr_training_iteration_per_update", 
        type=int, 
        default=2, 
    )
    parser.add_argument("--THER_replay_capacity", 
        type=float, 
        default=500, #250 #5000
    )
    parser.add_argument("--THER_lock_test_storage", type=str2bool, default=False)
    parser.add_argument("--THER_test_replay_capacity", 
        type=float, 
        default=50, #25 #1000
    )
    parser.add_argument("--THER_min_capacity", 
        type=float, 
        default=32, #1e4
    )
    parser.add_argument("--THER_test_min_capacity", 
        type=float, 
        default=12, #1e4
    )
    parser.add_argument("--THER_predictor_nbr_minibatches", 
        type=int, 
        default=8,
    )
    parser.add_argument("--THER_predictor_batch_size", 
        type=int, 
        default=256,
    )
    parser.add_argument("--THER_predictor_accuracy_threshold", 
        type=float, 
        default=0.75,
    )
    parser.add_argument("--THER_predictor_accuracy_safe_to_relabel_threshold", 
        type=float, 
        default=0.5,
    )
    parser.add_argument("--THER_predictor_test_train_split_interval",
        type=int,
        default=10,#3 #10 #5
    )

    parser.add_argument("--goal_oriented", type=str2bool, default="True",)
    parser.add_argument("--use_HER", type=str2bool, default="True",)
    parser.add_argument("--use_THER", type=str2bool, default="True",)
    parser.add_argument("--THER_use_THER", type=str2bool, default="True",)
    parser.add_argument("--THER_use_PER", type=str2bool, default="False",)
    parser.add_argument("--THER_episode_length_reward_shaping", type=str2bool, default="False",)
    parser.add_argument("--THER_observe_achieved_goal", type=str2bool, default="False",)
    parser.add_argument("--single_pick_episode", type=str2bool, default="False",)
    parser.add_argument("--THER_train_contrastively", type=str2bool, default="False",)
    parser.add_argument("--THER_rg_max_sentence_length", type=int, default=10)
    parser.add_argument("--THER_contrastive_training_nbr_neg_examples", type=int, default=0,)
    parser.add_argument("--THER_feedbacks_failure_reward", type=int, default=-1,)
    parser.add_argument("--THER_feedbacks_success_reward", type=int, default=0,)
    parser.add_argument("--THER_relabel_terminal", type=str2bool, default="True",)
    parser.add_argument("--THER_train_on_success", type=str2bool, default="False",)
    parser.add_argument("--THER_predict_PADs", type=str2bool, default="False",)
    parser.add_argument("--THER_filter_predicate_fn", type=str2bool, default="False",)
    parser.add_argument("--THER_filter_out_timed_out_episode", type=str2bool, default="False",)
    parser.add_argument("--THER_timing_out_episode_length_threshold", type=int, default=40,)
    parser.add_argument("--BabyAI_Bot_action_override", type=str2bool, default="False",)
    parser.add_argument("--MiniWorld_entity_visibility_oracle", type=str2bool, default="False",)
    parser.add_argument("--MiniWorld_entity_visibility_oracle_top_view", type=str2bool, default="False",)
    parser.add_argument("--language_guided_curiosity", type=str2bool, default="False",)
    parser.add_argument("--coverage_metric", type=str2bool, default="False",)
    parser.add_argument("--nbr_training_iteration_per_cycle", type=int, default=10)
    parser.add_argument("--nbr_episode_per_cycle", type=int, default=16)
    #parser.add_argument("--critic_arch_feature_dim", 
    #    type=int, 
    #    default=32,
    #)
    
    parser.add_argument("--use_ETHER", type=str2bool, default="True",)
    parser.add_argument("--ETHER_use_ETHER", type=str2bool, default="True",)
    parser.add_argument("--ETHER_use_supervised_training", type=str2bool, default="True",)
    parser.add_argument("--ETHER_use_continuous_feedback", type=str2bool, default=False,)
    parser.add_argument("--ETHER_listener_based_predicated_reward_fn", type=str2bool, default=False,)
    parser.add_argument("--ETHER_rg_sanity_check_compactness_ambiguity_metric", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_training_period", type=int, default=1024)
    parser.add_argument("--ETHER_rg_accuracy_threshold", type=float, default=75)
    parser.add_argument("--ETHER_rg_verbose", type=str2bool, default="True",)
    parser.add_argument("--ETHER_rg_use_cuda", type=str2bool, default="True",)
    parser.add_argument("--ETHER_exp_key", type=str, default="succ_s",)
    parser.add_argument("--ETHER_rg_with_semantic_grounding_metric", type=str2bool, default="False",)
    parser.add_argument("--ETHER_rg_use_semantic_cooccurrence_grounding", type=str2bool, default="False",)
    parser.add_argument("--ETHER_grounding_signal_key", type=str, default="info:desired_goal",)
    parser.add_argument("--ETHER_rg_semantic_cooccurrence_grounding_lambda", type=float, default=1.0)
    parser.add_argument("--ETHER_rg_semantic_cooccurrence_grounding_noise_magnitude", type=float, default=0.0)
    parser.add_argument("--ETHER_split_strategy", type=str, default="divider-1-offset-0",)
    parser.add_argument("--ETHER_replay_capacity", type=int, default=1024)
    parser.add_argument("--ETHER_rg_filter_out_non_unique", type=str2bool, default=False)
    parser.add_argument("--ETHER_lock_test_storage", type=str2bool, default=False)
    parser.add_argument("--ETHER_test_replay_capacity", type=int, default=512)
    parser.add_argument("--ETHER_test_train_split_interval",type=int, default=5)
    parser.add_argument("--ETHER_train_dataset_length", type=intOrNone, default=None)
    parser.add_argument("--ETHER_test_dataset_length", type=intOrNone, default=None)
    parser.add_argument("--ETHER_rg_object_centric_version", type=int, default=1)
    parser.add_argument("--ETHER_rg_descriptive_version", type=str, default=2)
    parser.add_argument("--ETHER_rg_with_color_jitter_augmentation", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_with_gaussian_blur_augmentation", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_egocentric_tr_degrees", type=float, default=15)
    parser.add_argument("--ETHER_rg_egocentric_tr_xy", type=float, default=10)
    parser.add_argument("--ETHER_rg_egocentric", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_nbr_train_distractors", type=int, default=7)
    parser.add_argument("--ETHER_rg_nbr_test_distractors", type=int, default=7)
    parser.add_argument("--ETHER_rg_descriptive", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_descriptive_ratio", type=float, default=0.0)
    parser.add_argument("--ETHER_rg_observability", type=str, default='partial')
    parser.add_argument("--ETHER_rg_max_sentence_length", type=int, default=10)
    parser.add_argument("--ETHER_rg_distractor_sampling", type=str, default='uniform')
    parser.add_argument("--ETHER_rg_object_centric", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_graphtype", type=str, default='straight_through_gumbel_softmax')
    parser.add_argument("--ETHER_rg_vocab_size", type=int, default=32)
    # TODO : integrate this feature in ArchiPredictorSpeaker ...
    parser.add_argument("--ETHER_rg_force_eos", type=str2bool, default=True)
    parser.add_argument("--ETHER_rg_symbol_embedding_size", type=int, default=64)
    parser.add_argument("--ETHER_rg_arch", type=str, default='BN+7x4x3xCNN')
    parser.add_argument("--ETHER_rg_shared_architecture", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_normalize_features", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_agent_loss_type", type=str, default='Hinge')

    parser.add_argument("--ETHER_rg_with_logits_mdl_principle", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_logits_mdl_principle_factor", type=float, default=1.0e-3)
    parser.add_argument("--ETHER_rg_logits_mdl_principle_accuracy_threshold", type=float, help='in percent.', default=10.0)
    
    parser.add_argument("--ETHER_rg_cultural_pressure_it_period", type=int, default=0)
    parser.add_argument("--ETHER_rg_cultural_speaker_substrate_size", type=int, default=1)
    parser.add_argument("--ETHER_rg_cultural_listener_substrate_size", type=int, default=1)
    parser.add_argument("--ETHER_rg_cultural_reset_strategy", type=str, default='uniformSL')
    #"oldestL", # "uniformSL" #"meta-oldestL-SGD"
    parser.add_argument("--ETHER_rg_cultural_pressure_meta_learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--ETHER_rg_iterated_learning_scheme", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_iterated_learning_period", type=int, default=5)
    parser.add_argument("--ETHER_rg_iterated_learning_rehearse_MDL", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_iterated_learning_rehearse_MDL_factor", type=float, default=1.0)
    
    parser.add_argument("--ETHER_rg_obverter_threshold_to_stop_message_generation", type=float, default=0.9)
    parser.add_argument("--ETHER_rg_obverter_nbr_games_per_round", type=int, default=20)
    parser.add_argument("--ETHER_rg_use_obverter_sampling", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_obverter_sampling_round_alternation_only", type=str2bool, default=False)
    
    parser.add_argument("--ETHER_rg_batch_size", type=int, default=32)
    parser.add_argument("--ETHER_rg_dataloader_num_worker", type=int, default=8)
    parser.add_argument("--ETHER_rg_learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--ETHER_rg_weight_decay", type=float, default=0.0)
    parser.add_argument("--ETHER_rg_dropout_prob", type=float, default=0.0)
    parser.add_argument("--ETHER_rg_emb_dropout_prob", type=float, default=0.0)
    parser.add_argument("--ETHER_rg_homoscedastic_multitasks_loss", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_use_feat_converter", type=str2bool, default=True)
    parser.add_argument("--ETHER_rg_use_curriculum_nbr_distractors", type=str2bool, default=False)
    parser.add_argument("--ETHER_rg_init_curriculum_nbr_distractors", type=int, default=1)
    parser.add_argument("--ETHER_rg_nbr_experience_repetition", type=int, default=1)
    parser.add_argument("--ETHER_rg_agent_nbr_latent_dim", type=int, default=32)
    parser.add_argument("--ETHER_rg_symbol_processing_nbr_hidden_units", type=int, default=512)
    
    parser.add_argument("--ETHER_rg_mini_batch_size", type=int, default=32)
    parser.add_argument("--ETHER_rg_optimizer_type", type=str, default='adam')
    parser.add_argument("--ETHER_rg_nbr_epoch_per_update", type=int, default=3)

    parser.add_argument("--ETHER_rg_metric_epoch_period", type=int, default=10024)
    parser.add_argument("--ETHER_rg_dis_metric_epoch_period", type=int, default=10024)
    parser.add_argument("--ETHER_rg_metric_batch_size", type=int, default=16)
    parser.add_argument("--ETHER_rg_metric_fast", type=str2bool, default=True)
    parser.add_argument("--ETHER_rg_parallel_TS_worker", type=int, default=8)
    parser.add_argument("--ETHER_rg_nbr_train_points", type=int, default=1024)
    parser.add_argument("--ETHER_rg_nbr_eval_points", type=int, default=512)
    parser.add_argument("--ETHER_rg_metric_resampling", type=str2bool, default=True)
    parser.add_argument("--ETHER_rg_dis_metric_resampling", type=str2bool, default=True)
    parser.add_argument("--ETHER_rg_seed", type=int, default=1)
    parser.add_argument("--ETHER_rg_metric_active_factors_only", type=str2bool, default=True)
    
    parser.add_argument("--use_ELA", type=str2bool, default="False",)
    parser.add_argument("--ELA_use_ELA", type=str2bool, default="False",)
    parser.add_argument("--ELA_reward_extrinsic_weight", type=float, default=1.0,)
    parser.add_argument("--ELA_reward_intrinsic_weight", type=float, default=1.0,)
    parser.add_argument("--ELA_feedbacks_failure_reward", type=float, default=0,)
    parser.add_argument("--ELA_feedbacks_success_reward", type=float, default=1,)
    parser.add_argument("--ELA_rg_sanity_check_compactness_ambiguity_metric", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_training_period", type=int, default=1024)
    parser.add_argument("--ELA_rg_accuracy_threshold", type=float, default=75)
    parser.add_argument("--ELA_rg_verbose", type=str2bool, default="True",)
    parser.add_argument("--ELA_rg_use_cuda", type=str2bool, default="True",)
    parser.add_argument("--ELA_exp_key", type=str, default="succ_s",)
    parser.add_argument("--ELA_rg_with_semantic_grounding_metric", type=str2bool, default="False",)
    parser.add_argument("--ELA_rg_use_semantic_cooccurrence_grounding", type=str2bool, default="False",)
    parser.add_argument("--ELA_grounding_signal_key", type=str, default="info:desired_goal",)
    parser.add_argument("--ELA_rg_semantic_cooccurrence_grounding_lambda", type=float, default=1.0)
    parser.add_argument("--ELA_rg_semantic_cooccurrence_grounding_noise_magnitude", type=float, default=0.0)
    parser.add_argument("--ELA_split_strategy", type=str, default="divider-1-offset-0",)
    parser.add_argument("--ELA_replay_capacity", type=int, default=1024)
    parser.add_argument("--ELA_lock_test_storage", type=str2bool, default=False)
    parser.add_argument("--ELA_test_replay_capacity", type=int, default=512)
    parser.add_argument("--ELA_test_train_split_interval",type=int, default=5)
    parser.add_argument("--ELA_train_dataset_length", type=intOrNone, default=None)
    parser.add_argument("--ELA_test_dataset_length", type=intOrNone, default=None)
    parser.add_argument("--ELA_rg_object_centric_version", type=int, default=1)
    parser.add_argument("--ELA_rg_descriptive_version", type=str, default=2)
    parser.add_argument("--ELA_rg_with_color_jitter_augmentation", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_with_gaussian_blur_augmentation", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_egocentric_tr_degrees", type=float, default=15)
    parser.add_argument("--ELA_rg_egocentric_tr_xy", type=float, default=10)
    parser.add_argument("--ELA_rg_egocentric", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_nbr_train_distractors", type=int, default=7)
    parser.add_argument("--ELA_rg_nbr_test_distractors", type=int, default=7)
    parser.add_argument("--ELA_rg_descriptive", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_descriptive_ratio", type=float, default=0.0)
    parser.add_argument("--ELA_rg_observability", type=str, default='partial')
    parser.add_argument("--ELA_rg_max_sentence_length", type=int, default=10)
    parser.add_argument("--ELA_rg_distractor_sampling", type=str, default='uniform')
    parser.add_argument("--ELA_rg_object_centric", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_graphtype", type=str, default='straight_through_gumbel_softmax')
    parser.add_argument("--ELA_rg_vocab_size", type=int, default=32)
    # TODO : integrate this feature in ArchiPredictorSpeaker ...
    parser.add_argument("--ELA_rg_force_eos", type=str2bool, default=True)
    parser.add_argument("--ELA_rg_symbol_embedding_size", type=int, default=64)
    parser.add_argument("--ELA_rg_arch", type=str, default='BN+7x4x3xCNN')
    parser.add_argument("--ELA_rg_shared_architecture", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_normalize_features", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_agent_loss_type", type=str, default='Hinge')

    parser.add_argument("--ELA_rg_with_logits_mdl_principle", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_logits_mdl_principle_factor", type=float, default=1.0e-3)
    parser.add_argument("--ELA_rg_logits_mdl_principle_accuracy_threshold", type=float, help='in percent.', default=10.0)
    
    parser.add_argument("--ELA_rg_cultural_pressure_it_period", type=int, default=0)
    parser.add_argument("--ELA_rg_cultural_speaker_substrate_size", type=int, default=1)
    parser.add_argument("--ELA_rg_cultural_listener_substrate_size", type=int, default=1)
    parser.add_argument("--ELA_rg_cultural_reset_strategy", type=str, default='uniformSL')
    #"oldestL", # "uniformSL" #"meta-oldestL-SGD"
    parser.add_argument("--ELA_rg_cultural_pressure_meta_learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--ELA_rg_iterated_learning_scheme", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_iterated_learning_period", type=int, default=5)
    parser.add_argument("--ELA_rg_iterated_learning_rehearse_MDL", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_iterated_learning_rehearse_MDL_factor", type=float, default=1.0)
    
    parser.add_argument("--ELA_rg_obverter_threshold_to_stop_message_generation", type=float, default=0.9)
    parser.add_argument("--ELA_rg_obverter_nbr_games_per_round", type=int, default=20)
    parser.add_argument("--ELA_rg_use_obverter_sampling", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_obverter_sampling_round_alternation_only", type=str2bool, default=False)
    
    parser.add_argument("--ELA_rg_batch_size", type=int, default=32)
    parser.add_argument("--ELA_rg_dataloader_num_worker", type=int, default=8)
    parser.add_argument("--ELA_rg_learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--ELA_rg_weight_decay", type=float, default=0.0)
    parser.add_argument("--ELA_rg_dropout_prob", type=float, default=0.0)
    parser.add_argument("--ELA_rg_emb_dropout_prob", type=float, default=0.0)
    parser.add_argument("--ELA_rg_homoscedastic_multitasks_loss", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_use_feat_converter", type=str2bool, default=True)
    parser.add_argument("--ELA_rg_use_curriculum_nbr_distractors", type=str2bool, default=False)
    parser.add_argument("--ELA_rg_init_curriculum_nbr_distractors", type=int, default=1)
    parser.add_argument("--ELA_rg_nbr_experience_repetition", type=int, default=1)
    parser.add_argument("--ELA_rg_agent_nbr_latent_dim", type=int, default=32)
    parser.add_argument("--ELA_rg_symbol_processing_nbr_hidden_units", type=int, default=512)
    
    parser.add_argument("--ELA_rg_mini_batch_size", type=int, default=32)
    parser.add_argument("--ELA_rg_optimizer_type", type=str, default='adam')
    parser.add_argument("--ELA_rg_nbr_epoch_per_update", type=int, default=3)

    parser.add_argument("--ELA_rg_metric_epoch_period", type=int, default=10024)
    parser.add_argument("--ELA_rg_dis_metric_epoch_period", type=int, default=10024)
    parser.add_argument("--ELA_rg_metric_batch_size", type=int, default=16)
    parser.add_argument("--ELA_rg_metric_fast", type=str2bool, default=True)
    parser.add_argument("--ELA_rg_parallel_TS_worker", type=int, default=8)
    parser.add_argument("--ELA_rg_nbr_train_points", type=int, default=1024)
    parser.add_argument("--ELA_rg_nbr_eval_points", type=int, default=512)
    parser.add_argument("--ELA_rg_metric_resampling", type=str2bool, default=True)
    parser.add_argument("--ELA_rg_dis_metric_resampling", type=str2bool, default=True)
    parser.add_argument("--ELA_rg_seed", type=int, default=1)
    parser.add_argument("--ELA_rg_metric_active_factors_only", type=str2bool, default=True)
    
    parser.add_argument("--time_limit", type=int, default=400,) 
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
    
    if dargs['THER_contrastive_training_nbr_neg_examples'] != 0:
        dargs['THER_train_contrastively'] = True

    if dargs["ETHER_rg_sanity_check_compactness_ambiguity_metric"]:
        import ipdb; ipdb.set_trace()
        dargs["ETHER_grounding_signal_key"] = "info:visible_entities_widx"
        dargs["MiniWorld_entity_visibility_oracle"] = True
        dargs["MiniWorld_entity_visibility_oracle_top_view"] = True
        dargs["ETHER_rg_use_semantic_cooccurrence_grounding"] = False
        print("WARNING :: sanity check in progress for compactness ambiguity metric.")
        print("WARNING :: therefore DISABLING the semantic cooccurrence grounding.")
    
    if dargs["ETHER_listener_based_predicated_reward_fn"]:
        print("WARNING: Listener-based predicated reward fn but NO DESCRIPTIVE RG.")
    
    if dargs["ETHER_rg_obverter_sampling_round_alternation_only"]:
        dargs["ETHER_rg_use_obverter_sampling"] = True

    if dargs['language_guided_curiosity']:
        dargs['coverage_metric'] = True
        dargs["MiniWorld_entity_visibility_oracle"] = True
    
    print(dargs)

    #from gpuutils import GpuUtils
    #GpuUtils.allocate(required_memory=6000, framework="torch")
    
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
        if args.ETHER_rg_max_sentence_length != agent_config['THER_max_sentence_length']:
            dargs['ETHER_rg_max_sentence_length'] = agent_config['THER_max_sentence_length']
            print(f"WARNING: ETHER rg max sentence length is different ({args.ETHER_rg_max_sentence_length}) than config THER max sentence length value, thus, updating it to: {dargs['ETHER_rg_max_sentence_length']}")
            import ipdb; ipdb.set_trace()
        if args.ETHER_rg_vocab_size < agent_config['THER_vocab_size']:
            dargs['ETHER_rg_vocab_size'] = agent_config['THER_vocab_size']
            print(f"WARNING: ETHER rg vocab size is lower ({args.ETHER_rg_vocab_size}) than necessary, updating to: {dargs['ETHER_rg_vocab_size']}")
            import ipdb; ipdb.set_trace()
        for k,v in dargs.items():
            task_config[k] = v
            agent_config[k] = v
            
            if k in task_config.get('env-config', {}):
                task_config['env-config'][k] = v
 
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

if __name__ == '__main__':
    main()
