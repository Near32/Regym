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
from symbolic_behaviour_benchmark.rule_based_agents import build_WrappedPositionallyDisentangledSpeakerAgent 

from regym.util.wrappers import ClipRewardEnv, PreviousRewardActionInfoMultiAgentWrapper

import ray

from regym.modules import EnvironmentModule, CurrentAgentsModule
from regym.modules import MARLEnvironmentModule, RLAgentModule

from regym.modules import ReconstructionFromHiddenStateModule, MultiReconstructionFromHiddenStateModule
from rl_hiddenstate_policy import RLHiddenStatePolicy

from regym.pubsub_manager import PubSubManager

import wandb
import argparse


def make_rl_pubsubmanager(
    agents,
    config, 
    logger=None,
    load_path=None,
    save_path=None,
    speaker_rec=False,
    speaker_rec_biasing=False, 
    listener_rec=False,
    listener_rec_biasing=False, 
    listener_comm_rec=False,
    listener_comm_rec_biasing=False, 
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
    pipelined = True #False
    use_multi_rec = False
    #if len(sys.argv) > 2:
    #  pipelined = any(['pipelined' in arg for arg in sys.argv])
    if len(sys.argv) >2:
        use_multi_rec = any(['multi_rec' in arg for arg in sys.argv])

    modules = config.pop("modules")

    cam_id = "current_agents"
    modules[cam_id] = CurrentAgentsModule(
        id=cam_id,
        agents=agents
    )

    if pipelined:
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
   
    def build_signal_to_reconstruct_from_trajectory_fn(
        traj: List[List[Any]],
        player_id:int,
        ) -> List[torch.Tensor]:
        labels = []
        for exp in traj[player_id]:
            labels.append(torch.from_numpy((1+exp[0])*0.5))
        return labels
    def build_comm_to_reconstruct_from_trajectory_fn(
        traj: List[List[Any]],
        player_id:int,
        ) -> List[torch.Tensor]:
        likelihoods = []
        previous_com = None
        for exp in traj[player_id]:
            current_com = torch.from_numpy(exp[-1]['communication_channel'])
            if previous_com is None:    previous_com = current_com

            target_pred = torch.cat([previous_com, current_com], dim=-1)
            likelihoods.append(target_pred)

            previous_com = current_com
        return likelihoods


    rec_p0_id = "Reconstruction_player0"
    rec_p0_input_stream_ids = {
      "logs_dict":"logs_dict",
      "losses_dict":"losses_dict",
      "epoch":"signals:epoch",
      "mode":"signals:mode",

      "trajectories":f"modules:{envm_id}:trajectories",
      "filtering_signal":f"modules:{envm_id}:new_trajectories_published",

      "current_agents":"modules:current_agents:ref",  
    }

    if speaker_rec_biasing:
      print("WARNING: Biasing for Speaker's Reconstruction.")
    else:
      print("WARNING: NOT biasing Speaker's Reconstruction.")
    
    rec_p0_config = {
      "biasing":speaker_rec_biasing,
      "nbr_players":len(agents),
      "player_id":0,
      'use_cuda':True,
      "signal_to_reconstruct_dim": 4*3,
      "hiddenstate_policy": RLHiddenStatePolicy(
          agent=agents[0],
          node_id_to_extract=node_id_to_extract,
      ),
      "build_signal_to_reconstruct_from_trajectory_fn": build_signal_to_reconstruct_from_trajectory_fn,
    }
    
    if speaker_rec and not(use_multi_rec):
      modules[rec_p0_id] = ReconstructionFromHiddenStateModule(
        id=rec_p0_id,
        config=rec_p0_config,
        input_stream_ids=rec_p0_input_stream_ids,
      )

    rec_p1_id = "Reconstruction_player1"
    rec_p1_input_stream_ids = {
      "logs_dict":"logs_dict",
      "losses_dict":"losses_dict",
      "epoch":"signals:epoch",
      "mode":"signals:mode",

      "trajectories":f"modules:{envm_id}:trajectories",
      "filtering_signal":f"modules:{envm_id}:new_trajectories_published",

      "current_agents":"modules:current_agents:ref",  
    }
    
    if listener_rec_biasing:
      print("WARNING: Biasing for Listener's Reconstruction.")
    else:
      print("WARNING: NOT biasing Listener's Reconstruction.")
    
    rec_p1_config = {
      "biasing":listener_rec_biasing,
      "nbr_players":len(agents),
      "player_id":1,
      'use_cuda':True,
      "signal_to_reconstruct_dim": 4*3,
      "hiddenstate_policy": RLHiddenStatePolicy(
          agent=agents[-1],
          node_id_to_extract=node_id_to_extract,
      ),
      "build_signal_to_reconstruct_from_trajectory_fn": build_signal_to_reconstruct_from_trajectory_fn,
    }
    
    if listener_rec and not(use_multi_rec):
      modules[rec_p1_id] = ReconstructionFromHiddenStateModule(
        id=rec_p1_id,
        config=rec_p1_config,
        input_stream_ids=rec_p1_input_stream_ids,
      )
    
    comm_rec_p1_id = "CommReconstruction_player1"
    comm_rec_p1_input_stream_ids = {
      "logs_dict":"logs_dict",
      "losses_dict":"losses_dict",
      "epoch":"signals:epoch",
      "mode":"signals:mode",

      "trajectories":f"modules:{envm_id}:trajectories",
      "filtering_signal":f"modules:{envm_id}:new_trajectories_published",

      "current_agents":"modules:current_agents:ref",  
    }

    if listener_comm_rec_biasing:
      print("WARNING: Biasing for Listener's Communication Reconstruction.")
    else:
      print("WARNING: NOT biasing Listener's Communication Reconstruction.")
    
    def comm_accuracy_pre_process_fn(
        pred:torch.Tensor, 
        target:torch.Tensor,
        ):
        # Reshape into (sentence_length, vocab_size):
        target = target.reshape(-1, 7)
        pred = pred.reshape(-1, 7)
        
        # Retrieve target idx:
        target_idx = target.max(dim=-1, keepdim=True)[1]
        
        pred_distr = pred.softmax(dim=-1)

        #acc = ((pred-0.1<=target).float()+(pred+0.1>=target).float())==2).gather(
        '''
        acc = (pred_distr>=0.5).float().gather(
            dim=-1,
            index=target_idx,
        )
        '''
        pred_idx = pred.max(dim=-1, keepdim=True)[1]
        acc = (target_idx == pred_idx).float().reshape(1,-1)
        # (1, sentence_length)
        
        return acc

    comm_rec_p1_config = {
      "biasing":listener_comm_rec_biasing,
      "nbr_players":len(agents),
      "player_id":1,
      'use_cuda':True,
      "signal_to_reconstruct_dim": 7*2,
      "hiddenstate_policy": RLHiddenStatePolicy(
          agent=agents[-1],
          node_id_to_extract=node_id_to_extract, 
      ),
      "build_signal_to_reconstruct_from_trajectory_fn": build_comm_to_reconstruct_from_trajectory_fn,
      "accuracy_pre_process_fn":comm_accuracy_pre_process_fn,
    }
    
    if listener_comm_rec and not(use_multi_rec):
        modules[comm_rec_p1_id] = ReconstructionFromHiddenStateModule(
            id=comm_rec_p1_id,
            config=comm_rec_p1_config,
            input_stream_ids=comm_rec_p1_input_stream_ids,
        )
    
    if use_multi_rec:
        if speaker_rec:
            raise NotImplementedError
        multi_rec_p1_id = 'multi_rec_p1'
        rec_dicts = {}
        rec_p1_config = {
            "signal_to_reconstruct_dim":4*3,
            "build_signal_to_reconstruct_from_trajectory_fn":build_signal_to_reconstruct_from_trajectory_fn,
        }
        rec_dicts[rec_p1_id] = rec_p1_config
        comm_rec_p1_config = {
            "signal_to_reconstruct_dim": 7*2,
            "build_signal_to_reconstruct_from_trajectory_fn": build_comm_to_reconstruct_from_trajectory_fn,
            "accuracy_pre_process_fn":comm_accuracy_pre_process_fn,
        }
        rec_dicts[comm_rec_p1_id] = comm_rec_p1_config
        modules[multi_rec_p1_id] = MultiReconstructionFromHiddenStateModule(
            id=multi_rec_p1_id,
            config={
                "biasing":listener_rec_biasing or listener_comm_rec_biasing,
                "nbr_players":len(agents),
                "player_id":1,
                "use_cuda":True,
                "hiddenstate_policy":RLHiddenStatePolicy(
                    agent=agents[-1],
                    node_id_to_extract=node_id_to_extract,
                ),
                "rec_dicts": rec_dicts,
            },
            input_stream_ids=rec_p1_input_stream_ids,
        )


    pipelines = config.pop("pipelines")
    
    pipelines["rl_loop_0"] = [
        envm_id,
    ]
    if pipelined:
      for rlam_id in rlam_ids:
        pipelines['rl_loop_0'].append(rlam_id)
    if use_multi_rec and (listener_rec or listener_comm_rec):
        pipelines["rl_loop_0"].append(multi_rec_p1_id)
    else:
        if speaker_rec:
            pipelines["rl_loop_0"].append(rec_p0_id)
        if listener_rec:
            pipelines["rl_loop_0"].append(rec_p1_id)
        if listener_comm_rec:
            pipelines["rl_loop_0"].append(comm_rec_p1_id)
     
 

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
                       speaker_rec=False,
                       listener_rec=False,
                       listener_comm_rec=False,
                       speaker_rec_biasing=False, 
                       listener_rec_biasing=False, 
                       listener_comm_rec_biasing=False, 
                       node_id_to_extract="hidden",
                       ):
    pubsub = True #False

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
      config['save_traj_length_divider'] =1
      config['sad'] = sad 
      config['vdn'] = vdn
      config['otherplay'] = otherplay
      config['nbr_players'] = 2

      # Hooks:
      ## S2B accuracy hook:
      acc_buffers = {}
      nbr_actors = task.env.get_nbr_envs()
      def acc_hook(
        sum_writer,
        env,
        agents,
        env_output_dict,
        obs_count,
        input_streams_dict,
        outputs_stream_dict,
        ):
        logs_dict = input_streams_dict['logs_dict']
        info = env_output_dict['succ_info']
        for iidx in range(len(info[0])):
          s2b_mode = info[0][iidx]['mode']
          end_of_mode = info[0][iidx]['end_of_mode']
          if end_of_mode:
            if s2b_mode not in acc_buffers:  acc_buffers[s2b_mode] = []
            acc_buffers[s2b_mode].append(info[0][iidx]['running_accuracy'])
            #logs_dict[f"S2B/Accuracy/{iidx}/{s2b_mode}"] = info[0][iidx]['running_accuracy']
        #sum_writer.add_scalar(f"S2B/Accuracy/{iidx}/{s2b_mode}", info[0][iidx]['running_accuracy'], obs_count)
        for s2b_mode in acc_buffers:
          if len(acc_buffers[s2b_mode])>=nbr_actors:
            values = np.asarray(acc_buffers[s2b_mode])
            meanv = values.mean()
            stdv = values.std()
            logs_dict[f"S2B/Accuracy/Mean/{s2b_mode}"] = meanv #sum(acc_buffers[s2b_mode])/nbr_actors
            logs_dict[f"S2B/Accuracy/Std/{s2b_mode}"] = stdv #sum(acc_buffers[s2b_mode])/nbr_actors
            median_value = np.nanpercentile(
                values,
                q=50,
                axis=None,
                interpolation="nearest"
            )
            q1_value = np.nanpercentile(
                values,
                q=25,
                axis=None,
                interpolation="lower"
            )
            q3_value = np.nanpercentile(
                values,
                q=75,
                axis=None,
                interpolation="higher"
            )
            iqr = q3_value-q1_value
              
            logs_dict[f"S2B/Accuracy/Median/{s2b_mode}"] = median_value
            logs_dict[f"S2B/Accuracy/Q1/{s2b_mode}"] = q1_value
            logs_dict[f"S2B/Accuracy/Q3/{s2b_mode}"] = q3_value
            logs_dict[f"S2B/Accuracy/IQR/{s2b_mode}"] = iqr

            #sum_writer.add_histogram(f"S2B/Accuracy/{s2b_mode}", values, obs_count*nbr_actors)

            acc_buffers[s2b_mode] = []
        return   
      
      step_hooks.append(acc_hook)
      config['step_hooks'] = step_hooks

      pubsubmanager = make_rl_pubsubmanager(
        agents=agents,
        config=config,
        speaker_rec=speaker_rec,
        listener_rec=listener_rec,
        listener_comm_rec=listener_comm_rec,
        speaker_rec_biasing=speaker_rec_biasing,
        listener_rec_biasing=listener_rec_biasing,
        listener_comm_rec_biasing=listener_comm_rec_biasing,
        logger=sum_writer,
        node_id_to_extract=node_id_to_extract,
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

    try:
        for agent in trained_agents:
            agent.save(with_replay_buffer=save_replay_buffer)
            print(f"Agent saved at: {agent.save_path}")
    except Exception as e:
        print(e)

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
    
    test_only = task_config.get('test_only', False)
    path_suffix = task_config.get('path_suffix', None)
    if path_suffix=='None':  path_suffix=None
    pubsub = task_config.get('pubsub', True)
    
    speaker_rec = task_config.get('speaker_rec', False)
    listener_rec = task_config.get('listener_rec', False)
    listener_comm_rec = task_config.get('listener_comm_rec', False)
    speaker_rec_biasing = task_config.get('speaker_rec_biasing', False)
    listener_rec_biasing = task_config.get('listener_rec_biasing', False)
    listener_comm_rec_biasing = task_config.get('listener_comm_rec_biasing', False)
    node_id_to_extract=  task_config.get('node_id_to_extract', "hidden") #"memory"
    
    player2_harvest = task_config.get('player2_harvest', False)
    
    use_rule_based_agent = task_config.get('use_rule_based_agent', False )
    use_speaker_rule_based_agent = task_config.get('use_speaker_rule_based_agent', False)
    
    if use_rule_based_agent:
      agent_config['vdn'] = False
      agent_config['sad'] = False 
      task_config['vdn'] = False 
      task_config['vdn'] = False 

    if len(sys.argv) > 2:
      override_nite = [idx for idx, arg in enumerate(sys.argv) if "--node_id_to_extract" in arg]
      if len(override_nite):
          node_id_to_extract = sys.argv[override_nite[0]+1]
          print(f"NEW NODE ID TO EXTRACT FOR REC: {node_id_to_extract}")

      override_seed_argv_idx = [idx for idx, arg in enumerate(sys.argv) if '--new_seed' in arg]
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

    if test_only:
      base_path = os.path.join(base_path,"TESTING")
    else:
      base_path = os.path.join(base_path,"TRAINING")
    
    if use_rule_based_agent:
      base_path = os.path.join(base_path, f"WithPosDis{'Speaker' if use_speaker_rule_based_agent else 'Listener'}RBAgent")

    if pubsub:
      base_path = os.path.join(base_path,"PUBSUB")
    else:
      base_path = os.path.join(base_path,"NOPUBSUB")
    
    if speaker_rec:
      base_path = os.path.join(base_path,f"SpeakerReconstructionFrom-{node_id_to_extract}-{'+Biasing-1p3' if speaker_rec_biasing else ''}-BigArch")
    if listener_rec:
      base_path = os.path.join(base_path,f"ListenerReconstructionFrom-{node_id_to_extract}-{'+Biasing-1p0' if listener_rec_biasing else ''}-BigArch")
    if listener_comm_rec:
      base_path = os.path.join(base_path,f"ListenerCommunicationChannelReconstructionFrom-{node_id_to_extract}-{'+Biasing-1p0' if listener_comm_rec_biasing else ''}-BigArch")
     
  
    if task_config["otherplay"]:
      base_path = os.path.join(base_path,"OtherPlay")
    
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
        if use_rule_based_agent:
            if use_speaker_rule_based_agent:
                rb_agent = build_WrappedPositionallyDisentangledSpeakerAgent( 
                    player_idx=0,
                    action_space_dim=task.env.action_space.n, 
                    vocab_size=task.env.unwrapped_env.unwrapped.vocab_size,
                    max_sentence_length=task.env.unwrapped_env.unwrapped.max_sentence_length,
                    nbr_communication_rounds=task.env.unwrapped_env.unwrapped.nbr_communication_rounds,
                    nbr_latents=task.env.unwrapped_env.unwrapped.nbr_latents,
        
                )
                agents = [rb_agent, agent]
            else:
                rb_agent = build_WrappedPositionallyDisentangledListenerAgent( 
                    player_idx=1,
                    action_space_dim=task.env.action_space.n, 
                    vocab_size=task.env.unwrapped_env.unwrapped.vocab_size,
                    max_sentence_length=task.env.unwrapped_env.unwrapped.max_sentence_length,
                    nbr_communication_rounds=task.env.unwrapped_env.unwrapped.nbr_communication_rounds,
                    nbr_latents=task.env.unwrapped_env.unwrapped.nbr_latents,
        
                )
                agents = [agent, rb_agent]
        else:
            agents = [agent, agent.get_async_actor(training=player2_harvest)]
            # We can create non-training or training async actors.
            # If traininging, then their experience is added to the replay buffer
            # of the main agent, which might have some advantanges
            # -given that it proposes decorrelated data-, but it may
            # also have unknown disadvantages. Needs proper investigation.
    
    config = {
        'task':task_config, 
        'agent': agent_config,
        'seed': seed,
    }
    wandb.init(project='META_RG_S2B', config=config)
    #wandb.watch(agents[-1].algorithm.model, log='all', log_freq=100, idx=None, log_graph=True)
    
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
      speaker_rec=speaker_rec,
      listener_rec=listener_rec,
      listener_comm_rec=listener_comm_rec,
      speaker_rec_biasing=speaker_rec_biasing,
      listener_rec_biasing=listener_rec_biasing,
      listener_comm_rec_biasing=listener_comm_rec_biasing,
      node_id_to_extract=node_id_to_extract,
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

    parser = argparse.ArgumentParser(description="S2B - Test.")
    parser.add_argument("--config", 
        type=str, 
        default="./s2b_2shots_r2d2_dnc_sad_vdn_benchmark_config.yaml",
    )
    
    #parser.add_argument("--speaker_rec", type=str, default="False",)
    #parser.add_argument("--listener_rec", type=str, default="False",)
    #parser.add_argument("--listener_comm_rec", type=str, default="False",)
    #parser.add_argument("--speaker_rec_biasing", type=str, default="False",)
    #parser.add_argument("--listener_rec_biasing", type=str, default="False",)
    #parser.add_argument("--listener_comm_rec_biasing", type=str, default="False",)
    #parser.add_argument("--node_id_to_extract", type=str, default="hidden",) #"memory"
    #parser.add_argument("--player2_harvest", type=str, default="False",)
    parser.add_argument("--use_rule_based_agent", type=str, default="False ",)
    parser.add_argument("--use_speaker_rule_based_agent", type=str, default="False",)
    
    parser.add_argument("--seed", 
        type=int, 
        default=10,
    )
 
    parser.add_argument("--path_suffix", 
        type=str, 
        default="",
    )
    parser.add_argument("--simplified_DNC", 
        type=str, 
        default="False",
    )
    parser.add_argument("--learning_rate", 
        type=float, 
        help="learning rate",
        default=1e-3,
    )
    parser.add_argument("--weights_decay_lambda", 
        type=float, 
        default=0.0,
    )
    parser.add_argument("--weights_entropy_lambda", 
        type=float, 
        default=0.0,
    )
    parser.add_argument("--DNC_sparse_K", 
        type=int, 
        default=0,
    )
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
        default=5e5,
    )


    args = parser.parse_args()
    
    args.sequence_replay_overlap_length = min(
        args.sequence_replay_overlap_length,
        args.sequence_replay_unroll_length-5,
    )

    args.simplified_DNC = True if "Tr" in args.simplified_DNC else False
    args.use_rule_based_agent = True if "Tr" in args.use_rule_based_agent else False
    args.use_speaker_rule_based_agent = True if "Tr" in args.use_speaker_rule_based_agent else False
    
    dargs = vars(args)
    
    if args.sequence_replay_burn_in_ratio != 0.0:
        dargs['sequence_replay_burn_in_length'] = int(args.sequence_replay_burn_in_ratio*args.sequence_replay_unroll_length)
        dargs['burn_in'] = True 
    
    dargs['seed'] = int(dargs['seed'])
    
    print(dargs)

    from gpuutils import GpuUtils
    GpuUtils.allocate(required_memory=10000, framework="torch")
    
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
            benchmarking_interval=int(float(experiment_config['benchmarking_interval'])),
            benchmarking_episodes=int(float(experiment_config['benchmarking_episodes'])),
            benchmarking_record_episode_interval=int(float(experiment_config['benchmarking_record_episode_interval'])) if experiment_config['benchmarking_record_episode_interval']!='None' else None,
            train_observation_budget=int(float(experiment_config['train_observation_budget'])),
            base_path=path,
            seed=experiment_config['seed'],
        )

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
