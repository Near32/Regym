from typing import Dict, List 

import os
import math
import copy
import time
from tqdm import tqdm
import numpy as np

import regym
from tensorboardX import SummaryWriter
from regym.util.wrappers import VDNVecEnvWrapper
from regym.util.wrappers import SADVecEnvWrapper

from regym.rl_algorithms.utils import _extract_from_rnn_states

from torch.multiprocessing import Process 
import ray 

import sys
import gc
import pdb
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
#forkedPdb = ForkedPdb()

from regym.modules.module import Module
from regym.rl_loops.multiagent_loops.wandb_marl_loop import test_agent

import wandb 


def build_MARLEnvironmentModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None) -> Module:
    return MARLEnvironmentModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class MARLEnvironmentModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):
        
        default_input_stream_ids = {
            #"logger":"modules:logger:ref",
            "logs_dict":"logs_dict",
            
            "iteration":"signals:iteration",

            "current_agents":"modules:current_agents:ref",
            "player_0":"modules:rl_agent_0:ref",
        }

        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_id, default_stream in default_input_stream_ids.items():
                if default_id not in input_stream_ids.keys():
                    input_stream_ids[default_id] = default_stream

        super(MARLEnvironmentModule, self).__init__(
            id=id,
            type="MARLEnvironmentModule",
            config=config,
            input_stream_ids=input_stream_ids
        )

        self.init = False 
        
        self.task = self.config['task']
        self.env = self.task.env

        if self.config.get('sad', False):
            self.env = SADVecEnvWrapper(self.env, nbr_actions=self.task.action_dim, otherplay=self.config.get('otherplay', False))
        if self.config.get('vdn', False):
            self.env = VDNVecEnvWrapper(self.env, nbr_players=self.config['nbr_players'])

        self.test_env = self.task.test_env
        if self.config.get('sad', False):
            self.test_env = SADVecEnvWrapper(self.test_env, nbr_actions=self.task.action_dim, otherplay=self.config.get('otherplay', False))
        if self.config.get('vdn', False):
            self.test_env = VDNVecEnvWrapper(self.test_env, nbr_players=self.config['nbr_players'])
        
        # Create placeholders for players:
        self.nbr_agents = self.config['nbr_players']
        if self.config.get('vdn', False):
            self.nbr_agents = 1 

        for player_idx in range(self.nbr_agents):
            setattr(self, f"player_{player_idx}", dict())

    def initialisation(self, input_streams_dict: Dict[str,object]) -> None:
        self.init = True
        print("Initialization of MARL Environment Module: ...") 

        self.observations = None 
        self.info = None 

        self.agents = input_streams_dict["current_agents"].agents
        self.sad = self.config.get('sad', False)
        self.vdn = self.config.get('vdn', False)


        self.obs_key = "observations"
        self.info_key = "info"
        self.action_key = "actions"
        self.reward_key = "reward" 
        self.done_key = "done" 
        self.succ_obs_key = "succ_observations"
        self.succ_info_key = "succ_info"
        if self.vdn:
            self.obs_key = "vdn_observations"
            self.info_key = "vdn_info"
            self.action_key = "vdn_actions"
            self.reward_key = "vdn_reward" 
            self.done_key = "vdn_done" 
            self.succ_obs_key = "vdn_succ_observations"
            self.succ_info_key = "vdn_succ_info"

        self.nbr_actors = self.env.get_nbr_envs()
        self.nbr_players = self.config['nbr_players']

        self.done = [False]*self.nbr_actors
        
        for agent in self.agents:
            agent.set_nbr_actor(self.nbr_actors)

        self.per_actor_per_player_trajectories = [
            [
                list() for p in range(self.nbr_players)
            ]
            for a in range(self.nbr_actors)
        ]
        self.trajectories = list()
        self.total_returns = list()
        self.positive_total_returns = list()
        self.total_int_returns = list()
        self.episode_lengths = list()

        self.obs_count = self.agents[0].get_experience_count() if hasattr(self.agents[0], "get_experience_count") else 0
        self.episode_count = 0
        self.episode_count_record = 0
        self.sample_episode_count = 0

        self.epoch = 0 

        self.pbar = tqdm(
            total=self.config['max_obs_count'], 
            position=0,
        )
        self.pbar.update(self.obs_count)

        print("Initialization of MARL Environment Module: DONE")
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}
        outputs_stream_dict["new_trajectories_published"] = False 
        outputs_stream_dict['reset_actors'] = []

        if not self.init:
            self.initialisation(input_streams_dict)

        if self.observations is None:
            env_reset_output_dict = self.env.reset(env_configs=self.config.get('env_configs', None))
            self.observations = env_reset_output_dict[self.obs_key]
            self.info = env_reset_output_dict[self.info_key]
            if self.vdn:
                self.nonvdn_observations = env_reset_output_dict["observations"]
                self.nonvdn_info = env_reset_output_dict["info"]
            
            outputs_stream_dict[self.obs_key] = copy.deepcopy(self.observations)
            outputs_stream_dict[self.info_key] = copy.deepcopy(self.info)
            outputs_stream_dict[self.action_key] = None 
            outputs_stream_dict[self.reward_key] = None 
            outputs_stream_dict[self.done_key] = None
            outputs_stream_dict[self.succ_obs_key] = None
            outputs_stream_dict[self.succ_info_key] = None

            if self.vdn:
                outputs_stream_dict["observations"] = copy.deepcopy(self.nonvdn_observations)
                outputs_stream_dict["info"] = copy.deepcopy(self.nonvdn_info)
                outputs_stream_dict["actions"] = None
                outputs_stream_dict["reward"] = None 
                outputs_stream_dict["done"] = None 
                outputs_stream_dict["succ_observations"] = None
                outputs_stream_dict["succ_info"] = None

            for pidx in range(self.nbr_agents):
                pidx_d = getattr(self, f"player_{pidx}")
                pidx_d['observations'] = None
                pidx_d['infos'] = None
                pidx_d['actions'] = None
                pidx_d['succ_observations'] = self.observations[pidx]
                pidx_d['succ_infos'] = self.info[pidx]  
                pidx_d['rewards'] = None
                pidx_d['dones'] = None
            
            outputs_stream_dict["signals:mode"] = 'train'
            outputs_stream_dict["signals:epoch"] = self.epoch
            outputs_stream_dict["signals:done_training"] = False
            
            return copy.deepcopy(outputs_stream_dict)

        """
        actions = [
            getattr(f'player_{player_idx}').actions
            for player_idx in range(self.nbr_agents)
        ]
        """
        actions = [
            input_streams_dict[f'player_{player_idx}'].actions
            for player_idx in range(self.nbr_agents)
        ]

        env_output_dict = self.env.step(actions)
        succ_observations = env_output_dict[self.succ_obs_key]
        reward = env_output_dict[self.reward_key]
        done = env_output_dict[self.done_key]
        succ_info = env_output_dict[self.succ_info_key]

        if self.vdn:
            nonvdn_actions = env_output_dict['actions']
            nonvdn_succ_observations = env_output_dict['succ_observations']
            nonvdn_reward = env_output_dict['reward']
            nonvdn_done = env_output_dict['done']
            nonvdn_succ_info = env_output_dict['succ_info']

        if self.sad and isinstance(actions[0], dict):
            actions = [
                a["action"]
                for a in actions
            ]
        
        for hook in self.config['step_hooks']:
            hook(
                None, #self.sum_writer,
                self.env, 
                self.agents, 
                env_output_dict, 
                self.obs_count, 
                input_streams_dict,
                outputs_stream_dict
            )

        for actor_index in range(self.nbr_actors):
            self.obs_count += 1
            self.pbar.update(1)

            # Bookkeeping of the actors whose episode just ended:
            done_condition = ('real_done' in succ_info[0][actor_index] \
                and succ_info[0][actor_index]['real_done']) \
            or ('real_done' not in succ_info[0][actor_index] \
                and done[actor_index])
            if done_condition:
                self.update_count = self.agents[0].get_update_count()
                self.episode_count += 1
                self.episode_count_record += 1
                env_reset_output_dict = self.env.reset(env_configs=self.config.get('env_configs', None), env_indices=[actor_index])
                succ_observations = env_reset_output_dict[self.obs_key]
                succ_info = env_reset_output_dict[self.info_key]
                if self.vdn:
                    nonvdn_succ_observations = env_reset_output_dict['observations']
                    nonvdn_succ_info = env_reset_output_dict['info']

                """
                for agent_idx, agent in enumerate(self.agents):
                    agent.reset_actors(indices=[actor_index])
                """
                outputs_stream_dict['reset_actors'].append(actor_index)

                # Logging:
                self.trajectories.append(self.per_actor_per_player_trajectories[actor_index])

                # Only care about logging player 0:
                player_id = 0 
                traj = self.trajectories[-1][player_id]
                self.total_returns.append(sum([ exp[2] for exp in traj]))
                self.positive_total_returns.append(sum([ exp[2] if exp[2]>0 else 0.0 for exp in traj]))
                self.total_int_returns.append(sum([ exp[3] for exp in traj]))
                self.episode_lengths.append(len(traj))

                wandb.log({'Training/TotalReturn':  self.total_returns[-1], "episode_count":self.episode_count}, commit=False)
                wandb.log({'PerObservation/TotalReturn':  self.total_returns[-1], "obs_count":self.obs_count}, commit=False)
                wandb.log({'PerUpdate/TotalReturn':  self.total_returns[-1], "update_count":self.update_count}, commit=False)
                
                wandb.log({'Training/PositiveTotalReturn':  self.positive_total_returns[-1], "episode_count":self.episode_count}, commit=False)
                wandb.log({'PerObservation/PositiveTotalReturn':  self.positive_total_returns[-1], "obs_count":self.obs_count}, commit=False)
                wandb.log({'PerUpdate/PositiveTotalReturn':  self.positive_total_returns[-1], "update_count":self.update_count}, commit=False)
                
                if actor_index == 0:
                    self.sample_episode_count += 1
                #wandb.log({f'data/reward_{actor_index}':  total_returns[-1]}) # sample_episode_count, commit=False)
                #wandb.log({f'PerObservation/Actor_{actor_index}_Reward':  total_returns[-1]}) # obs_count, commit=False)
                #wandb.log({f'PerObservation/Actor_{actor_index}_PositiveReward':  positive_total_returns[-1]}) # obs_count, commit=False)
                #wandb.log({f'PerUpdate/Actor_{actor_index}_Reward':  total_returns[-1], "update_count":self.update_count}, commit=False)
                #wandb.log({'Training/TotalIntReturn':  total_int_returns[-1]}) # episode_count, commit=False)

                if len(self.trajectories) >= self.nbr_actors:
                    mean_total_return = sum( self.total_returns) / len(self.trajectories)
                    std_ext_return = math.sqrt( sum( [math.pow( r-mean_total_return ,2) for r in self.total_returns]) / len(self.total_returns) )
                    mean_positive_total_return = sum( self.positive_total_returns) / len(self.trajectories)
                    std_ext_positive_return = math.sqrt( sum( [math.pow( r-mean_positive_total_return ,2) for r in self.positive_total_returns]) / len(self.positive_total_returns) )
                    mean_total_int_return = sum( self.total_int_returns) / len(self.trajectories)
                    std_int_return = math.sqrt( sum( [math.pow( r-mean_total_int_return ,2) for r in self.total_int_returns]) / len(self.total_int_returns) )
                    mean_episode_length = sum( self.episode_lengths) / len(self.trajectories)
                    std_episode_length = math.sqrt( sum( [math.pow( l-mean_episode_length ,2) for l in self.episode_lengths]) / len(self.episode_lengths) )

                    wandb.log({'PerEpisodeBatch/StdIntReturn':  std_int_return, "per_actor_training_step":self.episode_count // self.nbr_actors}, commit=False)
                    wandb.log({'PerEpisodeBatch/StdExtReturn':  std_ext_return, "per_actor_training_step":self.episode_count // self.nbr_actors}, commit=False)

                    wandb.log({'PerEpisodeBatch/MeanTotalReturn':  mean_total_return, "per_actor_training_step":self.episode_count // self.nbr_actors}, commit=False)
                    wandb.log({'PerObservation/MeanTotalReturn':  mean_total_return, "obs_count":self.obs_count}, commit=False)
                    wandb.log({'PerUpdate/MeanTotalReturn':  mean_total_return, "update_count":self.update_count}, commit=False)
                    wandb.log({'PerEpisodeBatch/MeanPositiveTotalReturn':  mean_positive_total_return, "per_actor_training_step":self.episode_count // self.nbr_actors}, commit=False)
                    wandb.log({'PerObservation/MeanPositiveTotalReturn':  mean_positive_total_return, "obs_count":self.obs_count}, commit=False)
                    wandb.log({'PerUpdate/MeanPositiveTotalReturn':  mean_positive_total_return, "update_count":self.update_count}, commit=False)
                    wandb.log({'PerEpisodeBatch/MeanTotalIntReturn':  mean_total_int_return, "per_actor_training_step":self.episode_count // self.nbr_actors}, commit=False)

                    wandb.log({'PerEpisodeBatch/MeanEpisodeLength':  mean_episode_length, "per_actor_training_step":self.episode_count // self.nbr_actors}, commit=False)
                    wandb.log({'PerObservation/MeanEpisodeLength':  mean_episode_length, "obs_count":self.obs_count}, commit=False)
                    wandb.log({'PerUpdate/MeanEpisodeLength':  mean_episode_length, "update_count":self.update_count}, commit=False)
                    wandb.log({'PerEpisodeBatch/StdEpisodeLength':  std_episode_length, "per_actor_training_step":self.episode_count // self.nbr_actors}, commit=False)
                    wandb.log({'PerObservation/StdEpisodeLength':  std_episode_length, "obs_count":self.obs_count}, commit=False)
                    wandb.log({'PerUpdate/StdEpisodeLength':  std_episode_length, "update_count":self.update_count}, commit=False)

                    # bookkeeping:
                    outputs_stream_dict["trajectories"] = copy.deepcopy(self.trajectories)
                    outputs_stream_dict["new_trajectories_published"] = True
                    self.epoch += 1
                    
                    # reset :
                    self.trajectories = list()
                    self.total_returns = list()
                    self.positive_total_returns = list()
                    self.total_int_returns = list()
                    self.episode_lengths = list()

                self.per_actor_per_player_trajectories[actor_index] = [
                    list() for p in range(self.nbr_players)
                ]

            if self.vdn:
                obs = self.nonvdn_observations
                act = nonvdn_actions
                succ_obs = nonvdn_succ_observations
                rew = nonvdn_reward
                d = nonvdn_done
                info = self.nonvdn_info
            else:
                obs = self.observations
                act = actions
                succ_obs = succ_observations
                rew = reward
                d = done
                info = self.info
            
            for player_index in range(self.nbr_players):
                pa_obs = obs[player_index][actor_index:actor_index+1]
                pa_a = act[player_index][actor_index:actor_index+1]
                pa_r = rew[player_index][actor_index:actor_index+1]
                pa_succ_obs = succ_obs[player_index][actor_index:actor_index+1]
                pa_done = d[actor_index:actor_index+1]
                pa_int_r = 0.0
                
                """
                pa_info = _extract_from_rnn_states(
                    self.info[player_index],
                    actor_index,
                    post_process_fn=None
                )
                """
                pa_info = info[player_index][actor_index]

                """
                if getattr(agent.algorithm, "use_rnd", False):
                    get_intrinsic_reward = getattr(agent, "get_intrinsic_reward", None)
                    if callable(get_intrinsic_reward):
                        pa_int_r = agent.get_intrinsic_reward(actor_index)
                """    
                self.per_actor_per_player_trajectories[actor_index][player_index].append( 
                    (pa_obs, pa_a, pa_r, pa_int_r, pa_succ_obs, pa_done, pa_info) 
                )


            if self.config['test_nbr_episode'] != 0 \
            and self.obs_count % self.config['test_obs_interval'] == 0:
                save_traj = False
                if (self.config['benchmarking_record_episode_interval'] is not None \
                    and self.config['benchmarking_record_episode_interval']>0):
                    #save_traj = (self.obs_count%benchmarking_record_episode_interval==0)
                    save_traj = (self.episode_count_record // self.nbr_actors > self.config['benchmarking_record_episode_interval'])
                    if save_traj:
                        self.episode_count_record = 0

                # TECHNICAL DEBT: clone_agent.get_update_count is failing because the update count param is None
                # haven't figured out why is the cloning function making it None...
                test_agent(
                    env=self.test_env,
                    agents=[agent.clone(training=False) for agent in self.agents],
                    update_count=self.agents[0].get_update_count(),
                    nbr_episode=self.config['test_nbr_episode'],
                    #sum_writer=self.sum_writer,
                    iteration=self.obs_count,
                    base_path=self.config['base_path'],
                    save_traj=save_traj,
                    render_mode=self.config['render_mode'],
                    save_traj_length_divider=self.config['save_traj_length_divider'],
                    obs_key=self.obs_key,
                    succ_obs_key=self.succ_obs_key,
                    reward_key=self.reward_key,
                    done_key=self.done_key,
                    info_key=self.info_key,
                    succ_info_key=self.succ_info_key,
                )

                if self.obs_count % 10000 == 0:
                    for agent in self.agents:
                      if not hasattr(agent, 'save'):    continue
                      agent.save(minimal=True)
                      print(f"Agent {agent} saved at: {agent.save_path}")
                    
        wandb.log()

        outputs_stream_dict[self.obs_key] = copy.deepcopy(self.observations)
        outputs_stream_dict[self.info_key] = copy.deepcopy(self.info)
        outputs_stream_dict[self.action_key] = actions 
        outputs_stream_dict[self.reward_key] = reward 
        outputs_stream_dict[self.done_key] = done 
        outputs_stream_dict[self.succ_obs_key] = succ_observations
        outputs_stream_dict[self.succ_info_key] = succ_info

        if self.vdn:
            outputs_stream_dict["observations"] = copy.deepcopy(self.nonvdn_observations)
            outputs_stream_dict["info"] = copy.deepcopy(self.nonvdn_info)
            outputs_stream_dict["actions"] = nonvdn_actions 
            outputs_stream_dict["reward"] = nonvdn_reward 
            outputs_stream_dict["done"] = nonvdn_done 
            outputs_stream_dict["succ_observations"] = nonvdn_succ_observations
            outputs_stream_dict["succ_info"] = nonvdn_succ_info

        # Prepare player dicts for RLAgent modules:
        for pidx in range(self.nbr_agents):
            pidx_d = getattr(self, f"player_{pidx}")
            pidx_d['observations'] = self.observations[pidx]
            pidx_d['infos'] = self.info[pidx] 
            pidx_d['actions'] = actions[pidx]
            pidx_d['succ_observations'] = succ_observations[pidx]
            pidx_d['succ_infos'] = succ_info[pidx] 
            pidx_d['rewards'] = reward[pidx]
            pidx_d['dones'] = done

        self.observations = copy.deepcopy(succ_observations)
        self.info = copy.deepcopy(succ_info)
        if self.vdn:
            self.nonvdn_observations = copy.deepcopy(nonvdn_succ_observations)
            self.nonvdn_info = copy.deepcopy(nonvdn_succ_info)


        outputs_stream_dict["signals:mode"] = 'train'
        outputs_stream_dict["signals:epoch"] = self.epoch

        if self.obs_count >= self.config["max_obs_count"]:
            outputs_stream_dict["signals:done_training"] = True 
            outputs_stream_dict["signals:trained_agents"] = self.agents 
            
            self.env.close()
            self.test_env.close()
            self.init = False

            return outputs_stream_dict 
        else:
            outputs_stream_dict["signals:done_training"] = False
        
        return copy.deepcopy(outputs_stream_dict)
            


    
