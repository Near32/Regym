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
from regym.rl_loops.multiagent_loops.marl_loop import test_agent


def build_EnvironmentModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None) -> Module:
    return EnvironmentModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class EnvironmentModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):
        
        default_input_stream_ids = {
            #"logger":"modules:logger:ref",
            #"logs_dict":"logs_dict",
            
            "iteration":"signals:iteration",

            "current_agents":"modules:current_agents:ref",
        }

        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_id, default_stream in default_input_stream_ids.items():
                if default_id not in input_stream_ids.keys():
                    input_stream_ids[default_id] = default_stream

        super(EnvironmentModule, self).__init__(
            id=id,
            type="EnvironmentModule",
            config=config,
            input_stream_ids=input_stream_ids
        )

        self.init = False 
        
        self.task = self.config['task']
        self.env = self.task.env

        if self.config.get('sad', False):
            self.env = SADVecEnvWrapper(self.env, nbr_actions=self.task.action_dim)
        if self.config.get('vdn', False):
            self.env = VDNVecEnvWrapper(self.env, nbr_players=self.config['nbr_players'])

        self.test_env = self.task.test_env
        if self.config.get('sad', False):
            self.test_env = SADVecEnvWrapper(self.test_env, nbr_actions=self.task.action_dim)
        if self.config.get('vdn', False):
            self.test_env = VDNVecEnvWrapper(self.test_env, nbr_players=self.config['nbr_players'])
        

    def initialisation(self, input_streams_dict: Dict[str,object]) -> None:
        self.init = True
        print("Initialization of Environment Module: ...") 

        self.observations = None 
        self.info = None 

        self.agents = input_streams_dict["current_agents"].agents
        self.sad = self.config.get('sad', False)
        self.vdn = self.config.get('vdn', False)

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

        if isinstance(self.config['sum_writer'], str):
            sum_writer_path = os.path.join(self.config['sum_writer'], 'actor.log')
            self.sum_writer = SummaryWriter(sum_writer_path, flush_secs=1)
        else:
            self.sum_writer = self.config['sum_writer']

        for agent in self.agents:
            agent_algo = getattr(agent, "algorithm", None)
            if agent_algo is None:  continue
            if agent.training:
                agent_algo.summary_writer = self.sum_writer
            else:
                agent_algo.summary_writer = None 

        self.epoch = 0 

        print("Initialization of Environment Module: DONE")
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}
        outputs_stream_dict["new_trajectories_published"] = False 

        if not self.init:
            self.initialisation(input_streams_dict)

        if self.observations is None:
            self.observations, self.info = self.env.reset(env_configs=self.config.get('env_configs', None))
    
        actions = [
            agent.take_action(
                state=self.observations[agent_idx],
                infos=self.info[agent_idx]
            )
            for agent_idx, agent in enumerate(self.agents)
        ]
        
        succ_observations, reward, done, succ_info = self.env.step(actions)


        if self.config['training']:
            for agent_idx, agent in enumerate(self.agents):
                if agent.training:
                    agent.handle_experience(
                        s=self.observations[agent_idx],
                        a=actions[agent_idx],
                        r=reward[agent_idx],
                        succ_s=succ_observations[agent_idx],
                        done=done,
                        infos=self.info[agent_idx],
                    )

        if self.sad:
            actions = [
                a["action"]
                for a in actions
            ]

        for actor_index in range(self.nbr_actors):
            self.obs_count += 1
            # pbar.update(1)

            for hook in self.config['step_hooks']:
                for agent in self.agents:
                    hook(self.env, agent, self.obs_count)

            # Bookkeeping of the actors whose episode just ended:
            done_condition = ('real_done' in succ_info[0][actor_index] \
                and succ_info[0][actor_index]['real_done']) \
            or ('real_done' not in succ_info[0][actor_index] \
                and done[actor_index])
            if done_condition:
                self.update_count = self.agents[0].get_update_count()
                self.episode_count += 1
                self.episode_count_record += 1
                succ_observations, succ_info = self.env.reset(env_configs=self.config.get('env_configs', None), env_indices=[actor_index])
                for agent_idx, agent in enumerate(self.agents):
                    agent.reset_actors(indices=[actor_index])
                
                # Logging:
                self.trajectories.append(self.per_actor_per_player_trajectories[actor_index])

                # Only care about logging player 0:
                player_id = 0 
                traj = self.trajectories[-1][player_id]
                self.total_returns.append(sum([ exp[2] for exp in traj]))
                self.positive_total_returns.append(sum([ exp[2] if exp[2]>0 else 0.0 for exp in traj]))
                self.total_int_returns.append(sum([ exp[3] for exp in traj]))
                self.episode_lengths.append(len(traj))

                if self.sum_writer is not None:
                    self.sum_writer.add_scalar('Training/TotalReturn', self.total_returns[-1], self.episode_count)
                    self.sum_writer.add_scalar('PerObservation/TotalReturn', self.total_returns[-1], self.obs_count)
                    self.sum_writer.add_scalar('PerUpdate/TotalReturn', self.total_returns[-1], self.update_count)
                    
                    self.sum_writer.add_scalar('Training/PositiveTotalReturn', self.positive_total_returns[-1], self.episode_count)
                    self.sum_writer.add_scalar('PerObservation/PositiveTotalReturn', self.positive_total_returns[-1], self.obs_count)
                    self.sum_writer.add_scalar('PerUpdate/PositiveTotalReturn', self.positive_total_returns[-1], self.update_count)
                    
                    if actor_index == 0:
                        self.sample_episode_count += 1
                    #sum_writer.add_scalar(f'data/reward_{actor_index}', total_returns[-1], sample_episode_count)
                    #sum_writer.add_scalar(f'PerObservation/Actor_{actor_index}_Reward', total_returns[-1], obs_count)
                    #sum_writer.add_scalar(f'PerObservation/Actor_{actor_index}_PositiveReward', positive_total_returns[-1], obs_count)
                    #sum_writer.add_scalar(f'PerUpdate/Actor_{actor_index}_Reward', total_returns[-1], self.update_count)
                    #sum_writer.add_scalar('Training/TotalIntReturn', total_int_returns[-1], episode_count)
                    self.sum_writer.flush()

                if len(self.trajectories) >= self.nbr_actors:
                    mean_total_return = sum( self.total_returns) / len(self.trajectories)
                    std_ext_return = math.sqrt( sum( [math.pow( r-mean_total_return ,2) for r in self.total_returns]) / len(self.total_returns) )
                    mean_positive_total_return = sum( self.positive_total_returns) / len(self.trajectories)
                    std_ext_positive_return = math.sqrt( sum( [math.pow( r-mean_positive_total_return ,2) for r in self.positive_total_returns]) / len(self.positive_total_returns) )
                    mean_total_int_return = sum( self.total_int_returns) / len(self.trajectories)
                    std_int_return = math.sqrt( sum( [math.pow( r-mean_total_int_return ,2) for r in self.total_int_returns]) / len(self.total_int_returns) )
                    mean_episode_length = sum( self.episode_lengths) / len(self.trajectories)
                    std_episode_length = math.sqrt( sum( [math.pow( l-mean_episode_length ,2) for l in self.episode_lengths]) / len(self.episode_lengths) )

                    if self.sum_writer is not None:
                        self.sum_writer.add_scalar('Training/StdIntReturn', std_int_return, self.episode_count // self.nbr_actors)
                        self.sum_writer.add_scalar('Training/StdExtReturn', std_ext_return, self.episode_count // self.nbr_actors)

                        self.sum_writer.add_scalar('Training/MeanTotalReturn', mean_total_return, self.episode_count // self.nbr_actors)
                        self.sum_writer.add_scalar('PerObservation/MeanTotalReturn', mean_total_return, self.obs_count)
                        self.sum_writer.add_scalar('PerUpdate/MeanTotalReturn', mean_total_return, self.update_count)
                        self.sum_writer.add_scalar('Training/MeanPositiveTotalReturn', mean_positive_total_return, self.episode_count // self.nbr_actors)
                        self.sum_writer.add_scalar('PerObservation/MeanPositiveTotalReturn', mean_positive_total_return, self.obs_count)
                        self.sum_writer.add_scalar('PerUpdate/MeanPositiveTotalReturn', mean_positive_total_return, self.update_count)
                        self.sum_writer.add_scalar('Training/MeanTotalIntReturn', mean_total_int_return, self.episode_count // self.nbr_actors)

                        self.sum_writer.add_scalar('Training/MeanEpisodeLength', mean_episode_length, self.episode_count // self.nbr_actors)
                        self.sum_writer.add_scalar('PerObservation/MeanEpisodeLength', mean_episode_length, self.obs_count)
                        self.sum_writer.add_scalar('PerUpdate/MeanEpisodeLength', mean_episode_length, self.update_count)
                        self.sum_writer.add_scalar('Training/StdEpisodeLength', std_episode_length, self.episode_count // self.nbr_actors)
                        self.sum_writer.add_scalar('PerObservation/StdEpisodeLength', std_episode_length, self.obs_count)
                        self.sum_writer.add_scalar('PerUpdate/StdEpisodeLength', std_episode_length, self.update_count)
                        self.sum_writer.flush()

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

            
            for player_index in range(self.nbr_players):
                pa_obs = self.observations[player_index][actor_index:actor_index+1]
                pa_a = actions[player_index][actor_index:actor_index+1]
                pa_r = reward[player_index][actor_index:actor_index+1]
                pa_succ_obs = succ_observations[player_index][actor_index:actor_index+1]
                pa_done = done[actor_index:actor_index+1]
                pa_int_r = 0.0
                
                """
                pa_info = _extract_from_rnn_states(
                    self.info[player_index],
                    actor_index,
                    post_process_fn=None
                )
                """
                pa_info = self.info[player_index][actor_index]

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
                    sum_writer=self.sum_writer,
                    iteration=self.obs_count,
                    base_path=self.config['base_path'],
                    save_traj=save_traj,
                    render_mode=self.config['render_mode'],
                    save_traj_length_divider=self.config['save_traj_length_divider'],
                )

        outputs_stream_dict["observations"] = copy.deepcopy(self.observations)
        outputs_stream_dict["info"] = copy.deepcopy(self.info)
        outputs_stream_dict["actions"] = actions 
        outputs_stream_dict["reward"] = reward 
        outputs_stream_dict["done"] = done 
        outputs_stream_dict["succ_observations"] = succ_observations
        outputs_stream_dict["succ_info"] = succ_info

        self.observations = copy.deepcopy(succ_observations)
        self.info = copy.deepcopy(succ_info)

        outputs_stream_dict["signals:mode"] = 'train'
        outputs_stream_dict["signals:epoch"] = self.epoch
        
        if self.obs_count >= self.config["max_obs_count"]:
            outputs_stream_dict["signals:done_training"] = True 
            outputs_stream_dict["signals:trained_agents"] = self.agents 

            if self.sum_writer is not None:
                self.sum_writer.flush()
            
            self.env.close()
            self.test_env.close()
            self.init = False 

        return copy.deepcopy(outputs_stream_dict)
            


    
