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
from regym.util.wrappers import SADVecEnvWrapper as SADEnvWrapper

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
        
        self.task = self.config['task']
        self.env = self.task.env

        if self.config.get('sad', False):
            self.env = SADEnvWrapper(self.env, nbr_actions=self.task.action_dim)
        if self.config.get('vdn', False):
            self.env = VDNVecEnvWrapper(self.env, nbr_players=self.config['nbr_players'])

        self.test_env = self.task.test_env
        if self.config.get('sad', False):
            self.test_env = SADEnvWrapper(self.test_env, nbr_actions=self.task.action_dim)
        if self.config.get('vdn', False):
            self.test_env = VDNVecEnvWrapper(self.test_env, nbr_players=self.config['nbr_players'])
        

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}

        agents = input_streams_dict["current_agents"].agents
        env = self.env 
        test_env = self.test_env
        sad = self.config.get('sad', False)
        vdn = self.config.get('vdn', False)

        observations, info = self.env.reset(env_configs=self.config['env_configs'])
        nbr_actors = self.env.get_nbr_envs()
        
        done = [False]*nbr_actors
    
        per_actor_trajectories = [list() for i in range(nbr_actors)]
        trajectories = list()
        total_returns = list()
        positive_total_returns = list()
        total_int_returns = list()
        episode_lengths = list()

        obs_count = agents[0].get_experience_count() if hasattr(agents[0], "get_experience_count") else 0
        episode_count = 0
        episode_count_record = 0
        sample_episode_count = 0

        """
        pbar = tqdm(
            total=self.config['max_obs_count'], 
            position=0,
        )
        pbar.update(obs_count)
        """

        if isinstance(self.config['sum_writer'], str):
            sum_writer_path = os.path.join(self.config['sum_writer'], 'actor.log')
            sum_writer = SummaryWriter(sum_writer_path, flush_secs=1)
            for agent in agents:
                if agent.training:
                    agent.algorithm.summary_writer = sum_writer
                else:
                    agent.algorithm.summary_writer = None 

        # Assumption: :param intput_stream_dict: is a reference to a dictionnary
        # that is updated outside of this function and therefore keeps up-to-date
        # information about subscribed data stream.

        while True: 
            actions = [
                agent.take_action(
                    state=observations[agent_idx],
                    infos=info[agent_idx]
                )
                for agent_idx, agent in enumerate(agents)
            ]
            
            succ_observations, reward, done, succ_info = env.step(actions)


            if self.config['training']:
                for agent_idx, agent in enumerate(agents):
                    if agent.training:
                        agent.handle_experience(
                            s=observations[agent_idx],
                            a=actions[agent_idx],
                            r=reward[agent_idx],
                            succ_s=succ_observations[agent_idx],
                            done=done,
                            infos=info[agent_idx],
                        )

            if sad:
                actions = [
                    a["action"]
                    for a in actions
                ]

            for actor_index in range(nbr_actors):
                obs_count += 1
                # pbar.update(1)

                for hook in self.config['step_hooks']:
                    for agent in agents:
                        hook(env, agent, obs_count)

                # Bookkeeping of the actors whose episode just ended:
                done_condition = ('real_done' in succ_info[0][actor_index] and succ_info[0][actor_index]['real_done']) or ('real_done' not in succ_info[0][actor_index] and done[actor_index])
                if done_condition:
                    update_count = agents[0].get_update_count()
                    episode_count += 1
                    episode_count_record += 1
                    succ_observations, succ_info = env.reset(env_configs=env_configs, env_indices=[actor_index])
                    for agent_idx, agent in enumerate(agents):
                        agent.reset_actors(indices=[actor_index])
                    
                    # Logging:
                    trajectories.append(per_actor_trajectories[actor_index])
                    total_returns.append(sum([ exp[2] for exp in trajectories[-1]]))
                    positive_total_returns.append(sum([ exp[2] if exp[2]>0 else 0.0 for exp in trajectories[-1]]))
                    total_int_returns.append(sum([ exp[3] for exp in trajectories[-1]]))
                    episode_lengths.append(len(trajectories[-1]))

                    if sum_writer is not None:
                        sum_writer.add_scalar('Training/TotalReturn', total_returns[-1], episode_count)
                        sum_writer.add_scalar('PerObservation/TotalReturn', total_returns[-1], obs_count)
                        sum_writer.add_scalar('PerUpdate/TotalReturn', total_returns[-1], update_count)
                        
                        sum_writer.add_scalar('Training/PositiveTotalReturn', positive_total_returns[-1], episode_count)
                        sum_writer.add_scalar('PerObservation/PositiveTotalReturn', positive_total_returns[-1], obs_count)
                        sum_writer.add_scalar('PerUpdate/PositiveTotalReturn', positive_total_returns[-1], update_count)
                        
                        if actor_index == 0:
                            sample_episode_count += 1
                        #sum_writer.add_scalar(f'data/reward_{actor_index}', total_returns[-1], sample_episode_count)
                        #sum_writer.add_scalar(f'PerObservation/Actor_{actor_index}_Reward', total_returns[-1], obs_count)
                        #sum_writer.add_scalar(f'PerObservation/Actor_{actor_index}_PositiveReward', positive_total_returns[-1], obs_count)
                        #sum_writer.add_scalar(f'PerUpdate/Actor_{actor_index}_Reward', total_returns[-1], update_count)
                        #sum_writer.add_scalar('Training/TotalIntReturn', total_int_returns[-1], episode_count)
                        sum_writer.flush()

                    if len(trajectories) >= nbr_actors:
                        mean_total_return = sum( total_returns) / len(trajectories)
                        std_ext_return = math.sqrt( sum( [math.pow( r-mean_total_return ,2) for r in total_returns]) / len(total_returns) )
                        mean_positive_total_return = sum( positive_total_returns) / len(trajectories)
                        std_ext_positive_return = math.sqrt( sum( [math.pow( r-mean_positive_total_return ,2) for r in positive_total_returns]) / len(positive_total_returns) )
                        mean_total_int_return = sum( total_int_returns) / len(trajectories)
                        std_int_return = math.sqrt( sum( [math.pow( r-mean_total_int_return ,2) for r in total_int_returns]) / len(total_int_returns) )
                        mean_episode_length = sum( episode_lengths) / len(trajectories)
                        std_episode_length = math.sqrt( sum( [math.pow( l-mean_episode_length ,2) for l in episode_lengths]) / len(episode_lengths) )

                        if sum_writer is not None:
                            sum_writer.add_scalar('Training/StdIntReturn', std_int_return, episode_count // nbr_actors)
                            sum_writer.add_scalar('Training/StdExtReturn', std_ext_return, episode_count // nbr_actors)

                            sum_writer.add_scalar('Training/MeanTotalReturn', mean_total_return, episode_count // nbr_actors)
                            sum_writer.add_scalar('PerObservation/MeanTotalReturn', mean_total_return, obs_count)
                            sum_writer.add_scalar('PerUpdate/MeanTotalReturn', mean_total_return, update_count)
                            sum_writer.add_scalar('Training/MeanPositiveTotalReturn', mean_positive_total_return, episode_count // nbr_actors)
                            sum_writer.add_scalar('PerObservation/MeanPositiveTotalReturn', mean_positive_total_return, obs_count)
                            sum_writer.add_scalar('PerUpdate/MeanPositiveTotalReturn', mean_positive_total_return, update_count)
                            sum_writer.add_scalar('Training/MeanTotalIntReturn', mean_total_int_return, episode_count // nbr_actors)

                            sum_writer.add_scalar('Training/MeanEpisodeLength', mean_episode_length, episode_count // nbr_actors)
                            sum_writer.add_scalar('PerObservation/MeanEpisodeLength', mean_episode_length, obs_count)
                            sum_writer.add_scalar('PerUpdate/MeanEpisodeLength', mean_episode_length, update_count)
                            sum_writer.add_scalar('Training/StdEpisodeLength', std_episode_length, episode_count // nbr_actors)
                            sum_writer.add_scalar('PerObservation/StdEpisodeLength', std_episode_length, obs_count)
                            sum_writer.add_scalar('PerUpdate/StdEpisodeLength', std_episode_length, update_count)
                            sum_writer.flush()

                        # reset :
                        trajectories = list()
                        total_returns = list()
                        positive_total_returns = list()
                        total_int_returns = list()
                        episode_lengths = list()

                    per_actor_trajectories[actor_index] = list()

                # Only care about agent 0's trajectory:
                pa_obs = observations[0][actor_index]
                pa_a = actions[0][actor_index]
                pa_r = reward[0][actor_index]
                pa_succ_obs = succ_observations[0][actor_index]
                pa_done = done[actor_index]
                pa_int_r = 0.0

                """
                if getattr(agent.algorithm, "use_rnd", False):
                    get_intrinsic_reward = getattr(agent, "get_intrinsic_reward", None)
                    if callable(get_intrinsic_reward):
                        pa_int_r = agent.get_intrinsic_reward(actor_index)
                """
                per_actor_trajectories[actor_index].append( (pa_obs, pa_a, pa_r, pa_int_r, pa_succ_obs, pa_done) )


                if self.config['test_nbr_episode'] != 0 \
                and obs_count % self.config['test_obs_interval'] == 0:
                    save_traj = False
                    if (self.config['benchmarking_record_episode_interval'] is not None \
                        and self.config['benchmarking_record_episode_interval']>0):
                        #save_traj = (obs_count%benchmarking_record_episode_interval==0)
                        save_traj = (episode_count_record // nbr_actors > self.config['benchmarking_record_episode_interval'])
                        if save_traj:
                            episode_count_record = 0

                    # TECHNICAL DEBT: clone_agent.get_update_count is failing because the update count param is None
                    # haven't figured out why is the cloning function making it None...
                    test_agent(
                        env=test_env,
                        agents=[agent.clone(training=False) for agent in agents],
                        update_count=agent.get_update_count(),
                        nbr_episode=self.config['test_nbr_episode'],
                        sum_writer=sum_writer,
                        iteration=obs_count,
                        base_path=self.config['base_path'],
                        save_traj=save_traj,
                        render_mode=self.config['render_mode'],
                        save_traj_length_divider=self.config['save_traj_length_divider'],
                    )

            outputs_stream_dict["observations"] = observations
            outputs_stream_dict["info"] = info
            outputs_stream_dict["actions"] = actions 
            outputs_stream_dict["reward"] = reward 
            outputs_stream_dict["done"] = done 
            outputs_stream_dict["succ_observations"] = succ_observations
            outputs_stream_dict["succ_info"] = succ_info

            yield outputs_stream_dict

            observations = copy.deepcopy(succ_observations)
            info = copy.deepcopy(succ_info)

            if obs_count >= max_obs_count:  break

        outputs_stream_dict["signals:done_training"] = True 
        outputs_stream_dict["signals:trained_agents"] = agents 

        if sum_writer is not None:
            sum_writer.flush()
        
        env.close()
        test_env.close()

        return outputs_stream_dict
            


    
