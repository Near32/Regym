from typing import Dict, Any, Optional, List
import os
import math
import copy
import time
from tqdm import tqdm
import numpy as np

import regym
from regym.util import save_traj_with_graph
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

import wandb

def run_episode_parallel(
    env,
    agents,
    training,
    max_episode_length=1e30,
    env_configs=None,
    save_traj=False,
    render_mode="rgb_array",
    obs_key="observations",
    succ_obs_key="succ_observations",
    reward_key="reward",
    done_key="done",
    info_key="info",
    succ_info_key="succ_info"):
    '''
    Runs a single multi-agent rl loop until termination.
    The observations vector is of length n, where n is the number of agents
    observations[i] corresponds to the oberservation of agent i.
    :param env: ParallelEnv wrapper around an OpenAI gym environment
    :param agent: Agent policy used to take actionsin the environment and to process simulated experiences
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :param max_episode_length: Maximum expisode duration meassured in steps.
    :param env_configs: configuration dictionnary to use when resetting the environments.
    :returns: Trajectory (o,a,r,o')
    
    N.B.: only care about agent 0's trajectory.
    '''
    #observations, info = env.reset(env_configs=env_configs)
    env_reset_output_dict = env.reset(env_configs=env_configs)
    observations = env_reset_output_dict[obs_key]
    info = env_reset_output_dict[info_key]
        
    nbr_actors = env.get_nbr_envs()
    
    for agent in agents:
        agent.set_nbr_actor(nbr_actors)
        agent.reset_actors()
    done = [False]*nbr_actors
    previous_done = copy.deepcopy(done)

    per_actor_trajectories = [list() for i in range(nbr_actors)]
    #generator = tqdm(range(int(max_episode_length))) if max_episode_length != math.inf else range(int(1e20))
    #for step in generator:
    for step in range(int(max_episode_length)):
        realdone = []
        actions = [ 
            agent.take_action(
                state=observations[agent_idx],
                infos=info[agent_idx]
            ) 
            for agent_idx, agent in enumerate(agents)
        ] 
        #succ_observations, reward, done, succ_info = env.step(actions, only_progress_non_terminated=True)
        env_output_dict = env.step(actions, only_progress_non_terminated=True)
        succ_observations = env_output_dict[succ_obs_key]
        reward = env_output_dict[reward_key]
        done = env_output_dict[done_key]
        succ_info = env_output_dict[succ_info_key]

        if training:
            for agent_idx, agent in enumerate(agents):
                agent.handle_experience(
                    s=observations[agent_idx],
                    a=actions[agent_idx],
                    r=reward[agent_idx],
                    succ_s=succ_observations[agent_idx],
                    done=done,
                    infos=info[agent_idx]
                )

        batch_index = -1
        batch_idx_done_actors_among_not_done = []
        for actor_index in range(nbr_actors):
            if previous_done[actor_index]:
                continue
            #batch_index +=1
            # since `only_progress_non_terminated=True`:
            batch_index = actor_index 

            # Bookkeeping of the actors whose episode just ended:
            d = done[actor_index]
            if ('real_done' in succ_info[0][actor_index]):
                d = succ_info[0][actor_index]['real_done']
                realdone.append(d)

            if d and not(previous_done[actor_index]):
                batch_idx_done_actors_among_not_done.append(batch_index)

            # Only care about agent 0's trajectory:
            pa_obs = observations[0][batch_index]
            pa_info = info[0][batch_index]
            if save_traj:
                pa_obs = env.render(render_mode, env_indices=[batch_index])[0]
            pa_a = actions[0][batch_index]
            pa_r = reward[0][batch_index]
            pa_succ_obs = succ_observations[0][batch_index]
            pa_done = done[actor_index]
            pa_int_r = 0.0
            
            """
            if getattr(agent.algorithm, "use_rnd", False):
                get_intrinsic_reward = getattr(agent, "get_intrinsic_reward", None)
                if callable(get_intrinsic_reward):
                    pa_int_r = agent.get_intrinsic_reward(actor_index)
            """
            if not previous_done[actor_index]:
                per_actor_trajectories[actor_index].append((
                    pa_obs, 
                    pa_a, 
                    pa_r, 
                    pa_int_r, 
                    #pa_succ_obs, 
                    pa_done, 
                    pa_info,
                ))
                #per_actor_trajectories[actor_index].append( (pa_obs, pa_a, pa_r, pa_int_r, pa_succ_obs, pa_done, pa_info) )

        observations = copy.deepcopy(succ_observations)
        info = copy.deepcopy(succ_info)

        """
        if len(batch_idx_done_actors_among_not_done):
            # Regularization of the agents' next observations:
            batch_idx_done_actors_among_not_done.sort(reverse=True)
            for batch_idx in batch_idx_done_actors_among_not_done:
                observations = [
                    np.concatenate( [obs[:batch_idx,...], obs[batch_idx+1:,...]], axis=0)
                    for obs in observations
                ]
        """

        previous_done = copy.deepcopy(done)

        alldone = all(done)
        allrealdone = False
        """
        for agent_idx in range(len(info)):
            for idx in reversed(range(len(info[agent_idx]))):
                if info[agent_idx][idx] is None:   del info[agent_idx][idx]
        """

        if len(realdone):
            allrealdone =  all(realdone)
        if alldone or allrealdone:
            break

    return per_actor_trajectories


def test_agent(
    env, 
    env_configs,
    agents, 
    update_count, 
    nbr_episode, 
    iteration, 
    base_path, 
    nbr_save_traj=1, 
    save_traj=False,
    render_mode="rgb_array",
    save_traj_length_divider=1,
    obs_key="observations",
    succ_obs_key="succ_observations",
    reward_key="reward",
    done_key="done",
    info_key="info",
    succ_info_key="succ_info",
    requested_metrics: List[str] = []
    ) -> Optional[Dict]:
    '''
    Available metrics to be requested:
    - 'total_return',
    - 'mean_total_return',
    - 'std_ext_return',
    - 'total_int_return',
    - 'mean_total_int_return',
    - 'std_int_return',
    - 'episode_lengths',
    - 'mean_episode_length',
    - 'episode_lengths',
    :returns: Dictionary containing values specified in :param: requested_metrics
    '''
    max_episode_length = 1e4
    env.set_nbr_envs(nbr_episode)

    trajectory = run_episode_parallel(
        env,
        agents,
        training=False,
        max_episode_length=max_episode_length,
        env_configs=env_configs,
        save_traj=save_traj,
        render_mode=render_mode,
        obs_key=obs_key,
        succ_obs_key=succ_obs_key,
        reward_key=reward_key,
        done_key=done_key,
        info_key=info_key,
        succ_info_key=succ_info_key,
    )

    total_return = [ sum([ exp[2] for exp in t]) for t in trajectory]
    positive_total_return = [ sum([ exp[2] if exp[2]>0 else 0.0 for exp in t]) for t in trajectory]
    mean_total_return = sum( total_return) / len(trajectory)
    std_ext_return = math.sqrt( sum( [math.pow( r-mean_total_return ,2) for r in total_return]) / len(total_return) )
    mean_positive_total_return = sum( positive_total_return) / len(trajectory)
    std_ext_positive_return = math.sqrt( sum( [math.pow( r-mean_positive_total_return ,2) for r in positive_total_return]) / len(positive_total_return) )

    total_int_return = [ sum([ exp[3] for exp in t]) for t in trajectory]
    mean_total_int_return = sum( total_int_return) / len(trajectory)
    std_int_return = math.sqrt( sum( [math.pow( r-mean_total_int_return ,2) for r in total_int_return]) / len(total_int_return) )

    #update_count = agent.get_update_count()

    for idx, (ext_ret, ext_pos_ret, int_ret) in enumerate(zip(total_return, positive_total_return, total_int_return)):
        wandb.log({'PerObservation/Testing/TotalReturn':  ext_ret}, commit=False) # iteration*len(trajectory)+idx)
        wandb.log({'PerObservation/Testing/PositiveTotalReturn':  ext_pos_ret}, commit=False) # iteration*len(trajectory)+idx)
        wandb.log({'PerObservation/Testing/TotalIntReturn':  int_ret}, commit=False) # iteration*len(trajectory)+idx)
        wandb.log({'PerUpdate/Testing/TotalReturn':  ext_ret}, commit=False) # update_count)
        wandb.log({'PerUpdate/Testing/PositiveTotalReturn':  ext_pos_ret}, commit=False) # update_count)
        wandb.log({'PerUpdate/Testing/TotalIntReturn':  int_ret}, commit=False) # update_count)

    wandb.log({'PerObservation/Testing/StdIntReturn':  std_int_return}, commit=False) # iteration)
    wandb.log({'PerObservation/Testing/StdExtReturn':  std_ext_return}, commit=False) # iteration)
    wandb.log({'PerObservation/Testing/StdExtPosReturn':  std_ext_positive_return}, commit=False) # iteration)

    wandb.log({'PerUpdate/Testing/StdIntReturn':  std_int_return}, commit=False) # update_count)
    wandb.log({'PerUpdate/Testing/StdExtReturn':  std_ext_return}, commit=False) # update_count)
    wandb.log({'PerUpdate/Testing/StdExtPosReturn':  std_ext_positive_return}, commit=False) # update_count)

    episode_lengths = [ len(t) for t in trajectory]
    mean_episode_length = sum( episode_lengths) / len(trajectory)
    std_episode_length = math.sqrt( sum( [math.pow( l-mean_episode_length ,2) for l in episode_lengths]) / len(trajectory) )

    trajectory_metrics = populate_metrics_dictionary(
        total_return, mean_total_return, std_ext_return,
        total_int_return, mean_total_int_return, std_int_return,
        episode_lengths, mean_episode_length, std_episode_length,
        requested_metrics
    )

    wandb.log({'PerObservation/Testing/MeanTotalReturn':  mean_total_return}, commit=False) # iteration)
    wandb.log({'PerObservation/Testing/MeanPositiveTotalReturn':  mean_positive_total_return}, commit=False) # iteration)
    wandb.log({'PerObservation/Testing/MeanTotalIntReturn':  mean_total_int_return}, commit=False) # iteration)

    wandb.log({'PerUpdate/Testing/MeanTotalReturn':  mean_total_return}, commit=False) # update_count)
    wandb.log({'PerUpdate/Testing/MeanPositiveTotalReturn':  mean_positive_total_return}, commit=False) # update_count)
    wandb.log({'PerUpdate/Testing/MeanTotalIntReturn':  mean_total_int_return}, commit=False) # update_count)

    wandb.log({'PerObservation/Testing/MeanEpisodeLength':  mean_episode_length}, commit=False) # iteration)
    wandb.log({'PerObservation/Testing/StdEpisodeLength':  std_episode_length}, commit=False) # iteration)

    wandb.log({'PerUpdate/Testing/MeanEpisodeLength':  mean_episode_length}, commit=False) # update_count)
    wandb.log({'PerUpdate/Testing/StdEpisodeLength':  std_episode_length}, commit=False) # update_count)

    if save_traj:
        for actor_idx in range(nbr_save_traj):
            gif_traj = [ exp[0] for exp in trajectory[actor_idx]]
            gif_data = [np.cumsum([ exp[2] for exp in trajectory[actor_idx]])]
            begin = time.time()
            save_traj_with_graph(
                gif_traj, 
                gif_data, 
                divider=save_traj_length_divider,
                episode=iteration, 
                actor_idx=actor_idx, 
                path=base_path
            )
            end = time.time()
            eta = end-begin
            print(f'{actor_idx+1} / {nbr_save_traj} :: Time: {eta} sec.')
    return trajectory_metrics


def populate_metrics_dictionary(total_return, mean_total_return, std_ext_return,
                                total_int_return, mean_total_int_return, std_int_return,
                                episode_lengths, mean_episode_length, std_episode_length,
                                requested_metrics: List[str] = []) -> Dict[str, Any]:
    trajectory_metrics = {}
    if 'total_return' in requested_metrics:
        trajectory_metrics['total_return'] = total_return
    if 'mean_total_return' in requested_metrics:
        trajectory_metrics['mean_total_return'] = mean_total_return
    if 'std_ext_return' in requested_metrics:
        trajectory_metrics['std_ext_return'] = std_ext_return

    if 'total_int_return' in requested_metrics:
        trajectory_metrics['total_int_return'] = total_int_return
    if 'mean_total_int_return' in requested_metrics:
        trajectory_metrics['mean_total_int_return'] = mean_total_int_return
    if 'std_int_return' in requested_metrics:
        trajectory_metrics['std_int_return'] = std_int_return

    if 'episode_lengths' in requested_metrics:
        trajectory_metrics['episode_lengths'] = episode_lengths
    if 'mean_episode_length' in requested_metrics:
        trajectory_metrics['mean_episode_length'] = mean_episode_length
    if 'episode_lengths' in requested_metrics:
        trajectory_metrics['std_episode_length'] = std_episode_length
    return trajectory_metrics


def async_gather_experience_parallel(
    task,
    agents,
    training,
    max_obs_count=1e7,
    max_update_count=1e7,
    test_obs_interval=1e4,
    test_nbr_episode=10,
    env_configs=None,
    base_path='./',
    benchmarking_record_episode_interval=None,
    save_traj_length_divider=1,
    step_hooks=[],
    sad=False):
    '''
    Runs a single multi-agent rl loop until the number of observation, `max_obs_count`, is reached.
    The observations vector is of length n, where n is the number of agents.
    observations[i] corresponds to the oberservation of agent i.
    :param env: ParallelEnv wrapper around an OpenAI gym environment
    :param agent: Agent policy used to take actionsin the environment and to process simulated experiences
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :param max_obs_count: Maximum number of observations to gather data for.
    :param test_obs_interval: Integer, interval between two testing of the agent in the test environment.
    :param test_nbr_episode: Integer, nbr of episode to test the agent with.
    :param env_configs: configuration dictionnary to use when resetting the environments.
    :param base_path: Path where to save gifs.
    :param benchmarking_record_episode_interval: None if not gif ought to be made, otherwise Integer.
    :returns:
    '''
    async_agents = [agent.get_async_actor() for agent in agents]
    # gathering_proc = Process(
    #     target=gather_experience_parallel,
    #     kwargs={
    #         "task":task,
    #         "agent":async_actor,
    #         "training":training,
    #         "max_obs_count":max_obs_count,
    #         "test_obs_interval":test_obs_interval,
    #         "test_nbr_episode":test_nbr_episode,
    #         "env_configs":env_configs,
    #         "base_path":base_path,
    #         "benchmarking_record_episode_interval":benchmarking_record_episode_interval,
    #         "step_hooks":step_hooks
    #     },
    # )
    # gathering_proc.start()

    gathering_proc_id = gather_experience_parallel_ray.remote(
        task=task,
        agents=async_agents,
        training=training,
        max_obs_count=max_obs_count,
        test_obs_interval=test_obs_interval,
        test_nbr_episode=test_nbr_episode,
        env_configs=env_configs,
        base_path=base_path,
        benchmarking_record_episode_interval=benchmarking_record_episode_interval,
        save_traj_length_divider=save_traj_length_divider,
        step_hooks=step_hooks,
        sad=sad,
    )

    pbar = tqdm(total=max_update_count, position=1)
    nbr_updates = 0
    #while gathering_proc.is_alive():
    while nbr_updates <= max_update_count:
        for agent_idx, agent in enumerate(agents):
            if agent.training:
                #assert agent_idx==0
                nbr_updates = agent.train()
            if nbr_updates is None: nbr_updates = 1
            pbar.update(nbr_updates)

    return agent 


def async_gather_experience_parallel1(
    task,
    agents,
    training,
    max_obs_count=1e7,
    max_update_count=2e6,
    test_obs_interval=1e4,
    test_nbr_episode=10,
    env_configs=None,
    base_path='./',
    benchmarking_record_episode_interval=None,
    save_traj_length_divider=1,
    step_hooks=[],
    sad=False):
    '''
    Runs a single multi-agent rl loop until the number of observation, `max_obs_count`, is reached.
    The observations vector is of length n, where n is the number of agents.
    observations[i] corresponds to the oberservation of agent i.
    :param env: ParallelEnv wrapper around an OpenAI gym environment
    :param agent: Agent policy used to take actionsin the environment and to process simulated experiences
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :param max_obs_count: Maximum number of observations to gather data for.
    :param test_obs_interval: Integer, interval between two testing of the agent in the test environment.
    :param test_nbr_episode: Integer, nbr of episode to test the agent with.
    :param env_configs: configuration dictionnary to use when resetting the environments.
    :param base_path: Path where to save gifs.
    :param benchmarking_record_episode_interval: None if not gif ought to be made, otherwise Integer.
    :returns:
    '''
    async_agents = [agent.get_async_actor() for agent in agents]
    
    # learner_proc = Process(
    #     target=learner_loop,
    #     kwargs={
    #         "agent":agent,
    #         "max_update_count":max_update_count
    #     }
    # )
    # learner_proc.start()
    learner_proc = learner_loop.remote(
        agents=agents,
        max_update_count=max_update_count
    )
    
    kwargs={
        "task":task,
        "agents":async_agents,
        "training":training,
        "max_obs_count":max_obs_count,
        "test_obs_interval":test_obs_interval,
        "test_nbr_episode":test_nbr_episode,
        "env_configs":env_configs,
        "base_path":base_path,
        "benchmarking_record_episode_interval":benchmarking_record_episode_interval,
        "save_traj_length_divider":save_traj_length_divider,
        "step_hooks":step_hooks,
        "sad":sad
    }
    gather_experience_parallel(**kwargs)
    
    return agents

@ray.remote(num_gpus=0.5)
def learner_loop(
        agents,
        max_update_count):
    #pbar = tqdm(total=max_update_count, position=1)
    total_nbr_updates = 0
    while total_nbr_updates < max_update_count:
        for agent_idx, agent in enumerate(agents):
            if agent.training:
                #assert agent_idx==0
                nbr_updates = agent.train()
                if nbr_updates is None: nbr_updates = 1
                #pbar.update(nbr_updates)
                total_nbr_updates += nbr_updates
                
        #print(f"Training: {total_nbr_updates} / {max_update_count}")
    return agents

#@ray.remote(num_gpus=0.5)
def gather_experience_parallel(
    task,
    agents,
    training,
    max_obs_count=1e7,
    test_obs_interval=1e4,
    test_nbr_episode=10,
    env_configs=None,
    base_path='./',
    benchmarking_record_episode_interval=None,
    save_traj_length_divider=1,
    render_mode="rgb_array",
    step_hooks=[],
    sad=False,
    vdn=False,
    otherplay=False,
    nbr_players=2,
    obs_key="observations",
    succ_obs_key="succ_observations",
    reward_key="reward",
    done_key="done",
    info_key="info",
    succ_info_key="succ_info",
    ):
    '''
    Runs a self-play multi-agent rl loop until the number of observation, `max_obs_count`, is reached.
    The observations vector is of length n, where n is the number of agents.
    observations[i] corresponds to the oberservation of agent i.
    :param env: ParallelEnv wrapper around an OpenAI gym environment
    :param agents: List of Agent policy used to take actions in the environment and to process simulated experiences
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :param max_obs_count: Maximum number of observations to gather data for.
    :param test_obs_interval: Integer, interval between two testing of the agent in the test environment.
    :param test_nbr_episode: Integer, nbr of episode to test the agent with.
    :param env_configs: configuration dictionnary to use when resetting the environments.
    :param base_path: Path where to save gifs.
    :param benchmarking_record_episode_interval: None if not gif ought to be made, otherwise Integer.
    :returns:

    N.B.: only logs agent 0's trajectory.
    '''
    assert nbr_players==2, "Not implemented with more than 2 players..."

    env = task.env
    if sad:
        env = SADEnvWrapper(env, nbr_actions=task.action_dim, otherplay=otherplay)
    if vdn:
        env = VDNVecEnvWrapper(env, nbr_players=nbr_players)

    test_env = task.test_env
    if sad:
        test_env = SADEnvWrapper(test_env, nbr_actions=task.action_dim, otherplay=otherplay)
    if vdn:
        test_env = VDNVecEnvWrapper(test_env, nbr_players=nbr_players)
    
    #observations, info = env.reset(env_configs=env_configs)
    env_reset_output_dict = env.reset(env_configs=env_configs)
    observations = env_reset_output_dict[obs_key]
    info = env_reset_output_dict[info_key]

    nbr_actors = env.get_nbr_envs()
    for agent in agents:
        agent.set_nbr_actor(nbr_actors)
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

    pbar = tqdm(total=max_obs_count, position=0)
    pbar.update(obs_count)

    while True:
        actions = [
            agent.take_action(
                state=observations[agent_idx],
                infos=info[agent_idx]
            )
            for agent_idx, agent in enumerate(agents)
        ]
        
        #succ_observations, reward, done, succ_info = env.step(actions)
        env_output_dict = env.step(actions)
        succ_observations = env_output_dict[succ_obs_key]
        reward = env_output_dict[reward_key]
        done = env_output_dict[done_key]
        succ_info = env_output_dict[succ_info_key]

        if training:
            for agent_idx, agent in enumerate(agents):
                if agent.training:
                    #assert agent_idx==0
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
            wandb.log({"obs_count": obs_count}, commit=False)
            pbar.update(1)

            for hook in step_hooks:
                for agent in agents:
                    hook(env, agent, obs_count)

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


            #////////////////////////////////////////////////////////////////////////////////////////
            # Bookkeeping of the actors whose episode just ended:
            #////////////////////////////////////////////////////////////////////////////////////////
            done_condition = ('real_done' in succ_info[0][actor_index] and succ_info[0][actor_index]['real_done']) or ('real_done' not in succ_info[0][actor_index] and done[actor_index])
            if done_condition:
                update_count = agents[0].get_update_count()
                wandb.log({"update_count": update_count}, commit=False)
                episode_count += 1
                episode_count_record += 1
                #succ_observations, succ_info = env.reset(env_configs=env_configs, env_indices=[actor_index])
                env_reset_output_dict = env.reset(env_configs=config.get('env_configs', None), env_indices=[actor_index])
                succ_observations = env_reset_output_dict[obs_key]
                succ_info = env_reset_output_dict[info_key]
                
                for agent_idx, agent in enumerate(agents):
                    agent.reset_actors(indices=[actor_index])
                
                # Logging:
                trajectories.append(per_actor_trajectories[actor_index])
                total_returns.append(sum([ exp[2] for exp in trajectories[-1]]))
                positive_total_returns.append(sum([ exp[2] if exp[2]>0 else 0.0 for exp in trajectories[-1]]))
                total_int_returns.append(sum([ exp[3] for exp in trajectories[-1]]))
                episode_lengths.append(len(trajectories[-1]))

                wandb.log({'Training/TotalReturn':  total_returns[-1]}, commit=False) # episode_count)
                wandb.log({'PerObservation/TotalReturn':  total_returns[-1]}, commit=False) # obs_count)
                wandb.log({'PerUpdate/TotalReturn':  total_returns[-1]}, commit=False) # update_count)
                
                wandb.log({'Training/PositiveTotalReturn':  positive_total_returns[-1]}, commit=False) # episode_count)
                wandb.log({'PerObservation/PositiveTotalReturn':  positive_total_returns[-1]}, commit=False) # obs_count)
                wandb.log({'PerUpdate/PositiveTotalReturn':  positive_total_returns[-1]}, commit=False) # update_count)
                
                if actor_index == 0:
                    sample_episode_count += 1
                #wandb.log({f'data/reward_{actor_index}':  total_returns[-1]}) # sample_episode_count)
                #wandb.log({f'PerObservation/Actor_{actor_index}_Reward':  total_returns[-1]}) # obs_count)
                #wandb.log({f'PerObservation/Actor_{actor_index}_PositiveReward':  positive_total_returns[-1]}) # obs_count)
                #wandb.log({f'PerUpdate/Actor_{actor_index}_Reward':  total_returns[-1]}) # update_count)
                #wandb.log({'Training/TotalIntReturn':  total_int_returns[-1]}) # episode_count)

                if len(trajectories) >= nbr_actors:
                    mean_total_return = sum( total_returns) / len(trajectories)
                    std_ext_return = math.sqrt( sum( [math.pow( r-mean_total_return ,2) for r in total_returns]) / len(total_returns) )
                    mean_positive_total_return = sum( positive_total_returns) / len(trajectories)
                    std_ext_positive_return = math.sqrt( sum( [math.pow( r-mean_positive_total_return ,2) for r in positive_total_returns]) / len(positive_total_returns) )
                    mean_total_int_return = sum( total_int_returns) / len(trajectories)
                    std_int_return = math.sqrt( sum( [math.pow( r-mean_total_int_return ,2) for r in total_int_returns]) / len(total_int_returns) )
                    mean_episode_length = sum( episode_lengths) / len(trajectories)
                    std_episode_length = math.sqrt( sum( [math.pow( l-mean_episode_length ,2) for l in episode_lengths]) / len(episode_lengths) )

                    wandb.log({'Training/StdIntReturn':  std_int_return}, commit=False) # episode_count // nbr_actors)
                    wandb.log({'Training/StdExtReturn':  std_ext_return}, commit=False) # episode_count // nbr_actors)

                    wandb.log({'Training/MeanTotalReturn':  mean_total_return}, commit=False) # episode_count // nbr_actors)
                    wandb.log({'PerObservation/MeanTotalReturn':  mean_total_return}, commit=False) # obs_count)
                    wandb.log({'PerUpdate/MeanTotalReturn':  mean_total_return}, commit=False) # update_count)
                    wandb.log({'Training/MeanPositiveTotalReturn':  mean_positive_total_return}, commit=False) # episode_count // nbr_actors)
                    wandb.log({'PerObservation/MeanPositiveTotalReturn':  mean_positive_total_return}, commit=False) # obs_count)
                    wandb.log({'PerUpdate/MeanPositiveTotalReturn':  mean_positive_total_return}, commit=False) # update_count)
                    wandb.log({'Training/MeanTotalIntReturn':  mean_total_int_return}, commit=False) # episode_count // nbr_actors)

                    wandb.log({'Training/MeanEpisodeLength':  mean_episode_length}, commit=False) # episode_count // nbr_actors)
                    wandb.log({'PerObservation/MeanEpisodeLength':  mean_episode_length}, commit=False) # obs_count)
                    wandb.log({'PerUpdate/MeanEpisodeLength':  mean_episode_length}, commit=False) # update_count)
                    wandb.log({'Training/StdEpisodeLength':  std_episode_length}, commit=False) # episode_count // nbr_actors)
                    wandb.log({'PerObservation/StdEpisodeLength':  std_episode_length}, commit=False) # obs_count)
                    wandb.log({'PerUpdate/StdEpisodeLength':  std_episode_length}, commit=False) # update_count)

                    # reset :
                    trajectories = list()
                    total_returns = list()
                    positive_total_returns = list()
                    total_int_returns = list()
                    episode_lengths = list()

                per_actor_trajectories[actor_index] = list()

            #////////////////////////////////////////////////////////////////////////////////////////
            #////////////////////////////////////////////////////////////////////////////////////////

            if test_nbr_episode != 0 and obs_count % test_obs_interval == 0:
                save_traj = False
                if (benchmarking_record_episode_interval is not None and benchmarking_record_episode_interval>0):
                    #save_traj = (obs_count%benchmarking_record_episode_interval==0)
                    save_traj = (episode_count_record // nbr_actors > benchmarking_record_episode_interval)
                    if save_traj:
                        episode_count_record = 0

                # TECHNICAL DEBT: clone_agent.get_update_count is failing because the update count param is None
                # haven't figured out why is the cloning function making it None...
                test_agent(
                    env=test_env,
                    agents=[agent.clone(training=False) for agent in agents],
                    update_count=agent.get_update_count(),
                    nbr_episode=test_nbr_episode,
                    iteration=obs_count,
                    base_path=base_path,
                    save_traj=save_traj,
                    render_mode=render_mode,
                    save_traj_length_divider=save_traj_length_divider
                )

        observations = copy.deepcopy(succ_observations)
        info = copy.deepcopy(succ_info)

        if obs_count >= max_obs_count:  break

    env.close()
    test_env.close()

    return agent

gather_experience_parallel_ray = ray.remote(gather_experience_parallel)
