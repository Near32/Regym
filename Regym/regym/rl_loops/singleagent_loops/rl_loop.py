import os
import math
import copy
import time
from tqdm import tqdm
import numpy as np

import regym
from tensorboardX import SummaryWriter
from regym.util import save_traj_with_graph

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


def run_episode(env, agent, training, max_episode_length=math.inf):
    '''
    Runs a single episode of a single-agent rl loop until termination.
    :param env: OpenAI gym environment
    :param agent: Agent policy used to take actions in the environment and to process simulated experiences
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :param max_episode_length: Maximum expisode duration meassured in steps.
    :returns: Episode trajectory. list of (o,a,r,o')
    '''
    observation = env.reset()
    done = False
    trajectory = []
    generator = tqdm(range(int(max_episode_length))) if max_episode_length != math.inf else range(int(1e20))
    for step in generator:
        action = agent.take_action(observation)
        succ_observation, reward, done, info = env.step(action)
        trajectory.append((observation, action, reward, succ_observation, done))
        if training: agent.handle_experience(observation, action, reward, succ_observation, done)
        observation = succ_observation
        if done:
            break

    return trajectory


def run_episode_parallel(
    env,
    agent,
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
    '''
    #observations, info = env.reset(env_configs=env_configs)
    env_reset_output_dict = env.reset(env_configs=env_configs)
    observations = env_reset_output_dict[obs_key]
    info = env_reset_output_dict[info_key]
    
    nbr_actors = env.get_nbr_envs()
    agent.set_nbr_actor(nbr_actors)
    agent.reset_actors()
    done = [False]*nbr_actors
    previous_done = copy.deepcopy(done)

    per_actor_trajectories = [list() for i in range(nbr_actors)]
    #generator = tqdm(range(int(max_episode_length))) if max_episode_length != math.inf else range(int(1e20))
    #for step in generator:
    for step in range(int(max_episode_length)):
        realdone = []
        action = agent.take_action(
            state=observations,
            infos=info
        )
        
        #succ_observations, reward, done, succ_info = env.step(action, only_progress_non_terminated=True)
        env_output_dict = env.step(actions, only_progress_non_terminated=True)
        succ_observations = env_output_dict[succ_obs_key]
        reward = env_output_dict[reward_key]
        done = env_output_dict[done_key]
        succ_info = env_output_dict[succ_info_key]

        if training:
            agent.handle_experience(
                s=observations,
                a=action,
                r=reward,
                succ_s=succ_observations,
                done=done,
                infos=info
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
            if ('real_done' in succ_info[actor_index]):
                d = succ_info[actor_index]['real_done']
                realdone.append(d)

            if d and not(previous_done[actor_index]):
                batch_idx_done_actors_among_not_done.append(batch_index)

            pa_obs = observations[batch_index]
            pa_info = info[0][batch_index]
            if save_traj:
                pa_obs = env.render(render_mode, env_indices=[batch_index])[0]
            
            pa_a = action[batch_index]
            pa_r = reward[batch_index]
            pa_succ_obs = succ_observations[batch_index]
            pa_done = done[actor_index]
            pa_int_r = 0.0

            if getattr(agent.algorithm, "use_rnd", False):
                get_intrinsic_reward = getattr(agent, "get_intrinsic_reward", None)
                if callable(get_intrinsic_reward):
                    pa_int_r = agent.get_intrinsic_reward(actor_index)
            
            if not previous_done[actor_index]:
                per_actor_trajectories[actor_index].append( (pa_obs, pa_a, pa_r, pa_int_r, pa_succ_obs, pa_done, pa_info) )

        observations = copy.deepcopy(succ_observations)
        info = copy.deepcopy(succ_info)

        """
        if len(batch_idx_done_actors_among_not_done):
            # Regularization of the agents' next observations:
            batch_idx_done_actors_among_not_done.sort(reverse=True)
            for batch_idx in batch_idx_done_actors_among_not_done:
                observations = np.concatenate( [observations[:batch_idx,...], observations[batch_idx+1:,...]], axis=0)
        """

        previous_done = copy.deepcopy(done)

        alldone = all(done)
        allrealdone = False
        """
        for idx in reversed(range(len(info))):
            if info[idx] is None:   del info[idx]
        """

        if len(realdone):
            allrealdone =  all(realdone)
        if alldone or allrealdone:
            break

    return per_actor_trajectories

def test_agent(
    env, 
    agent, 
    update_count, 
    nbr_episode, 
    sum_writer, 
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
    ):
    max_episode_length = 1e4
    env.set_nbr_envs(nbr_episode)

    trajectory = run_episode_parallel(
        env,
        agent,
        training=False,
        max_episode_length=max_episode_length,
        env_configs=None,
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
    mean_total_return = sum( total_return) / len(trajectory)
    std_ext_return = math.sqrt( sum( [math.pow( r-mean_total_return ,2) for r in total_return]) / len(total_return) )

    total_int_return = [ sum([ exp[3] for exp in t]) for t in trajectory]
    mean_total_int_return = sum( total_int_return) / len(trajectory)
    std_int_return = math.sqrt( sum( [math.pow( r-mean_total_int_return ,2) for r in total_int_return]) / len(total_int_return) )

    #update_count = agent.get_update_count()

    if sum_writer is not None:
        for idx, (ext_ret, int_ret) in enumerate(zip(total_return, total_int_return)):
            sum_writer.add_scalar('PerObservation/Testing/TotalReturn', ext_ret, iteration*len(trajectory)+idx)
            sum_writer.add_scalar('PerObservation/Testing/TotalIntReturn', int_ret, iteration*len(trajectory)+idx)
            sum_writer.add_scalar('PerUpdate/Testing/TotalReturn', ext_ret, update_count)
            sum_writer.add_scalar('PerUpdate/Testing/TotalIntReturn', int_ret, update_count)

        sum_writer.add_scalar('PerObservation/Testing/StdIntReturn', std_int_return, iteration)
        sum_writer.add_scalar('PerObservation/Testing/StdExtReturn', std_ext_return, iteration)

        sum_writer.add_scalar('PerUpdate/Testing/StdIntReturn', std_int_return, update_count)
        sum_writer.add_scalar('PerUpdate/Testing/StdExtReturn', std_ext_return, update_count)

    episode_lengths = [ len(t) for t in trajectory]
    mean_episode_length = sum( episode_lengths) / len(trajectory)
    std_episode_length = math.sqrt( sum( [math.pow( l-mean_episode_length ,2) for l in episode_lengths]) / len(trajectory) )

    if sum_writer is not None:
        sum_writer.add_scalar('PerObservation/Testing/MeanTotalReturn', mean_total_return, iteration)
        sum_writer.add_scalar('PerObservation/Testing/MeanTotalIntReturn', mean_total_int_return, iteration)

        sum_writer.add_scalar('PerUpdate/Testing/MeanTotalReturn', mean_total_return, update_count)
        sum_writer.add_scalar('PerUpdate/Testing/MeanTotalIntReturn', mean_total_int_return, update_count)

        sum_writer.add_scalar('PerObservation/Testing/MeanEpisodeLength', mean_episode_length, iteration)
        sum_writer.add_scalar('PerObservation/Testing/StdEpisodeLength', std_episode_length, iteration)

        sum_writer.add_scalar('PerUpdate/Testing/MeanEpisodeLength', mean_episode_length, update_count)
        sum_writer.add_scalar('PerUpdate/Testing/StdEpisodeLength', std_episode_length, update_count)

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


def async_gather_experience_parallel(
    task,
    agent,
    training,
    max_obs_count=1e7,
    max_update_count=1e7,
    test_obs_interval=1e4,
    test_nbr_episode=10,
    env_configs=None,
    sum_writer=None,
    base_path='./',
    benchmarking_record_episode_interval=None,
    step_hooks=[]):
    raise NotImplementedError('needs to be updated with dictionnary output from vec_env..')
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
    :param sum_writer: SummaryWriter.
    :param base_path: Path where to save gifs.
    :param benchmarking_record_episode_interval: None if not gif ought to be made, otherwise Integer.
    :returns:
    '''
    async_actor = agent.get_async_actor()
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
    #         "sum_writer":sum_writer,
    #         "base_path":base_path,
    #         "benchmarking_record_episode_interval":benchmarking_record_episode_interval,
    #         "step_hooks":step_hooks
    #     },
    # )
    # gathering_proc.start()

    gathering_proc_id = gather_experience_parallel.remote(
        task=task,
        agent=async_actor,
        training=training,
        max_obs_count=max_obs_count,
        test_obs_interval=test_obs_interval,
        test_nbr_episode=test_nbr_episode,
        env_configs=env_configs,
        sum_writer=sum_writer,
        base_path=base_path,
        benchmarking_record_episode_interval=benchmarking_record_episode_interval,
        step_hooks=step_hooks
    )

    if isinstance(sum_writer, str):
        sum_writer_path = os.path.join(sum_writer, 'learner.log')
        sum_writer = SummaryWriter(sum_writer_path, flush_secs=1)
        agent.algorithm.summary_writer = sum_writer

    pbar = tqdm(total=max_update_count, position=1)
    nbr_updates = 0
    #while gathering_proc.is_alive():
    while nbr_updates <= max_update_count:
        nbr_updates = agent.train()
        if nbr_updates is None: nbr_updates = 1
        pbar.update(nbr_updates)
        sum_writer.flush()

    sum_writer.flush()

    return agent 


def async_gather_experience_parallel1(
    task,
    agent,
    training,
    max_obs_count=1e7,
    max_update_count=2e6,
    test_obs_interval=1e4,
    test_nbr_episode=10,
    env_configs=None,
    sum_writer=None,
    base_path='./',
    benchmarking_record_episode_interval=None,
    step_hooks=[]):
    raise NotImplementedError('needs to be updated with dictionnary output from vec_env..')
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
    :param sum_writer: SummaryWriter.
    :param base_path: Path where to save gifs.
    :param benchmarking_record_episode_interval: None if not gif ought to be made, otherwise Integer.
    :returns:
    '''
    async_actor = agent.get_async_actor()
    
    # learner_proc = Process(
    #     target=learner_loop,
    #     kwargs={
    #         "sum_writer":sum_writer,
    #         "agent":agent,
    #         "max_update_count":max_update_count
    #     }
    # )
    # learner_proc.start()
    learner_proc = learner_loop.remote(
        sum_writer=sum_writer,
        agent=agent,
        max_update_count=max_update_count
    )
    
    kwargs={
        "task":task,
        "agent":async_actor,
        "training":training,
        "max_obs_count":max_obs_count,
        "test_obs_interval":test_obs_interval,
        "test_nbr_episode":test_nbr_episode,
        "env_configs":env_configs,
        "sum_writer":sum_writer,
        "base_path":base_path,
        "benchmarking_record_episode_interval":benchmarking_record_episode_interval,
        "step_hooks":step_hooks
    }
    gather_experience_parallel(**kwargs)
    
    return agent 

@ray.remote(num_gpus=0.5)
def learner_loop(
        sum_writer,
        agent,
        max_update_count):
    if isinstance(sum_writer, str):
        sum_writer_path = os.path.join(sum_writer, 'learner.log')
        sum_writer = SummaryWriter(sum_writer_path, flush_secs=1)
        agent.algorithm.summary_writer = sum_writer

    #pbar = tqdm(total=max_update_count, position=1)
    total_nbr_updates = 0
    while total_nbr_updates < max_update_count:
        nbr_updates = agent.train()
        if nbr_updates is None: nbr_updates = 1
        #pbar.update(nbr_updates)
        total_nbr_updates += nbr_updates
        
        sum_writer.flush()
        #print(f"Training: {total_nbr_updates} / {max_update_count}")

    sum_writer.flush()

    return agent

#@ray.remote(num_gpus=0.5)
def gather_experience_parallel(
    task,
    agent,
    training,
    max_obs_count=1e7,
    test_obs_interval=1e4,
    test_nbr_episode=10,
    env_configs=None,
    sum_writer=None,
    base_path='./',
    benchmarking_record_episode_interval=None,
    save_traj_length_divider=1,
    render_mode="rgb_array",
    step_hooks=[],
    sad=False,
    vdn=False,
    nbr_players=2,
    obs_key="observations",
    succ_obs_key="succ_observations",
    reward_key="reward",
    done_key="done",
    info_key="info",
    succ_info_key="succ_info",
    ):
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
    :param sum_writer: SummaryWriter.
    :param base_path: Path where to save gifs.
    :param benchmarking_record_episode_interval: None if not gif ought to be made, otherwise Integer.
    :returns:
    '''
    env = task.env
    test_env = task.test_env

    #observations, info = env.reset(env_configs=env_configs)
    env_reset_output_dict = env.reset(env_configs=env_configs)
    observations = env_reset_output_dict[obs_key]
    info = env_reset_output_dict[info_key]

    nbr_actors = env.get_nbr_envs()
    agent.set_nbr_actor(nbr_actors)
    agent.reset_actors()
    done = [False]*nbr_actors

    per_actor_trajectories = [list() for i in range(nbr_actors)]
    trajectories = list()
    total_returns = list()
    total_int_returns = list()
    episode_lengths = list()

    obs_count = agent.get_experience_count() if hasattr(agent, "get_experience_count") else 0
    episode_count = 0
    sample_episode_count = 0

    pbar = tqdm(total=max_obs_count, position=0)
    pbar.update(obs_count)

    if isinstance(sum_writer, str):
        sum_writer_path = os.path.join(sum_writer, 'actor.log')
        sum_writer = SummaryWriter(sum_writer_path, flush_secs=1)
        #agent.algorithm.summary_writer = sum_writer
        if agent.training:
            agent.algorithm.unwrapped.summary_writer = sum_writer
        else:
            algo = getattr(agent, "algorithm", None)
            if algo is not None:
                agent.algorithm.unwrapped.summary_writer = None 
    
    while True:
        action = agent.take_action(
            state=observations,
            infos=info
        )
        
        #succ_observations, reward, done, succ_info = env.step(action)
        env_output_dict = env.step(action)
        succ_observations = env_output_dict[succ_obs_key]
        reward = env_output_dict[reward_key]
        done = env_output_dict[done_key]
        succ_info = env_output_dict[succ_info_key]

        if training:
            agent.handle_experience(
                s=observations,
                a=action,
                r=reward,
                succ_s=succ_observations,
                done=done,
                infos=info
            )

        for actor_index in range(nbr_actors):
            obs_count += 1
            pbar.update(1)

            for hook in step_hooks:
                hook(env, agent, obs_count)

            pa_obs = observations[actor_index]
            pa_a = action[actor_index]
            pa_r = reward[actor_index]
            pa_succ_obs = succ_observations[actor_index]
            pa_done = done[actor_index]
            pa_int_r = 0.0

            if getattr(agent.algorithm, "use_rnd", False):
                get_intrinsic_reward = getattr(agent, "get_intrinsic_reward", None)
                if callable(get_intrinsic_reward):
                    pa_int_r = agent.get_intrinsic_reward(actor_index)

            per_actor_trajectories[actor_index].append( (pa_obs, pa_a, pa_r, pa_int_r, pa_succ_obs, pa_done) )

            #////////////////////////////////////////////////////////////////////////////////////////
            # Bookkeeping of the actors whose episode just ended:
            #////////////////////////////////////////////////////////////////////////////////////////
            if done[actor_index]:
                agent.reset_actors(indices=[actor_index])

            done_condition = ('real_done' in succ_info[actor_index] and succ_info[actor_index]['real_done']) or ('real_done' not in succ_info[actor_index] and done[actor_index])
            if done_condition:
                update_count = agent.get_update_count()
                episode_count += 1
                #succ_observations, succ_info = env.reset(env_configs=env_configs, env_indices=[actor_index])
                env_reset_output_dict = env.reset(env_configs=env_configs, env_indices=[actor_index])
                succ_observations = env_reset_output_dict[obs_key]
                succ_info = env_reset_output_dict[info_key]
                
                agent.reset_actors(indices=[actor_index])
                #################
                
                # Logging:
                trajectories.append(per_actor_trajectories[actor_index])
                total_returns.append(sum([ exp[2] for exp in trajectories[-1]]))
                total_int_returns.append(sum([ exp[3] for exp in trajectories[-1]]))
                episode_lengths.append(len(trajectories[-1]))

                if sum_writer is not None:
                    sum_writer.add_scalar('Training/TotalReturn', total_returns[-1], episode_count)
                    sum_writer.add_scalar('PerObservation/TotalReturn', total_returns[-1], obs_count)
                    sum_writer.add_scalar('PerUpdate/TotalReturn', total_returns[-1], update_count)
                    if actor_index == 0:
                        sample_episode_count += 1
                    ##sum_writer.add_scalar(f'data/reward_{actor_index}', total_returns[-1], sample_episode_count)
                    #sum_writer.add_scalar(f'PerObservation/Actor_{actor_index}_Reward', total_returns[-1], obs_count)
                    #sum_writer.add_scalar(f'PerUpdate/Actor_{actor_index}_Reward', total_returns[-1], update_count)
                    #sum_writer.add_scalar('Training/TotalIntReturn', total_int_returns[-1], episode_count)
                    sum_writer.flush()

                if len(trajectories) >= nbr_actors:
                    mean_total_return = sum( total_returns) / len(trajectories)
                    std_ext_return = math.sqrt( sum( [math.pow( r-mean_total_return ,2) for r in total_returns]) / len(total_returns) )
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
                    total_int_returns = list()
                    episode_lengths = list()

                per_actor_trajectories[actor_index] = list()

            #////////////////////////////////////////////////////////////////////////////////////////
            #////////////////////////////////////////////////////////////////////////////////////////

            if test_nbr_episode != 0 and obs_count % test_obs_interval == 0:
                save_traj = False
                if (benchmarking_record_episode_interval is not None and benchmarking_record_episode_interval>0):
                    save_traj = (obs_count%benchmarking_record_episode_interval==0)
                # TECHNICAL DEBT: clone_agent.get_update_count is failing because the update count param is None
                # haven't figured out why is the cloning function making it None...
                test_agent(
                    env=test_env,
                    agent=agent.clone(training=False),
                    update_count=agent.get_update_count(),
                    nbr_episode=test_nbr_episode,
                    sum_writer=sum_writer,
                    iteration=obs_count,
                    base_path=base_path,
                    save_traj=save_traj,
                    render_mode=render_mode,
                    save_traj_length_divider=save_traj_length_divider
                )

        observations = copy.deepcopy(succ_observations)
        info = copy.deepcopy(succ_info)

        if obs_count >= max_obs_count:  break


    if sum_writer is not None:
        sum_writer.flush()
    
    env.close()
    test_env.close()

    return agent
