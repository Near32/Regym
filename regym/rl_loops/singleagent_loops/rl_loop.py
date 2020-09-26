import math
import copy
import time
from tqdm import tqdm
import numpy as np
from regym.util import save_traj_with_graph

from torch.multiprocessing import Process 


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


def run_episode_parallel(env,
                            agent,
                            training,
                            max_episode_length=1e30,
                            env_configs=None):
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
    observations = env.reset(env_configs=env_configs)

    nbr_actors = env.get_nbr_envs()
    agent.set_nbr_actor(nbr_actors)
    agent.reset_actors()
    done = [False]*nbr_actors
    previous_done = copy.deepcopy(done)

    per_actor_trajectories = [list() for i in range(nbr_actors)]
    #generator = tqdm(range(int(max_episode_length))) if max_episode_length != math.inf else range(int(1e20))
    #for step in generator:
    for step in range(int(max_episode_length)):
        action = agent.take_action(observations)
        succ_observations, reward, done, info = env.step(action)

        if training:
            agent.handle_experience(observations,
                                    action,
                                    reward,
                                    succ_observations,
                                    done,
                                    infos=info)

        batch_index = -1
        batch_idx_done_actors_among_not_done = []
        for actor_index in range(nbr_actors):
            if previous_done[actor_index]:
                continue
            batch_index +=1

            # Bookkeeping of the actors whose episode just ended:
            d = done[actor_index]
            if ('real_done' in info[actor_index]):
                d = info[actor_index]['real_done']

            if d and not(previous_done[actor_index]):
                batch_idx_done_actors_among_not_done.append(batch_index)

            pa_obs = observations[batch_index]
            pa_a = action[batch_index]
            pa_r = reward[batch_index]
            pa_succ_obs = succ_observations[batch_index]
            pa_done = done[actor_index]
            pa_int_r = 0.0
            if getattr(agent.algorithm, "use_rnd", False):
                get_intrinsic_reward = getattr(agent, "get_intrinsic_reward", None)
                if callable(get_intrinsic_reward):
                    pa_int_r = agent.get_intrinsic_reward(actor_index)
            per_actor_trajectories[actor_index].append( (pa_obs, pa_a, pa_r, pa_int_r, pa_succ_obs, pa_done) )

        observations = copy.deepcopy(succ_observations)
        if len(batch_idx_done_actors_among_not_done):
            # Regularization of the agents' next observations:
            batch_idx_done_actors_among_not_done.sort(reverse=True)
            for batch_idx in batch_idx_done_actors_among_not_done:
                observations = np.concatenate( [observations[:batch_idx,...], observations[batch_idx+1:,...]], axis=0)

        previous_done = copy.deepcopy(done)

        alldone = all(done)
        allrealdone = False
        for idx in reversed(range(len(info))):
            if info[idx] is None:   del info[idx]

        if len(info):
            allrealdone =  all([i['real_done'] if 'real_done' in i else False for i in info])
        if alldone or allrealdone:
            break

    return per_actor_trajectories

def test_agent(env, agent, nbr_episode, sum_writer, iteration, base_path, nbr_save_traj=1, save_traj=False):
    max_episode_length = 1e4
    env.set_nbr_envs(nbr_episode)

    trajectory = run_episode_parallel(env,
                                      agent,
                                      training=False,
                                      max_episode_length=max_episode_length,
                                      env_configs=None)

    total_return = [ sum([ exp[2] for exp in t]) for t in trajectory]
    mean_total_return = sum( total_return) / len(trajectory)
    std_ext_return = math.sqrt( sum( [math.pow( r-mean_total_return ,2) for r in total_return]) / len(total_return) )

    total_int_return = [ sum([ exp[3] for exp in t]) for t in trajectory]
    mean_total_int_return = sum( total_int_return) / len(trajectory)
    std_int_return = math.sqrt( sum( [math.pow( r-mean_total_int_return ,2) for r in total_int_return]) / len(total_int_return) )

    update_count = agent.get_update_count()

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
            save_traj_with_graph(gif_traj, gif_data, episode=iteration, actor_idx=actor_idx, path=base_path)
            end = time.time()
            eta = end-begin
            print(f'{actor_idx+1} / {nbr_save_traj} :: Time: {eta} sec.')


def async_gather_experience_parallel(
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
    step_hooks=[]):
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

    gathering_proc = Process(
        target=gather_experience_parallel,
        args=(
            task,
            agent.get_async_actor(),
            training,
            max_obs_count,
            test_obs_interval,
            test_nbr_episode,
            env_configs,
            sum_writer,
            base_path,
            benchmarking_record_episode_interval,
            step_hooks),
    )
    gathering_proc.start()

    while gathering_proc.is_alive():
        agent.train()

    return agent 


def gather_experience_parallel(task,
                                agent,
                                training,
                                max_obs_count=1e7,
                                test_obs_interval=1e4,
                                test_nbr_episode=10,
                                env_configs=None,
                                sum_writer=None,
                                base_path='./',
                                benchmarking_record_episode_interval=None,
                                step_hooks=[]):
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

    observations = env.reset(env_configs=env_configs)

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

    pbar = tqdm(total=max_obs_count)

    while True:
        action = agent.take_action(observations)
        succ_observations, reward, done, info = env.step(action)

        if training:
            agent.handle_experience(observations,
                                    action,
                                    reward,
                                    succ_observations,
                                    done,
                                    infos=info)

        for actor_index in range(nbr_actors):
            obs_count += 1
            pbar.update(1)
            
            for hook in step_hooks:
                hook(env, agent, obs_count)

            # Bookkeeping of the actors whose episode just ended:
            if done[actor_index]:
                agent.reset_actors(indices=[actor_index])

            if ('real_done' in info[actor_index] and info[actor_index]['real_done'])\
                or ('real_done' not in info[actor_index] and done[actor_index]):
                update_count = agent.get_update_count()
                episode_count += 1
                succ_observations[actor_index] = env.reset(env_configs=env_configs, env_indices=[actor_index])
                agent.reset_actors(indices=[actor_index])

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
                        sum_writer.add_scalar('data/reward', total_returns[-1], sample_episode_count)
                        sum_writer.add_scalar('PerObservation/Actor0Reward', total_returns[-1], obs_count)
                        sum_writer.add_scalar('PerUpdate/Actor0Reward', total_returns[-1], update_count)
                    sum_writer.add_scalar('Training/TotalIntReturn', total_int_returns[-1], episode_count)

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

                    # reset :
                    trajectories = list()
                    total_returns = list()
                    total_int_returns = list()
                    episode_lengths = list()

                per_actor_trajectories[actor_index] = list()

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


            if test_nbr_episode != 0 and obs_count % test_obs_interval == 0:
                save_traj = False
                if (benchmarking_record_episode_interval is not None and benchmarking_record_episode_interval>0):
                    save_traj = (obs_count%benchmarking_record_episode_interval==0)
                test_agent(env=test_env,
                            agent=agent.clone(training=False),
                            nbr_episode=test_nbr_episode,
                            sum_writer=sum_writer,
                            iteration=obs_count,
                            base_path=base_path,
                            save_traj=save_traj)

        observations = copy.deepcopy(succ_observations)

        if obs_count >= max_obs_count:  break

    return agent
