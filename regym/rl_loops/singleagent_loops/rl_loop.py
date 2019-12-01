import math
import copy
from OpenGL import GL
from tqdm import tqdm
import numpy as np

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

def run_episode_parallel(env, agent, training, max_episode_length=math.inf, env_configs=None):
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
                                    done)
        
        batch_index = -1
        batch_idx_done_actors_among_not_done = []
        for actor_index in range(nbr_actors):
            if previous_done[actor_index]:
                continue
            batch_index +=1
            
            # Bookkeeping of the actors whose episode just ended:
            if done[actor_index] and not(previous_done[actor_index]):
                batch_idx_done_actors_among_not_done.append(batch_index)
                
            pa_obs = observations[batch_index]
            pa_a = action[batch_index]
            pa_r = reward[batch_index]
            pa_succ_obs = succ_observations[batch_index]
            pa_done = done[actor_index]
            pa_int_r = 0.0
            if agent.algorithm.use_rnd:
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

        if all(done): break

    return per_actor_trajectories


def gather_experience_parallel(env, agent, training, max_obs_count=1e7, env_configs=None, sum_writer=None):
    '''
    Runs a single multi-agent rl loop until the number of observation, `max_obs_count`, is reached.
    The observations vector is of length n, where n is the number of agents.
    observations[i] corresponds to the oberservation of agent i.
    :param env: ParallelEnv wrapper around an OpenAI gym environment
    :param agent: Agent policy used to take actionsin the environment and to process simulated experiences
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :param max_obs_count: Maximum number of observations to gather data for.
    :param env_configs: configuration dictionnary to use when resetting the environments.
    :param sum_writer: SummaryWriter.
    :returns: 
    '''
    observations = env.reset(env_configs=env_configs)

    nbr_actors = env.get_nbr_envs()
    agent.set_nbr_actor(nbr_actors)
    done = [False]*nbr_actors
    
    per_actor_trajectories = [list() for i in range(nbr_actors)]
    trajectories = list()
    total_returns = list()
    total_int_returns = list()
    episode_lengths = list()

    obs_count = 0
    episode_count = 0

    pbar = tqdm(total=max_obs_count)
    
    while True:
        action = agent.take_action(observations)
        succ_observations, reward, done, info = env.step(action)

        if training:
            agent.handle_experience(observations, 
                                    action, 
                                    reward, 
                                    succ_observations, 
                                    done)
        
        for actor_index in range(nbr_actors):
            obs_count += 1
            pbar.update(1)
    
            # Bookkeeping of the actors whose episode just ended:
            if done[actor_index]:
                episode_count += 1
                succ_observations[actor_index] = env.reset(env_configs=env_configs, env_indices=[actor_index])
                agent.reset_actors(indices=[actor_index])

                # Logging:
                trajectories.append(per_actor_trajectories[actor_index])
                total_returns.append(sum([ exp[2] for exp in trajectories[-1]]))
                total_int_returns.append(sum([ exp[3] for exp in trajectories[-1]]))
                episode_lengths.append(len(trajectories[-1]))
                
                sum_writer.add_scalar('Training/TotalReturn', total_returns[-1], episode_count)
                sum_writer.add_scalar('Training/TotalIntReturn', total_int_returns[-1], episode_count)
                
                if len(trajectories) >= nbr_actors:
                    mean_total_return = sum( total_returns) / len(trajectories)
                    std_ext_return = math.sqrt( sum( [math.pow( r-mean_total_return ,2) for r in total_returns]) / len(total_returns) )
                    mean_total_int_return = sum( total_int_returns) / len(trajectories)
                    std_int_return = math.sqrt( sum( [math.pow( r-mean_total_int_return ,2) for r in total_int_returns]) / len(total_int_returns) )
                    mean_episode_length = sum( episode_lengths) / len(trajectories)
                    std_episode_length = math.sqrt( sum( [math.pow( l-mean_episode_length ,2) for l in episode_lengths]) / len(episode_lengths) )
                
                    sum_writer.add_scalar('Training/StdIntReturn', std_int_return, episode_count // nbr_actors)
                    sum_writer.add_scalar('Training/StdExtReturn', std_ext_return, episode_count // nbr_actors)

                    sum_writer.add_scalar('Training/MeanTotalReturn', mean_total_return, episode_count // nbr_actors)
                    sum_writer.add_scalar('Training/MeanTotalIntReturn', mean_total_int_return, episode_count // nbr_actors)
                    
                    sum_writer.add_scalar('Training/MeanEpisodeLength', mean_episode_length, episode_count // nbr_actors)
                    sum_writer.add_scalar('Training/StdEpisodeLength', std_episode_length, episode_count // nbr_actors)

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
            if agent.algorithm.use_rnd:
                get_intrinsic_reward = getattr(agent, "get_intrinsic_reward", None)
                if callable(get_intrinsic_reward):
                    pa_int_r = agent.get_intrinsic_reward(actor_index)
            per_actor_trajectories[actor_index].append( (pa_obs, pa_a, pa_r, pa_int_r, pa_succ_obs, pa_done) )

        observations = copy.deepcopy(succ_observations)
        
        if obs_count >= max_obs_count:  break

    return agent