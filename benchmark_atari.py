import logging
import yaml
import os
import sys
import yaml
from typing import Dict


import torch
import numpy as np
import math 
import gym
import random
import cv2 
import time

import regym
from regym.environments import parse_environment
from regym.rl_loops.singleagent_loops import rl_loop
from regym.util.experiment_parsing import initialize_algorithms
from regym.util.experiment_parsing import filter_relevant_agent_configurations

from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt 
import matplotlib.animation as anim

from collections import deque
from functools import partial

gif_interval = 100


def make_gif_with_graph(trajectory, data, episode=0, actor_idx=0, path='./', divider=4):
    print("GIF Making: ...", end='\r')
    fig = plt.figure()
    imgs = []
    gd = []
    for idx, (state, d) in enumerate(zip(trajectory,data)):
        if state.shape[-1] != 3:
            # handled Stacked images...
            per_image_first_channel_indices = range(0,state.shape[-1]+1,3)
            ims = [ state[...,idx_begin:idx_end] for idx_begin, idx_end in zip(per_image_first_channel_indices,per_image_first_channel_indices[1:])]
            for img in ims:
                imgs.append( img)
                gd.append(d)
        else:
            imgs.append(state)
            gd.append(d)

    gifimgs = []
    for idx,img in enumerate(imgs):
        if idx%divider: continue
        plt.subplot(211)
        gifimg = plt.imshow(img, animated=True)
        ax = plt.subplot(212)
        x = np.arange(0,idx,1)
        y = np.asarray(gd[:idx])
        ax.set_xlim(left=0,right=idx+10)
        line = ax.plot(x, y, color='blue', marker='o', linestyle='dashed',linewidth=2, markersize=10)
        
        gifimgs.append([gifimg]+line)
        
    gif = anim.ArtistAnimation(fig, gifimgs, interval=200, blit=True, repeat_delay=None)
    path = os.path.join(path, f'./traj-ep{episode}-actor{actor_idx}.gif')
    print("GIF Saving: ...", end='\r')
    gif.save(path, dpi=None, writer='imagemagick')
    print("GIF Saving: DONE.", end='\r')
    #plt.show()
    plt.close(fig)
    print("GIF Making: DONE.", end='\r')


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


# # Wrappers:
# # Observation Wrappers:
'''
Adapted from:
https://github.com/chainer/chainerrl/blob/master/chainerrl/wrappers/atari_wrappers.py
'''
class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
    
    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=-1)
        if dtype is not None:
            out = out.astype(dtype)
        return out


# ## FrameSkipping/Stacking:
'''
Adapted from:
https://github.com/minerllabs/baselines/blob/master/general/chainerrl/baselines/env_wrappers.py

MIT License

Copyright (c) Kevin Denamganaï

Modifications:
Combination of the two wrappers into one.
'''
class FrameSkipStack(gym.Wrapper):
    """
    Return a stack of framed composed of every 'skip'-th repeat.
    The rewards are summed over the skipped and stackedd frames.
    
    This wrapper assumes:
    - the observation space of the environment to be frames solely.
    - the frames are concatenated on the last axis, i.e. the channel axis.
    """
    def __init__(self, env, skip=8, stack=4):
        gym.Wrapper.__init__(self,env)
        self.skip = skip if skip is not None else 0
        self.stack = stack if stack is not None else 1
        
        self.observations = deque([], maxlen=self.stack)
        
        assert(isinstance(self.env.observation_space, gym.spaces.Box))
        
        low_obs_space = np.repeat(self.env.observation_space.low, self.stack, axis=-1)
        high_obs_space = np.repeat(self.env.observation_space.high, self.stack, axis=-1)
        self.observation_space = gym.spaces.Box(low=low_obs_space, high=high_obs_space, dtype=self.env.observation_space.dtype)
    
    def _get_obs(self):
        assert(len(self.observations) == self.stack)
        return LazyFrames(list(self.observations))
        
    def reset(self, **args):
        obs = self.env.reset()
        for _ in range(self.stack):
            self.observations.append(obs)
        return self._get_obs()
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:break
        self.observations.append(obs)
        return self._get_obs(), total_reward, done, info


class FrameResizeWrapper(gym.ObservationWrapper):
    """
    """
    def __init__(self, env, size=(64, 64)):
        gym.ObservationWrapper.__init__(self, env=env)
        
        self.size = size
        if isinstance(self.size, int):
            self.size = (self.size, self.size)

        low = np.zeros((*self.size, 3))
        high  = 255*np.ones((*self.size, 3))
        
        self.observation_space = gym.spaces.Box(low=low, high=high)
    
    def observation(self, observation):
        obs = cv2.resize(observation, self.size)
        return obs



def pixelwrap_env(env, size, skip=None, stack=None):
    # Observations:
    #wrapped_env = GrayScaleObservation(env=env) 
    #wrapped_env = PixelObservationWrapper(env=env)
    wrapped_env = FrameResizeWrapper(env, size=size) 
    if skip is not None or stack is not None:
        wrapped_env = FrameSkipStack(env=wrapped_env, skip=skip, stack=stack)
    
    return wrapped_env

def train_and_evaluate(agent, task, sum_writer, base_path, offset_episode_count=0, nbr_episodes=1e4):

    nbr_observations = 1e7
    trained_agent = rl_loop.gather_experience_parallel(task.env,
                                                       agent,
                                                       training=True,
                                                       max_obs_count=nbr_observations,
                                                       env_configs=None,
                                                       sum_writer=sum_writer)
    
    task.env.close()
    
    raise 

    nbr_observations = 1e7
    max_episode_length = 1e4
    nbr_actors = task.env.get_nbr_envs()
    global gif_interval

    for i in tqdm(range(offset_episode_count, int(nbr_episodes))):
        trajectory = rl_loop.run_episode_parallel(task.env, 
                                                  agent, 
                                                  training=True, 
                                                  max_episode_length=max_episode_length,
                                                  env_configs=None)
        
        total_return = [ sum([ exp[2] for exp in t]) for t in trajectory]
        mean_total_return = sum( total_return) / len(trajectory)
        std_ext_return = math.sqrt( sum( [math.pow( r-mean_total_return ,2) for r in total_return]) / len(total_return) )
        
        total_int_return = [ sum([ exp[3] for exp in t]) for t in trajectory]
        mean_total_int_return = sum( total_int_return) / len(trajectory)
        std_int_return = math.sqrt( sum( [math.pow( r-mean_total_int_return ,2) for r in total_int_return]) / len(total_int_return) )

        for idx, (ext_ret, int_ret) in enumerate(zip(total_return, total_int_return)):
            sum_writer.add_scalar('Training/TotalReturn', ext_ret, i*len(trajectory)+idx)
            sum_writer.add_scalar('Training/TotalIntReturn', int_ret, i*len(trajectory)+idx)
        
        sum_writer.add_scalar('Training/StdIntReturn', std_int_return, i)
        sum_writer.add_scalar('Training/StdExtReturn', std_ext_return, i)

        episode_lengths = [ len(t) for t in trajectory]
        mean_episode_length = sum( episode_lengths) / len(trajectory)
        std_episode_length = math.sqrt( sum( [math.pow( l-mean_episode_length ,2) for l in episode_lengths]) / len(trajectory) )
        
        sum_writer.add_scalar('Training/MeanTotalReturn', mean_total_return, i)
        sum_writer.add_scalar('Training/MeanTotalIntReturn', mean_total_int_return, i)
        
        sum_writer.add_scalar('Training/MeanEpisodeLength', mean_episode_length, i)
        sum_writer.add_scalar('Training/StdEpisodeLength', std_episode_length, i)

        # Update configs:
        agent.episode_count += 1

        '''
        if i%gif_interval == 1:
            for actor_idx in range(min(4,nbr_actors)): 
                gif_traj = [ exp[0] for exp in trajectory[actor_idx]]
                gif_data = [ exp[2] for exp in trajectory[actor_idx]]
                #gif_data = [ exp[3] for exp in trajectory[actor_idx]]
                begin = time.time()
                #make_gif(gif_traj, episode=i, actor_idx=actor_idx, path=base_path)
                make_gif_with_graph(gif_traj, gif_data, episode=i, actor_idx=actor_idx, path=base_path)
                end = time.time()
                eta = end-begin
                print(f'Time: {eta} sec.')
        '''
    task.env.close()


def training_process(agent_config: Dict, task_config: Dict,
                     benchmarking_episodes: int, train_observation_budget: int,
                     base_path: str, seed: int):
    if not os.path.exists(base_path): os.makedirs(base_path)

    np.random.seed(seed)
    torch.manual_seed(seed)

    pixel_wrapping_fn = partial(pixelwrap_env,
                          size=task_config['observation_resize_dim'], 
                          skip=task_config['nbr_frame_skipping'], 
                          stack=task_config['nbr_frame_stacking'])

    task = parse_environment(task_config['env-id'],
                             nbr_parallel_env=task_config['nbr_actors'],
                             wrapping_fn=pixel_wrapping_fn)

    sum_writer = SummaryWriter(base_path)
    save_path = os.path.join(base_path,f"./{task_config['agent-id']}.agent")
    agent, offset_episode_count = check_path_for_agent(save_path)
    if agent is None: 
        agent = initialize_algorithms(environment=task,
                                      agent_configurations={task_config['agent-id']: agent_config})[0]
    agent.save_path = save_path
    regym.rl_algorithms.PPO.ppo.summary_writer = sum_writer
    
    train_and_evaluate(agent=agent,
                       task=task,
                       sum_writer=sum_writer,
                       base_path=base_path,
                       offset_episode_count=offset_episode_count)


def load_configs(config_file_path: str):
    all_configs = yaml.load(open(config_file_path))

    agents_config = all_configs['agents']
    experiment_config = all_configs['experiment']
    envs_config = experiment_config['tasks']

    return experiment_config, agents_config, envs_config


def test():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Atari Benchmark')

    config_file_path = './config.yaml'#sys.argv[1]
    experiment_config, agents_config, tasks_configs = load_configs(config_file_path)

    # Generate path for experiment
    base_path = experiment_config['experiment_id']
    if not os.path.exists(base_path): os.mkdir(base_path)

    for task_config in tasks_configs:
        agent_name = task_config['agent-id']
        env_name = task_config['env-id']
        path = f'{base_path}/{agent_name}/{env_name}'
        training_process(agents_config[task_config['agent-id']], task_config,
                         benchmarking_episodes=experiment_config['benchmarking_episodes'],
                         train_observation_budget=experiment_config['train_observation_budget'],
                         base_path=path,
                         seed=experiment_config['seed'])

if __name__ == '__main__':
    test()