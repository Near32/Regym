from typing import Dict, List, Callable
from copy import deepcopy
import pickle
import yaml
import os
import sys
import time
from functools import partial

import torch
from tqdm import tqdm
import numpy as np
import gym
import minerl
from tensorboardX import SummaryWriter

import regym

from regym.util.minerl import get_action_set, generate_action_parser, MineRLTrajectoryBasedEnv, trajectory_based_rl_loop, get_good_demo_names

from regym.environments import generate_task
from regym.environments.vec_env import VecEnv
from regym.rl_loops.singleagent_loops import rl_loop
from regym.util.experiment_parsing import initialize_agents

from regym.util.wrappers import LazyFrames, FrameSkipStack, ContinuingTimeLimit, minerl2020_wrap_env

#import aicrowd_helper

#from utility.parser import Parser
#import logging
#import coloredlogs
#coloredlogs.install(logging.DEBUG)
#logger = logging.getLogger(__name__)

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4*24*60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

"""
# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
parser = Parser('performance/',
                allowed_environment=MINERL_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)
"""

'''
Adapted from:
https://github.com/minerllabs/baselines/blob/master/general/chainerrl/baselines/env_wrappers.py

MIT License

Copyright (c) Kevin Denamganaï.

Modifications:
Removed every element but the pov observation...
Subsequent version will add the inventory, in some way...
'''
class POVObservationWrapper(gym.ObservationWrapper):
  """
  Returns a frame/gym.space.Box with multiple channels that account for alone:

  The parameter region_size is used to build squares of information that each corresponds
  to a different element in the 'inventory', or in the 'equipped_items'.
  """
  def __init__(self, env, scaling=True):
    gym.ObservationWrapper.__init__(self, env=env)

    self.scaling = scaling

    pov_space = self.env.observation_space.spaces['pov']
    low_dict = {'pov':pov_space.low}
    high_dict = {'pov':pov_space.high}
    self.scaler_dict = {'pov': (high_dict['pov']-low_dict['pov']) / 255.0}

    if False: #'vector' in self.env.observation_space:
      vector_space = self.env.observation_space.spaces['vector']
      low_dict['vector'] = vector_space.low
      high_dict['vector'] = vector_space.high
      self.scaler_dict['vector'] = (high_dict['vector']-low_dict['vector']) / 255.0

    low = self.observation(low_dict)
    high = self.observation(high_dict)

    self.observation_space = gym.spaces.Box(low=low, high=high)

  def observation(self, observation):
    obs = observation['pov']
    # Scaling requires float32 type but then it makes it incompatible with PIL images...
    #obs = obs.astype(np.float32)
    #obs /= self.scaler_dict['pov']
    pov_dtype = obs.dtype

    if False: #'vector' in observation:
      vector_scale = observation['vector'] / self._compass_angle_scale
      print(f"Vector scaled: {vector_scale}.")
      raise NotImplementedError
      vector_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=pov_dtype) * vector_scaled
      obs = np.concatenate([obs, compass_channel], axis=-1)

    return obs

"""
MIT License

Copyright (c) Kevin Denamganaï

"""
class DictActionWrapper(gym.ActionWrapper):
  """Convert MineRL env's `Dict` action space as a continuous action space.

  Parameters
  ----------
  env
  Wrapping gym environment.
  """

  def __init__(self, env):
    super().__init__(env)

    wrapping_action_space = self.env.action_space.spaces['vector']
    low = wrapping_action_space.low
    high = wrapping_action_space.high

    self.action_space = gym.spaces.Box(low=low, high=high)

  def action(self, action):
    return {'vector': action}



class _ContinuingTimeLimit(gym.Wrapper):
  """TimeLimit wrapper for continuing environments.
  This is similar gym.wrappers.TimeLimit, which sets a time limit for
  each episode, except that done=False is returned and that
  info['needs_reset'] is set to True when past the limit.
  Code that calls env.step is responsible for checking the info dict, the
  fourth returned value, and resetting the env if it has the 'needs_reset'
  key and its value is True.
  Args:
    env (gym.Env): Env to wrap.
    max_episode_steps (int): Maximum number of timesteps during an episode,
      after which the env needs a reset.
  """

  def __init__(self, env, max_episode_steps):
    super(ContinuingTimeLimit, self).__init__(env)
    self._max_episode_steps = max_episode_steps

    self._elapsed_steps = None

  def step(self, action):
    assert self._elapsed_steps is not None,\
      "Cannot call env.step() before calling reset()"
    observation, reward, done, info = self.env.step(action)
    self._elapsed_steps += 1

    if self._max_episode_steps <= self._elapsed_steps:
      info['needs_reset'] = True

    return observation, reward, done, info

  def reset(self):
    self._elapsed_steps = 0
    return self.env.reset()


def wrap_env(env,
       skip=None,
       stack=None,
       scaling=True,
       #region_size=8,
       ):
  if isinstance(env, gym.wrappers.TimeLimit):
    #logger.info('Detected `gym.wrappers.TimeLimit`! Unwrap it and re-wrap our own time limit.')
    env = env.env
    max_episode_steps = env.spec.max_episode_steps
    #logger.info(f"TimeLimit : {max_episode_steps}")
    env = ContinuingTimeLimit(env, max_episode_steps=max_episode_steps)
  # Observations:
  wrapped_env = POVObservationWrapper(env=env, scaling=scaling)
  if skip is not None or stack is not None:
    wrapped_env = FrameSkipStack(env=wrapped_env, skip=skip, stack=stack)
  # Actions:
  wrapped_env = DictActionWrapper(env=wrapped_env)
  return wrapped_env

def minerl_wrap_env(env,
                    size=84,
                    skip=None,
                    stack=None,
                    scaling=True,
                    #region_size=8,
                    grayscale=False,
                    reward_scheme='None'):
  if isinstance(env, gym.wrappers.TimeLimit):
    #logger.info('Detected `gym.wrappers.TimeLimit`! Unwrap it and re-wrap our own time limit.')
    env = env.env
    max_episode_steps = env.spec.max_episode_steps
    #max_episode_steps = env.env.spec.max_episode_steps
    assert( max_episode_steps == 8e3)
    env = ContinuingTimeLimit(env, max_episode_steps=max_episode_steps)

  # Observations:
  env = POVObservationWrapper(env=env, scaling=scaling)

  penalizing = ('penalizing' in reward_scheme)
  if penalizing: reward_scheme = reward_scheme.replace("penalizing", "")
  if reward_scheme == 'single_reward_episode':
    env = SingleRewardWrapper(env=env, penalizing=penalizing)
  elif 'progressive' in reward_scheme:
    reward_scheme = reward_scheme.replace("progressive", "")
    nbr_episode = 1e4
    try:
      reward_scheme = reward_scheme.replace("_", "")
      nbr_episode = float(reward_scheme)
      print(f"Reward Scheme :: Progressive :: nbr_episode = {nbr_episode}")
    except Exception as e:
      print(f'Reward Scheme :: number of episode not understood... ({reward_scheme})')
    env = ProgressivelyMultiRewardWrapper(env=env, penalizing=penalizing, nbr_episode=nbr_episode)

  if skip is not None or stack is not None:
    env = FrameSkipStack(env=env, skip=skip, stack=stack)
  # Actions:
  env = DictActionWrapper(env=env)

  return env


VERBOSE = False


def lr_setter(env, agent, value):
  global VERBOSE
  agent.algorithm.optimizer.lr = value
  if VERBOSE: print(f"LR Decay: {agent.algorithm.optimizer.lr}")

def ppo_clip_setter(env, agent, value):
  global VERBOSE
  agent.algorithm.kwargs['ppo_ratio_clip'] = max(value, 1e-8)
  if VERBOSE: print(f"PPO Clip Ratio Decay: {agent.algorithm.kwargs['ppo_ratio_clip']}")


class LinearInterpolationHook(object):
  """Hook to set a linearly interpolated value.
  Args:
  total_steps (int): Number of total steps.
  start_value (float): Start value.
  stop_value (float): Stop value.
  setter (callable): (env, agent, value) -> None
  """

  def __init__(self, total_steps, start_value, stop_value, setter):
    self.total_steps = total_steps
    self.start_value = start_value
    self.stop_value = stop_value
    self.setter = setter

  def __call__(self, env, agent, step):
    value = np.interp(step,
      [1, self.total_steps],
      [self.start_value, self.stop_value])
    self.setter(env, agent, value)


def check_path_for_agent(filepath, restore=True):
  #filepath = os.path.join(path,filename)
  agent = None
  offset_episode_count = 0
  if restore and os.path.isfile(filepath):
    print('==> loading checkpoint {}'.format(filepath))
    agent = torch.load(filepath)
    offset_episode_count = agent.episode_count
    #setattr(agent, 'episode_count', offset_episode_count)
    print('==> loaded checkpoint {}'.format(filepath))
  return agent, offset_episode_count


class MineRLTrajectoryEnvironmentCreator():
    def __init__(self, task_name, trajectory_names: List[str], wrapping_fn=None, action_parser: Callable=lambda x:x):
        self.trajectory_names = trajectory_names
        self.wrapping_fn = wrapping_fn

        self.next_env_pointer = 0  # Next environment index to create

        self.envs = []
        for trajectory_name in self.trajectory_names:
            data_pipeline = minerl.data.make(task_name)
            data_iterator = data_pipeline.load_data(trajectory_name)
            self.envs.append(MineRLTrajectoryBasedEnv(data_iterator, action_parser=action_parser))

    def __call__(self, worker_id=None, seed=0):
        env = self.envs[self.next_env_pointer]
        self.next_env_pointer = (self.next_env_pointer + 1) % len(self.trajectory_names)

        env.seed(seed)
        if self.wrapping_fn is not None: env = self.wrapping_fn(env=env)
        return env


from sklearn.metrics import pairwise_distances

def action_parser(action, action_set):
  true_action = action['vector'] if isinstance(action, dict) else action
  dis = pairwise_distances(action_set, true_action.reshape(1, -1))
  discrete_action = np.argmin(dis, axis=0)
  # (1,)
  return discrete_action

def load_demonstrations_into_replay_buffer(
      agent, 
      action_set,
      task_name: str, 
      seed: int, 
      wrapping_fn: Callable):
    if os.path.exists('good_demo_names.pickle'):
        good_demo_names = pickle.load(open('good_demo_names.pickle', 'rb'))
    else:
        good_demo_names = get_good_demo_names(
            task_name,
            path=None,
            score_percent=0.45
        )
        pickle.dump(good_demo_names, open('good_demo_names.pickle', "wb"))

    # Action set
    #continuous_to_discrete_action_parser = generate_action_parser(action_set)
    continuous_to_discrete_action_parser = partial(action_parser,
      action_set=action_set
    )

    next_batch_trajectory_names = []
    for i, demo_name in enumerate(good_demo_names):
        next_batch_trajectory_names += [demo_name]

        if (len(next_batch_trajectory_names) == agent.nbr_actor) or ((i + 1) == len(good_demo_names)):
            env_creator = MineRLTrajectoryEnvironmentCreator(
                task_name=task_name,
                trajectory_names=deepcopy(next_batch_trajectory_names),
                wrapping_fn=wrapping_fn,
                action_parser=continuous_to_discrete_action_parser
            )
            next_batch_trajectory_names = []

            vec_env = VecEnv(
                env_creator=env_creator,
                nbr_parallel_env=agent.nbr_actor,
                seed=seed,
                gathering=False, #True,
                video_recording_episode_period=None,
                video_recording_dirpath=None,
            )

            # Load demoonstrations to agent's replay buffer
            trajectory_based_rl_loop(
                agent=agent,
                minerl_trajectory_env=vec_env,
                action_parser=continuous_to_discrete_action_parser
            )

def train_and_evaluate(agent: object,
                       task: object,
                       sum_writer: object,
                       base_path: str,
                       offset_episode_count: int = 0,
                       nbr_max_observations: int = 1e7,
                       test_obs_interval: int = 1e4,
                       test_nbr_episode: int = 10,
                       benchmarking_record_episode_interval: int = None,
                       step_hooks = None):
  
  async = False
  if len(sys.argv) > 2:
    async = any(['async' in arg for arg in sys.argv])

  if async:
    trained_agent = rl_loop.async_gather_experience_parallel(
      task,
      agent,
      training=True,
      max_obs_count=nbr_max_observations,
      env_configs=None,
      sum_writer=sum_writer,
      base_path=base_path,
      test_obs_interval=test_obs_interval,
      test_nbr_episode=test_nbr_episode,
      benchmarking_record_episode_interval=benchmarking_record_episode_interval,
      step_hooks=step_hooks
    )
  else:    
    trained_agent = rl_loop.gather_experience_parallel(
      task,
      agent,
      training=True,
      max_obs_count=nbr_max_observations,
      env_configs=None,
      sum_writer=sum_writer,
      base_path=base_path,
      test_obs_interval=test_obs_interval,
      test_nbr_episode=test_nbr_episode,
      benchmarking_record_episode_interval=benchmarking_record_episode_interval,
      step_hooks=step_hooks
    )

  save_replay_buffer = False
  if len(sys.argv) > 2:
    save_replay_buffer = any(['save_replay_buffer' in arg for arg in sys.argv])

  trained_agent.save(with_replay_buffer=save_replay_buffer)
  print(f"Agent saved at: {trained_agent.save_path}")
  
  task.env.close()
  task.test_env.close()

  return trained_agent


def training_process(agent_config: Dict,
                     task_config: Dict,
                     benchmarking_interval: int = 1e4,
                     benchmarking_episodes: int = 10,
                     benchmarking_record_episode_interval: int = None,
                     train_observation_budget: int = 1e7,
                     base_path: str = './',
                     video_recording_episode_period_training: int = None,
                     video_recording_episode_period_benchmarking: int = None,
                     seed: int = 0):
  if not os.path.exists(base_path): os.makedirs(base_path)

  np.random.seed(seed)
  torch.manual_seed(seed)

  #torch.backends.cudnn.deterministic = True
  #torch.backends.cudnn.benchmark = False

  """
  pixel_wrapping_fn = partial(minerl_wrap_env,
    size=task_config['observation_resize_dim'],
    skip=task_config['nbr_frame_skipping'],
    stack=task_config['nbr_frame_stacking'],
    scaling=task_config['scaling'],
    grayscale=task_config['grayscale'],
    reward_scheme=task_config['reward_scheme']
  )
  """

  if os.path.exists('action_set.pickle'):
    action_set = pickle.load(open('action_set.pickle', 'rb'))
  else:
    action_set = get_action_set(
      task_name=task_config['env-id'],
      path=None,
      n_clusters=task_config['n_clusters'],
    )
    pickle.dump(action_set, open('action_set.pickle', "wb"))


  pixel_wrapping_fn = partial(minerl2020_wrap_env,
    action_set=action_set,
    skip=task_config['nbr_frame_skipping'],
    stack=task_config['nbr_frame_stacking'],
    previous_reward_action=True,
    trajectory_wrapping=False
  )

  test_pixel_wrapping_fn = pixel_wrapping_fn

  preloading_wrapping_fn = partial(minerl2020_wrap_env,
    action_set=action_set,
    skip=task_config['nbr_frame_skipping'],
    stack=task_config['nbr_frame_stacking'],
    previous_reward_action=True,
    trajectory_wrapping=True
  )

  task = generate_task(
    task_config['env-id'],
    nbr_parallel_env=task_config['nbr_actor'],
    wrapping_fn=pixel_wrapping_fn,
    test_wrapping_fn=test_pixel_wrapping_fn,
    seed=seed,
    test_seed=100+seed,
    train_video_recording_episode_period=video_recording_episode_period_training,
    train_video_recording_dirpath=os.path.join(base_path, 'recordings/train/'),
    test_video_recording_episode_period=video_recording_episode_period_benchmarking,
    test_video_recording_dirpath=os.path.join(base_path, 'recordings/test/'),
    gathering=True
  )


  agent_config['nbr_actor'] = task_config['nbr_actor']

  #sum_writer = SummaryWriter(base_path)
  regym.RegymSummaryWriterPath = base_path #regym.RegymSummaryWriter = GlobalSummaryWriter(base_path)
  sum_writer =  base_path
    
  save_path = os.path.join(base_path,f"./{task_config['agent-id']}.agent")
  agent, offset_episode_count = check_path_for_agent(save_path, restore=False)
  if agent is None:
    agent = initialize_agents(task=task,
                              agent_configurations={task_config['agent-id']: agent_config})[0]

  agent.save_path = save_path

  #regym.rl_algorithms.algorithms.DQN.dqn.summary_writer = sum_writer

  step_hooks = []
  '''
  lr_hook = LinearInterpolationHook(train_observation_budget, agent.algorithm.kwargs['learning_rate'], 0, lr_setter)
  step_hooks.append(lr_hook)
  print(f"Learning Rate Decay Hooked: {lr_hook}")

  if isinstance(agent, regym.rl_algorithms.agents.PPOAgent):
    clip_hook = LinearInterpolationHook(train_observation_budget, agent.algorithm.kwargs['ppo_ratio_clip'], 0, ppo_clip_setter)
    step_hooks.append(clip_hook)
    print(f"PPO Clip Ratio Decay Hooked: {clip_hook}")
  '''
  async_actor_agent = agent.get_async_actor()
  async_actor_agent = load_demonstrations_into_replay_buffer(
      async_actor_agent,
      action_set,
      task_name=task_config['env-id'],
      seed=seed,
      wrapping_fn=preloading_wrapping_fn)

  import ipdb; ipdb.set_trace()
  # Verify that agent's storages contains data

  if task_config['pre_train_on_demonstrations']:
      raise NotImplementedError

  trained_agent = train_and_evaluate(
    agent=agent,
    task=task,
    sum_writer=sum_writer,
    base_path=base_path,
    offset_episode_count=offset_episode_count,
    nbr_max_observations=train_observation_budget,
    test_obs_interval=benchmarking_interval,
    test_nbr_episode=benchmarking_episodes,
    benchmarking_record_episode_interval=benchmarking_record_episode_interval,
    step_hooks=step_hooks
  )

  return trained_agent, task

def load_configs(config_file_path: str):
  all_configs = yaml.load(open(config_file_path))

  agents_config = all_configs['agents']
  experiment_config = all_configs['experiment']
  envs_config = experiment_config['tasks']

  return experiment_config, agents_config, envs_config


def training(config_file_path):
  #logging.basicConfig(level=logging.INFO)
  #logger = logging.getLogger('MineRL Training.')

  experiment_config, agents_config, tasks_configs = load_configs(config_file_path)

  # Generate path for experiment
  base_path = experiment_config['experiment_id']
  if not os.path.exists(base_path): os.mkdir(base_path)

  trained_agents = []
  tasks = []
  for task_config in tasks_configs:
    agent_name = task_config['agent-id']
    env_name = task_config['env-id']
    run_name = task_config['run-id']
    path = f'{base_path}/{env_name}/{run_name}/{agent_name}'
    print(f"Path: -- {path} --")
    trained_agent, task = training_process(
      agents_config[task_config['agent-id']],
      task_config,
      benchmarking_interval=int(float(experiment_config['benchmarking_interval'])),
      benchmarking_episodes=int(float(experiment_config['benchmarking_episodes'])),
      benchmarking_record_episode_interval=int(float(experiment_config['benchmarking_record_episode_interval'])),
      train_observation_budget=int(float(experiment_config['train_observation_budget'])),
      base_path=path,
      seed=experiment_config['seed']
    )
    trained_agents.append(trained_agent)
    tasks.append(task)

  return trained_agents, tasks

def main(config_file_path: str):
  on_csgpu = False 
  if len(sys.argv) > 2:
      on_csgpu = any(['csgpu' in arg for arg in sys.argv])
  if on_csgpu:
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["JRE_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/jre"
    os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  
  return training(config_file_path)

if __name__ == '__main__':
  config_file_path = sys.argv[1]

  async = False 
  __spec__ = None
  if len(sys.argv) > 2:
      async = any(['async' in arg for arg in sys.argv])
  if async:
      torch.multiprocessing.freeze_support()
      torch.multiprocessing.set_start_method("forkserver", force=True)

  from torch.multiprocessing import Manager
  regym.RegymManager = Manager()

  main(config_file_path)

