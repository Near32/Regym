from typing import Dict, Any

import gym

from .gym_parser import parse_gym_environment
from .unity_parser import parse_unity_environment
from .parallel_env import ParallelEnv
from .vec_env import VecEnv
from .utils import EnvironmentCreator
from .task import Task, EnvType


def generate_task(env_name: str, 
                  env_type: EnvType = EnvType.SINGLE_AGENT, 
                  nbr_parallel_env: int = 1, 
                  wrapping_fn: object = None, 
                  test_wrapping_fn: object = None,
                  env_config: Dict[str,Any] = {},
                  test_env_config: Dict[str,Any] = {},
                  seed: int = 0,
                  test_seed: int = 1,
                  train_video_recording_episode_period: int = None,
                  train_video_recording_dirpath: str = './tmp/recordings/train/',
                  train_video_recording_render_mode: str = 'rgb_array',
                  test_video_recording_episode_period: int = None,
                  test_video_recording_dirpath: str = './tmp/recordings/teset/',
                  test_video_recording_render_mode: str = 'rgb_array',
                  static: bool = False,
                  gathering: bool = False) -> Task:
    '''
    Returns a regym.environments.Task by creating an environment derived from :param: env_name
    and extracting relevant information used to build regym.rl_algorithms.agents from the Task.
    If :param: env_name matches a registered OpenAI Gym environment it will create it from there
    If :param: env_name points to a (platform specific) UnityEnvironment executable, it will generate a Unity environment
    In the case of :param: env_name being detected as both an OpenAI Gym and Unity environmet, an error will be raised
    
    Note: :param wrapping_fn: is only usable with gym environment, not with Unity environments.

    :param env_name: String identifier for the environment to be created
    :param env_type: Determines whether the parameter is (single/multi)-agent
                     and how are the environment processes these actions
                     (i.e all actions simultaneously, or sequentially)
    :param nbr_parallel_env: number of environment to create and experience in parallel.
    :param wrapping_fn: Function used to wrap the environment.
    :param test_wrapping_fn: Function used to wrap the test environment.
    :param seed: int to seed the environment with...
    :param test_seed: int to seed the test environment with...
    :param gathering: Bool specifying whether we are gathering experience or running evaluation episodes.
    :returns: Task created from :param: env_name
    '''
    if env_name is None: raise ValueError('Parameter \'env_name\' was None')
    try:
        is_gym_environment = any([env_name == spec.id for spec in gym.envs.registry.all()]) # Checks if :param: env_name was registered
    except Exception as e:
        print(f"WARNING: OpenAI gym version does not allow access to registry.all(): {e}")
        print(F"WARNING: trying while assuming it is a dict...")
        is_gym_environment = any([env_name == spec for spec in gym.envs.registry.keys()])

    is_gymnasium_environment = False
    try:
        import gymnasium
        is_gymnasium_environment = any([env_name == spec for spec in gymnasium.envs.registry.keys()]) # Checks if :param: env_name was registered
    except Exception as e:
        print(f"WARNING: exception while checking whether the environment is from gymnasium : {e}")
    is_unity_environment = check_for_unity_executable(env_name)

    task = None
    env = None
    if is_gym_environment and is_unity_environment: 
        raise ValueError(f'{env_name} exists as both a Gym and an Unity environment. Rename Unity environment to remove duplicate problem.')
    elif is_gym_environment or is_gymnasium_environment: 
        if is_gym_environment:
            env = gym.make(env_name, **env_config)
        else:
            env = gymnasium.make(env_name, **env_config)
            print(f"WARNING: As a gymnasium environment, seeding is only available upon reset.")
            print(f"WARNING: As a result, seeding is meant to enforce episode-level similarity, rather than environment-level control of the randomness.")
        if wrapping_fn is not None: 
            env = wrapping_fn(env=env)
        if is_gym_environment:
            task = parse_gym_environment(env, env_type)
        else:
            from .gymnasium_parser import parse_gymnasium_environment
            task = parse_gymnasium_environment(env, env_type)
    elif is_unity_environment: 
        task = parse_unity_environment(env_name, env_type)
    else: 
        raise ValueError(f'Environment \'{env_name}\' was not recognized as either a Gym nor a Unity environment')

    env_creator = EnvironmentCreator(env_name, is_unity_environment, is_gym_environment, is_gymnasium_environment, wrapping_fn=wrapping_fn, env_config=env_config)
    test_env_creator = EnvironmentCreator(env_name, is_unity_environment, is_gym_environment, is_gymnasium_environment, wrapping_fn=test_wrapping_fn, env_config=test_env_config)

    task = Task(task.name, 
                #ParallelEnv(env_creator, nbr_parallel_env, seed=seed), 
                VecEnv(
                    env_creator, 
                    nbr_parallel_env,
                    single_agent=(env_type==EnvType.SINGLE_AGENT), 
                    seed=seed, 
                    static=static,
                    gathering=gathering,
                    video_recording_episode_period=train_video_recording_episode_period,
                    video_recording_dirpath=train_video_recording_dirpath,
                    video_recording_render_mode=train_video_recording_render_mode,
                    initial_env=env,
                ), 
                env_type,
                VecEnv(
                    test_env_creator, 
                    nbr_parallel_env,
                    single_agent=(env_type==EnvType.SINGLE_AGENT), 
                    seed=test_seed,
                    static=static,
                    gathering=False,
                    video_recording_episode_period=test_video_recording_episode_period,
                    video_recording_dirpath=test_video_recording_dirpath,
                    video_recording_render_mode=test_video_recording_render_mode,
                ),
                task.state_space_size, 
                task.action_space_size, 
                task.observation_shape, 
                task.observation_type, 
                task.action_dim, 
                task.action_type, 
                task.hash_function,
                task.goal_shape,
                task.goal_type)

    return task

def check_for_unity_executable(env_name):
    '''
    Checks if :param: env_name points to a Unity Executable
    :param env_name: String identifier for the environment to be created
    :returns: Boolean whether :param: env_name is a Unity executable
    '''
    import os, platform
    valid_extensions = {'Linux': '.x86_64', 'Darwin': '.app', 'Windows': '.exe'}
    return os.path.isfile(env_name + valid_extensions[platform.system()])
