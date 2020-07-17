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
                  seed: int = 0,
                  test_seed: int = 1,
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
    is_gym_environment = any([env_name == spec.id for spec in gym.envs.registry.all()]) # Checks if :param: env_name was registered
    is_unity_environment = check_for_unity_executable(env_name)

    task = None
    if is_gym_environment and is_unity_environment: raise ValueError(f'{env_name} exists as both a Gym and an Unity environment. Rename Unity environment to remove duplicate problem.')
    elif is_gym_environment: 
        env = gym.make(env_name)
        if wrapping_fn is not None: env = wrapping_fn(env=env)
        task = parse_gym_environment(env, env_type)
        env.close()
    elif is_unity_environment: task = parse_unity_environment(env_name, env_type)
    else: raise ValueError(f'Environment \'{env_name}\' was not recognized as either a Gym nor a Unity environment')

    env_creator = EnvironmentCreator(env_name, is_unity_environment, is_gym_environment, wrapping_fn=wrapping_fn)
    test_env_creator = EnvironmentCreator(env_name, is_unity_environment, is_gym_environment, wrapping_fn=test_wrapping_fn)

    task = Task(task.name, 
                #ParallelEnv(env_creator, nbr_parallel_env, seed=seed), 
                VecEnv(env_creator, nbr_parallel_env, seed=seed, gathering=gathering), 
                env_type,
                VecEnv(test_env_creator, nbr_parallel_env, seed=test_seed,gathering=False),
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
