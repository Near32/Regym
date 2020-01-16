import logging
import yaml
import os
import sys
from typing import Dict
from tensorboardX import SummaryWriter
from tqdm import tqdm
from functools import partial

import torch
import numpy as np

import regym
from regym.environments import generate_task
from regym.rl_loops.singleagent_loops import rl_loop
from regym.util.experiment_parsing import initialize_agents
from regym.util.wrappers import minerl_wrap_env

import gym
import minerl


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


def train_and_evaluate(agent: object, 
                       task: object, 
                       sum_writer: object, 
                       base_path: str, 
                       offset_episode_count: int = 0, 
                       nbr_max_observations: int = 1e7,
                       test_obs_interval: int = 1e4,
                       test_nbr_episode: int = 10):
    trained_agent = rl_loop.gather_experience_parallel(task,
                                                       agent,
                                                       training=True,
                                                       max_obs_count=nbr_max_observations,
                                                       env_configs=None,
                                                       sum_writer=sum_writer,
                                                       base_path=base_path,
                                                       test_obs_interval=test_obs_interval,
                                                       test_nbr_episode=test_nbr_episode)
    
    task.env.close()
    task.test_env.close()


def training_process(agent_config: Dict, 
                     task_config: Dict,
                     benchmarking_interval: int = 1e4,
                     benchmarking_episodes: int = 10, 
                     train_observation_budget: int = 1e7,
                     base_path: str = './', 
                     seed: int = 0):
    if not os.path.exists(base_path): os.makedirs(base_path)

    np.random.seed(seed)
    torch.manual_seed(seed)

    pixel_wrapping_fn = partial(minerl_wrap_env,
                                size=task_config['observation_resize_dim'], 
                                skip=task_config['nbr_frame_skipping'], 
                                stack=task_config['nbr_frame_stacking'],
                                scaling=True,
                                observation_wrapper=task_config['observation_wrapper'],
                                action_wrapper=task_config['action_wrapper'],
                                grayscale=task_config['grayscale'],
                                )
    
    task = generate_task(task_config['env-id'],
                         nbr_parallel_env=task_config['nbr_actor'],
                         wrapping_fn=pixel_wrapping_fn,
                         test_wrapping_fn=pixel_wrapping_fn,
                         gathering=True)

    agent_config['nbr_actor'] = task_config['nbr_actor']

    sum_writer = SummaryWriter(base_path)
    save_path = os.path.join(base_path,f"./{task_config['agent-id']}.agent")
    agent, offset_episode_count = check_path_for_agent(save_path)
    if agent is None: 
        agent = initialize_agents(task=task,
                                  agent_configurations={task_config['agent-id']: agent_config})[0]
    agent.save_path = save_path
    regym.rl_algorithms.PPO.ppo.summary_writer = sum_writer
    regym.rl_algorithms.A2C.a2c.summary_writer = sum_writer
    
    train_and_evaluate(agent=agent,
                       task=task,
                       sum_writer=sum_writer,
                       base_path=base_path,
                       offset_episode_count=offset_episode_count,
                       nbr_max_observations=train_observation_budget,
                       test_obs_interval=benchmarking_interval,
                       test_nbr_episode=benchmarking_episodes)


def load_configs(config_file_path: str):
    all_configs = yaml.load(open(config_file_path))

    agents_config = all_configs['agents']
    experiment_config = all_configs['experiment']
    envs_config = experiment_config['tasks']

    return experiment_config, agents_config, envs_config


def training():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('MineRL Training.')

    config_file_path = "./minerl_config.yaml"
    experiment_config, agents_config, tasks_configs = load_configs(config_file_path)

    # Generate path for experiment
    base_path = experiment_config['experiment_id']
    if not os.path.exists(base_path): os.mkdir(base_path)

    for task_config in tasks_configs:
        agent_name = task_config['agent-id']
        env_name = task_config['env-id']
        run_name = task_config['run-id']
        path = f'{base_path}/{env_name}/{run_name}/{agent_name}'
        print(f"Path: -- {path} --")
        training_process(agents_config[task_config['agent-id']], task_config,
                         benchmarking_interval=experiment_config['benchmarking_interval'],
                         benchmarking_episodes=experiment_config['benchmarking_episodes'],
                         train_observation_budget=int(float(experiment_config['train_observation_budget'])),
                         base_path=path,
                         seed=experiment_config['seed'])


if __name__ == '__main__':
    import os
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["JRE_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd/jre"
    os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    training()
