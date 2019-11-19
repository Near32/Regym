import logging
import yaml
import os
import sys
import yaml
from typing import Dict


import torch
import numpy as np
from regym.environments import parse_environment
from regym.util.experiment_parsing import initialize_algorithms
from regym.util.experiment_parsing import filter_relevant_agent_configurations


def training_process(agent_config: Dict, env_config: Dict,
                     benchmarking_episodes: int, train_observation_budget: int,
                     base_path: str, seed: int):
    if not os.path.exists(base_path): os.makedirs(base_path)

    np.random.seed(seed)
    torch.manual_seed(seed)
    # TODO Create tensorboardx loggers

    # TODO pass env_config['nbr_actors'] and extra parameters to parse environment
    task = parse_environment(env_config['env-id'])

    # TODO
    # There is a preprocessed_observation_shape hyperparameter
    # that must be set in the config file
    agent = initialize_algorithms(env_config['env-id'],
                                  {env_config['agent-id']: agent_config})
    # TODO Train and test


def load_configs(config_file_path: str):
    all_configs = yaml.load(open(config_file_path))

    agents_config = all_configs['agents']
    experiment_config = all_configs['experiment']
    envs_config = experiment_config['tasks']

    return experiment_config, agents_config, envs_config


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Atari Benchmark')

    config_file_path = sys.argv[1]
    experiment_config, agents_config, envs_configs = load_configs(config_file_path)

    # Generate path for experiment
    base_path = experiment_config['experiment_id']
    if not os.path.exists(base_path): os.mkdir(base_path)

    for env_config in envs_configs:
        agent_name = env_config['agent-id']
        env_name = env_config['env-id']
        path = f'{base_path}/{agent_name}/{env_name}'
        training_process(agents_config[env_config['agent-id']], env_config,
                         benchmarking_episodes=experiment_config['benchmarking_episodes'],
                         train_observation_budget=experiment_config['train_observation_budget'],
                         base_path=path,
                         seed=experiment_config['seed'])
