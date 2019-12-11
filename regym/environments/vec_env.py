import gym
import numpy as np 
import copy
import time 
from .utils import EnvironmentCreator


class VecEnv():
    def __init__(self, env_creator, nbr_parallel_env, nbr_frame_stacking=1, single_agent=True, worker_id=None):
        self.env_creator = env_creator
        self.nbr_parallel_env = nbr_parallel_env
        self.nbr_frame_stacking = nbr_frame_stacking
        self.env_processes = None
        self.single_agent = single_agent

        self.env_queues = [None]*self.nbr_parallel_env
        self.env_configs = [None]*self.nbr_parallel_env
        self.env_processes = [None]*self.nbr_parallel_env
        if worker_id is None:
            self.worker_ids = [None]*self.nbr_parallel_env
        elif isinstance(worker_id, int):
            self.worker_ids = [worker_id]*self.nbr_parallel_env
        elif isinstance(worker_id, list):
            self.worker_ids = worker_id
        else:
            raise NotImplementedError

        self.dones = [False]*self.nbr_parallel_env
        self.previous_dones = copy.deepcopy(self.dones)

    def get_nbr_envs(self):
        return self.nbr_parallel_env

    def launch_env_process(self, idx, worker_id_offset=0):
        self.env_queues[idx] = {'in':list(), 'out':list()}
        wid = self.worker_ids[idx]
        if wid is not None: wid += worker_id_offset
        self.env_processes[idx] = self.env_creator(worker_id=wid)
        time.sleep(10)

    def clean(self, idx):
        self.env_processes[idx].close()

    def check_update_reset_env_process(self, idx, env_configs=None, reset=False):
        p = self.env_processes[idx]
        if p is None:
            self.launch_env_process(idx)
            print('Launching environment {}...'.format(idx))
        
        if reset:
            if env_configs is not None: 
                self.env_configs[idx] = env_configs[idx]
            env_config = copy.deepcopy(self.env_configs[idx]) 
            if env_config is not None and 'worker_id' in env_config: 
                env_config.pop('worker_id')
            if env_config is None:
                out = self.env_processes[idx].reset()
            else:
                out = self.env_processes[idx].reset(env_config)
            self.env_queues[idx]['out'] = out

    def get_from_queue(self, idx, exhaust_first_when_failure=False):
        out = self.env_queues[idx]['out']
        return out

    def put_action_in_queue(self, action, idx):
        self.env_queues[idx]['out'] = self.env_processes[idx].step(action)

    def reset(self, env_configs=None, env_indices=None) :
        if env_indices is None: env_indices = range(self.nbr_parallel_env)
        
        if env_configs is not None: 
            self.worker_ids = [ env_config.pop('worker_id', None) for env_config in env_configs]
         
        for idx in env_indices:
            self.check_update_reset_env_process(idx, env_configs=env_configs, reset=True)

        observations = [self.get_from_queue(idx) for idx in env_indices] 
        
        if self.single_agent:
            observations = [ np.concatenate([obs]*self.nbr_frame_stacking, axis=-1) for obs in observations]
            per_env_obs = np.concatenate( [ np.array(obs).reshape(1, *(obs.shape)) for obs in observations], axis=0)
        else:
            per_env_obs = [ np.concatenate( [ np.array(obs[idx_agent]).reshape(1, *(obs[idx_agent].shape)) for obs in observations], axis=0) for idx_agent in range(len(observations[0]) ) ]
        
        for idx in env_indices:
            self.dones[idx] = False
        self.previous_dones = copy.deepcopy(self.dones)

        return per_env_obs

    def step(self, action_vector):
        observations = []
        rewards = []
        infos = []
        
        batch_env_index = -1
        for env_index in range(len(self.env_queues) ):
            if self.dones[env_index]:
                continue
            batch_env_index += 1
            
            if self.single_agent:
                pa_a = action_vector[batch_env_index]
            else:
                pa_a = [ action_vector[idx_agent][batch_env_index] for idx_agent in range( len(action_vector) ) ]
            
            for i in range(self.nbr_frame_stacking):
                self.put_action_in_queue(action=pa_a, idx=env_index)

        for env_index in range(len(self.env_queues) ):
            if self.dones[env_index]:
                infos.append(None)
                continue
            
            obses = []
            rs = []
            dones = []
            infs = []
            for i in range(self.nbr_frame_stacking):
                experience = self.get_from_queue(idx=env_index, exhaust_first_when_failure=True)
                obs, r, done, info = experience
                obses.append(obs)
                rs.append(r)
                dones.append(done)
                infs.append(info)

            obs = np.concatenate(obses, axis=-1)
            r = sum(rs)
            done = any(dones)
            info = infs 

            observations.append( obs )
            rewards.append( r )
            self.dones[env_index] = done
            infos.append(info)
        
        self.previous_dones = copy.deepcopy(self.dones[env_index]) 
            
        if self.single_agent:
            per_env_obs = np.concatenate( [ np.array(obs).reshape(1, *(obs.shape)) for obs in observations], axis=0)
            per_env_reward = np.concatenate( [ np.array(r).reshape(-1) for r in rewards], axis=0)
        else:
            per_env_obs = [ np.concatenate( [ np.array(obs[idx_agent]).reshape(1,-1) for obs in observations], axis=0) for idx_agent in range(len(observations[0]) ) ]
            per_env_reward = [ np.concatenate( [ np.array(r[idx_agent]).reshape((-1)) for r in rewards], axis=0) for idx_agent in range(len(rewards[0]) ) ]

        return per_env_obs, per_env_reward, self.dones, infos

    def close(self) :
        # Tell the processes to terminate themselves:
        if self.env_processes is not None:
            for env_index in range(len(self.env_processes)):
                self.env_processes[env_index].close()
        

class VecEnvironmentCreationFunction():

    def __init__(self, environment_name_cli, nbr_parallel_env):
        valid_environments = ['RockPaperScissors-v0','RoboschoolSumo-v0','RoboschoolSumoWithRewardShaping-v0']
        if environment_name_cli not in valid_environments:
            raise ValueError("Unknown environment {}\t valid environments: {}".format(environment_name_cli, valid_environments))
        self.environment_name = environment_name_cli
        self.nbr_parallel_env = nbr_parallel_env

    def __call__(self):
        return VecEnv(self.environment_name, self.nbr_parallel_env)