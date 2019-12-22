import gym
import numpy as np 
import copy
import time 
from .utils import EnvironmentCreator

import torch
# https://github.com/pytorch/pytorch/issues/11201:
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.multiprocessing import Process, Queue
from threading import Thread

import gc 

import sys
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

forkedPdb = ForkedPdb()


USE_PROC =True

#def env_worker(env, queue_in, queue_out, worker_id=None):
def env_worker(envCreator, queue_in, queue_out, worker_id=None, seed=0):
    continuer = True
    env = envCreator(worker_id=worker_id, seed=seed)

    obs = None
    r = None
    done = None
    info = None

    try:
        while continuer:
            instruction = queue_in.get()

            if isinstance(instruction,bool):
                continuer = False
            elif isinstance(instruction, str) or isinstance(instruction, tuple):
                env_config = instruction[-1]
                if env_config is None: observations = env.reset()
                else:  observations = env.reset(env_config) 
                done = False
                queue_out.put(observations)
            else :
                pa_a = instruction
                obs, r, done, info = env.step( pa_a)
                #env.render()
                queue_out.put( [obs,r,done,info] )
    except Exception as e:
        print(e)
        #forkedPdb.set_trace()
    finally:
        env.close()


class ParallelEnv():
    def __init__(self, env_creator, nbr_parallel_env, single_agent=True, seed=0, gathering=True):
        self.gathering = gathering
        self.seed = seed
        self.env_creator = env_creator
        self.nbr_parallel_env = nbr_parallel_env
        self.single_agent = single_agent

        self.env_queues = [None]*self.nbr_parallel_env
        self.env_configs = [None]*self.nbr_parallel_env
        self.env_processes = [None]*self.nbr_parallel_env
        self.worker_ids = [None]*self.nbr_parallel_env
        self.count_failures = [0]*self.nbr_parallel_env
        self.env_actions = [None]*self.nbr_parallel_env

        self.dones = [False]*self.nbr_parallel_env
        self.previous_dones = copy.deepcopy(self.dones)

    def seed(self, seed):
        self.seed = seed 

    def get_nbr_envs(self):
        return self.nbr_parallel_env

    def set_nbr_envs(self, nbr_parallel_env):
        if self.nbr_parallel_env != nbr_parallel_env:
            self.nbr_parallel_env = nbr_parallel_env
            self.close()

    def launch_env_process(self, idx, worker_id_offset=0):
        global USE_PROC
        self.env_queues[idx] = {'in':Queue(), 'out':Queue()}
        
        wid = self.worker_ids[idx]
        if wid is not None: wid += worker_id_offset
        seed = self.seed+idx
        
        if USE_PROC:
            p = Process(target=env_worker, args=(self.env_creator, *(self.env_queues[idx].values()), wid, seed) )
        else:
            p = Thread(target=env_worker, args=(self.env_creator, *(self.env_queues[idx].values()), wid, seed) )
        p.start()
        
        self.env_processes[idx] = p

    def clean(self, idx):
        self.env_processes[idx].terminate()
        self.env_processes[idx] = None
        self.env_queues[idx] = None
        gc.collect()

    def check_update_reset_env_process(self, idx, env_configs=None, reset=False):
        p = self.env_processes[idx]
        
        if p is None:
            self.launch_env_process(idx)
            print('Launching environment {}...'.format(idx))
        elif not(p.is_alive()):
            self.clean(idx)
            # Waiting for the sockets to detach:
            time.sleep(2)
            # Relaunching again...
            if self.count_failures[idx] == 0:
                self.count_failures[idx] = 1
            elif self.count_failures[idx] == 1:
                self.count_failures[idx] = -1
            elif self.count_failures[idx] == -1:
                self.count_failures[idx] = 0
            worker_id_offset = self.count_failures[idx]*self.nbr_parallel_env
            self.launch_env_process(idx, worker_id_offset=worker_id_offset)
            print('Relaunching environment {}...'.format(idx))
            self.env_queues[idx]['in'].put( ('reset', env_config))
            print('Resetting environment {}...'.format(idx))
        
        if reset:
            if env_configs is not None: 
                self.env_configs[idx] = env_configs[idx]
            env_config = copy.deepcopy(self.env_configs[idx]) 
            if env_config is not None and 'worker_id' in env_config: env_config.pop('worker_id')
            self.env_queues[idx]['in'].put( ('reset', env_config))

    def get_from_queue(self, idx, exhaust_first_when_failure=False):
        out = None
        while out is None:
            try:
                # Block/wait for at most 60 seconds:
                out = self.env_queues[idx]['out'].get(block=True,timeout=60)
            except Exception as e:
                print('Exception: {}'.format(e))
                # Otherwise, we assume that there is an issue with the environment
                # And thus we relaunch it, after waiting sufficiently to be able to do so:
                print('Environment {} encountered an issue.'.format(idx))
                self.check_update_reset_env_process(idx=idx, env_configs=None, reset=True)
                if exhaust_first_when_failure:
                    out = None 
                    exhaust = self.env_queues[idx]['out'].get(block=True,timeout=None)
                    self.put_action_in_queue(action=self.env_actions[idx], idx=idx)
                    
        return out

    def put_action_in_queue(self, action, idx):
        self.env_actions[idx] = action
        self.env_queues[idx]['in'].put(action)

    def reset(self, env_configs=None, env_indices=None) :
        if env_indices is None: env_indices = range(self.nbr_parallel_env)
        
        if env_configs is not None: 
            self.worker_ids = [ env_config.pop('worker_id', None) for env_config in env_configs]
         
        for idx in env_indices:
            self.check_update_reset_env_process(idx, env_configs=env_configs, reset=True)

        observations = [ self.get_from_queue(idx) for idx in env_indices] 

        if self.single_agent:
            per_env_obs = np.concatenate( [ np.expand_dims(np.array(obs), axis=0) for obs in observations], axis=0)
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
            if not(self.gathering) and self.dones[env_index]:
                continue
            batch_env_index += 1
            
            if self.single_agent:
                pa_a = action_vector[batch_env_index]
            else:
                pa_a = [ action_vector[idx_agent][batch_env_index] for idx_agent in range( len(action_vector) ) ]
            
            self.put_action_in_queue(action=pa_a, idx=env_index)

        for env_index in range(len(self.env_queues) ):
            if not(self.gathering) and self.dones[env_index]:
                infos.append(None)
                continue
            
            experience = self.get_from_queue(idx=env_index, exhaust_first_when_failure=True)
            obs, r, done, info = experience
            
            observations.append( obs )
            rewards.append( r )
            self.dones[env_index] = done
            infos.append(info)
            
        self.previous_dones = copy.deepcopy(self.dones[env_index]) 
            
        if self.single_agent:
            per_env_obs = np.concatenate( [ np.expand_dims(np.array(obs), axis=0) for obs in observations], axis=0)
            per_env_reward = np.concatenate( [ np.array(r).reshape(-1) for r in rewards], axis=0)
        else:
            per_env_obs = [ np.concatenate( [ np.array(obs[idx_agent]).reshape(1,-1) for obs in observations], axis=0) for idx_agent in range(len(observations[0]) ) ]
            per_env_reward = [ np.concatenate( [ np.array(r[idx_agent]).reshape((-1)) for r in rewards], axis=0) for idx_agent in range(len(rewards[0]) ) ]

        return per_env_obs, per_env_reward, self.dones, infos

    def close(self) :
        # Tell the processes to terminate themselves:
        if self.env_processes is not None:
            for env_index in range(len(self.env_processes)):
                if self.env_processes[env_index] is None: continue

                self.env_queues[env_index]['in'].put(False)
                self.env_processes[env_index].join()
                self.env_processes[env_index].terminate()
                self.env_processes[env_index] = None
                
                self.env_queues[env_index]['in'].close()
                self.env_queues[env_index]['in'] = None
                self.env_queues[env_index]['out'].close()
                self.env_queues[env_index]['out'] = None
                gc.collect()

        self.env_queues = [None]*self.nbr_parallel_env
        self.env_configs = [None]*self.nbr_parallel_env
        self.env_processes = [None]*self.nbr_parallel_env
        self.worker_ids = [None]*self.nbr_parallel_env
        self.count_failures = [0]*self.nbr_parallel_env
        self.env_actions = [None]*self.nbr_parallel_env

        self.dones = [False]*self.nbr_parallel_env
        self.previous_dones = copy.deepcopy(self.dones)

class ParallelEnvironmentCreationFunction():

    def __init__(self, environment_name_cli, nbr_parallel_env):
        valid_environments = ['RockPaperScissors-v0','RoboschoolSumo-v0','RoboschoolSumoWithRewardShaping-v0']
        if environment_name_cli not in valid_environments:
            raise ValueError("Unknown environment {}\t valid environments: {}".format(environment_name_cli, valid_environments))
        self.environment_name = environment_name_cli
        self.nbr_parallel_env = nbr_parallel_env

    def __call__(self):
        return ParallelEnv(self.environment_name, self.nbr_parallel_env)