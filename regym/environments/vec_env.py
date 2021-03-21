import gym
import numpy as np 
import copy
import time 
from .utils import EnvironmentCreator
from ..util.wrappers import PeriodicVideoRecorderWrapper

class VecEnv():
    def __init__(self, 
                 env_creator, 
                 nbr_parallel_env, 
                 single_agent=True, 
                 worker_id=None, 
                 seed=0, 
                 gathering=True,
                 video_recording_episode_period=None,
                 video_recording_dirpath='./tmp/regym/video_recording/',
                 video_recording_render_mode='rgb_array',
                 initial_env=None):
        
        self.video_recording_episode_period = video_recording_episode_period
        self.video_recording_dirpath = video_recording_dirpath
        self.video_recording_render_mode = video_recording_render_mode

        self.gathering = gathering
        self.seed = seed
        self.env_creator = env_creator
        self.nbr_parallel_env = nbr_parallel_env
        self.single_agent = single_agent

        self.initial_env = initial_env
        
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

    @property
    def observation_space(self):
        if self.env_processes[0] is None:
            self.launch_env_process(idx=0)
        return self.env_processes[0].observation_space

    @property
    def action_space(self):
        if self.env_processes[0] is None:
            self.launch_env_process(idx=0)
        return self.env_processes[0].action_space
    
    def seed(self, seed):
        self.seed = seed 

    def get_nbr_envs(self):
        return self.nbr_parallel_env

    def set_nbr_envs(self, nbr_parallel_env):
        if self.nbr_parallel_env != nbr_parallel_env:
            self.nbr_parallel_env = nbr_parallel_env
            self.close()

    def launch_env_process(self, idx, worker_id_offset=0):
        self.env_queues[idx] = {'in':list(), 'out':list()}
        wid = self.worker_ids[idx]
        if wid is not None: wid += worker_id_offset
        seed = self.seed+idx+1
        """
        if idx==0 and self.initial_env is not None:
            self.env_processes[idx] = self.initial_env
            self.initial_env = None
        else:
            self.env_processes[idx] = self.env_creator(worker_id=wid, seed=seed)
        """
        if idx==0 and self.initial_env is not None:
            self.initial_env = None
        self.env_processes[idx] = self.env_creator(worker_id=wid, seed=seed)
        
        if idx==0 and self.video_recording_episode_period is not None:
            self.env_processes[idx] = PeriodicVideoRecorderWrapper(
                env=self.env_processes[idx], 
                base_dirpath=self.video_recording_dirpath,
                video_recording_episode_period=self.video_recording_episode_period,
                render_mode=self.video_recording_render_mode,
            )

    def sample(self):
        sampled_actions = np.concatenate([env_proc.action_space.sample().reshape(self.nbr_parallel_env,-1) for env_proc in self.env_processes], axis=0)
        return sampled_actions

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

    def render(self, render_mode="rgb_array", env_indices=None) :
        if env_indices is None: env_indices = range(self.nbr_parallel_env)
        
        observations = []
        for idx in env_indices:
            obs = self.env_processes[idx].render(render_mode)
            observations.append(obs)

        return observations

    def reset(self, env_configs=None, env_indices=None) :
        if env_indices is None: env_indices = range(self.nbr_parallel_env)
        
        if env_configs is not None: 
            self.worker_ids = [ env_config.pop('worker_id', None) for env_config in env_configs]
        
        # Reset environments: 
        for idx in env_indices:
            self.check_update_reset_env_process(idx, env_configs=env_configs, reset=True)

        observations = []
        infos = []
        #for idx in env_indices:
        for idx in range(self.nbr_parallel_env):
            data = self.get_from_queue(idx) 
            if isinstance(data, tuple):
                if len(data) == 2:
                    obs, info = data
                elif len(data) == 4:
                    # not an environment that have just been resetted:
                    # obs, reward, done, info:
                    obs, reward, done, info = data
                else:
                    raise NotImplementedError
            else:
                obs, info = data, None 
            
            observations.append(obs)
            infos.append(info)

        
        if self.single_agent:
            per_env_obs = np.concatenate( [ np.expand_dims(np.array(obs), axis=0) for obs in observations], axis=0)
            per_env_infos = infos
        else:
            # agent/player x actor/env x ...
            per_env_obs = [ 
                np.concatenate([ 
                    #np.array(obs[idx_agent]).reshape(1,-1) 
                    np.expand_dims(obs[idx_agent], axis=0) if obs[idx_agent].shape[0]!=1 else obs[idx_agent] 
                    for obs in observations
                    ], 
                    axis=0
                ) 
                for idx_agent in range(len(observations[0])) 
            ]
            per_env_infos = [ 
                [ 
                    info[idx_agent] 
                    for info in infos
                ]
                for idx_agent in range(len(infos[0])) 
            ]
            
        for idx in env_indices:
            self.dones[idx] = False
        self.init_reward = []

        return copy.deepcopy([per_env_obs, per_env_infos])

    def step(self, action_vector, only_progress_non_terminated=True):
        observations = []
        rewards = []
        infos = []
        dones = []
        
        batch_env_index = -1
        for env_index in range(len(self.env_queues) ):
            if not(self.gathering) and self.dones[env_index] and not(only_progress_non_terminated):
                continue
            batch_env_index += 1
            
            if self.single_agent:
                pa_a = action_vector[batch_env_index]
            else:
                pa_a = [ action_vector[idx_agent][batch_env_index] for idx_agent in range( len(action_vector) ) ]
            
            if only_progress_non_terminated and self.dones[env_index]:  continue 

            self.put_action_in_queue(action=pa_a, idx=env_index)

        for env_index in range(len(self.env_queues) ):
            if not(self.gathering) and self.dones[env_index] and not(only_progress_non_terminated):
                infos.append(None)
                continue
            
            experience = self.get_from_queue(idx=env_index, exhaust_first_when_failure=True)
            obs, r, done, info = experience

            if len(self.init_reward)<len(self.env_queues):
                # Zero-out this initial reward:
                init_r = copy.deepcopy(r)
                if isinstance(init_r, list):
                    for ridx in range(len(init_r)):
                        init_r[ridx] = 0*init_r[ridx]  
                self.init_reward.append(init_r)

            observations.append( obs )
            rewards.append( r )

            if only_progress_non_terminated and self.dones[env_index] and not(all(self.dones)):
                done=False
                rewards[-1] = self.init_reward[env_index]
            else:
                self.dones[env_index] = done 

            dones.append(done)
            infos.append(info)
        
            
        if self.single_agent:
            per_env_obs = np.concatenate( [ np.expand_dims(np.array(obs), axis=0) for obs in observations], axis=0)
            per_env_reward = np.concatenate( [ np.array(r).reshape(-1) for r in rewards], axis=0)
            per_env_infos = infos
        else:
            # agent/player x actor/env x ...
            per_env_obs = [ 
                np.concatenate([ 
                    #np.array(obs[idx_agent]).reshape(1,-1) 
                    np.expand_dims(obs[idx_agent], axis=0) if obs[idx_agent].shape[0]!=1 else obs[idx_agent] 
                    for obs in observations
                    ], 
                    axis=0
                ) 
                for idx_agent in range(len(observations[0])) 
            ]
            per_env_reward = [ 
                np.concatenate([ 
                    np.array(r[idx_agent]).reshape((-1)) 
                    for r in rewards
                    ], 
                    axis=0
                ) 
                for idx_agent in range(len(rewards[0])) 
            ]
            per_env_infos = [ 
                [ 
                    info[idx_agent] 
                    for info in infos
                ]
                for idx_agent in range(len(infos[0])) 
            ]

        return copy.deepcopy([per_env_obs, per_env_reward, dones, per_env_infos])

    def close(self) :
        if self.env_processes is not None:
            for env_index in range(len(self.env_processes)):
                if self.env_processes[env_index] is None: continue
                self.env_processes[env_index].close()

        self.env_queues = [None]*self.nbr_parallel_env
        self.env_configs = [None]*self.nbr_parallel_env
        self.env_processes = [None]*self.nbr_parallel_env
        self.worker_ids = [None]*self.nbr_parallel_env
        
        self.dones = [False]*self.nbr_parallel_env
