from typing import Dict, List

import time
import copy 
from collections import deque 
from functools import partial 

import ray

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from . import dqn_loss, ddqn_loss

import regym
from ..algorithm import Algorithm
from ...replay_buffers import PrioritizedReplayStorage, ReplayStorage

from ...networks import hard_update, soft_update, random_sample
from regym.rl_algorithms.utils import archi_concat_fn, _extract_rnn_states_from_batch_indices, _concatenate_hdict, _concatenate_list_hdict
from regym.thirdparty.Archi.Archi.model import Model as ArchiModel

import wandb
summary_writer = None 

import torch.multiprocessing as mp
#import multiprocessing as mp
#from multiprocessing.managers import BaseManager
from multiprocessing.managers import SyncManager
#from multiprocessing import Manager as AlgoManager
#class AlgoManager(BaseManager):  
#    pass
#AlgoManager.register('dict', dict)
#AlgoManager.register('PrioritizedReplayStorage', PrioritizedReplayStorage)

#from regym import BuildProxy
#PRSProxy = BuildProxy(PrioritizedReplayStorage)

from multiprocessing.managers import NamespaceProxy
import types
class PRSProxy(NamespaceProxy):
    _exposed_ = tuple(dir(PrioritizedReplayStorage))

    def __getattr__(self, name):
        result = super().__getattr__(name)
        if isinstance(result, types.MethodType):
            def wrapper(*args, **kwargs):
                self._callmethod(name, args, kwargs)
            return wrapper
        return result
    def __len__(self):
        callmethod = object.__getattribute__(self, "_callmethod")        
        return callmethod("__len__")

SyncManager.register('PrioritizedReplayStorage', PrioritizedReplayStorage, PRSProxy)

import logging 


class DQNAlgorithm(Algorithm):
    def __init__(self, kwargs, model, target_model=None, optimizer=None, loss_fn=dqn_loss.compute_loss, sum_writer=None, name='dqn_algo'):
        '''
        '''
        super(DQNAlgorithm, self).__init__(name=name)

        self.train_request_count = 0 

        self.kwargs = copy.deepcopy(kwargs)        
        self.use_cuda = kwargs["use_cuda"]
        self.nbr_actor = self.kwargs['nbr_actor']
        
        self.double = self.kwargs['double']
        self.dueling = self.kwargs['dueling']
        self.noisy = self.kwargs['noisy']
        self.n_step = self.kwargs['n_step'] if 'n_step' in self.kwargs else 1
        
        if self.n_step > 1:
            self.n_step_buffers = [deque(maxlen=self.n_step) for _ in range(self.nbr_actor)]

        self.use_PER = self.kwargs['use_PER']
        
        self.goal_oriented = self.kwargs['goal_oriented'] if 'goal_oriented' in self.kwargs else False
        self.use_HER = self.kwargs['use_HER'] if 'use_HER' in self.kwargs else False

        assert (self.use_HER and self.goal_oriented) or not(self.goal_oriented)

        self.model = model
        if self.kwargs['use_cuda']:
            self.model = self.model.cuda()

        if target_model is None:
            target_model = copy.deepcopy(self.model)

        self.target_model = target_model
        #self.target_model.share_memory()

        hard_update(self.target_model, self.model)
        if self.use_cuda:
            self.target_model = self.target_model.cuda()

        
        if optimizer is None:
            parameters = self.model.parameters()
            # Tuning learning rate with respect to the number of actors:
            # Following: https://arxiv.org/abs/1705.04862
            lr = float(kwargs['learning_rate']) 
            if kwargs['lr_account_for_nbr_actor']:
                lr *= self.nbr_actor
            print(f"Learning rate: {lr}")
            self.optimizer = optim.Adam(
                parameters, 
                lr=lr, 
                betas=(0.9,0.999), 
                eps=float(kwargs['adam_eps']),
                weight_decay=float(kwargs.get("adam_weight_decay", 0.0)),
            )
        else: self.optimizer = optimizer

        self.loss_fn = loss_fn
        print(f"WARNING: loss_fn is {self.loss_fn}")
            
        
        # DEPRECATED in order to allow extra_inputs infos 
        # stored in the rnn_states that acts as frame_states...
        #self.recurrent = False
        self.recurrent = True
        # TECHNICAL DEBT: check for recurrent property by looking at the modules in the model rather than relying on the kwargs that may contain
        # elements that do not concern the model trained by this algorithm, given that it is now use-able inside I2A...
        self.recurrent_nn_submodule_names = [hyperparameter for hyperparameter, value in self.kwargs.items() if isinstance(value, str) and 'RNN' in value]
        if len(self.recurrent_nn_submodule_names): self.recurrent = True

        self.storages = None
        self.use_mp = False 
        if self.use_mp:
            #torch.multiprocessing.freeze_support()
            if not hasattr(regym, 'AlgoManager'):
                torch.multiprocessing.set_start_method("forkserver")#, force=True)
                regym.AlgoManager = mp.Manager()
            #regym.RegymManager = SyncManager()
            #regym.RegymManager.start()

            regym.samples = regym.AlgoManager.dict()
            regym.sampling_config = regym.AlgoManager.dict()
            regym.sampling_config['stay_on'] = True
            regym.sampling_config['keep_sampling'] = False
            regym.sampling_config['minibatch_size'] = 32
            
            if not hasattr(self, 'keys'):
                self.keys = ['s', 'a', 'r', 'non_terminal']
                self.keys_to_retrieve = ['s', 'a', 'succ_s', 'r', 'non_terminal']
                if self.recurrent:
                    self.keys += ['rnn_states']
                    self.keys_to_retrieve += ['rnn_states', 'next_rnn_states']
            
            self.kremap = {
                's':'states',
                'a':'actions',
                'r':'rewards',
                'non_terminal':'non_terminals',
                'succ_s':'next_states',
            }
            self.reset_storages()
            
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            
            regym.sampling_process = mp.Process(
                target=self._mp_sampling,
                kwargs={
                    'storages':self.storages,
                    'sampling_config':regym.sampling_config,
                    'samples':regym.samples,
                },
            )
            regym.sampling_process.start()
        else:
            self.reset_storages()

        self.min_capacity = int(float(kwargs["min_capacity"]))
        self.batch_size = int(kwargs["batch_size"])
        self.nbr_minibatches = int(kwargs["nbr_minibatches"])

        self.TAU = self.kwargs.get('tau', 'None')
        if self.TAU == "None":
            assert 'inverted_tau' in self.kwargs
            self.inverted_TAU = self.kwargs.get('inverted_tau', 'None')
            if self.inverted_TAU == "None":
                raise NotImplementedError
            else:
                self.inverted_TAU = float(self.inverted_TAU)
            self.use_target_update_interval = True
        else:
            self.TAU = float(self.TAU)
            self.use_target_update_interval = False

        if self.use_target_update_interval:
            self.target_update_interval = int(self.inverted_TAU)
            self.target_update_count = 0
        
        self.GAMMA = float(kwargs["discount"])
        
        """
        self.epsend = float(kwargs['epsend'])
        self.epsstart = float(kwargs['epsstart'])
        self.epsdecay = float(kwargs['epsdecay'])
        self.eps = self.epsstart
        """

        # Eps-greedy approach blends in two different schemes, from two different papers:
        # - Ape-X eps-greedy scheme,
        # - DQN eps-greedy scheme 
        #   (retrieved by setting eps_greedy_alpha=0.0, i.e. all actors have the same epsilon). 
        self.eps_greedy_alpha = float(kwargs['eps_greedy_alpha']) if 'eps_greedy_alpha' in kwargs else 0.0
        self.reset_epsilon()

        global summary_writer
        if sum_writer is not None: summary_writer = sum_writer
        self.summary_writer = summary_writer
        if regym.RegymManager is not None:
            from regym import RaySharedVariable
            try:
                self._param_update_counter = ray.get_actor(f"{self.name}.param_update_counter")
                self._param_obs_counter = ray.get_actor(f"{self.name}.param_obs_counter")
            except ValueError:  # Name is not taken.
                self._param_update_counter = RaySharedVariable.options(name=f"{self.name}.param_update_counter").remote(0)
                self._param_obs_counter = RaySharedVariable.options(name=f"{self.name}.param_obs_counter").remote(0)
        else:
            from regym import SharedVariable
            self._param_update_counter = SharedVariable(0)
            self._param_obs_counter = SharedVariable(0)
        
           

    def parameters(self):
        return self.model.parameters()
        
    @property
    def param_update_counter(self):
        if isinstance(self._param_update_counter, ray.actor.ActorHandle):
            return ray.get(self._param_update_counter.get.remote())    
        else:
            return self._param_update_counter.get()

    @param_update_counter.setter
    def param_update_counter(self, val):
        if isinstance(self._param_update_counter, ray.actor.ActorHandle):
            self._param_update_counter.set.remote(val) 
        else:
            self._param_update_counter.set(val)
    
    def set_optimizer(self, optimizer):
        self.optimizer.load_state_dict(optimizer.state_dict())

    def get_optimizer(self):
        return self.optimizer

    @property
    def param_obs_counter(self):
        if isinstance(self._param_obs_counter, ray.actor.ActorHandle):
            return ray.get(self._param_obs_counter.get.remote())    
        else:
            return self._param_obs_counter.get()

    @param_obs_counter.setter
    def param_obs_counter(self, val):
        if isinstance(self._param_obs_counter, ray.actor.ActorHandle):
            self._param_obs_counter.set.remote(val) 
        else:
            self._param_obs_counter.set(val)
    
    def get_models(self):
        return {'model': self.model, 'target_model': self.target_model}

    def set_models(self, models_dict):
        if "model" in models_dict:
            hard_update(self.model, models_dict["model"])
        if "target_model" in models_dict:
            hard_update(self.target_model, models_dict["target_model"])
    
    def get_nbr_actor(self):
        nbr_actor = self.nbr_actor
        return nbr_actor

    def set_nbr_actor(self, nbr_actor):
        self.nbr_actor = nbr_actor
        self.reset_epsilon()

    def get_update_count(self):
        return self.param_update_counter
    
    def get_obs_count(self):
        return self.param_obs_counter

    def reset_epsilon(self):
        self.epsend = self.kwargs['epsend']
        self.epsstart = self.kwargs['epsstart']
        self.epsdecay = self.kwargs['epsdecay']
        if not isinstance(self.epsend, list): self.epsend = [float(self.epsend)]
        if not isinstance(self.epsstart, list): self.epsstart = [float(self.epsstart)]
        if not isinstance(self.epsdecay, list): self.epsdecay = [float(self.epsdecay)]
        
        # Ape-X eps-greedy scheme is used to setup the missing values:
        # i.e. if there is only one value specified in the yaml file, 
        # then the effective initialisation of the eps-greedy scheme 
        # will be that of the Ape-X paper.
        while len(self.epsend) < self.nbr_actor:   
            self.epsend.append(
                np.power(
                    self.epsend[0], 
                    1+self.eps_greedy_alpha*(len(self.epsend)/(self.nbr_actor-1))
                )
            )
        while len(self.epsstart) < self.nbr_actor:   
            self.epsstart.append(
                np.power(
                    self.epsstart[0], 
                    1+self.eps_greedy_alpha*(len(self.epsstart)/(self.nbr_actor-1))
                )
            )

        # Decaying epsilon scheme can still be applied independently of the initialisation scheme.
        # e.g. setting epsend to your actural epsilon value, and epsdecay to 1, with any value of epsstart.
        while len(self.epsdecay) < self.nbr_actor:   self.epsdecay.append(self.epsdecay[0])

        self.epsend = np.array(self.epsend)[:self.nbr_actor]
        self.epsstart = np.array(self.epsstart)[:self.nbr_actor]
        self.epsdecay = np.array(self.epsdecay)[:self.nbr_actor]
        
        self.eps = self.epsstart
        
    def get_epsilon(self, nbr_steps, strategy='exponential'):
        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer
        
        if 'exponential' in strategy:
            self.eps = self.epsend + (self.epsstart-self.epsend) * np.exp(-1.0 * nbr_steps / self.epsdecay)
        else:
            self.eps = self.epsend + max(0, (self.epsstart-self.epsend)/((float(nbr_steps)/self.epsdecay)+1))

        """
        if self.summary_writer is not None:
            for actor_i in range(self.eps.shape[0]):
                self.summary_writer.add_scalar(f'Training/Eps_Actor_{actor_i}', self.eps[actor_i], nbr_steps)
        """
        return self.eps 

    def reset_storages(self, nbr_actor: int=None):
        if nbr_actor is not None:
            self.nbr_actor = nbr_actor

            if self.n_step > 1:
                self.n_step_buffers = [deque(maxlen=self.n_step) for _ in range(self.nbr_actor)]

        if self.storages is not None:
            for storage in self.storages: storage.reset()

        self.storages = []
        keys = ['s', 'a', 'r', 'non_terminal']
        if self.recurrent:  keys += ['rnn_states']

        circular_keys={'succ_s':'s'}
        circular_offsets={'succ_s':self.n_step}
        if self.recurrent:
            circular_keys.update({'next_rnn_states':'rnn_states'})
            circular_offsets.update({'next_rnn_states':1})

        beta_increase_interval = None
        if 'PER_beta_increase_interval' in self.kwargs and self.kwargs['PER_beta_increase_interval']!='None':
            beta_increase_interval = float(self.kwargs['PER_beta_increase_interval'])  
        
 
        for i in range(self.nbr_actor):
            if self.kwargs['use_PER']:
                if self.use_mp:
                    rp_fn = regym.AlgoManager.PrioritizedReplayStorage
                else:
                    rp_fn = PrioritizedReplayStorage
                rp = rp_fn(
                    capacity=self.kwargs['replay_capacity']//self.nbr_actor,
                    alpha=self.kwargs['PER_alpha'],
                    beta=self.kwargs['PER_beta'],
                    beta_increase_interval=beta_increase_interval,
                    keys=keys,
                    circular_keys=circular_keys,                 
                    circular_offsets=circular_offsets
                )
            else:
                rp = ReplayStorage(
                    capacity=self.kwargs['replay_capacity']//self.nbr_actor,
                    keys=keys,
                    circular_keys=circular_keys,                 
                    circular_offsets=circular_offsets
                )
            self.storages.append(rp)
            

    def stored_experiences(self):
        self.train_request_count += 1
        if isinstance(self.storages[0], ray.actor.ActorHandle):
            nbr_stored_experiences = sum([ray.get(storage.__len__.remote()) for storage in self.storages])
        else:
            nbr_stored_experiences = sum([len(storage) for storage in self.storages])

        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer
        wandb.log({'PerTrainingRequest/NbrStoredExperiences':  nbr_stored_experiences}, commit=False) # self.train_request_count)
        
        return nbr_stored_experiences

    def _compute_truncated_n_step_return(self, actor_index=0):
        '''
        Compute n-step return for the first element of `self.n_step_buffer` deque.
        '''
        torch.set_grad_enabled(False)

        truncated_n_step_return = self.n_step_buffers[actor_index][-1]['r']
        for exp_dict in reversed(list(self.n_step_buffers[actor_index])[:-1]):
            truncated_n_step_return = exp_dict['r'] + self.GAMMA * truncated_n_step_return * exp_dict['non_terminal']
        return truncated_n_step_return

    def store(self, exp_dict, actor_index=0):
        '''
        Compute n-step returns, for each actor, separately,
        and then store the experience in the relevant-actor's storage.        
        '''
        if self.n_step>1:
            # Append to deque:
            self.n_step_buffers[actor_index].append(exp_dict)
            if len(self.n_step_buffers[actor_index]) < self.n_step:
                return
            # Compute n-step return of the first element of deque:
            truncated_n_step_return = self._compute_truncated_n_step_return()
            # Retrieve the first element of deque:
            current_exp_dict = copy.deepcopy(self.n_step_buffers[actor_index][0])
            current_exp_dict['r'] = truncated_n_step_return
        else:
            current_exp_dict = exp_dict
        """
        # depr : goal update
        if self.goal_oriented and 'g' not in current_exp_dict:
            current_exp_dict['g'] = current_exp_dict['goals']['desired_goals']['s']
        """

        if self.use_PER:
            init_sampling_priority = None 
            self.storages[actor_index].add(current_exp_dict, priority=init_sampling_priority)
        else:
            self.storages[actor_index].add(current_exp_dict)
        
        self.param_obs_counter += 1 

    def train(self, minibatch_size:int=None):
        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer

        if minibatch_size is None:  minibatch_size = self.batch_size

        start = time.time()
        if self.use_mp:
            samples = self._retrieve_values_from_storages(minibatch_size=self.nbr_minibatches*minibatch_size)
        else:
            samples = self.retrieve_values_from_storages(minibatch_size=self.nbr_minibatches*minibatch_size)
        #samples = self.retrieve_values_from_storages(minibatch_size=minibatch_size)
        end = time.time()

        wandb.log({'PerUpdate/TimeComplexity/RetrieveValuesFn':  end-start}, commit=False) # self.param_update_counter)


        if self.noisy \
        and hasattr(self.model, "reset_noise"):
            self.model.reset_noise()
            self.target_model.reset_noise()

        start = time.time()
        self.optimize_model(minibatch_size, samples)
        end = time.time()
        
        wandb.log({'PerUpdate/TimeComplexity/OptimizeModelFn':  end-start}, commit=False) # self.param_update_counter)
        
        if self.use_target_update_interval:
            self.target_update_count += self.nbr_actor
            if self.target_update_count > self.target_update_interval:
                self.target_update_count = 0
                hard_update(self.target_model,self.model)
        elif self.use_HER and self.kwargs.get("HER_soft_update", False):
            soft_update(self.target_model, self.model, tau=0.95) 
        else:
            soft_update(self.target_model, self.model, tau=self.TAU)

    def _mp_sampling(self, storages, sampling_config, samples):
        logger = logging.getLogger()
        while sampling_config['stay_on']:
            if not sampling_config['keep_sampling']:   
                logger.info("SamplingProcess: waiting...")
                time.sleep(1)
                continue
            logger.info("SamplingProcess: sampling")
            start = time.time()
            sample_d = self.retrieve_values_from_storages(
                minibatch_size=sampling_config['minibatch_size'],
                storages=storages,
            )
            end = time.time()
            logger.info(f"SamplingProcess: sampling : DONE in {end-start} sec")

            logger.info("SamplingProcess: updating")
            start = time.time()
            samples.update(sample_d)
            end = time.time()
            logger.info(f"SamplingProcess: updating : DONE in {end-start} sec")
    
    def _retrieve_values_from_storages(self, minibatch_size: int, storages: ReplayStorage=None):
        regym.sampling_config['minibatch_size'] = minibatch_size
        regym.sampling_config['keep_sampling'] = True
        while len(regym.samples) == 0:
            print("WARNING: waiting for sampling process...")
            time.sleep(1)
        return regym.samples

    def retrieve_values_from_storages(
        self, 
        minibatch_size: int, 
        storages: ReplayStorage=None,
        keys=None,
    ):
        '''
        Each storage stores in their key entries either numpy arrays or hierarchical dictionnaries of numpy arrays.
        This function samples from each storage, concatenate the sampled elements on the batch dimension,
        and maintains the hierarchy of dictionnaries.
        '''
        torch.set_grad_enabled(False)
        if storages is None: storages = self.storages

        if keys is None:
            keys= self.keys_to_retrieve
            #['s', 'a', 'succ_s', 'r', 'non_terminal']

        fulls = {}
        if self.use_PER:
            fulls['importanceSamplingWeights'] = []

        for key in keys:    fulls[key] = []

        using_ray = isinstance(storages[0], ray.actor.ActorHandle)
        for storage in storages:
            # Check that there is something in the storage 
            if using_ray:
                storage_size = ray.get(storage.__len__.remote())
            else:
                storage_size = len(storage)
                
            if storage_size <= 1: continue
            #if len(storage) <= 1: continue
            if self.use_PER:
                if using_ray:
                    sample, importanceSamplingWeights = ray.get(
                        storage.sample.remote(batch_size=minibatch_size, keys=keys)
                    )
                else:
                    sample, importanceSamplingWeights = storage.sample(batch_size=minibatch_size, keys=keys)
                importanceSamplingWeights = torch.from_numpy(importanceSamplingWeights)
                fulls['importanceSamplingWeights'].append(importanceSamplingWeights)
            else:
                sample = storage.sample(batch_size=minibatch_size, keys=keys)
            
            values = {}
            for key, value in zip(keys, sample):
                value = value.tolist()
                if isinstance(value[0], dict): 
                    value = _concatenate_list_hdict(
                        lhds=value, 
                        #concat_fn=partial(torch.cat, dim=0),   # concatenate on the unrolling dimension (axis=1).
                        concat_fn=archi_concat_fn,
                        preprocess_fn=(lambda x:x),
                    )
                else:
                    value = torch.cat(value, dim=0)
                values[key] = value 

            for key, value in values.items():
                fulls[key].append(value)
        
        for key, value in fulls.items():
            if len(value) >1:
                if isinstance(value[0], dict):
                    value = _concatenate_list_hdict(
                        lhds=value, 
                        concat_fn=partial(torch.cat, dim=0),   # concatenate on the unrolling dimension (axis=1).
                        preprocess_fn=(lambda x:x),
                    )
                else:
                    value = torch.cat(value, dim=0)
            else:
                value = value[0]

            fulls[key] = value
        
        return fulls

    def optimize_model(self, minibatch_size: int, samples: Dict, optimisation_minibatch_size:int=None):
        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer
        
        if optimisation_minibatch_size is None:
            optimisation_minibatch_size = minibatch_size*self.nbr_actor

        start = time.time()
        torch.set_grad_enabled(True)
        self.model.train(True)
        
        beta = 1.0
        if self.use_PER:
            if hasattr(self.storages[0].get_beta, "remote"):
                beta_id = self.storages[0].get_beta.remote()
                beta = ray.get(beta_id)
            else:
                beta = self.storages[0].get_beta()

        # For each actor, there is one mini_batch update:
        #sampler = list(random_sample(np.arange(states.size(0)), minibatch_size))
        sampler = list(random_sample(np.arange(samples['s'].size(0)), minibatch_size))
        nbr_minibatches = len(sampler)
        nbr_sampled_element_per_storage = self.nbr_minibatches*minibatch_size 
        list_batch_indices = [storage_idx*nbr_sampled_element_per_storage+np.arange(nbr_sampled_element_per_storage) \
                                for storage_idx, _ in enumerate(self.storages)]
        array_batch_indices = np.concatenate(list_batch_indices, axis=0)
        sampled_batch_indices = []
        sampled_losses_per_item = []
        
        self.optimizer.zero_grad()

        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()
            sampled_batch_indices.append(batch_indices)

            sampled_samples = {}
            for k in samples:
                out_k = k
                if k in self.kremap:
                    out_k = self.kremap[k]
                
                v = samples[k]
                if v is None:   
                    sampled_samples[out_k] = None
                    continue
                if 'rnn' in k:
                    v = _extract_rnn_states_from_batch_indices(
                        v, 
                        batch_indices, 
                        use_cuda=self.kwargs['use_cuda'],
                    )
                elif self.kwargs['use_cuda']:
                    v = v[batch_indices].cuda() 
                else: 
                    v = v[batch_indices]
                
                sampled_samples[out_k] = v
                # (batch_size, unroll_dim, ...)

            #self.optimizer.zero_grad()
            
            if self.use_HER and 'HER_target_clamping' not in self.kwargs:
                raise NotImplementedError
	
            self.kwargs["logging"] = False # (self.param_update_counter % 32) == 0
            loss, loss_per_item = self.loss_fn(
                samples=sampled_samples,
                models=self.get_models(),
                summary_writer=self.summary_writer,
                iteration_count=self.param_update_counter,
                
                gamma=self.GAMMA,
                PER_running_beta=beta,
                **self.kwargs,
            )
            
            (loss/nbr_minibatches).backward(retain_graph=False)
            '''
            loss.backward(retain_graph=False)
            if self.kwargs['gradient_clip'] > 1e-3:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs['gradient_clip'])
            self.optimizer.step()
            '''

            if self.use_PER:
                sampled_losses_per_item.append(loss_per_item)
                #wandb_data = copy.deepcopy(wandb.run.history._data)
                #wandb.run.history._data = {}
                wandb.log({
                    'PerUpdate/ImportanceSamplingMean':  sampled_samples['importanceSamplingWeights'].cpu().mean().item(),
                    'PerUpdate/ImportanceSamplingStd':  sampled_samples['importanceSamplingWeights'].cpu().std().item(),
                    'PerUpdate/PER_Beta':  beta
                    },
                    commit=False,
                ) # self.param_update_counter)
                #wandb.run.history._data = wandb_data

            self.param_update_counter += 1 

        if self.kwargs['gradient_clip'] > 1e-3:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs['gradient_clip'])
        self.optimizer.step()
        
        self.model.train(False)
        torch.set_grad_enabled(False)

        if self.use_PER :
            sampled_batch_indices = np.concatenate(sampled_batch_indices, axis=0)
            # let us align the batch indices with the losses:
            array_batch_indices = array_batch_indices[sampled_batch_indices]
            # Now we can iterate through the losses and retrieve what 
            # storage and what batch index they were associated with:
            self._update_replay_buffer_priorities(
                sampled_losses_per_item=sampled_losses_per_item, 
                array_batch_indices=array_batch_indices,
                minibatch_size=nbr_sampled_element_per_storage,#minibatch_size,
            )

        end = time.time()
        wandb.log({'PerUpdate/TimeComplexity/OptimizationLoss':  end-start}, commit=False) # self.param_update_counter)


    def compute_td_error(self, samples: Dict):
        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer
        
        start = time.time()
        torch.set_grad_enabled(False)
        self.model.train(False)
        
        beta = 1.0
        
        batch_indices = torch.arange(samples['s'].shape[0])
        sampled_samples = {}
        for k in samples:
            out_k = k
            if k in self.kremap:
                out_k = self.kremap[k]
            
            v = samples[k]
            if v is None:   
                sampled_samples[out_k] = None
                continue
            if 'rnn' in k:
                v = _extract_rnn_states_from_batch_indices(
                    v, 
                    batch_indices, 
                    use_cuda=self.kwargs['use_cuda'],
                )
            elif self.kwargs['use_cuda']:
                v = v[batch_indices].cuda() 
            else: 
                v = v[batch_indices]
            
            sampled_samples[out_k] = v
            # (batch_size, unroll_dim, ...)
         
        loss, loss_per_item = self.loss_fn(
            samples=sampled_samples,
            models=self.get_models(),
            summary_writer=self.summary_writer,
            iteration_count=self.param_update_counter,
            
            gamma=self.GAMMA,
            PER_running_beta=beta,
            **self.kwargs,
        )
            
        '''
        loss, loss_per_item = self.loss_fn(
            sampled_states, 
            sampled_actions, 
            sampled_next_states,
            sampled_rewards,
            sampled_non_terminals,
            rnn_states=sampled_rnn_states,
            next_rnn_states=sampled_next_rnn_states,
            goals=sampled_goals,
            gamma=self.GAMMA,
            model=self.model,
            target_model=self.target_model,
            weights_decay_lambda=self.weights_decay_lambda,
            weights_entropy_lambda=self.weights_entropy_lambda,
            use_PER=self.use_PER,
            PER_beta=beta,
            importanceSamplingWeights=sampled_importanceSamplingWeights,
            HER_target_clamping=self.kwargs['HER_target_clamping'] if 'HER_target_clamping' in self.kwargs else False,
            iteration_count=self.param_update_counter,
            summary_writer=None,
            kwargs=self.kwargs
        )
        '''

        end = time.time()
        
        wandb.log({'PerUpdate/TimeComplexity/TDErrorComputation':  end-start}, commit=False) # self.param_update_counter)
        
        return loss, loss_per_item 

    def sample_from_rnn_states(self, rnn_states, next_rnn_states, batch_indices, use_cuda):
        sampled_rnn_states = _extract_rnn_states_from_batch_indices(rnn_states, batch_indices, use_cuda=self.kwargs['use_cuda'])
        sampled_next_rnn_states = _extract_rnn_states_from_batch_indices(next_rnn_states, batch_indices, use_cuda=self.kwargs['use_cuda'])
        return sampled_rnn_states, sampled_next_rnn_states

    def _update_replay_buffer_priorities(self, 
                                         sampled_losses_per_item: List[torch.Tensor], 
                                         array_batch_indices: List,
                                         minibatch_size: int):
        '''
        Updates the priorities of each sampled elements from their respective storages.

        TODO: update to useing Ray and get_tree_indices
        '''
        # losses corresponding to sampled batch indices: 
        sampled_losses_per_item = torch.cat(sampled_losses_per_item, dim=0).cpu().detach().numpy()
        for sloss, arr_bidx in zip(sampled_losses_per_item, array_batch_indices):
            storage_idx = arr_bidx//minibatch_size
            el_idx_in_batch = arr_bidx%minibatch_size
            el_idx_in_storage = self.storages[storage_idx].tree_indices[el_idx_in_batch]
            new_priority = self.storages[storage_idx].priority(sloss)
            self.storages[storage_idx].update(idx=el_idx_in_storage, priority=new_priority)

    def clone(self, with_replay_buffer: bool=False, clone_proxies: bool=False, minimal=False):        
        if self.storages is None:
            self.reset_storages()
        if not(with_replay_buffer): 
            storages = self.storages
            self.storages = None
            
        sum_writer = self.summary_writer
        self.summary_writer = None
        
        param_update_counter = self._param_update_counter
        self._param_update_counter = None 

        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model)
        
        if isinstance(self.model, ArchiModel):
            self.model.reset()
        if isinstance(self.target_model, ArchiModel):
            self.target_model.reset()
        
        param_obs_counter = self._param_obs_counter
        self._param_obs_counter = None 

        cloned_algo = copy.deepcopy(self)
         
        if minimal:
            cloned_algo.target_model = None

        if not(with_replay_buffer): 
            self.storages = storages
            # the following line might increase the size of the clone algo:
            if not minimal:
                cloned_algo.reset_storages()

        self.summary_writer = sum_writer
        
        self._param_update_counter = param_update_counter
        cloned_algo._param_update_counter = param_update_counter

        self._param_obs_counter = param_obs_counter
        cloned_algo._param_obs_counter = param_obs_counter

        # Goes through all variables 'Proxy' (dealing with multiprocessing)
        # contained in this class and removes them from clone
        if not(clone_proxies):
            proxy_key_values = [
                (key, value) 
                for key, value in cloned_algo.__dict__.items() 
                if ('Proxy' in str(type(value)))
            ]
            for key, value in proxy_key_values:
                setattr(cloned_algo, key, None)

        return cloned_algo

    def async_actor(self):        
        storages = self.storages
        self.storages = None
        
        sum_writer = self.summary_writer
        self.summary_writer = None
        
        param_update_counter = self._param_update_counter
        self._param_update_counter = None 

        param_obs_counter = self._param_obs_counter
        self._param_obs_counter = None 

        cloned_algo = copy.deepcopy(self)
        
        self.storages = storages
        cloned_algo.storages = storages

        self.summary_writer = sum_writer
        cloned_algo.summary_writer = sum_writer

        self._param_update_counter = param_update_counter
        cloned_algo._param_update_counter = param_update_counter

        self._param_obs_counter = param_obs_counter
        cloned_algo._param_obs_counter = param_obs_counter

        return cloned_algo
