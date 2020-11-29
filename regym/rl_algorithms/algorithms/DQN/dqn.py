from typing import Dict, List

import copy 
from collections import deque 
from functools import partial 

import ray
# TODO : change every storage to use remote ray storages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from . import dqn_loss, ddqn_loss

import regym
from ..algorithm import Algorithm
from ...replay_buffers import PrioritizedReplayStorage, ReplayStorage
from ...networks import hard_update, random_sample
from regym.rl_algorithms.utils import _extract_rnn_states_from_batch_indices, _concatenate_hdict, _concatenate_list_hdict



summary_writer = None 


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

        self.weights_decay_lambda = float(self.kwargs['weights_decay_lambda'])
        self.weights_entropy_lambda = float(self.kwargs['weights_entropy_lambda']) if 'weights_entropy_lambda' in self.kwargs else 0.0
        
        
        self.model = model
        if self.kwargs['use_cuda']:
            self.model = self.model.cuda()

        if target_model is None:
            target_model = copy.deepcopy(self.model)

        self.target_model = target_model
        self.target_model.share_memory()

        hard_update(self.target_model, self.model)
        if self.use_cuda:
            self.target_model = self.target_model.cuda()

        
        if optimizer is None:
            parameters = self.model.parameters()
            # Tuning learning rate with respect to the number of actors:
            # Following: https://arxiv.org/abs/1705.04862
            lr = kwargs['learning_rate'] 
            if kwargs['lr_account_for_nbr_actor']:
                lr *= self.nbr_actor
            print(f"Learning rate: {lr}")
            self.optimizer = optim.Adam(parameters, lr=lr, betas=(0.9,0.999), eps=kwargs['adam_eps'])
        else: self.optimizer = optimizer

        self.loss_fn = loss_fn
        print(f"WARNING: loss_fn is {self.loss_fn}")
            
            
        self.recurrent = False
        # TECHNICAL DEBT: check for recurrent property by looking at the modules in the model rather than relying on the kwargs that may contain
        # elements that do not concern the model trained by this algorithm, given that it is now use-able inside I2A...
        self.recurrent_nn_submodule_names = [hyperparameter for hyperparameter, value in self.kwargs.items() if isinstance(value, str) and 'RNN' in value]
        if len(self.recurrent_nn_submodule_names): self.recurrent = True

        self.storages = None
        self.reset_storages()

        self.min_capacity = int(float(kwargs["min_capacity"]))
        self.batch_size = int(kwargs["batch_size"])

        self.TAU = float(self.kwargs['tau'])
        self.target_update_interval = int(1.0/self.TAU)
        self.target_update_count = 0
        self.GAMMA = float(kwargs["discount"])
        
        self.epsend = float(kwargs['epsend'])
        self.epsstart = float(kwargs['epsstart'])
        self.epsdecay = float(kwargs['epsdecay'])
        self.eps = self.epsstart
        
        global summary_writer
        if sum_writer is not None: summary_writer = sum_writer
        self.summary_writer = summary_writer
        if regym.RegymManager is not None:
            #self._param_update_counter = regym.RegymManager.Value(int, 0, lock=False)
            from regym import SharedVariable
            try:
                self._param_update_counter = ray.get_actor(f"{self.name}.param_update_counter")
            except ValueError:  # Name is not taken.
                self._param_update_counter = SharedVariable.options(name=f"{self.name}.param_update_counter").remote(0)

        else:
            self._param_update_counter =0 

    @property
    def param_update_counter(self):
        if isinstance(self._param_update_counter, int):
            return self._param_update_counter
        else:
            return ray.get(self._param_update_counter.get.remote())    

    @param_update_counter.setter
    def param_update_counter(self, val):
        if isinstance(self._param_update_counter, int):
            self._param_update_counter = val
        else:
            self._param_update_counter.set.remote(val) 
    
    def get_models(self):
        return {'model': self.model, 'target_model': self.target_model}

    def set_models(self, models_dict):
        if "model" in models_dict:
            hard_update(self.model, models_dict["model"])
        if "target_model" in models_dict:
            hard_update(self.target_model, models_dict["target_model"])
    
    def get_nbr_actor(self):
        return self.nbr_actor

    def get_update_count(self):
        return self.param_update_counter

    def get_epsilon(self, nbr_steps, strategy='exponential'):
        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer
        
        if 'exponential' in strategy:
            self.eps = self.epsend + (self.epsstart-self.epsend) * np.exp(-1.0 * nbr_steps / self.epsdecay)
        else:
            self.eps = self.epsend + max(0, (self.epsstart-self.epsend)/((float(nbr_steps)/self.epsdecay)+1))

        if summary_writer is not None:
            summary_writer.add_scalar('Training/Eps', self.eps, nbr_steps)

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
        if self.goal_oriented:    keys += ['g']
        
        circular_keys={'succ_s':'s'}
        circular_offsets={'succ_s':self.n_step}
        if self.recurrent:
            circular_keys.update({'next_rnn_states':'rnn_states'})
            circular_offsets.update({'next_rnn_states':1})

        beta_increase_interval = float(self.kwargs['PER_beta_increase_interval'])  if 'PER_beta_increase_interval' in self.kwargs else 1e4
        
        for i in range(self.nbr_actor):
            if self.kwargs['use_PER']:
                self.storages.append(
                    PrioritizedReplayStorage(
                        capacity=self.kwargs['replay_capacity']//self.nbr_actor,
                        alpha=self.kwargs['PER_alpha'],
                        beta=self.kwargs['PER_beta'],
                        beta_increase_interval=beta_increase_interval,
                        keys=keys,
                        circular_keys=circular_keys,                 
                        circular_offsets=circular_offsets
                    )
                )
            else:
                self.storages.append(
                    ReplayStorage(
                        capacity=self.kwargs['replay_capacity']//self.nbr_actor,
                        keys=keys,
                        circular_keys=circular_keys,                 
                        circular_offsets=circular_offsets
                    )
                )

    def stored_experiences(self):
        self.train_request_count += 1
        nbr_stored_experiences = sum([ray.get(storage.__len__.remote()) for storage in self.storages])
        
        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('PerTrainingRequest/NbrStoredExperiences', nbr_stored_experiences, self.train_request_count)
        return nbr_stored_experiences

    def _compute_truncated_n_step_return(self, actor_index=0):
        '''
        Compute n-step return for the first element of `self.n_step_buffer` deque.
        '''
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
        
        if self.goal_oriented and 'g' not in current_exp_dict:
            current_exp_dict['g'] = current_exp_dict['goals']['desired_goals']['s']

        if self.use_PER:
            init_sampling_priority = None 
            self.storages[actor_index].add(current_exp_dict, priority=init_sampling_priority)
        else:
            self.storages[actor_index].add(current_exp_dict)

    def train(self, minibatch_size:int=None):
        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer

        if minibatch_size is None:  minibatch_size = self.batch_size

        self.target_update_count += self.nbr_actor

        start = time.time()
        samples = self.retrieve_values_from_storages(minibatch_size=minibatch_size)
        end = time.time()

        if self.summary_writer is not None:
            self.summary_writer.add_scalar('PerUpdate/TimeComplexity/RetrieveValuesFn', end-start, self.param_update_counter)


        if self.noisy:  
            self.model.reset_noise()
            self.target_model.reset_noise()

        self.optimize_model(minibatch_size, samples)
        
        if self.target_update_count > self.target_update_interval:
            self.target_update_count = 0
            hard_update(self.target_model,self.model)

    def retrieve_values_from_storages(self, minibatch_size: int):
        '''
        Each storage stores in their key entries either numpy arrays or hierarchical dictionnaries of numpy arrays.
        This function samples from each storage, concatenate the sampled elements on the batch dimension,
        and maintains the hierarchy of dictionnaries.
        '''
        keys=['s', 'a', 'succ_s', 'r', 'non_terminal']

        fulls = {}
        
        if self.use_PER:
            fulls['importanceSamplingWeights'] = []

        if self.recurrent:
            keys += ['rnn_states', 'next_rnn_states']
        
        if self.goal_oriented:
            keys += ['g']
        
        for key in keys:    fulls[key] = []

        for storage in self.storages:
            # Check that there is something in the storage 
            if len(storage) <= 1: continue
            if self.use_PER:
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
                        concat_fn=partial(torch.cat, dim=0),   # concatenate on the unrolling dimension (axis=1).
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

    def optimize_model(self, minibatch_size: int, samples: Dict):
        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer
        
        #beta = self.storages[0].get_beta() if self.use_PER else 1.0
        beta = 1.0
        if self.use_PER:
            beta_id = self.storages[0].get_beta.remote()
            beta = ray.get(beta_id)

        states = samples['s']
        actions = samples['a']
        next_states = samples['succ_s']
        rewards = samples['r']
        non_terminals = samples['non_terminal']

        rnn_states = samples['rnn_states'] if 'rnn_states' in samples else None
        next_rnn_states = samples['next_rnn_states'] if 'next_rnn_states' in samples else None
        goals = samples['g'] if 'g' in samples else None

        importanceSamplingWeights = samples['importanceSamplingWeights'] if 'importanceSamplingWeights' in samples else None

        # For each actor, there is one mini_batch update:
        sampler = random_sample(np.arange(states.size(0)), minibatch_size)
        list_batch_indices = [storage_idx*minibatch_size+np.arange(minibatch_size) \
                                for storage_idx, _ in enumerate(self.storages)]
        array_batch_indices = np.concatenate(list_batch_indices, axis=0)
        sampled_batch_indices = []
        sampled_losses_per_item = []

        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()
            sampled_batch_indices.append(batch_indices)

            sampled_rnn_states = None
            sampled_next_rnn_states = None
            if self.recurrent:
                sampled_rnn_states, sampled_next_rnn_states = self.sample_from_rnn_states(
                    rnn_states, 
                    next_rnn_states, 
                    batch_indices, 
                    use_cuda=self.kwargs['use_cuda']
                )
                # (batch_size, unroll_dim, ...)

            sampled_goals = None
            if self.goal_oriented:
                sampled_goals = goals[batch_indices].cuda() if self.kwargs['use_cuda'] else goals[batch_indices]

            sampled_importanceSamplingWeights = None
            if self.use_PER:
                sampled_importanceSamplingWeights = importanceSamplingWeights[batch_indices].cuda() if self.kwargs['use_cuda'] else importanceSamplingWeights[batch_indices]
            
            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_next_states = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]
            sampled_rewards = rewards[batch_indices].cuda() if self.kwargs['use_cuda'] else rewards[batch_indices]
            sampled_non_terminals = non_terminals[batch_indices].cuda() if self.kwargs['use_cuda'] else non_terminals[batch_indices]
            # (batch_size, unroll_dim, ...)

            self.optimizer.zero_grad()
            
            loss, loss_per_item = self.loss_fn(sampled_states, 
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
                                          summary_writer=self.summary_writer,
                                          kwargs=self.kwargs)
            
            loss.backward(retain_graph=False)
            if self.kwargs['gradient_clip'] > 1e-3:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs['gradient_clip'])
            self.optimizer.step()

            if self.use_PER:
                sampled_losses_per_item.append(loss_per_item)
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar('PerUpdate/ImportanceSamplingMean', sampled_importanceSamplingWeights.cpu().mean().item(), self.param_update_counter)
                    self.summary_writer.add_scalar('PerUpdate/ImportanceSamplingStd', sampled_importanceSamplingWeights.cpu().std().item(), self.param_update_counter)
                    self.summary_writer.add_scalar('PerUpdate/PER_Beta', beta, self.param_update_counter)

            self.param_update_counter += 1 

        if self.use_PER :
            sampled_batch_indices = np.concatenate(sampled_batch_indices, axis=0)
            # let us align the batch indices with the losses:
            array_batch_indices = array_batch_indices[sampled_batch_indices]
            # Now we can iterate through the losses and retrieve what 
            # storage and what batch index they were associated with:
            self._update_replay_buffer_priorities(
                sampled_losses_per_item=sampled_losses_per_item, 
                array_batch_indices=array_batch_indices,
                minibatch_size=minibatch_size,
            )


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
        if not(with_replay_buffer): 
            storages = self.storages
            self.storages = None
            
        sum_writer = self.summary_writer
        self.summary_writer = None
        
        param_update_counter = self._param_update_counter
        self._param_update_counter = None 

        cloned_algo = copy.deepcopy(self)
        
        if minimal:
            cloned_algo.target_model = None

        if not(with_replay_buffer): 
            self.storages = storages
        
        self.summary_writer = sum_writer
        
        self._param_update_counter = param_update_counter
        cloned_algo._param_update_counter = param_update_counter

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

        cloned_algo = copy.deepcopy(self)
        
        self.storages = storages
        cloned_algo.storages = storages

        self.summary_writer = sum_writer
        cloned_algo.summary_writer = sum_writer

        self._param_update_counter = param_update_counter
        cloned_algo._param_update_counter = param_update_counter

        return cloned_algo
