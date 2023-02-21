import copy 
from collections import deque 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from . import dqn_ther_loss, ddqn_ther_loss

from ..algorithm import Algorithm
from ...replay_buffers import ReplayBuffer, PrioritizedReplayBuffer, EXP, EXPPER
from ...replay_buffers import PrioritizedReplayStorage, ReplayStorage
from ...networks import hard_update, random_sample


summary_writer = None 


class THERAlgorithm(Algorithm):
    def __init__(self, kwargs, model, predictor, target_model=None, optimizer=None, sum_writer=None):
        '''
        '''
        self.kwargs = copy.deepcopy(kwargs)        
        self.use_cuda = kwargs["use_cuda"]

        self.double = self.kwargs['double']
        self.dueling = self.kwargs['dueling']
        self.noisy = self.kwargs['noisy']
        self.n_step = self.kwargs['n_step'] if 'n_step' in self.kwargs else 1
        if self.n_step > 1:
            self.n_step_buffer = deque(maxlen=self.n_step)

        self.use_PER = self.kwargs['use_PER']
        
        self.goal_oriented = self.kwargs['goal_oriented'] if 'goal_oriented' in self.kwargs else False
        self.use_HER = self.kwargs['use_HER'] if 'use_HER' in self.kwargs else False
        assert((self.use_HER and self.goal_oriented) or self.goal_oriented)

        self.weights_decay_lambda = float(self.kwargs['weights_decay_lambda'])
        
        self.nbr_actor = self.kwargs['nbr_actor']
        
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

        self.predictor = predictor
        
        if optimizer is None:
            parameters = list(self.model.parameters())+list(self.predictor.parameters())
            # Tuning learning rate with respect to the number of actors:
            # Following: https://arxiv.org/abs/1705.04862
            lr = kwargs['learning_rate'] 
            if kwargs['lr_account_for_nbr_actor']:
                lr *= self.nbr_actor
            print(f"Learning rate: {lr}")
            self.optimizer = optim.Adam(parameters, lr=lr, betas=(0.9,0.999), eps=kwargs['adam_eps'])
        else: self.optimizer = optimizer

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
        self.param_update_counter = 0
    
    def get_models(self):
        return {'model': self.model, 'target_model': self.target_model}

    def parameters(self):
        return self.model.parameters()+self.predictor.parameters()
    
    def get_nbr_actor(self):
        return self.nbr_actor

    def get_update_count(self):
        return self.param_update_counter

    def get_epsilon(self, nbr_steps, strategy='exponential'):
        global summary_writer
        self.summary_writer = summary_writer
        
        if 'exponential' in strategy:
            self.eps = self.epsend + (self.epsstart-self.epsend) * np.exp(-1.0 * nbr_steps / self.epsdecay)
        else:
            self.eps = self.epsend + max(0, (self.epsstart-self.epsend)/((float(nbr_steps)/self.epsdecay)+1))

        if summary_writer is not None:
            summary_writer.add_scalar('Training/Eps', self.eps, nbr_steps)

        return self.eps 

    def reset_storages(self, nbr_actor=None):
        if nbr_actor is not None:
            self.nbr_actor = nbr_actor

        if self.storages is not None:
            for storage in self.storages: storage.reset()

        self.storages = []
        keys = ['s', 'a', 'r', 'non_terminal']
        if self.recurrent:  keys += ['rnn_states', 'next_rnn_states']
        if self.goal_oriented:    keys += ['g']
        
        for i in range(self.nbr_actor):
            if self.kwargs['use_PER']:
                self.storages.append(PrioritizedReplayStorage(capacity=self.kwargs['replay_capacity'],
                                                                alpha=self.kwargs['PER_alpha'],
                                                                beta=self.kwargs['PER_beta'],
                                                                keys=keys,
                                                                circular_offsets={'succ_s':self.n_step})
                )
            else:
                self.storages.append(ReplayStorage(capacity=self.kwargs['replay_capacity'],
                                                   keys=keys,
                                                   circular_offsets={'succ_s':self.n_step})
                )
            
    def _compute_truncated_n_step_return(self):
        truncated_n_step_return = self.n_step_buffer[-1]['r']
        for exp_dict in reversed(list(self.n_step_buffer)[:-1]):
            truncated_n_step_return = exp_dict['r'] + self.GAMMA * truncated_n_step_return * exp_dict['non_terminal']
        return truncated_n_step_return

    def store(self, exp_dict, actor_index=0):
        if self.n_step>1:
            self.n_step_buffer.append(exp_dict)
            if len(self.n_step_buffer) < self.n_step:
                return
            truncated_n_step_return = self._compute_truncated_n_step_return()
            current_exp_dict = copy.deepcopy(exp_dict)
            current_exp_dict['r'] = truncated_n_step_return
        else:
            current_exp_dict = exp_dict    
        
        if self.goal_oriented and 'g' not in exp_dict:
            exp_dict['g'] = exp_dict['goals']['desired_goals']['s']

        if self.use_PER:
            init_sampling_priority = None 
            self.storages[actor_index].add(current_exp_dict, priority=init_sampling_priority)
        else:
            self.storages[actor_index].add(current_exp_dict)

    def train(self, minibatch_size=None):
        if minibatch_size is None:  minibatch_size = self.batch_size

        self.target_update_count += self.nbr_actor

        samples = self.retrieve_values_from_storages(minibatch_size=minibatch_size)
        if self.recurrent: samples['rnn_states'] = self.reformat_rnn_states(samples['rnn_states'])
        
        if self.noisy:  
            self.model.reset_noise()
            self.target_model.reset_noise()

        self.optimize_model(minibatch_size, samples)
        
        if self.target_update_count > self.target_update_interval:
            self.target_update_count = 0
            hard_update(self.target_model,self.model)

    def retrieve_values_from_storages(self, minibatch_size):
        keys=['s', 'a', 'succ_s', 'r', 'non_terminal']

        fulls = {}
        
        if self.use_PER:
            fulls['importanceSamplingWeights'] = []

        if self.recurrent:
            keys += ['rnn_states']
        
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
                value = torch.cat(value.tolist(), dim=0)
                values[key] = value 

            for key, value in values.items():
                fulls[key].append(value)

        for key, value in fulls.items():
            fulls[key] = torch.cat(value, dim=0)
        
        return fulls

    def optimize_model(self, minibatch_size, samples):
        global summary_writer
        self.summary_writer = summary_writer

        beta = self.storages[0].beta if self.use_PER else 1.0
        
        states = samples['s']
        actions = samples['a']
        next_states = samples['succ_s']
        rewards = samples['r']
        non_terminals = samples['non_terminal']

        rnn_states = samples['rnn_state'] if 'rnn_state' in samples else None
        goals = samples['g'] if 'g' in samples else None

        importanceSamplingWeights = samples['importanceSamplingWeights'] if 'importanceSamplingWeights' in samples else None

        # What is this? create dictionary to store length of each part of the recurrent submodules of the current model
        nbr_layers_per_rnn = None
        if self.recurrent:
            nbr_layers_per_rnn = {recurrent_submodule_name: len(rnn_states[recurrent_submodule_name]['hidden'])
                                  for recurrent_submodule_name in rnn_states}

        # For each actor, there is one mini_batch update:
        sampler = random_sample(np.arange(states.size(0)), minibatch_size)
        list_batch_indices = [storage_idx*minibatch_size+np.arange(minibatch_size) \
                                for storage_idx, storage in enumerate(self.storages)]
        array_batch_indices = np.concatenate(list_batch_indices, axis=0)
        sampled_batch_indices = []
        sampled_losses_per_item = []

        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()
            sampled_batch_indices.append(batch_indices)

            sampled_rnn_states = None
            if self.recurrent:
                sampled_rnn_states = self.calculate_rnn_states_from_batch_indices(rnn_states, batch_indices, nbr_layers_per_rnn)

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
            
            self.optimizer.zero_grad()
            if self.double or self.dueling:
                loss, loss_per_item = ddqn_ther_loss.compute_loss(sampled_states, 
                                              sampled_actions, 
                                              sampled_next_states,
                                              sampled_rewards,
                                              sampled_non_terminals,
                                              rnn_states=sampled_rnn_states,
                                              goals=sampled_goals,
                                              gamma=self.GAMMA,
                                              model=self.model,
                                              predictor=self.predictor,
                                              target_model=self.target_model,
                                              weights_decay_lambda=self.weights_decay_lambda,
                                              use_PER=self.use_PER,
                                              PER_beta=beta,
                                              importanceSamplingWeights=sampled_importanceSamplingWeights,
                                              use_HER=self.use_HER,
                                              iteration_count=self.param_update_counter,
                                              summary_writer=summary_writer)
            else:
                loss, loss_per_item = dqn_ther_loss.compute_loss(sampled_states, 
                                              sampled_actions, 
                                              sampled_next_states,
                                              sampled_rewards,
                                              sampled_non_terminals,
                                              rnn_states=sampled_rnn_states,
                                              goals=sampled_goals,
                                              gamma=self.GAMMA,
                                              model=self.model,
                                              predictor=self.predictor,
                                              target_model=self.target_model,
                                              weights_decay_lambda=self.weights_decay_lambda,
                                              use_PER=self.use_PER,
                                              PER_beta=beta,
                                              importanceSamplingWeights=sampled_importanceSamplingWeights,
                                              use_HER=self.use_HER,
                                              iteration_count=self.param_update_counter,
                                              summary_writer=summary_writer)

            loss.backward(retain_graph=False)
            if self.kwargs['gradient_clip'] > 1e-3:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs['gradient_clip'])
                nn.utils.clip_grad_norm_(self.predictor.parameters(), self.kwargs['gradient_clip'])
            self.optimizer.step()

            if self.use_PER:
                sampled_losses_per_item.append(loss_per_item)

            if summary_writer is not None:
                self.param_update_counter += 1 

        if self.use_PER :
            # losses corresponding to sampled batch indices: 
            sampled_losses_per_item = torch.cat(sampled_losses_per_item, dim=0).cpu().detach().numpy()
            sampled_batch_indices = np.concatenate(sampled_batch_indices, axis=0)
            # let us align the batch indices with the losses:
            array_batch_indices = array_batch_indices[sampled_batch_indices]
            # Now we can iterate through the losses and retrieve what 
            # storage and what batch index they were associated with:
            for sloss, arr_bidx in zip(sampled_losses_per_item, array_batch_indices):
                storage_idx = arr_bidx//minibatch_size
                el_idx_in_batch = arr_bidx%minibatch_size
                el_idx_in_storage = self.storages[storage_idx].tree_indices[el_idx_in_batch]
                new_priority = self.storages[storage_idx].priority(sloss)
                self.storages[storage_idx].update(idx=el_idx_in_storage, priority=new_priority)

    def clone(self):        
        storages = self.storages
        self.storages = None
        sum_writer = self.summary_writer
        self.summary_writer = None
        cloned_algo = copy.deepcopy(self)
        self.storages = storages
        self.summary_writer = sum_writer
        return cloned_algo

