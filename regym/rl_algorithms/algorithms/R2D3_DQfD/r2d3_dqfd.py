from typing import Dict, List, Any, Optional, Callable

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import ray 
import time 
import copy
from tqdm import tqdm 

import regym
from ...networks import hard_update, random_sample

from regym.rl_algorithms.algorithms.R2D3 import R2D3Algorithm
from regym.rl_algorithms.algorithms.R2D3_DQfD import r2d3_large_margin_classification_loss
from regym.rl_algorithms.algorithms.DQN import DQNAlgorithm
from regym.rl_algorithms.replay_buffers import PrioritizedReplayStorage, ReplayStorage

from regym.rl_algorithms.utils import _concatenate_list_hdict

sum_writer = None

class R2D3DQfDAlgorithm(R2D3Algorithm):

    def __init__(self, kwargs: Dict[str, Any], 
                 model: nn.Module,
                 target_model: Optional[nn.Module] = None,
                 expert_buffer: ReplayStorage = None,
                 optimizer=None,
                 loss_fn: Callable = r2d3_large_margin_classification_loss.compute_loss,
                 sum_writer=None,
                 name='r2d3_dqfd_algo'):
        R2D3Algorithm.__init__(
            self=self,
            kwargs=kwargs, 
            model=model, 
            target_model=target_model, 
            optimizer=optimizer, 
            loss_fn=loss_fn, 
            sum_writer=sum_writer,
            name=name
        )
        
        self.demo_ratio = kwargs['demo_ratio']  # Should be small (around: 1 / 256)
        
        self.expert_buffer = expert_buffer
        #self.expert_buffer.reset_priorities()
        for idx in range(len(self.expert_buffer)-1):
            self.expert_buffer.update(idx=idx+1, priority=0.1)
        self.expert_buffer._iteration = 0
        self.expert_buffer._beta = 0.1

        for i in range(len(self.storages)):
            self.storages[i] = copy.deepcopy(self.expert_buffer)
            
  
    # NOTE: we are overriding this function from R2D3Algorithm
    def retrieve_values_from_storages(self, minibatch_size: int=32, pretraining=False):
        '''
        We sample from both replay buffers (expert_buffer and agent
        collected experiences) according to property self.demo_ratio
        '''
        demo_ratio = self.demo_ratio
        if pretraining:
            demo_ratio = 0.5
        num_demonstration_samples = (minibatch_size*len(self.storages)) * demo_ratio / (1 - demo_ratio)  

        '''
        Each storage stores in their key entries either numpy arrays or hierarchical dictionnaries of numpy arrays.
        This function samples from each storage, concatenate the sampled elements on the batch dimension,
        and maintains the hierarchy of dictionnaries.
        '''
        
        
        keys=['s', 'a', 'succ_s', 'r', 'non_terminal']

        fulls = {}
        
        fulls['demo_transition_mask'] = []

        if self.use_PER:
            fulls['importanceSamplingWeights'] = []

        if self.recurrent:
            keys += ['rnn_states', 'next_rnn_states']
        
        if self.goal_oriented:
            keys += ['g']
        
        for key in keys:    fulls[key] = []

        using_ray = False
        if hasattr(self.storages[0].__len__, "remote"):
            using_ray = True
            
        if using_ray:
            storage_sizes_ids = [storage.__len__.remote() for storage in self.storages]
            storage_sizes = []
            storage_samples_ids = []
            for storage_idx, storage in enumerate(self.storages):
                # Check that there is something in the storage 
                storage_size = ray.get(storage_sizes_ids[storage_idx])
                storage_sizes.append(storage_size)
                if storage_size <= 1: continue
                if self.use_PER:
                    storage_samples_ids.append(storage.sample.remote(batch_size=minibatch_size, keys=keys))
        else:
            storage_sizes = [len(storage) for storage in self.storages]

        # adding the expert_buffer:
        if self.expert_buffer is not None:
            storage_sizes.append(len(self.expert_buffer))
        else:
            storage_sizes.append(0)

        '''
        Sample from experience buffers
        '''
        for storage_idx, storage in enumerate(self.storages+[self.expert_buffer]):
            demo_transition_mask = torch.zeros(minibatch_size)
            
            nbr_sampling_values = minibatch_size
            if storage_idx == len(self.storages):
                nbr_sampling_values = num_demonstration_samples
                demo_transition_mask = torch.ones(minibatch_size)
            
            fulls['demo_transition_mask'].append(demo_transition_mask)

            # Check that there is something in the storage
            storage_size = storage_sizes[storage_idx]
            if storage is None or storage_size <= 1: continue
            if self.use_PER:
                if using_ray and storage_idx != len(self.storages):
                    sample, importanceSamplingWeights = ray.get(storage_samples_ids[storage_idx])
                    #sample, importanceSamplingWeights = ray.get(storage.sample.remote(batch_size=nbr_sampling_values, keys=keys))
                else:
                    sample, importanceSamplingWeights = storage.sample(batch_size=minibatch_size, keys=keys)
                importanceSamplingWeights = torch.from_numpy(importanceSamplingWeights)
                fulls['importanceSamplingWeights'].append(importanceSamplingWeights)
            else:
                sample = storage.sample(batch_size=num_replay_buffer_samples, keys=keys)
            
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

        start = time.time()

        #beta = self.storages[0].get_beta() if self.use_PER else 1.0
        beta = 1.0
        using_ray = False
        if self.use_PER:
            if hasattr(self.storages[0].get_beta, "remote"):
                beta_id = self.storages[0].get_beta.remote()
                beta = ray.get(beta_id)
                using_ray = True
            else:
                beta = self.storages[0].get_beta()

        states = samples['s']
        actions = samples['a']
        next_states = samples['succ_s']
        rewards = samples['r']
        non_terminals = samples['non_terminal']

        rnn_states = samples['rnn_states'] if 'rnn_states' in samples else None
        next_rnn_states = samples['next_rnn_states'] if 'next_rnn_states' in samples else None
        goals = samples['g'] if 'g' in samples else None

        importanceSamplingWeights = samples['importanceSamplingWeights'] if 'importanceSamplingWeights' in samples else None
        demo_transition_mask = samples['demo_transition_mask']

        # For each actor, there is one mini_batch update:
        sampler = random_sample(np.arange(states.size(0)), minibatch_size)
        list_batch_indices = [storage_idx*minibatch_size+np.arange(minibatch_size) \
                                for storage_idx, _ in enumerate(self.storages)]
        
        if self.expert_buffer is not None:
            list_batch_indices += [len(self.storages)*minibatch_size+np.arange(states.size(0)-minibatch_size*len(self.storages))]
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
            
            sampled_demo_transition_mask = demo_transition_mask[batch_indices].cuda() if self.kwargs['use_cuda'] else demo_transition_mask[batch_indices]
            
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
                                          demo_transition_mask=sampled_demo_transition_mask,
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
            if using_ray:
                self._update_replay_buffer_priorities_ray(
                    sampled_losses_per_item=sampled_losses_per_item, 
                    array_batch_indices=array_batch_indices,
                    minibatch_size=minibatch_size,
                )
            else:
                self._update_replay_buffer_priorities(
                    sampled_losses_per_item=sampled_losses_per_item, 
                    array_batch_indices=array_batch_indices,
                    minibatch_size=minibatch_size,
                )

        end = time.time()
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('PerUpdate/TimeComplexity/OptimizationLoss', end-start, self.param_update_counter)
            self.summary_writer.flush()
