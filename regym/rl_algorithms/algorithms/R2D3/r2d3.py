from typing import Dict, List, Any, Optional, Callable

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import ray 
import time 

import regym
from ...networks import hard_update, random_sample

from regym.rl_algorithms.algorithms.DQN import dqn_loss, ddqn_loss
from regym.rl_algorithms.algorithms.R2D2 import R2D2Algorithm, r2d2_loss
from regym.rl_algorithms.algorithms.DQN import DQNAlgorithm
from regym.rl_algorithms.replay_buffers import PrioritizedReplayStorage, ReplayStorage

from regym.rl_algorithms.utils import _concatenate_list_hdict

sum_writer = None

class R2D3Algorithm(R2D2Algorithm):

    def __init__(self, kwargs: Dict[str, Any], 
                 model: nn.Module,
                 target_model: Optional[nn.Module] = None,
                 expert_buffer: ReplayStorage = None,
                 optimizer=None,
                 loss_fn: Callable = r2d2_loss.compute_loss,
                 sum_writer=None,
                 name='r2d3_algo'):
        R2D2Algorithm.__init__(
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

    def build_expert_buffer(self):
        
        keys = ['s', 'a', 'r', 'non_terminal']
        if self.recurrent:  keys += ['rnn_states']
        if self.goal_oriented:    keys += ['g']
        
        circular_keys={'succ_s':'s'}
        circular_offsets={'succ_s':self.n_step}
        if self.recurrent:
            circular_keys.update({'next_rnn_states':'rnn_states'})
            circular_offsets.update({'next_rnn_states':1})

        beta_increase_interval = float(self.kwargs['PER_beta_increase_interval'])  if 'PER_beta_increase_interval' in self.kwargs else 1e4
            
        if self.kwargs['use_PER']:
            self.expert_buffer = PrioritizedReplayStorage(
                capacity=self.kwargs['replay_capacity']//self.nbr_actor,
                alpha=self.kwargs['PER_alpha'],
                beta=self.kwargs['PER_beta'],
                beta_increase_interval=beta_increase_interval,
                keys=keys,
                circular_keys=circular_keys,                 
                circular_offsets=circular_offsets
            )
        else:
            self.expert_buffer = ReplayStorage(
                capacity=self.kwargs['replay_capacity']//self.nbr_actor,
                keys=keys,
                circular_keys=circular_keys,                 
                circular_offsets=circular_offsets
            )

    # NOTE: we are overriding this function from DQNAlgorithm
    def retrieve_values_from_storages(self, minibatch_size: int=32, pretraining=False):
        '''
        We sample from both replay buffers (expert_buffer and agent
        collected experiences) according to property self.demo_ratio
        
        Each storage stores in their key entries either numpy arrays or hierarchical dictionnaries of numpy arrays.
        This function samples from each storage, concatenate the sampled elements on the batch dimension,
        and maintains the hierarchy of dictionnaries.
        '''
        demo_ratio = self.demo_ratio
        if pretraining:
            demo_ratio = 0.5

        num_demonstration_samples = (minibatch_size*len(self.storages)) * demo_ratio / (1 - demo_ratio)  
        
        keys=['s', 'a', 'succ_s', 'r', 'non_terminal']

        fulls = {}
        
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
            nbr_sampling_values = minibatch_size
            if storage_idx == len(self.storages):
                nbr_sampling_values = num_demonstration_samples
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
            beta_id = self.storages[0].get_beta.remote()
            beta = ray.get(beta_id)
            using_ray = True

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
    
    # NOTE: we are overriding this function from R2D2Algorithm
    def _update_replay_buffer_priorities(self, 
                                         sampled_losses_per_item: List[torch.Tensor], 
                                         array_batch_indices: List,
                                         minibatch_size: int):
        '''
        Updates the priorities of each sampled elements from their respective storages.
        '''
        nbr_policy_sample = minibatch_size*len(self.storages)
        # losses corresponding to sampled batch indices: 
        sampled_losses_per_item = torch.cat(sampled_losses_per_item, dim=0).cpu().detach().numpy()
        # (batch_size, unroll_dim, 1)
        unroll_length = self.sequence_replay_unroll_length - self.sequence_replay_burn_in_length

        #ps_tree_indices = [ray.get(storage.get_tree_indices.remote()) for storage in self.storages]
        ps_tree_indices = [storage.get_tree_indices() for storage in self.storages]
        
        for sloss, arr_bidx in zip(sampled_losses_per_item, array_batch_indices):
            storage_idx = arr_bidx//minibatch_size

            # Ids less than self.sample_split come from replay buffer
            if storage_idx < len(self.storages):
                el_idx_in_batch = arr_bidx%minibatch_size
                #el_idx_in_storage = self.storages[storage_idx].tree_indices[el_idx_in_batch]
                el_idx_in_storage = ps_tree_indices[storage_idx][el_idx_in_batch]
                # (unroll_dim,)
                #new_priority = ray.get(self.storages[storage_idx].sequence_priority.remote(sloss.reshape(unroll_length,)))
                new_priority = self.storages[storage_idx].sequence_priority(sloss.reshape(unroll_length,))
                #ray.get(self.storages[storage_idx].update.remote(idx=el_idx_in_storage, priority=new_priority))
                self.storages[storage_idx].update(idx=el_idx_in_storage, priority=new_priority)
            else:
                el_idx_in_batch = arr_bidx - nbr_policy_sample
                el_idx_in_storage = self.expert_buffer.tree_indices[el_idx_in_batch]
                # (unroll_dim,)
                new_priority = self.expert_buffer.sequence_priority(sloss.reshape(unroll_length,))
                self.expert_buffer.update(idx=el_idx_in_storage, priority=new_priority)
            
    # NOTE: we are overriding this function from R2D2Algorithm
    def _update_replay_buffer_priorities_ray(self, 
                                         sampled_losses_per_item: List[torch.Tensor], 
                                         array_batch_indices: List,
                                         minibatch_size: int):
        '''
        Updates the priorities of each sampled elements from their respective storages.
        '''
        nbr_policy_sample = minibatch_size*len(self.storages)
        # losses corresponding to sampled batch indices: 
        sampled_losses_per_item = torch.cat(sampled_losses_per_item, dim=0).cpu().detach().numpy()
        # (batch_size, unroll_dim, 1)
        unroll_length = self.sequence_replay_unroll_length - self.sequence_replay_burn_in_length

        ps_tree_indices_ids = [storage.get_tree_indices.remote() for storage in self.storages]
        
        new_priorities_pp_ids = []
        for sloss, arr_bidx in zip(sampled_losses_per_item, array_batch_indices):
            storage_idx = arr_bidx//minibatch_size

            # Ids less than self.sample_split come from replay buffer
            if storage_idx < len(self.storages):
                el_idx_in_batch = arr_bidx%minibatch_size
                new_sequence_priority = self.storages[storage_idx].sequence_priority.remote(sloss.reshape(unroll_length,))
            else:
                el_idx_in_batch = arr_bidx - nbr_policy_sample
                new_sequence_priority = self.expert_buffer.sequence_priority(sloss.reshape(unroll_length,))

            new_priorities_pp_ids.append( 
                [
                    storage_idx, 
                    el_idx_in_batch, 
                    new_sequence_priority    
                ]
            )
        
        ps_tree_indices = [ray.get(ps_tree_indices_id) for ps_tree_indices_id in ps_tree_indices_ids]
        
        for storage_idx, el_idx_in_batch, new_priority_id in new_priorities_pp_ids:
            #new_priority = self.storages[storage_idx].sequence_priority(sloss.reshape(unroll_length,))
            # Ids less than self.sample_split come from replay buffer
            if storage_idx < len(self.storages):
                el_idx_in_storage = ps_tree_indices[storage_idx][el_idx_in_batch]
                new_priority = ray.get(new_priority_id)
                self.storages[storage_idx].update.remote(idx=el_idx_in_storage, priority=new_priority)
            else:
                el_idx_in_storage = self.expert_buffer.tree_indices[el_idx_in_batch]
                new_priority = new_priority_id
                self.expert_buffer.update(idx=el_idx_in_storage, priority=new_priority)
                # (unroll_dim,)