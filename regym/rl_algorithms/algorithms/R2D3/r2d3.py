from typing import Dict, List, Any, Optional, Callable

from functools import partial

import numpy as np
import torch
import torch.nn as nn

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
                 sum_writer=None):
        super().__init__(
            kwargs=kwargs, 
            model=model, 
            target_model=target_model, 
            optimizer=optimizer, 
            loss_fn=loss_fn, 
            sum_writer=sum_writer
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

    # NOTE: we are overriding this function from R2D2Algorithm
    def retrieve_values_from_storages(self, minibatch_size: int):
        '''
        We sample from both replay buffers (expert_buffer and agent
        collected experiences) according to property self.demo_ratio
        '''
        sample_buffers = np.random.choice([1,0],size=minibatch_size,p=[self.demo_ratio,(1-self.demo_ratio)])
        num_demonstrations_samples = np.sum(sample_buffers)
        num_replay_buffer_samples = minibatch_size - num_demonstrations_samples

        self.sample_split = num_replay_buffer_samples

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

        '''
        Sample from experience buffers
        '''
        for storage in self.storages:
            # Check that there is something in the storage 
            if len(storage) <= 1: continue
            if self.use_PER:
                sample, importanceSamplingWeights = storage.sample(batch_size=num_replay_buffer_samples, keys=keys)
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
        
            '''
            Sample from expert buffer
            '''
            if num_demonstrations_samples > 0:
            
                if self.use_PER:
                    sample, importanceSamplingWeights = self.expert_buffer.sample(batch_size=num_demonstrations_samples, keys=keys)
                    importanceSamplingWeights = torch.from_numpy(importanceSamplingWeights)
                    fulls['importanceSamplingWeights'].append(importanceSamplingWeights)
                else:
                    sample = self.expert_buffer.sample(batch_size=num_demonstrations_samples, keys=keys)
                    
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
        
    
    # NOTE: we are overriding this function from R2D2Algorithm
    def _update_replay_buffer_priorities(self, 
                                         sampled_losses_per_item: List[torch.Tensor], 
                                         array_batch_indices: List,
                                         minibatch_size: int):
        '''
        Updates the priorities of each sampled elements from their respective storages.
        '''
        # losses corresponding to sampled batch indices: 
        sampled_losses_per_item = torch.cat(sampled_losses_per_item, dim=0).cpu().detach().numpy()
        # (batch_size, unroll_dim, 1)
        unroll_length = self.sequence_replay_unroll_length - self.sequence_replay_burn_in_length
        for sloss, arr_bidx in zip(sampled_losses_per_item, array_batch_indices):
            storage_idx = arr_bidx//minibatch_size
            el_idx_in_batch = arr_bidx%minibatch_size
            
            # Ids less than self.sample_split come from replay buffer
            if el_idx_in_batch < self.sample_split:
                el_idx_in_storage = self.storages[storage_idx].tree_indices[el_idx_in_batch]
                
                # (unroll_dim,)
                new_priority = self.storages[storage_idx].sequence_priority(sloss.reshape(unroll_length,))
                self.storages[storage_idx].update(idx=el_idx_in_storage, priority=new_priority)
            else:
                el_idx_in_storage = self.expert_buffer.tree_indices[el_idx_in_batch - self.sample_split + (storage_idx * (minibatch_size - self.sample_split))]
                
                # (unroll_dim,)
                new_priority = self.expert_buffer.sequence_priority(sloss.reshape(unroll_length,))
                self.expert_buffer.update(idx=el_idx_in_storage, priority=new_priority)
