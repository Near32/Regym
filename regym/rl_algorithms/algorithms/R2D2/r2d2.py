from typing import Dict, List, Any, Optional, Callable

import copy
from collections import deque 
from functools import partial 

import ray
import numpy as np
import torch
import torch.nn as nn

import regym
from regym.rl_algorithms.algorithms.algorithm import Algorithm
from regym.rl_algorithms.algorithms.R2D2 import r2d2_loss
from regym.rl_algorithms.algorithms.DQN import DQNAlgorithm
from regym.rl_algorithms.replay_buffers import ReplayStorage, PrioritizedReplayStorage, SharedPrioritizedReplayStorage
from regym.rl_algorithms.utils import _concatenate_hdict, _concatenate_list_hdict

sum_writer = None

store_on_terminal = False


class R2D2Algorithm(DQNAlgorithm):
    def __init__(self, 
                 kwargs: Dict[str, Any], 
                 model: nn.Module,
                 target_model: Optional[nn.Module] = None,
                 optimizer=None,
                 loss_fn: Callable = r2d2_loss.compute_loss,
                 sum_writer=None,
                 name='r2d2_algo',
                 single_storage=True):
        
        Algorithm.__init__(self=self, name=name)
        self.single_storage = single_storage

        self.sequence_replay_unroll_length = kwargs['sequence_replay_unroll_length']
        self.sequence_replay_overlap_length = kwargs['sequence_replay_overlap_length']
        self.sequence_replay_burn_in_length = kwargs['sequence_replay_burn_in_length']
        
        self.replay_buffer_capacity = kwargs['replay_capacity'] // (self.sequence_replay_unroll_length-self.sequence_replay_overlap_length)
        
        assert kwargs['n_step'] < kwargs['sequence_replay_unroll_length']-kwargs['sequence_replay_burn_in_length'], \
                "Sequence_replay_unroll_length-sequence_replay_burn_in_length needs to be set to a value greater \
                 than n_step return, in order to be able to compute the bellman target."
        
        super().__init__(
            kwargs=kwargs, 
            model=model, 
            target_model=target_model, 
            optimizer=optimizer, 
            loss_fn=loss_fn, 
            sum_writer=sum_writer
        )
        
        self.storage_buffer_refresh_period = 32
        self.storage_buffers = [list() for _ in range(self.nbr_actor)]

        self.sequence_replay_buffers = [deque(maxlen=self.sequence_replay_unroll_length) for _ in range(self.nbr_actor)]
        self.sequence_replay_buffers_count = [0 for _ in range(self.nbr_actor)]

    # NOTE: overridding from DQNAlgorithm
    def reset_storages(self, nbr_actor: int=None):
        if nbr_actor is not None:    
            self.nbr_actor = nbr_actor
        
            if self.n_step > 1:
                self.n_step_buffers = [deque(maxlen=self.n_step) for _ in range(self.nbr_actor)]

            self.sequence_replay_buffers = [deque(maxlen=self.sequence_replay_unroll_length) for _ in range(self.nbr_actor)]
            self.sequence_replay_buffers_count = [0 for _ in range(self.nbr_actor)]    
            
        nbr_storages = 1
        if not(self.single_storage):
            nbr_storages = self.nbr_actor
        storage_capacity = self.replay_buffer_capacity // nbr_storages
        
        self.storages = []
        keys = ['s', 'a', 'r', 'non_terminal']
        if self.recurrent:  keys += ['rnn_states']
        if self.goal_oriented:    keys += ['g']
        
        # TODO: WARNING: rnn states can be handled that way but it is meaningless since dealing with sequences...
        circular_keys={'succ_s':'s'}
        # On the contrary to DQNAlgorithm,
        # since we are dealing with batches of unrolled experiences,
        # succ_s ought to be the sequence of unrolled experiences that comes
        # directly after the current unrolled sequence s:
        circular_offsets={'succ_s':1}
        
        # TODO: WARNING: rnn states can be handled that way but it is meaningless since dealing with sequences...
        if self.recurrent:
            circular_keys.update({'next_rnn_states':'rnn_states'})
            circular_offsets.update({'next_rnn_states':1})

        beta_increase_interval = float(self.kwargs['PER_beta_increase_interval'])  if 'PER_beta_increase_interval' in self.kwargs else 1e4
        
        for i in range(nbr_storages):
            if self.kwargs['use_PER']:
                if regym.RegymManager is not None:
                    try:
                        storage = ray.get_actor(f"{self.name}.storage_{i}")
                    except ValueError:  # Name is not taken.
                        storage = SharedPrioritizedReplayStorage.options(
                            name=f"{self.name}.storage_{i}"
                        ).remote(
                        capacity=storage_capacity,
                        alpha=self.kwargs['PER_alpha'],
                        beta=self.kwargs['PER_beta'],
                        beta_increase_interval=beta_increase_interval,
                        eta=self.kwargs['sequence_replay_PER_eta'],
                        keys=keys,
                        circular_keys=circular_keys,                 
                        circular_offsets=circular_offsets
                    )
                else:
                    storage = PrioritizedReplayStorage(
                            capacity=storage_capacity,
                            alpha=self.kwargs['PER_alpha'],
                            beta=self.kwargs['PER_beta'],
                            beta_increase_interval=beta_increase_interval,
                            eta=self.kwargs['sequence_replay_PER_eta'],
                            keys=keys,
                            circular_keys=circular_keys,                 
                            circular_offsets=circular_offsets
                        )
                self.storages.append(storage)
            else:
                self.storages.append(
                    ReplayStorage(
                        capacity=storage_capacity,
                        keys=keys,
                        circular_keys=circular_keys,                 
                        circular_offsets=circular_offsets
                    )
                )

    def stored_experiences(self):
        self.train_request_count += 1
        if isinstance(self.storages[0], ray.actor.ActorHandle):
            nbr_stored_sequences = sum([ray.get(storage.__len__.remote()) for storage in self.storages])
        else:
            nbr_stored_sequences = sum([len(storage) for storage in self.storages])

        nbr_stored_experiences = nbr_stored_sequences*(self.sequence_replay_unroll_length-self.sequence_replay_overlap_length)

        global summary_writer
        if self.summary_writer is None:
            self.summary_writer = summary_writer
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('PerTrainingRequest/NbrStoredExperiences', nbr_stored_experiences, self.train_request_count)
        #print(f"Train request: {self.train_request_count} // nbr_exp stored: {nbr_stored_experiences}")
        return nbr_stored_experiences
    
    def _prepare_sequence_exp_dict(self, sequence_buffer):
        '''
        Returns a dictionnary of numpy arrays from the list of dictionnaries `sequence buffer`. 
        '''
        keys = sequence_buffer[0].keys()
        d = {}
        for key in keys:
            if key == 'info': continue
            # (batch_size=1, unroll_dim, ...)
            if isinstance(sequence_buffer[0][key], dict):
                values = [sequence_buffer[i][key] for i in range(len(sequence_buffer))]
                value = _concatenate_list_hdict(
                    lhds=values, 
                    concat_fn=partial(torch.cat, dim=1),   # concatenate on the unrolling dimension (axis=1).
                    preprocess_fn=lambda x: x.reshape(1, 1, -1),
                )
            else:
                value = torch.cat(
                    [
                        sequence_buffer[i][key].unsqueeze(dim=1)    # add unroll dim 
                        for i in range(len(sequence_buffer))
                    ],
                    axis=1
                )
            d[key] = value
        return d

    def _add_sequence_to_replay_storage(self, actor_index:int, override:bool=False):
        storage_index = actor_index
        if self.single_storage:
            storage_index = 0
        # Can we add the current sequence buffer to the replay storage?
        if not override and len(self.sequence_replay_buffers[actor_index]) < self.sequence_replay_unroll_length:
            return
        if override or self.sequence_replay_overlap_length == 0 or self.sequence_replay_buffers_count[actor_index] % self.sequence_replay_overlap_length == 0:
            current_sequence_exp_dict = self._prepare_sequence_exp_dict(list(self.sequence_replay_buffers[actor_index]))
            if self.use_PER:
                init_sampling_priority = None 
                if isinstance(self.storages[storage_index], ray.actor.ActorHandle):
                    ray.get(
                        self.storages[storage_index].add.remote(
                            current_sequence_exp_dict, 
                            priority=init_sampling_priority
                        )
                    )
                else:
                    self.storages[storage_index].add(
                        current_sequence_exp_dict, 
                        priority=init_sampling_priority
                    )
            else:
                self.storages[storage_index].add(current_sequence_exp_dict)

    # NOTE: overriding this function from DQNAlgorithm -
    def store(self, exp_dict, actor_index=0):
        '''
        Compute n-step returns, for each actor, separately,
        and then assembles experiences into sequences of experiences of length
        `self.sequence_replay_unroll_length`, with an overlap of `self.sequence_replay_overlap_length`.

        Note: No sequence being stored crosses the episode barrier. 
        If the input `exp_dict` is terminal, 
        then the n-step buffer is dumped entirely in the sequence buffer and the sequence is committed 
        to the relevant storage buffer.
        '''
        if self.n_step>1:
            # Append to deque:
            self.n_step_buffers[actor_index].append(exp_dict)
            if len(self.n_step_buffers[actor_index]) < self.n_step:
                return
        
        reached_end_of_episode = not(exp_dict['non_terminal'])
        nbr_experience_to_handle = 1
        if self.n_step > 1 and reached_end_of_episode:
            nbr_experience_to_handle = min(self.n_step, len(self.n_step_buffers[actor_index])) 

        for exp_it in range(nbr_experience_to_handle):
            if self.n_step>1:
                # Compute n-step return of the first element of deque:
                truncated_n_step_return = self._compute_truncated_n_step_return(actor_index=actor_index)
                # Retrieve the first element of deque:
                current_exp_dict = copy.deepcopy(self.n_step_buffers[actor_index][0])
                current_exp_dict['r'] = truncated_n_step_return
            else:
                current_exp_dict = exp_dict
            
            if self.goal_oriented and 'g' not in current_exp_dict:
                current_exp_dict['g'] = current_exp_dict['goals']['desired_goals']['s']

            # Store in relevant sequence buffer:
            self.sequence_replay_buffers[actor_index].append(current_exp_dict)
            self.sequence_replay_buffers_count[actor_index] += 1

            if nbr_experience_to_handle > 1:
                # If we need to dump the whole buffer into the sequence,
                # then here we make sure the next iteration of the loop will handle
                # the next element of the n_step buffer until it is empty. 
                self.n_step_buffers[actor_index].popleft()

            # Maybe add to replay storage?
            self._add_sequence_to_replay_storage(
                actor_index=actor_index, 
                override=(store_on_terminal and (exp_it==nbr_experience_to_handle-1) and reached_end_of_episode),
                # Only add if experience count handled, 
                # no longer cares about crossing the episode barrier as the loss handles it,
                # unless store_on_terminal is true
            )

        # Make sure the sequence buffer do not cross the episode barrier:
        # UPDATE: no longer care about this since the loss takes care of the episode barriers...
        # unless store_on_terminal is true
        if (store_on_terminal and reached_end_of_episode):
            self.sequence_replay_buffers[actor_index].clear()
            # Re-initialise the buffer count since the buffer is cleared out.
            # Otherwise some stored sequences could have length different than
            # unroll_length since reached_end_of_episode is not necessarily
            # synchronised with the modulo sequence_replay_overlap_length operation
            # that controls whether to store the current sequence.
            self.sequence_replay_buffers_count[actor_index] = 0

    # NOTE: we are overriding this function from DQNAlgorithm
    def _update_replay_buffer_priorities(self, 
                                         sampled_losses_per_item: List[torch.Tensor], 
                                         array_batch_indices: List,
                                         minibatch_size: int):
        '''
        Updates the priorities of each sampled elements from their respective storages.

        #TODO: update to use Ray and get_tree_indices...
        '''
        # losses corresponding to sampled batch indices: 
        sampled_losses_per_item = torch.cat(sampled_losses_per_item, dim=0).cpu().detach().numpy()
        # (batch_size, unroll_dim, 1)
        unroll_length = self.sequence_replay_unroll_length - self.sequence_replay_burn_in_length

        if isinstance(self.storages[0], ray.actor.ActorHandle):
            ps_tree_indices = [ray.get(storage.get_tree_indices.remote()) for storage in self.storages]
        else:
            ps_tree_indices = [storage.get_tree_indices() for storage in self.storages]
        
        for sloss, arr_bidx in zip(sampled_losses_per_item, array_batch_indices):
            storage_idx = arr_bidx//minibatch_size
            el_idx_in_batch = arr_bidx%minibatch_size

            el_idx_in_storage = ps_tree_indices[storage_idx][el_idx_in_batch]
            #el_idx_in_storage = self.storages[storage_idx].tree_indices[el_idx_in_batch]
            
            # (unroll_dim,)
            if isinstance(self.storages[0], ray.actor.ActorHandle):
                new_priority = ray.get(self.storages[storage_idx].sequence_priority.remote(sloss.reshape(unroll_length,)))
                ray.get(self.storages[storage_idx].update.remote(idx=el_idx_in_storage, priority=new_priority))
            else:
                new_priority = self.storages[storage_idx].sequence_priority(sloss.reshape(unroll_length,))
                self.storages[storage_idx].update(idx=el_idx_in_storage, priority=new_priority)

