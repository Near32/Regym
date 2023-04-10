from typing import Dict, List, Any, Optional, Callable

import copy
from collections import deque 
from functools import partial 

import ray
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt 

import regym
from regym.rl_algorithms.algorithms.algorithm import Algorithm
from regym.rl_algorithms.algorithms.R2D2 import r2d2_loss
from regym.rl_algorithms.algorithms.DQN import DQNAlgorithm
from regym.rl_algorithms.replay_buffers import ReplayStorage, PrioritizedReplayStorage, SharedPrioritizedReplayStorage
from regym.rl_algorithms.utils import archi_concat_fn, concat_fn, _concatenate_hdict, _concatenate_list_hdict

import wandb
sum_writer = None



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

        print(kwargs)

        self.sequence_replay_unroll_length = kwargs['sequence_replay_unroll_length']
        self.sequence_replay_overlap_length = kwargs['sequence_replay_overlap_length']
        self.sequence_replay_burn_in_length = kwargs['sequence_replay_burn_in_length']
        
        self.sequence_replay_store_on_terminal = kwargs["sequence_replay_store_on_terminal"]
        
        self.replay_buffer_capacity = kwargs['replay_capacity'] // (self.sequence_replay_unroll_length-self.sequence_replay_overlap_length)
        
        assert kwargs['n_step'] < kwargs['sequence_replay_unroll_length']-kwargs['sequence_replay_burn_in_length'], \
                "Sequence_replay_unroll_length-sequence_replay_burn_in_length needs to be set to a value greater \
                 than n_step return, in order to be able to compute the bellman target."
        
        # DEPRECATED in order to allow extra_inputs infos 
        # stored in the rnn_states that acts as frame_states...
        #self.recurrent = False
        self.recurrent = True
        
        # TECHNICAL DEBT: check for recurrent property by looking at the modules in the model rather than relying on the kwargs that may contain
        # elements that do not concern the model trained by this algorithm, given that it is now use-able inside I2A...
        self.recurrent_nn_submodule_names = [hyperparameter for hyperparameter, value in self.kwargs.items() if isinstance(value, str) and 'RNN' in value]

        self.keys = ['s', 'a', 'r', 'non_terminal']
        if self.recurrent:  
            self.keys += ['rnn_states']
            self.circular_keys.update({'next_rnn_states':'rnn_states'})
            self.circular_offsets.update({'next_rnn_states':1})
         
        # TODO: WARNING: rnn states can be handled that way but it is meaningless since dealing with sequences...
        self.circular_keys={'succ_s':'s'}
        # On the contrary to DQNAlgorithm,
        # since we are dealing with batches of unrolled experiences,
        # succ_s ought to be the sequence of unrolled experiences that comes
        # directly after the current unrolled sequence s:
        self.circular_offsets={'succ_s':1}
        
        super().__init__(
            kwargs=kwargs, 
            model=model, 
            target_model=target_model, 
            optimizer=optimizer, 
            loss_fn=loss_fn, 
            sum_writer=sum_writer
        )
        
        self.keys_to_retrieve = ['s', 'a', 'succ_s', 'r', 'non_terminal']
        if self.recurrent:  
            self.keys_to_retrieve += ['rnn_states', 'next_rnn_states']
        
        self.storage_buffer_refresh_period = 32
        self.storage_buffers = [list() for _ in range(self.nbr_actor)]
        self.sequence_replay_buffers = [deque(maxlen=self.sequence_replay_unroll_length) for _ in range(self.nbr_actor)]
        self.sequence_replay_buffers_count = [0 for _ in range(self.nbr_actor)]

    # NOTE: overridding from DQNAlgorithm
    def reset_storages(self, nbr_actor: int=None):
        if nbr_actor is not None:    
            self.nbr_actor = nbr_actor
        
            """
            if self.n_step > 1:
                self.n_step_buffers = [deque(maxlen=self.n_step) for _ in range(self.nbr_actor)]
            """

            self.storage_buffers = [list() for _ in range(self.nbr_actor)]
            self.sequence_replay_buffers = [deque(maxlen=self.sequence_replay_unroll_length) for _ in range(self.nbr_actor)]
            self.sequence_replay_buffers_count = [0 for _ in range(self.nbr_actor)]    
            
        nbr_storages = 1
        if not(self.single_storage):
            nbr_storages = self.nbr_actor
        storage_capacity = self.replay_buffer_capacity // nbr_storages
        
        self.storages = []
        beta_increase_interval = None
        if 'PER_beta_increase_interval' in self.kwargs and self.kwargs['PER_beta_increase_interval']!='None':
            beta_increase_interval = float(self.kwargs['PER_beta_increase_interval'])  

        self.pre_storage_sequence_exp_dict = []
        self.pre_storage_sequence_storage_idx = []

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
                        keys=self.keys,
                        circular_keys=self.circular_keys,                 
                        circular_offsets=self.circular_offsets
                    )
                else:
                    if self.use_mp:
                        rp_fn = regym.AlgoManager.PrioritizedReplayStorage
                    else:
                        rp_fn = PrioritizedReplayStorage
                    storage = rp_fn(
                            capacity=storage_capacity,
                            alpha=self.kwargs['PER_alpha'],
                            beta=self.kwargs['PER_beta'],
                            beta_increase_interval=beta_increase_interval,
                            eta=self.kwargs['sequence_replay_PER_eta'],
                            keys=self.keys,
                            circular_keys=self.circular_keys,
                            circular_offsets=self.circular_offsets,
                        )
                self.storages.append(storage)
            else:
                self.storages.append(
                    ReplayStorage(
                        capacity=storage_capacity,
                        keys=self.keys,
                        circular_keys=self.circular_keys,                 
                        circular_offsets=self.circular_offsets
                    )
                )

    def stored_experiences(self):
        self.train_request_count += 1
        if isinstance(self.storages[0], ray.actor.ActorHandle):
            nbr_stored_sequences = sum([ray.get(storage.__len__.remote()) for storage in self.storages])
        else:
            nbr_stored_sequences = sum([len(storage) for storage in self.storages])

        nbr_stored_experiences = nbr_stored_sequences*(self.sequence_replay_unroll_length-self.sequence_replay_overlap_length)

        wandb.log({'PerTrainingRequest/NbrStoredExperiences': nbr_stored_experiences}, commit=False) #, self.train_request_count)
        #print(f"Train request: {self.train_request_count} // nbr_exp stored: {nbr_stored_experiences}")
        return nbr_stored_experiences
    
    def _prepare_sequence_exp_dict(self, sequence_buffer):
        '''
        Returns a dictionnary of numpy arrays from the list of dictionnaries `sequence buffer`. 
        '''
        keys = sequence_buffer[0].keys()
        d = {}
        for key in keys:
            if 'info' in key: continue
            # (batch_size=1, unroll_dim, ...)
            if isinstance(sequence_buffer[0][key], dict):
                values = [sequence_buffer[i][key] for i in range(len(sequence_buffer))]
                value = _concatenate_list_hdict(
                    lhds=values, 
                    #concat_fn=partial(torch.cat, dim=1),   # concatenate on the unrolling dimension (axis=1).
                    #TODO: verify that unrolling on list is feasible:
                    #concat_fn=(lambda x: torch.cat(x, dim=1) if x[0].shape==x[1].shape else np.array(x, dtype=object)),
                    concat_fn=concat_fn,
                    #concat_fn=archi_concat_fn,
                    preprocess_fn=lambda x: x.clone().reshape(1, 1, *x.shape[1:]),
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
            # Verify the length of the sequence:
            while len(self.sequence_replay_buffers[actor_index]) != self.sequence_replay_unroll_length:
                # This can only happen when overriding, i.e. end of episode is reached and we store on end of episode:
                # Therefore we can pad the sequence with the last transition, that consist of a terminal transition:
                self.sequence_replay_buffers[actor_index].append(copy.deepcopy(self.sequence_replay_buffers[actor_index][-1]))

            current_sequence_exp_dict = self._prepare_sequence_exp_dict(list(self.sequence_replay_buffers[actor_index]))
            self.param_obs_counter += (self.sequence_replay_unroll_length-self.sequence_replay_overlap_length)
            if self.use_PER:
                if self.kwargs['PER_compute_initial_priority']:
                    """
                    Put the experience dict into a buffer until we have enough
                    to compute td_errors in batch.
                    """
                    self.pre_storage_sequence_exp_dict.append(current_sequence_exp_dict)
                    self.pre_storage_sequence_storage_idx.append(storage_index)
                    if len(self.pre_storage_sequence_exp_dict) < self.batch_size//self.sequence_replay_unroll_length:
                        return 

                    samples = {}
                    for exp_dict in self.pre_storage_sequence_exp_dict:
                        for key, value in exp_dict.items():
                            if key not in samples:  samples[key] = []
                            samples[key].append(value)

                    for key, value_list in samples.items():
                        if len(value_list) >1:
                            if isinstance(value_list[0], dict):
                                batched_values = _concatenate_list_hdict(
                                    lhds=value_list, 
                                    concat_fn=partial(torch.cat, dim=0),   # concatenate on the batch dimension (axis=0).
                                    preprocess_fn=(lambda x:x),
                                )
                            else:
                                batched_values = torch.cat(value_list, dim=0)
                        else:
                            batched_values = value_list[0]

                        samples[key] = batched_values

                    with torch.no_grad():
                        td_error_per_item = self.compute_td_error(samples=samples)[-1].cpu().detach().numpy()
                    
                    unroll_length = self.sequence_replay_unroll_length - self.sequence_replay_burn_in_length
                    for exp_dict_idx, (csed, cs_storage_idx) in enumerate(zip(self.pre_storage_sequence_exp_dict, self.pre_storage_sequence_storage_idx)):
                        if isinstance(self.storages[0], ray.actor.ActorHandle):
                            new_priority = ray.get(
                                self.storages[cs_storage_idx].sequence_priority.remote(
                                    td_error_per_item[exp_dict_idx].reshape(unroll_length,)
                                )
                            )
                        else:
                            new_priority = self.storages[cs_storage_idx].sequence_priority(
                                td_error_per_item[exp_dict_idx].reshape(unroll_length,)
                            )
                        
                        if isinstance(self.storages[cs_storage_idx], ray.actor.ActorHandle):
                            ray.get(
                                self.storages[cs_storage_idx].add.remote(
                                    csed, 
                                    priority=new_priority
                                )
                            )
                        else:
                            self.storages[cs_storage_idx].add(
                                csed, 
                                priority=new_priority
                            )

                    self.pre_storage_sequence_exp_dict = []
                    self.pre_storage_sequence_storage_idx = []
                else:
                    new_priority = None 
                    if isinstance(self.storages[storage_index], ray.actor.ActorHandle):
                        ray.get(
                            self.storages[storage_index].add.remote(
                                current_sequence_exp_dict, 
                                priority=new_priority
                            )
                        )
                    else:
                        self.storages[storage_index].add(
                            current_sequence_exp_dict, 
                            priority=new_priority
                        )
            else:
                self.storages[storage_index].add(current_sequence_exp_dict)

    # NOTE: overriding this function from DQNAlgorithm -
    def store(self, exp_dict, actor_index=0):
        '''
        Compute n-step returns, for each actor, separately,
        and then assembles experiences into sequences of experiences of length
        `self.sequence_replay_unroll_length`, with an overlap of 
        `self.sequence_replay_overlap_length`.

        Note: No sequence being stored crosses the episode barrier. 
        If the input `exp_dict` is terminal, 
        then the n-step buffer is dumped entirely in the sequence buffer
        and the sequence is committed to the relevant storage buffer.
        '''
        torch.set_grad_enabled(False)

        if False: #self.n_step>1:
            raise NotImplementedError
            # Append to deque:
            self.n_step_buffers[actor_index].append(copy.deepcopy(exp_dict))
            if len(self.n_step_buffers[actor_index]) < self.n_step:
                return
        
        # We assume non_terminal are the same for all players ==> torch.all :
        assert torch.all(exp_dict['non_terminal'].bool()) == torch.any(exp_dict['non_terminal'].bool())

        reached_end_of_episode = not(torch.all(exp_dict['non_terminal'].bool()))
        nbr_experience_to_handle = 1
        if False: #self.n_step > 1 and reached_end_of_episode:
            raise NotImplementedError
            nbr_experience_to_handle = min(self.n_step, len(self.n_step_buffers[actor_index])) 

        for exp_it in range(nbr_experience_to_handle):
            if False: #self.n_step>1:
                raise NotImplementedError
                # Compute n-step return of the first element of deque:
                truncated_n_step_return = self._compute_truncated_n_step_return(actor_index=actor_index)
                # Retrieve the first element of deque:
                current_exp_dict = copy.deepcopy(self.n_step_buffers[actor_index][0])
                
                current_exp_dict['r'] = truncated_n_step_return
                
                #condition_state = torch.all(self.n_step_buffers[actor_index][0]['s']==self.n_step_buffers[actor_index][-1]['s'])
            else:
                current_exp_dict = exp_dict
                wandb.log({'Training/Storing/CurrentExp/MaxReward':  exp_dict['r'].cpu().max().item()}, commit=True)
            """
            # depr : goal update
            if self.goal_oriented and 'g' not in current_exp_dict:
                current_exp_dict['g'] = current_exp_dict['goals']['desired_goals']['s']
            """

            # Store in relevant sequence buffer:
            self.sequence_replay_buffers[actor_index].append(current_exp_dict)
            self.sequence_replay_buffers_count[actor_index] += 1

            if nbr_experience_to_handle > 1:
                raise NotImplementedError
                # If we need to dump the whole buffer into the sequence,
                # then here we make sure the next iteration of the loop will handle
                # the next element of the n_step buffer until it is empty. 
                self.n_step_buffers[actor_index].popleft()

            # Maybe add to replay storage?
            self._add_sequence_to_replay_storage(
                actor_index=actor_index, 
                override=(self.sequence_replay_store_on_terminal and (exp_it==nbr_experience_to_handle-1) and reached_end_of_episode),
                # Only add if experience count handled, 
                # no longer cares about crossing the episode barrier as the loss handles it,
                # unless self.sequence_replay_store_on_terminal is true
            )

        # Make sure the sequence buffer do not cross the episode barrier:
        # UPDATE: no longer care about this since the loss takes care of the episode barriers...
        # unless self.sequence_replay_store_on_terminal is true
        if (self.sequence_replay_store_on_terminal and reached_end_of_episode):
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
        torch.set_grad_enabled(False)

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

