from typing import Dict, Optional, List 

import time
from functools import partial
import copy

import torch
import torch.optim as optim 
import torch.nn as nn 

import numpy as np
from regym.rl_algorithms.algorithms.algorithm import Algorithm 
from regym.rl_algorithms.networks import random_sample

from regym.rl_algorithms.algorithms.wrappers.algorithm_wrapper import AlgorithmWrapper

from regym.rl_algorithms.replay_buffers import PrioritizedReplayStorage, SplitReplayStorage, SplitPrioritizedReplayStorage
from regym.rl_algorithms.utils import archi_concat_fn, _extract_rnn_states_from_batch_indices, _concatenate_hdict, _concatenate_list_hdict, copy_hdict

import wandb 
import pandas as pd

from regym.rl_algorithms.algorithms.wrappers.reward_prediction_loss import compute_loss as RP_compute_loss_fn

class RewardPredictionAlgorithmWrapper(AlgorithmWrapper):
    def __init__(
        self, 
        algorithm, 
        predictor, 
        predictor_loss_fn=RP_compute_loss_fn, 
        ):
        """
        """
        
        super(RewardPredictionAlgorithmWrapper, self).__init__(algorithm=algorithm)
        self.hook_fns = []
        self.nbr_episode_success_range = 256

        self.predictor = predictor 
        if self.kwargs['use_cuda']:
            self.predictor = self.predictor.cuda()
        self.best_predictor = self.predictor.clone()

        self.predictor_loss_fn = predictor_loss_fn
        # Tuning learning rate with respect to the number of actors:
        # Following: https://arxiv.org/abs/1705.04862
        lr = self.kwargs['RP_predictor_learning_rate'] 
        if isinstance(lr, str): lr = float(lr)
        if self.kwargs['lr_account_for_nbr_actor']:
            lr *= self.nbr_actor
        
        self.predictor_optimizer = optim.Adam(
            self.predictor.parameters(), 
            lr=lr, betas=(0.9,0.999), 
            eps=float(self.kwargs.get('RP_adam_eps', 1.0e-5)),
            weight_decay=float(self.kwargs.get("RP_adam_weight_decay", 0.0)),
        )
        self.best_predictor_optimizer_sd = self.predictor_optimizer.state_dict()

        self.predictor_storages = None 
        self._reset_predictor_storages()

        self.episode_buffer = [[] for i in range(self.algorithm.get_nbr_actor())]
        self.episode_count = 0
        self.param_predictor_update_counter = 0

        self.nbr_buffered_predictor_experience = 0
        self.nbr_handled_predictor_experience = 0
        self.batch_size = self.kwargs['RP_predictor_batch_size']
        self.nbr_minibatches = self.kwargs['RP_predictor_nbr_minibatches']
        
    def _reset_predictor_storages(self):
        '''
        Creates 3 storages. One for negative rewards, one for null rewards,
        and one for positive rewards.
        '''

        if self.predictor_storages is not None:
            for storage in self.predictor_storages: storage.reset()
       
        nbr_storages = 3  

        self.predictor_storages = []
        keys = ['s', 'a', 'r', 'non_terminal']
        if self.recurrent:  keys += ['rnn_states', 'next_rnn_states']
        
        circular_keys= {} #{'succ_s':'s'}
        circular_offsets= {} #{'succ_s':1}
        keys.append('succ_s')
        
        beta_increase_interval = None
        if 'PER_beta_increase_interval' in self.kwargs and self.kwargs['PER_beta_increase_interval']!='None':
            beta_increase_interval = float(self.kwargs['PER_beta_increase_interval'])  

        for i in range(nbr_storages):
            if self.kwargs['RP_use_PER']:
                self.predictor_storages.append(
                    SplitPrioritizedReplayStorage(
                        capacity=int(self.kwargs['RP_replay_capacity']),
                        alpha=self.kwargs['RP_PER_alpha'],
                        beta=self.kwargs['RP_PER_beta'],
                        keys=keys,
                        circular_keys=circular_keys,
                        circular_offsets=circular_offsets,
                        test_train_split_interval=self.kwargs['RP_predictor_test_train_split_interval'],
                        test_capacity=int(self.kwargs['RP_test_replay_capacity']),
                        lock_test_storage=self.kwargs['RP_lock_test_storage'],
                    )
                )
            else:
                self.predictor_storages.append(
                    SplitReplayStorage(
                        capacity=int(self.kwargs['RP_replay_capacity']),
                        keys=keys,
                        circular_keys=circular_keys,
                        circular_offsets=circular_offsets,
                        test_train_split_interval=self.kwargs['RP_predictor_test_train_split_interval'],
                        test_capacity=int(self.kwargs['RP_test_replay_capacity']),
                        lock_test_storage=self.kwargs['RP_lock_test_storage'],
                    )
                )

    def store(self, exp_dict, actor_index=0):
        self.algorithm.store(exp_dict, actor_index=actor_index)
        
        self.episode_buffer[actor_index].append(exp_dict)
        self.nbr_buffered_predictor_experience += 1

        if not(exp_dict['non_terminal']):
            self.episode_count += 1
            episode_length = len(self.episode_buffer[actor_index])

            # Assumes non-successful rewards are non-positive:
            successful_traj = all(self.episode_buffer[actor_index][-1]['r']>0.5)

            episode_rewards = []
            per_episode_d2store = {}
            previous_d2stores = [] 

            for idx in range(episode_length):
                s = self.episode_buffer[actor_index][idx]['s']
                a = self.episode_buffer[actor_index][idx]['a']
                r = self.episode_buffer[actor_index][idx]['r']
                succ_s = self.episode_buffer[actor_index][idx]['succ_s']
                non_terminal = self.episode_buffer[actor_index][idx]['non_terminal']

                info = self.episode_buffer[actor_index][idx]['info']
                succ_info = self.episode_buffer[actor_index][idx]['succ_info']
                rnn_states = self.episode_buffer[actor_index][idx]['rnn_states']
                next_rnn_states = self.episode_buffer[actor_index][idx]['next_rnn_states']
                
                episode_rewards.append(r)

                d2store = {
                    's':s, 
                    'a':a, 
                    'r':r, 
                    'succ_s':succ_s, 
                    'non_terminal':non_terminal, 
                    'rnn_states':copy_hdict(rnn_states),
                    'next_rnn_states':copy_hdict(next_rnn_states),
                    'info': info,
                    'succ_info': succ_info,
                }

                if -1 not in per_episode_d2store: per_episode_d2store[-1] = []
                per_episode_d2store[-1].append(d2store)
                
                for hook_fn in self.hook_fns:
                    hook_fn(
                        exp_dict=d2store,
                        actor_index=actor_index,
                        negative=False,
                        self=self,
                    )

                self.predictor_store(
                    d2store, 
                    actor_index=actor_index, 
                    negative=False,
                )
                   
                wandb.log({'Training/RP_Predictor/DatasetSize': self.nbr_handled_predictor_experience}, commit=False) # self.param_predictor_update_counter)
                if self.algorithm.unwrapped.summary_writer is not None:
                    self.algorithm.unwrapped.summary_writer.add_scalar('Training/RP_Predictor/DatasetSize', self.nbr_handled_predictor_experience, self.param_predictor_update_counter)
                    
                if idx==(episode_length-1):
                    wandb.log({'PerEpisode/EpisodeLength': episode_length}, commit=False)
                    wandb.log({
                        'PerEpisode/HER_Success': float(r.item()>0.5), #1+her_r.mean().item(),
                    }, commit=False) 
                    wandb.log({'PerEpisode/OriginalFinalReward': r.mean().item()}, commit=False)
                    wandb.log({'PerEpisode/OriginalReturn': sum(episode_rewards)}, commit=False)
                    wandb.log({'PerEpisode/OriginalNormalizedReturn': sum(episode_rewards)/episode_length}, commit=False) # self.episode_count)
                    if not hasattr(self, "nbr_success"):  self.nbr_success = 0
                    if successful_traj: self.nbr_success += 1
                    if self.episode_count % self.nbr_episode_success_range == 0:
                        wandb.log({
                            'PerEpisode/SuccessRatio': float(self.nbr_success)/self.nbr_episode_success_range,
                            'PerEpisode/SuccessRatioIndex': int(self.episode_count//self.nbr_episode_success_range),
                            },
                            commit=False,
                        ) # self.episode_count)
                        self.nbr_success = 0

                    if self.algorithm.unwrapped.summary_writer is not None:
                        self.algorithm.unwrapped.summary_writer.add_histogram('PerEpisode/Rewards', episode_rewards, self.episode_count)

                
            # Now that we have all the different trajectories,
            # we can send them to the main algorithm as complete
            # whole trajectories, one experience at a time.
            for key in per_episode_d2store:
                wandb.log({f'PerEpisode/RP_traj_length/{key}': len(per_episode_d2store[key])}, commit=False)
            # Reset the relevant episode buffer:
            self.episode_buffer[actor_index] = []

        self.update_predictor()
	   
    def predictor_store(self, exp_dict, actor_index=0, negative=False):
        if exp_dict['r'].mean().item() == 0.0:
            actor_index = 1
        elif exp_dict['r'].mean().item() > 0:
            actor_index = 2
        else:
            actor_index = 0

        self.nbr_handled_predictor_experience += 1
        test_set = None
        if negative:    test_set = False
        if self.kwargs['RP_use_PER']:
            init_sampling_priority = None 
            self.predictor_storages[actor_index].add(exp_dict, priority=init_sampling_priority, test_set=test_set)
        else:
            self.predictor_storages[actor_index].add(exp_dict, test_set=test_set)

    def update_predictor(self):
        period_check = self.kwargs['RP_replay_period']
        period_count_check = self.nbr_buffered_predictor_experience
        
        # Update predictor:
        if not(self.nbr_handled_predictor_experience >= self.kwargs['RP_min_capacity']):
            return
        
        if not(period_count_check % period_check == 0):
            return 
        
        full_update = True
        for it in range(self.kwargs['RP_nbr_training_iteration_per_update']):
            self.test_acc = self.train_predictor()
            if self.test_acc >= self.kwargs['RP_predictor_accuracy_threshold']:
                full_update = False
                break
        wandb.log({f"Training/RP_Predictor/FullUpdate":int(full_update)}, commit=False)
         
    def train_predictor(self, minibatch_size=None):
        if minibatch_size is None:  minibatch_size = self.batch_size

        start = time.time()
        samples = self.retrieve_values_from_predictor_storages(minibatch_size=self.nbr_minibatches*minibatch_size)
        end = time.time()
        
        wandb.log({'PerRPPredictorUpdate/TimeComplexity/RetrieveValuesFn':  end-start}, commit=False) # self.param_update_counter)
        
        start = time.time()
        self.optimize_predictor(minibatch_size, samples)
        end = time.time()
        
        wandb.log({'PerRPPredictorUpdate/TimeComplexity/OptimizeModelFn':  end-start}, commit=False) # self.param_update_counter)
        
        full_train size = 0
        full_test_size = 0
        test_storage_size = self.predictor_storages[0].get_size(test=True) #.test_storage.current_size['s']  
        train_storage_size = self.predictor_storages[0].get_size(test=False) #test_storage.current_size['s']  
        full_train_size += train_storage_size
        full_test_size += test_storage_size
        wandb.log({'PerRPPredictorUpdate/TestStorageSize/Negative':  test_storage_size}, commit=False)
        wandb.log({'PerRPPredictorUpdate/TrainStorageSize/Negative':  train_storage_size}, commit=False)
        test_storage_size = self.predictor_storages[1].get_size(test=True) #.test_storage.current_size['s']  
        train_storage_size = self.predictor_storages[1].get_size(test=False) #test_storage.current_size['s']  
        full_train_size += train_storage_size
        full_test_size += test_storage_size
        wandb.log({'PerRPPredictorUpdate/TestStorageSize/Null':  test_storage_size}, commit=False)
        wandb.log({'PerRPPredictorUpdate/TrainStorageSize/Null':  train_storage_size}, commit=False)
        test_storage_size = self.predictor_storages[2].get_size(test=True) #.test_storage.current_size['s']  
        train_storage_size = self.predictor_storages[2].get_size(test=False) #test_storage.current_size['s']  
        full_train_size += train_storage_size
        full_test_size += test_storage_size
        wandb.log({'PerRPPredictorUpdate/TestStorageSize/Positive':  test_storage_size}, commit=False)
        wandb.log({'PerRPPredictorUpdate/TrainStorageSize/Positive':  train_storage_size}, commit=False)
        wandb.log({'PerRPPredictorUpdate/TestStorageSize/Whole':  full_test_size}, commit=False)
        wandb.log({'PerRPPredictorUpdate/TrainStorageSize/Whole':  full_train_size}, commit=False)
        if test_storage_size > self.kwargs['RP_test_min_capacity']:
            #test_samples = self.retrieve_values_from_predictor_storages(minibatch_size=minibatch_size, test=True)
            test_samples = self.retrieve_values_from_predictor_storages(minibatch_size=self.nbr_minibatches*minibatch_size, test=True)
            with torch.no_grad():
                updated_acc = self.test_predictor( self.predictor, minibatch_size, test_samples)
                best_acc = self.test_predictor( self.best_predictor, minibatch_size, test_samples)
        else:
            updated_acc = 0.0
            best_acc = 0.0
        
        successful_update = int(updated_acc >= best_acc)
        wandb.log({f"Training/RP_Predictor/SuccessfulUpdate":successful_update}, commit=False)
        if not successful_update:
            self.predictor.load_state_dict(self.best_predictor.state_dict(), strict=False)
            self.predictor_optimizer.load_state_dict(self.best_predictor_optimizer_sd)
            acc = best_acc
        else:
            self.best_predictor.load_state_dict(self.predictor.state_dict(), strict=False)
            self.best_predictor_optimizer_sd = self.predictor_optimizer.state_dict()
            acc = updated_acc 

        wandb.log({'PerRPPredictorUpdate/TestSentenceAccuracy': acc, "RP_predictor_update_count":self.param_predictor_update_counter}, commit=True)
        
        return acc 

    def retrieve_values_from_predictor_storages(self, minibatch_size, test=False):
        torch.set_grad_enabled(False)
        keys=['s', 'a', 'succ_s', 'r', 'non_terminal']

        fulls = {}
        
        if self.kwargs['RP_use_PER'] and not test:
            fulls['importanceSamplingWeights'] = []

        if self.recurrent:
            keys += ['rnn_states', 'next_rnn_states']
        
        for key in keys:    fulls[key] = []

        for storage in self.predictor_storages:
            # Check that there is something in the storage 
            if len(storage) <= 1: continue
            batch_size = minibatch_size
            if batch_size is None:
                batch_size = storage.get_size(test=test)

            if self.kwargs['RP_use_PER'] and not test:
                sample, importanceSamplingWeights = storage.sample(
                    batch_size=batch_size, 
                    keys=keys, 
                    test=test,
                    #replace=test,
                )
                importanceSamplingWeights = torch.from_numpy(importanceSamplingWeights)
                fulls['importanceSamplingWeights'].append(importanceSamplingWeights)
            else:
                sample = storage.sample(
                    batch_size=batch_size, 
                    keys=keys, 
                    test=test,
                    replace=test,
                )
            
            values = {}
            for key, value in zip(keys, sample):
                value = value.tolist()
                if isinstance(value[0], dict):   
                    value = _concatenate_list_hdict(
                        lhds=value, 
                        concat_fn=archi_concat_fn,
                        preprocess_fn=(lambda x:x),
                        #map_keys=['hidden', 'cell']
                    )
                else:
                    value = torch.cat(value, dim=0)
                values[key] = value 

            for key, value in values.items():
                fulls[key].append(value)
        
        for key, value in fulls.items():
            if len(value) > 1:
                if isinstance(value[0], dict):
                    value = _concatenate_list_hdict(
                        lhds=value,
                        concat_fn=partial(torch.cat, dim=0),
                        preprocess_fn=(lambda x:x),
                    )
                else:
                    value = torch.cat(value, dim=0)
            else:
                value = value[0]
            fulls[key] = value

        return fulls

    def optimize_predictor(self, minibatch_size, samples):
        start = time.time()
        torch.set_grad_enabled(True)
        self.predictor.train(True)

        beta = self.predictor_storages[0].beta if self.kwargs['RP_use_PER'] else 1.0
        
        states = samples['s']
        actions = samples['a']
        next_states = samples['succ_s']
        rewards = samples['r']
        non_terminals = samples['non_terminal']
        goals = samples['g'] if 'g' in samples else None

        rnn_states = samples['rnn_states'] if 'rnn_states' in samples else None
        next_rnn_states = samples['next_rnn_states'] if 'next_rnn_states' in samples else None
        
        importanceSamplingWeights = samples['importanceSamplingWeights'] if 'importanceSamplingWeights' in samples else None

        # For each actor, there is one mini_batch update:
        sampler = list(random_sample(np.arange(states.size(0)), minibatch_size))
        nbr_minibatches = len(sampler)
        nbr_sampled_element_per_storage = self.nbr_minibatches*minibatch_size
        '''
        list_batch_indices = [storage_idx*nbr_sampled_element_per_storage+np.arange(nbr_sampled_element_per_storage) \
                                for storage_idx, storage in enumerate(self.predictor_storages)]
        array_batch_indices = np.concatenate(list_batch_indices, axis=0)
        '''
        batch_idx2storage_idx = {}
        batch_idx2el_in_batch_idx = {}
        offset = 0
        for storage_idx, storage in enumerate(self.predictor_storages):
            if len(storage)==0:
                continue
            nbr_sampled_element = len(storage.tree_indices)
            for el_in_batch_idx in range(nbr_sampled_element):
                bidx = offset+el_in_batch_idx
                batch_idx2storage_idx[bidx] = storage_idx
                batch_idx2el_in_batch_idx[bidx] = el_in_batch_idx
            offset += nbr_sampled_element

        sampled_batch_indices = []
        sampled_losses_per_item = []
        
        self.predictor_optimizer.zero_grad()
        
        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()
            sampled_batch_indices.append(batch_indices)

            sampled_rnn_states = None
            sampled_next_rnn_states = None
            if self.recurrent:
                sampled_rnn_states = _extract_rnn_states_from_batch_indices(
                    rnn_states, 
                    batch_indices, 
                    use_cuda=self.kwargs['use_cuda'],
                )
                sampled_next_rnn_states = _extract_rnn_states_from_batch_indices(
                    next_rnn_states, 
                    batch_indices, 
                    use_cuda=self.kwargs['use_cuda'],
                )

            sampled_importanceSamplingWeights = None
            if importanceSamplingWeights is not None:
                sampled_importanceSamplingWeights = importanceSamplingWeights[batch_indices].cuda() if self.kwargs['use_cuda'] else importanceSamplingWeights[batch_indices]
            
            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_next_states = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]
            sampled_rewards = rewards[batch_indices].cuda() if self.kwargs['use_cuda'] else rewards[batch_indices]
            sampled_non_terminals = non_terminals[batch_indices].cuda() if self.kwargs['use_cuda'] else non_terminals[batch_indices]
            sampled_goals = None #DEPRECATED goals[batch_indices].cuda() if self.kwargs['use_cuda'] else goals[batch_indices]

            #self.predictor_optimizer.zero_grad()
            
            output_dict = self.predictor_loss_fn(
                sampled_states, 
                sampled_actions, 
                sampled_next_states,
                sampled_rewards,
                sampled_non_terminals,
                goals=sampled_goals,
                rnn_states=sampled_rnn_states,
                next_rnn_states=sampled_next_rnn_states,
                predictor=self.predictor,
                weights_decay_lambda=self.kwargs['RP_weights_decay_lambda'],
                use_PER=self.kwargs['RP_use_PER'],
                PER_beta=beta,
                importanceSamplingWeights=sampled_importanceSamplingWeights,
                iteration_count=self.param_predictor_update_counter,
                summary_writer=self.algorithm.unwrapped.summary_writer,
                phase="Training",
            )
            
            loss = output_dict['loss']
            #loss_per_item = output_dict['loss_per_item']
            loss_per_item = output_dict['loss_per_item'].detach()
            
            (loss/nbr_minibatches).backward(retain_graph=False)
            '''
            loss.backward(retain_graph=False)
            if self.kwargs['THER_gradient_clip'] > 1e-3:
                nn.utils.clip_grad_norm_(self.predictor.parameters(), self.kwargs['THER_gradient_clip'])
            self.predictor_optimizer.step()
            '''

            if importanceSamplingWeights is not None:
                sampled_losses_per_item.append(loss_per_item)
                #wandb_data = copy.deepcopy(wandb.run.history._data)
                #wandb.run.history._data = {}
                wandb.log({
                    'PerRPPredictorUpdate/ImportanceSamplingMean':  sampled_importanceSamplingWeights.cpu().mean().item(),
                    'PerRPPredictorUpdate/ImportanceSamplingStd':  sampled_importanceSamplingWeights.cpu().std().item(),
                    'PerRPPredictorUpdate/PER_Beta':  beta
                }) # self.param_update_counter)
                #wandb.run.history._data = wandb_data

            self.param_predictor_update_counter += 1 

        if self.kwargs['RP_gradient_clip'] > 1e-3:
            nn.utils.clip_grad_norm_(self.predictor.parameters(), self.kwargs['RP_gradient_clip'])
        self.predictor_optimizer.step()
        
        torch.set_grad_enabled(False)
        self.predictor.train(False)

        if importanceSamplingWeights is not None:
            # losses corresponding to sampled batch indices: 
            sampled_losses_per_item = torch.cat(sampled_losses_per_item, dim=0).cpu().detach().numpy()
            sampled_batch_indices = np.concatenate(sampled_batch_indices, axis=0)
            # let us align the batch indices with the losses:
            '''
            array_batch_indices = array_batch_indices[sampled_batch_indices]
            '''
            # Now we can iterate through the losses and retrieve what 
            # storage and what batch index they were associated with:
            '''
            for sloss, arr_bidx in zip(sampled_losses_per_item, array_batch_indices):
            '''
            for sloss, bidx in zip(sampled_losses_per_item, sampled_batch_indices):
                '''
                storage_idx = arr_bidx//nbr_sampled_element_per_storage
                el_idx_in_batch = arr_bidx%nbr_sampled_element_per_storage
                '''
                storage_idx = batch_idx2storage_idx[bidx]
                el_idx_in_batch = batch_idx2el_in_batch_idx[bidx]
                #storage_idx = arr_bidx//minibatch_size
                #el_idx_in_batch = arr_bidx%minibatch_size
                
                el_idx_in_storage = self.predictor_storages[storage_idx].tree_indices[el_idx_in_batch]
                new_priority = self.predictor_storages[storage_idx].priority(sloss)
                self.predictor_storages[storage_idx].update(idx=el_idx_in_storage, priority=new_priority)

        end = time.time()
        wandb.log({'PerRPPredictorUpdate/TimeComplexity/OptimizationLoss':  end-start}, commit=False) # self.param_update_counter)

    def test_predictor(self, predictor, minibatch_size, samples):
        training = predictor.training
        predictor.train(False)

        torch.set_grad_enabled(False)
        
        beta = self.predictor_storages[0].beta if self.kwargs['RP_use_PER'] else 1.0
        
        states = samples['s']
        actions = samples['a']
        next_states = samples['succ_s']
        rewards = samples['r']
        non_terminals = samples['non_terminal']
        goals = samples['g'] if 'g' in samples else None

        rnn_states = samples['rnn_states'] if 'rnn_states' in samples else None
        next_rnn_states = samples['next_rnn_states'] if 'next_rnn_states' in samples else None
        
        importanceSamplingWeights = samples['importanceSamplingWeights'] if 'importanceSamplingWeights' in samples else None

        # For each actor, there is one mini_batch update:
        sampler = random_sample(np.arange(states.size(0)), minibatch_size)
        list_batch_indices = [storage_idx*minibatch_size+np.arange(minibatch_size) \
                                for storage_idx, storage in enumerate(self.predictor_storages)]
        array_batch_indices = np.concatenate(list_batch_indices, axis=0)
        sampled_batch_indices = []
        sampled_losses_per_item = []

        running_acc = 0
        nbr_batches = 0
        for batch_indices in sampler:
            nbr_batches += 1
            batch_indices = torch.from_numpy(batch_indices).long()
            sampled_batch_indices.append(batch_indices)

            sampled_rnn_states = None
            sampled_next_rnn_states = None
            if self.recurrent:
                sampled_rnn_states = _extract_rnn_states_from_batch_indices(rnn_states, batch_indices, use_cuda=self.kwargs['use_cuda'])
                sampled_next_rnn_states = _extract_rnn_states_from_batch_indices(next_rnn_states, batch_indices, use_cuda=self.kwargs['use_cuda'])

            sampled_importanceSamplingWeights = None
            if importanceSamplingWeights is not None:
                sampled_importanceSamplingWeights = importanceSamplingWeights[batch_indices].cuda() if self.kwargs['use_cuda'] else importanceSamplingWeights[batch_indices]
            
            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_next_states = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]
            sampled_rewards = rewards[batch_indices].cuda() if self.kwargs['use_cuda'] else rewards[batch_indices]
            sampled_non_terminals = non_terminals[batch_indices].cuda() if self.kwargs['use_cuda'] else non_terminals[batch_indices]
            sampled_goals = None # DEPRECATED goals[batch_indices].cuda() if self.kwargs['use_cuda'] else goals[batch_indices]

            output_dict = self.predictor_loss_fn(
                sampled_states, 
                sampled_actions, 
                sampled_next_states,
                sampled_rewards,
                sampled_non_terminals,
                goals=sampled_goals,
                rnn_states=sampled_rnn_states,
                next_rnn_states=sampled_next_rnn_states,
                predictor=predictor,
                weights_decay_lambda=self.kwargs['RP_weights_decay_lambda'],
                use_PER=self.kwargs['RP_use_PER'],
                PER_beta=beta,
                importanceSamplingWeights=sampled_importanceSamplingWeights,
                iteration_count=self.param_predictor_update_counter,
                summary_writer=self.algorithm.unwrapped.summary_writer,
                phase="Testing",
            )
            
            loss = output_dict['loss']
            loss_per_item = output_dict['loss_per_item']
            
            accuracy = output_dict['accuracy']
            running_acc = running_acc + accuracy

            if self.kwargs['RP_use_PER']:
                sampled_losses_per_item.append(loss_per_item)

        '''
        if importanceSamplingWeights is not None:
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
                el_idx_in_storage = self.predictor_storages[storage_idx].get_test_storage().tree_indices[el_idx_in_batch]
                new_priority = self.predictor_storages[storage_idx].priority(sloss)
                self.predictor_storages[storage_idx].update(idx=el_idx_in_storage, priority=new_priority, test=True)
        '''

        predictor.train(training)

        running_acc = running_acc / nbr_batches
        return running_acc

    def clone(self, with_replay_buffer: bool=False, clone_proxies: bool=False, minimal: bool=False):        
        cloned_algo = RewardPredictionAlgorithmWrapper(
            algorithm=self.algorithm.clone(
                with_replay_buffer=with_replay_buffer,
                clone_proxies=clone_proxies,
                minimal=minimal
            ), 
            predictor=self.predictor, 
            predictor_loss_fn=self.predictor_loss_fn, 
        )
        return cloned_algo

