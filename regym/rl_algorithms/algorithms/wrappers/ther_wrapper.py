import torch
import torch.optim as optim 
import torch.nn as nn 

import numpy as np
from ..algorithm import Algorithm 
from ...networks import random_sample

from .algorithm_wrapper import AlgorithmWrapper

from .her_wrapper import state_eq_goal_reward_fn

from ...replay_buffers import PrioritizedReplayStorage, ReplayStorage


def predictor_based_goal_predicated_reward_fn(predictor, achieved_exp, desired_exp, epsilon=1e0):
    state = achieved_exp['succ_s']
    with torch.no_grad():
        achieved_goal = predictor(state).cpu()
    goal = desired_exp['goals']['desired_goals']['s']
    abs_fn = torch.abs
    dist = abs_fn(achieved_goal-goal).float().mean()
    if dist < epsilon:
        return torch.zeros(1), achieved_goal, dist
    else:
        return -torch.ones(1), achieved_goal, dist


class THERAlgorithmWrapper(AlgorithmWrapper):
    def __init__(self, algorithm, predictor, predictor_loss_fn, strategy="future-4", goal_predicated_reward_fn=None, rewards={'failure':0, 'success':1}):
        super(THERAlgorithmWrapper, self).__init__(algorithm=algorithm)
        self.rewards = rewards 
        self.all_relabeling_rewarded = True 
        self.relabling_when_successfull = True 

        self.predictor = predictor 
        if self.kwargs['use_cuda']:
            self.predictor = self.predictor.cuda()

        self.predictor_loss_fn = predictor_loss_fn
        print(f"WARNING: THER loss_fn is {self.predictor_loss_fn}")
        
        # Tuning learning rate with respect to the number of actors:
        # Following: https://arxiv.org/abs/1705.04862
        lr = self.kwargs['THER_predictor_learning_rate'] 
        if self.kwargs['lr_account_for_nbr_actor']:
            lr *= self.nbr_actor
        print(f"THER Predictor Learning rate: {lr}")
        self.predictor_optimizer = optim.Adam(self.predictor.parameters(), lr=lr, betas=(0.9,0.999), eps=self.kwargs['adam_eps'])
        
        self.predictor_storages = None 
        self._reset_predictor_storages()

        self.episode_buffer = [[] for i in range(self.algorithm.get_nbr_actor())]
        self.strategy = strategy
        assert( ('future' in self.strategy or 'final' in self.strategy) and '-' in self.strategy)
        self.k = int(self.strategy.split('-')[-1])    
        
        if goal_predicated_reward_fn is None:   goal_predicated_reward_fn = state_eq_goal_reward_fn
        self.goal_predicated_reward_fn = goal_predicated_reward_fn
        
        self.episode_count = 0
        self.param_predictor_update_counter = 0

        self.nbr_handled_predictor_experience = 0
        self.batch_size = self.kwargs['THER_predictor_batch_size']

    def _reset_predictor_storages(self):
        if self.predictor_storages is not None:
            for storage in self.predictor_storages: storage.reset()
        else:
            self.predictor_storages = []
            keys = ['s', 'a', 'r', 'non_terminal', 'g']
            if self.recurrent:  keys += ['rnn_states', 'next_rnn_states']
            
            for i in range(self.nbr_actor):
                if self.kwargs['THER_use_PER']:
                    self.predictor_storages.append(PrioritizedReplayStorage(capacity=self.kwargs['THER_replay_capacity'],
                                                                    alpha=self.kwargs['THER_PER_alpha'],
                                                                    beta=self.kwargs['THER_PER_beta'],
                                                                    keys=keys,
                                                                    circular_offsets={'succ_s':1})
                    )
                else:
                    self.predictor_storages.append(ReplayStorage(capacity=self.kwargs['THER_replay_capacity'],
                                                       keys=keys,
                                                       circular_offsets={'succ_s':1})
                    )

    def store(self, exp_dict, actor_index=0):
        self.episode_buffer[actor_index].append(exp_dict)
        
        if not(exp_dict['non_terminal']):
            episode_length = len(self.episode_buffer[actor_index])
            relabelling = True 
            if self.relabling_when_successfull:
                relabelling = all(self.episode_buffer[actor_index][-1]['r']>0)
            
            episode_rewards = []
            for idx in range(episode_length):
                s = self.episode_buffer[actor_index][idx]['s']
                a = self.episode_buffer[actor_index][idx]['a']
                
                #r = self.episode_buffer[actor_index][idx]['r']
                r = self.rewards['success']*torch.ones(1) if all(self.episode_buffer[actor_index][idx]['r']>0) else self.rewards['failure']*torch.ones(1)
                
                succ_s = self.episode_buffer[actor_index][idx]['succ_s']
                non_terminal = self.episode_buffer[actor_index][idx]['non_terminal']

                desired_goal = self.episode_buffer[actor_index][idx]['goals']['desired_goals']['s']
                
                rnn_states = None
                if self.recurrent:
                    rnn_states = self.episode_buffer[actor_index][idx]['rnn_states']

                episode_rewards.append(r)

                d2store = {'s':s, 
                           'a':a, 
                           'r':r, 
                           'succ_s':succ_s, 
                           'non_terminal':non_terminal, 
                           'g':desired_goal,
                           'rnn_states':rnn_states}
                self.algorithm.store(d2store, actor_index=actor_index)
                
                # Store data in predictor storages:
                if self.kwargs['THER_use_THER'] and all(r>self.rewards['failure']):
                    self.predictor_store(d2store, actor_index=actor_index)
                    self.algorithm.summary_writer.add_scalar('Training/THER_Predictor/DatasetSize', self.nbr_handled_predictor_experience, self.param_predictor_update_counter)
                    
                if self.algorithm.summary_writer is not None and all(non_terminal<=0.5):
                    self.episode_count += 1
                    self.algorithm.summary_writer.add_scalar('PerEpisode/Success', (self.rewards['success']==r).float().mean().item(), self.episode_count)
                    self.algorithm.summary_writer.add_histogram('PerEpisode/Rewards', episode_rewards, self.episode_count)    
                    
                if not(self.kwargs['THER_use_THER']) or not(relabelling):
                    continue 

                for k in range(self.k):
                    if 'final' in self.strategy:
                        #achieved_goal = self.episode_buffer[actor_index][-1]['goals']['achieved_goals']['s']
                        
                        achieved_exp = self.episode_buffer[actor_index][idx]
                        desired_exp = self.episode_buffer[actor_index][-1]
                        
                        new_r, achieved_goal, dist = self.goal_predicated_reward_fn(achieved_exp=achieved_exp, desired_exp=desired_exp)
                        new_r = self.rewards['success']*torch.ones(1) if self.all_relabeling_rewarded or all(new_r>-0.5) else self.rewards['failure']*torch.ones(1)
                        
                        new_non_terminal = torch.zeros(1) if all(new_r>self.rewards['failure']) else torch.ones(1)
                        
                        d2store_her = {'s':s, 
                                       'a':a, 
                                       'r':new_r, 
                                       'succ_s':succ_s, 
                                       'non_terminal':new_non_terminal, 
                                       'g':achieved_goal,
                                       'rnn_states':rnn_states}
                        
                        if self.algorithm.summary_writer is not None:
                            self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_final', new_r.mean().item(), self.algorithm.get_update_count())
                            self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_dist', dist.mean().item(), self.algorithm.get_update_count())
                    '''
                    if 'future' in self.strategy:
                        future_idx = np.random.randint(idx, episode_length)
                        #achieved_goal = self.episode_buffer[actor_index][future_idx]['goals']['achieved_goals']['s']
                        
                        achieved_exp = self.episode_buffer[actor_index][idx]
                        desired_exp = self.episode_buffer[actor_index][future_idx]
                        
                        new_r, achieved_goal = self.goal_predicated_reward_fn(achieved_exp=achieved_exp, desired_exp=desired_exp)
                        new_r = self.rewards['success']*torch.ones(1) if all(new_r>-0.5) else self.rewards['failure']*torch.ones(1)
                        
                        new_non_terminal = torch.zeros(1) if all(new_r>self.rewards['failure']) else torch.ones(1) 
                        d2store_her = {'s':s, 
                                       'a':a, 
                                       'r':new_r, 
                                       'succ_s':succ_s, 
                                       'non_terminal':new_non_terminal, 
                                       'g':achieved_goal,
                                       'rnn_states':rnn_states}
                        
                        if self.algorithm.summary_writer is not None:
                            self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_future', new_r.mean().item(), self.algorithm.get_update_count())
                    '''

                    self.algorithm.store(d2store_her, actor_index=actor_index)
            
            # Reset episode buffer:
            self.episode_buffer[actor_index] = []

            # Train predictor:
            if self.nbr_handled_predictor_experience >= self.kwargs['THER_min_capacity']:
                self.train_predictor()

    def predictor_store(self, exp_dict, actor_index=0):
        self.nbr_handled_predictor_experience += 1

        if self.kwargs['THER_use_PER']:
            init_sampling_priority = None 
            self.predictor_storages[actor_index].add(exp_dict, priority=init_sampling_priority)
        else:
            self.predictor_storages[actor_index].add(exp_dict)

    def train_predictor(self, minibatch_size=None):
        if minibatch_size is None:  minibatch_size = self.batch_size

        samples = self.retrieve_values_from_predictor_storages(minibatch_size=minibatch_size)
        
        self.optimize_predictor(minibatch_size, samples)
        
    def retrieve_values_from_predictor_storages(self, minibatch_size):
        keys=['s', 'a', 'succ_s', 'r', 'non_terminal', 'g']

        fulls = {}
        
        if self.kwargs['THER_use_PER']:
            fulls['importanceSamplingWeights'] = []

        if self.recurrent:
            keys += ['rnn_states']
        
        for key in keys:    fulls[key] = []

        for storage in self.predictor_storages:
            # Check that there is something in the storage 
            if len(storage) <= 1: continue
            if self.kwargs['THER_use_PER']:
                sample, importanceSamplingWeights = storage.sample(batch_size=minibatch_size, keys=keys)
                importanceSamplingWeights = torch.from_numpy(importanceSamplingWeights)
                fulls['importanceSamplingWeights'].append(importanceSamplingWeights)
            else:
                sample = storage.sample(batch_size=minibatch_size, keys=keys)
            
            values = {}
            for key, value in zip(keys, sample):
                value = value.tolist()
                if isinstance(value[0], dict):   
                    value = Algorithm._concatenate_hdict(value.pop(0), value, map_keys=['hidden', 'cell'])
                else:
                    value = torch.cat(value, dim=0)
                values[key] = value 

            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    fulls[key].append(value)
                else:
                    fulls[key] = value

        for key, value in fulls.items():
            if isinstance(value, dict):
                fulls[key] = value
            else:
                fulls[key] = torch.cat(value, dim=0)

        return fulls

    def optimize_predictor(self, minibatch_size, samples):
        beta = self.storages[0].beta if self.kwargs['THER_use_PER'] else 1.0
        
        states = samples['s']
        actions = samples['a']
        next_states = samples['succ_s']
        rewards = samples['r']
        non_terminals = samples['non_terminal']
        goals = samples['g']

        rnn_states = samples['rnn_states'] if 'rnn_states' in samples else None
        
        importanceSamplingWeights = samples['importanceSamplingWeights'] if 'importanceSamplingWeights' in samples else None

        # For each actor, there is one mini_batch update:
        sampler = random_sample(np.arange(states.size(0)), minibatch_size)
        list_batch_indices = [storage_idx*minibatch_size+np.arange(minibatch_size) \
                                for storage_idx, storage in enumerate(self.predictor_storages)]
        array_batch_indices = np.concatenate(list_batch_indices, axis=0)
        sampled_batch_indices = []
        sampled_losses_per_item = []

        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()
            sampled_batch_indices.append(batch_indices)

            sampled_rnn_states = None
            if self.recurrent:
                sampled_rnn_states = Algorithm._extract_rnn_states_from_batch_indices(rnn_states, batch_indices, use_cuda=self.kwargs['use_cuda'])

            sampled_importanceSamplingWeights = None
            if self.kwargs['THER_use_PER']:
                sampled_importanceSamplingWeights = importanceSamplingWeights[batch_indices].cuda() if self.kwargs['use_cuda'] else importanceSamplingWeights[batch_indices]
            
            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_next_states = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]
            sampled_rewards = rewards[batch_indices].cuda() if self.kwargs['use_cuda'] else rewards[batch_indices]
            sampled_non_terminals = non_terminals[batch_indices].cuda() if self.kwargs['use_cuda'] else non_terminals[batch_indices]
            sampled_goals = goals[batch_indices].cuda() if self.kwargs['use_cuda'] else goals[batch_indices]

            self.predictor_optimizer.zero_grad()
            
            loss, loss_per_item = self.predictor_loss_fn(sampled_states, 
                                          sampled_actions, 
                                          sampled_next_states,
                                          sampled_rewards,
                                          sampled_non_terminals,
                                          goals=sampled_goals,
                                          rnn_states=sampled_rnn_states,
                                          predictor=self.predictor,
                                          weights_decay_lambda=self.kwargs['THER_weights_decay_lambda'],
                                          use_PER=self.kwargs['THER_use_PER'],
                                          PER_beta=beta,
                                          importanceSamplingWeights=sampled_importanceSamplingWeights,
                                          iteration_count=self.param_predictor_update_counter,
                                          summary_writer=self.algorithm.summary_writer)
            
            loss.backward(retain_graph=False)
            if self.kwargs['gradient_clip'] > 1e-3:
                nn.utils.clip_grad_norm_(self.predictor.parameters(), self.kwargs['gradient_clip'])
            self.predictor_optimizer.step()

            if self.kwargs['THER_use_PER']:
                sampled_losses_per_item.append(loss_per_item)

            self.param_predictor_update_counter += 1 

        if self.kwargs['THER_use_PER']:
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
                el_idx_in_storage = self.predictor_storages[storage_idx].tree_indices[el_idx_in_batch]
                new_priority = self.predictor_storages[storage_idx].priority(sloss)
                self.predictor_storages[storage_idx].update(idx=el_idx_in_storage, priority=new_priority)


    def clone(self):
        return THERAlgorithmWrapper(algorithm=self.algorithm.clone(),
                                   predictor=predictor, 
                                   strategy=self.strategy, 
                                   goal_predicated_reward_fn=self.goal_predicated_reward_fn)