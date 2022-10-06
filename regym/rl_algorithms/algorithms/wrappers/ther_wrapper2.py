from typing import Dict, Optional, List 

import time
from functools import partial

import torch
import torch.optim as optim 
import torch.nn as nn 

import numpy as np
from regym.rl_algorithms.algorithms.algorithm import Algorithm 
from regym.rl_algorithms.networks import random_sample

from regym.rl_algorithms.algorithms.wrappers.algorithm_wrapper import AlgorithmWrapper

from regym.rl_algorithms.algorithms.wrappers.her_wrapper2 import state_eq_goal_reward_fn2

from regym.rl_algorithms.replay_buffers import PrioritizedReplayStorage, SplitReplayStorage, SplitPrioritizedReplayStorage
from regym.rl_algorithms.utils import archi_concat_fn, _extract_rnn_states_from_batch_indices, _concatenate_hdict, _concatenate_list_hdict, copy_hdict

import wandb 


def predictor_based_goal_predicated_reward_fn2(
    predictor, 
    achieved_exp, 
    target_exp, 
    _extract_goal_from_info_fn=None, 
    goal_key="achieved_goal",
    latent_goal_key=None,
    epsilon=1e0,
    feedbacks={"failure":-1, "success":0},
    reward_shape=[1,1],
    ):
    '''
    Relabelling an unsuccessful trajectory, so the desired_exp's goal is not interesting.
    We want to know the goal that is achieved on the desired_exp succ_s / desired_state.
    
    Comparison between the predicted goal of the achieved state and the desired state
    tells us whether the achieved state is achieving the relabelling goal.

    Returns -1 for failure and 0 for success
    '''
    target_latent_goal = None 

    state = achieved_exp['succ_s']
    target_state = target_exp['succ_s']
    with torch.no_grad():
        training = predictor.training
        predictor.train(False)
        achieved_pred_goal = predictor(state).cpu()
        target_pred_goal = predictor(target_state).cpu()
        predictor.train(training)
    abs_fn = torch.abs
    dist = abs_fn(achieved_pred_goal-target_pred_goal).float().mean()
    if dist < epsilon:
        return feedbacks["success"]*torch.ones(reward_shape), target_pred_goal, target_latent_goal
    else:
        return feedbacks["failure"]*torch.ones(reward_shape), target_pred_goal, target_latent_goal


class THERAlgorithmWrapper2(AlgorithmWrapper):
    def __init__(
        self, 
        algorithm, 
        extra_inputs_infos,
        predictor, 
        predictor_loss_fn, 
        strategy="future-4", 
        goal_predicated_reward_fn=None,
        _extract_goal_from_info_fn=None,
        achieved_goal_key_from_info="achieved_goal",
        target_goal_key_from_info="target_goal",
        achieved_latent_goal_key_from_info=None,
        target_latent_goal_key_from_info=None,
        filtering_fn="None",
        #rewards={'failure':-1, 'success':0}
        feedbacks={"failure":-1, "success":0},
        #rewards={'failure':0, 'success':1}
        ):
        """
        :param achieved_goal_key_from_info: Str of the key from the info dict
            used to retrieve the *achieved* goal from the *desired*/target
            experience's info dict.
        :param target_goal_key_from_info: Str of the key from the info dict
            used to replace the *target* goal into the HER-modified rnn/frame_states. 
        """
        
        super(THERAlgorithmWrapper2, self).__init__(algorithm=algorithm)
        
        if goal_predicated_reward_fn is None:   goal_predicated_reward_fn = state_eq_goal_reward_fn2
        if _extract_goal_from_info_fn is None:  _extract_goal_from_info_fn = self._extract_goal_from_info_default_fn

        self.extra_inputs_infos = extra_inputs_infos
        self.filtering_fn = filtering_fn 
 
        #self.rewards = rewards 
        self.feedbacks = feedbacks 
        self.test_acc = 0.0

        self.predictor = predictor 
        if self.kwargs['use_cuda']:
            self.predictor = self.predictor.cuda()

        self.predictor_loss_fn = predictor_loss_fn
        #print(f"WARNING: THER loss_fn is {self.predictor_loss_fn}")
        
        # Tuning learning rate with respect to the number of actors:
        # Following: https://arxiv.org/abs/1705.04862
        lr = self.kwargs['THER_predictor_learning_rate'] 
        if isinstance(lr, str): lr = float(lr)
        if self.kwargs['lr_account_for_nbr_actor']:
            lr *= self.nbr_actor
        #print(f"THER Predictor Learning rate: {lr}")
        
        self.predictor_optimizer = optim.Adam(
            self.predictor.parameters(), 
            lr=lr, betas=(0.9,0.999), 
            eps=self.kwargs['adam_eps']
        )
        
        self.predictor_storages = None 
        self._reset_predictor_storages()

        self.episode_buffer = [[] for i in range(self.algorithm.get_nbr_actor())]
        self.strategy = strategy
        assert( ('future' in self.strategy or 'final' in self.strategy) and '-' in self.strategy)
        self.k = int(self.strategy.split('-')[-1])    
        self.goal_predicated_reward_fn = goal_predicated_reward_fn
        self._extract_goal_from_info_fn = _extract_goal_from_info_fn
        self.achieved_goal_key_from_info = achieved_goal_key_from_info
        self.target_goal_key_from_info = target_goal_key_from_info
        self.achieved_latent_goal_key_from_info = achieved_latent_goal_key_from_info
        self.target_latent_goal_key_from_info = target_latent_goal_key_from_info

        self.episode_count = 0
        self.param_predictor_update_counter = 0

        self.nbr_buffered_predictor_experience = 0
        self.nbr_handled_predictor_experience = 0
        self.batch_size = self.kwargs['THER_predictor_batch_size']

    def _reset_predictor_storages(self):
        if self.predictor_storages is not None:
            for storage in self.predictor_storages: storage.reset()
       
        nbr_storages = 1  

        self.predictor_storages = []
        keys = ['s', 'a', 'r', 'non_terminal']
        if self.recurrent:  keys += ['rnn_states']

        circular_keys={'succ_s':'s'}
        circular_offsets={'succ_s':1}
        if self.recurrent:
            circular_keys.update({'next_rnn_states':'rnn_states'})
            circular_offsets.update({'next_rnn_states':1})

        beta_increase_interval = None
        if 'PER_beta_increase_interval' in self.kwargs and self.kwargs['PER_beta_increase_interval']!='None':
            beta_increase_interval = float(self.kwargs['PER_beta_increase_interval'])  

        for i in range(nbr_storages):
            if self.kwargs['THER_use_PER']:
                self.predictor_storages.append(
                    SplitPrioritizedReplayStorage(
                        capacity=self.kwargs['THER_replay_capacity'],
                        alpha=self.kwargs['THER_PER_alpha'],
                        beta=self.kwargs['THER_PER_beta'],
                        keys=keys,
                        circular_keys=circular_keys,
                        circular_offsets=circular_offsets,
                        test_train_split_interval=self.kwargs['THER_predictor_test_train_split_interval'],
                        test_capacity=self.kwargs['THER_test_replay_capacity']
                    )
                )
            else:
                self.predictor_storages.append(
                    SplitReplayStorage(
                        capacity=self.kwargs['THER_replay_capacity'],
                        keys=keys,
                        circular_keys=circular_keys,
                        circular_offsets=circular_offsets,
                        test_train_split_interval=self.kwargs['THER_predictor_test_train_split_interval'],
                        test_capacity=self.kwargs['THER_test_replay_capacity']
                    )
                )

    def _update_goals_in_rnn_states(
        self, 
        hdict:Dict, 
        goal_value:torch.Tensor, 
        latent_goal_value:Optional[torch.Tensor]=None,
        goal_key:Optional[str]='target_goal',
        latent_goal_key:Optional[str]=None,
        ):
        goals = {goal_key:goal_value}
        if latent_goal_key is not None: goals[latent_goal_key] = latent_goal_value
        for gkey, gvalue in goals.items():
            if gkey in self.extra_inputs_infos:
                if not isinstance(self.extra_inputs_infos[gkey]['target_location'][0], list):
                    self.extra_inputs_infos[gkey]['target_location'] = [self.extra_inputs_infos[gkey]['target_location']]
                for tl in self.extra_inputs_infos[gkey]['target_location']:
                    pointer = hdict
                    for child_node in tl:
                        if child_node not in pointer:
                            pointer[child_node] = {}
                        pointer = pointer[child_node]
                    pointer[gkey] = [gvalue]
        return hdict

    def _extract_goal_from_info_default_fn(
        self, 
        hdict:Dict, 
        goal_key:Optional[str]='achieved_goal',
        ):
        assert goal_key in hdict
        value = hdict[goal_key]
        postprocess_fn=(lambda x:torch.from_numpy(x).float() if isinstance(x, np.ndarray) else torch.ones(1, 1).float()*x)
        return postprocess_fn(value)

    def store(self, exp_dict, actor_index=0):
        self.episode_buffer[actor_index].append(exp_dict)
        self.nbr_buffered_predictor_experience += 1

        successful_traj = False

        if not(exp_dict['non_terminal']):
            self.episode_count += 1
            episode_length = len(self.episode_buffer[actor_index])

            # Assumes non-successful rewards are non-positive:
            successful_traj = all(self.episode_buffer[actor_index][-1]['r']>0)
            
            # Relabelling if unsuccessfull trajectory:
            relabelling = not successful_traj
            
            episode_rewards = []
            her_rs = []
            per_episode_d2store = {}

            for idx in range(episode_length):
                s = self.episode_buffer[actor_index][idx]['s']
                a = self.episode_buffer[actor_index][idx]['a']
                r = self.episode_buffer[actor_index][idx]['r']
                
                # Assumes failure rewards are non-positive:
                self.reward_shape = r.shape
                her_r = self.feedbacks['success']*torch.ones_like(r) if r.item()>0 else self.feedbacks['failure']*torch.ones_like(r)
                
                succ_s = self.episode_buffer[actor_index][idx]['succ_s']
                non_terminal = self.episode_buffer[actor_index][idx]['non_terminal']

                info = self.episode_buffer[actor_index][idx]['info']
                succ_info = self.episode_buffer[actor_index][idx]['succ_info']
                rnn_states = self.episode_buffer[actor_index][idx]['rnn_states']
                
                episode_rewards.append(r)
                her_rs.append(her_r)

                d2store = {
                    's':s, 
                    'a':a, 
                    'r':her_r, 
                    'succ_s':succ_s, 
                    'non_terminal':non_terminal, 
                    'rnn_states':copy_hdict(rnn_states),
                    'info': info,
                    'succ_info': succ_info,
                }

                if not(relabelling):
                    # Only insert this experience that way if successfull:
                    #self.algorithm.store(d2store, actor_index=actor_index)
                    if -1 not in per_episode_d2store: per_episode_d2store[-1] = []
                    per_episode_d2store[-1].append(d2store)
                
                # Store data in predictor storages if successfull:
                if self.kwargs['THER_use_THER'] and r.item()>0: #self.feedbacks['failure']:
                    self.predictor_store(d2store, actor_index=actor_index)
                    wandb.log({'Training/THER_Predictor/DatasetSize': self.nbr_handled_predictor_experience}, commit=False) # self.param_predictor_update_counter)
                    if self.algorithm.summary_writer is not None:
                        self.algorithm.summary_writer.add_scalar('Training/THER_Predictor/DatasetSize', self.nbr_handled_predictor_experience, self.param_predictor_update_counter)
                    
                #if all(non_terminal<=0.5) 
                if idx==(episode_length-1):
                    self.episode_count += 1
                    wandb.log({'PerEpisode/EpisodeLength': len(her_rs)}, commit=False)
                    
                    wandb.log({'PerEpisode/HER_Success': 1+her_r.mean().item()}, commit=False) 
                    wandb.log({'PerEpisode/HER_FinalReward': her_r.mean().item()}, commit=False) 
                    wandb.log({'PerEpisode/HER_Return': sum(her_rs)}, commit=False)
                    wandb.log({'PerEpisode/HER_NormalizedReturn': sum(her_rs)/len(her_r)}, commit=False)
                    wandb.log({'PerEpisode/OriginalFinalReward': r.mean().item()}, commit=False)
                    wandb.log({'PerEpisode/OriginalReturn': sum(episode_rewards)}, commit=False)
                    wandb.log({'PerEpisode/OriginalNormalizedReturn': sum(episode_rewards)/len(episode_rewards)}, commit=False) # self.episode_count)
                    if self.algorithm.summary_writer is not None:
                        self.algorithm.summary_writer.add_scalar('PerEpisode/Success', (self.rewards['success']==her_r).float().mean().item(), self.episode_count)
                        self.algorithm.summary_writer.add_histogram('PerEpisode/Rewards', episode_rewards, self.episode_count)

                
                # Are we relabelling?
                if not(self.kwargs['THER_use_THER']) or not(relabelling):
                    continue 
                
                # Is it safe to use the predictor:
                safe_relabelling = self.test_acc >= self.kwargs['THER_predictor_accuracy_safe_to_relabel_threshold']
                
                if safe_relabelling:
                    # Relabelling everything with the hindsight_goal computed on the fly, and set the reward accordingly:
                    for k in range(self.k):
                        if 'final' in self.strategy:
                            achieved_exp = self.episode_buffer[actor_index][idx]
                            target_exp = self.episode_buffer[actor_index][-1]
                            new_r, achieved_goal_from_target_exp, \
                            achieved_latent_goal_from_target_exp = self.goal_predicated_reward_fn(
                                achieved_exp=achieved_exp, 
                                target_exp=target_exp,
                                _extract_goal_from_info_fn=self._extract_goal_from_info_fn,
                                goal_key=self.achieved_goal_key_from_info,
                                latent_goal_key=self.achieved_latent_goal_key_from_info,
                                feedbacks=self.feedbacks,
                                reward_shape=self.reward_shape
                            )
                            
                            # Assumes new_r to be -1 for failure and 0 for success:
                            new_her_r = self.feedbacks['success']*torch.ones_like(r) if all(new_r>-0.5) else self.feedbacks['failure']*torch.ones_like(r)
                            
                            new_non_terminal = torch.zeros_like(non_terminal) if all(new_her_r>self.feedbacks['failure']) else torch.ones_like(non_terminal)
                            
                            d2store_her = {
                                's':s, 
                                'a':a, 
                                'r':new_her_r, 
                                'succ_s':succ_s, 
                                'non_terminal':new_non_terminal, 
                                'rnn_states': copy_hdict(
                                    self._update_goals_in_rnn_states(
                                        hdict=rnn_states,
                                        goal_value=achieved_goal_from_target_exp,
                                        latent_goal_value=achieved_latent_goal_from_target_exp,
                                        goal_key=self.target_goal_key_from_info,
                                        latent_goal_key=self.target_latent_goal_key_from_info,
                                    )
                                ),
                                'info': info,
                                'succ_info': succ_info,
                            }
                            
                            if self.algorithm.summary_writer is not None:
                                self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_final', new_her_r.mean().item(), self.algorithm.get_update_count())
                                #self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_dist', dist.mean().item(), self.algorithm.get_update_count())
                            wandb.log({'PerUpdate/HER_AfterRelabellingReward': new_her_r.mean().item()}, commit=True)
                            
                        if 'future' in self.strategy:
                            raise NotImplementedError
                            future_idx = np.random.randint(idx, episode_length)
                            achieved_exp = self.episode_buffer[actor_index][idx]
                            desired_exp = self.episode_buffer[actor_index][future_idx]
                            
                            new_r, achieved_pred_goal, desired_pred_goal, dist = self.goal_predicated_reward_fn(
                                achieved_exp=achieved_exp, 
                                desired_exp=desired_exp,
                                _extract_goals_from_info_fn=self._extract_goals_from_info,
                            )
                            
                            # Assumes new_r to be -1 for failure and 0 for success:
                            new_her_r = self.rewards['success']*torch.ones_like(r) if all(new_r>-0.5) else self.rewards['failure']*torch.ones_like(r)
                            
                            new_non_terminal = torch.zeros_like(non_terminal) if all(new_her_r>self.rewards['failure']) else torch.ones_like(non_terminal)
                            
                            d2store_her = {
                                's':s, 
                                'a':a, 
                                'r':new_her_r, 
                                'succ_s':succ_s, 
                                'non_terminal':new_non_terminal, 
                                
                                'rnn_states': copy_hdict(
                                    self._update_goals_in_rnn_states(
                                        hdict=rnn_states,
                                        goal_value=desired_pred_goal,
                                        goal_key='desired_goal',
                                    )
                                ),
                                'info': info,
                                #'g': desired_pred_goal,
                            }
                            
                        # Adding this relabelled experience to the replay buffer with 'proper' goal...
                        #self.algorithm.store(d2store_her, actor_index=actor_index)
                        valid_exp = True
                        if self.filtering_fn != "None":
                            kwargs = {
                                "d2store":d2store,
                                "episode_buffer":self.episode_buffer[actor_index],
                                "achieved_goal_from_target_exp":achieved_goal_from_target_exp,
                                "achieved_latent_goal_from_target_exp":achieved_latent_goal_from_target_exp,
                            }
                            valid_exp = self.filtering_fn(**kwargs)
                        if not valid_exp:   continue
                        
                        if k not in per_episode_d2store: per_episode_d2store[k] = []
                        per_episode_d2store[k].append(d2store_her)
            
            else:
                # safe relabelling is not possible...
                # what can we do instead?
                pass

            # Now that we have all the different trajectories,
            # we can send them to the main algorithm as complete
            # whole trajectories, one experience at a time.
            for key in per_episode_d2store:
                for didx, d2store in enumerate(per_episode_d2store[key]):
                    self.algorithm.store(d2store, actor_index=actor_index)
                wandb.log({f'PerEpisode/HER_traj_length/{key}': len(per_episode_d2store[key])}, commit=False)
            # Reset episode buffer:
            self.episode_buffer[actor_index] = []

        period_check = self.kwargs['THER_replay_period']
        period_count_check = self.nbr_buffered_predictor_experience
        
        # Update predictor:
        if self.nbr_handled_predictor_experience >= self.kwargs['THER_min_capacity']\
        and ((period_count_check % period_check == 0) or (self.kwargs['THER_train_on_success'] and successful_traj)):
            self.update_predictor()
            
    def predictor_store(self, exp_dict, actor_index=0):
        # WARNING : multi storage is deprecated!
        actor_index = 0
        self.nbr_handled_predictor_experience += 1

        if self.kwargs['THER_use_PER']:
            init_sampling_priority = None 
            self.predictor_storages[actor_index].add(exp_dict, priority=init_sampling_priority)
        else:
            self.predictor_storages[actor_index].add(exp_dict)

    def update_predictor(self):
        for it in range(self.kwargs['THER_nbr_training_iteration_per_update']):
            self.test_acc = self.train_predictor()
            if self.test_acc >= self.kwargs['THER_predictor_accuracy_threshold']:
                break
        
    def train_predictor(self, minibatch_size=None):
        if minibatch_size is None:  minibatch_size = self.batch_size

        start = time.time()
        samples = self.retrieve_values_from_predictor_storages(minibatch_size=minibatch_size)
        end = time.time()
        
        wandb.log({'PerTHERPredictorUpdate/TimeComplexity/RetrieveValuesFn':  end-start}, commit=False) # self.param_update_counter)
        
        start = time.time()
        # WARNING: trying to prevent overfitting or some kind of instability:
        if self.test_acc <= self.kwargs['THER_predictor_accuracy_threshold']:
            self.optimize_predictor(minibatch_size, samples)
        end = time.time()
        
        wandb.log({'PerTHERPredictorUpdate/TimeComplexity/OptimizeModelFn':  end-start}, commit=False) # self.param_update_counter)
        
        test_storage_size = self.predictor_storages[0].get_size(test=True) #.test_storage.current_size['s']  
        train_storage_size = self.predictor_storages[0].get_size(test=False) #test_storage.current_size['s']  
        wandb.log({'PerTHERPredictorUpdate/TestStorageSize':  test_storage_size}, commit=False)
        wandb.log({'PerTHERPredictorUpdate/TrainStorageSize':  train_storage_size}, commit=False)
        if test_storage_size > self.kwargs['THER_min_capacity']:
            test_samples = self.retrieve_values_from_predictor_storages(minibatch_size=minibatch_size, test=True)
            with torch.no_grad():
                acc = self.test_predictor(minibatch_size, test_samples)
        else:
            acc = 0.0
 
        wandb.log({'PerTHERPredictorUpdate/TestSentenceAccuracy': acc, "ther_predictor_update_count":self.param_predictor_update_counter}, commit=False)
        
        return acc 

    def retrieve_values_from_predictor_storages(self, minibatch_size, test=False):
        torch.set_grad_enabled(False)
        keys=['s', 'a', 'succ_s', 'r', 'non_terminal']

        fulls = {}
        
        if self.kwargs['THER_use_PER']:
            fulls['importanceSamplingWeights'] = []

        if self.recurrent:
            keys += ['rnn_states']
        
        for key in keys:    fulls[key] = []

        for storage in self.predictor_storages:
            # Check that there is something in the storage 
            if len(storage) <= 1: continue
            batch_size = minibatch_size
            if batch_size is None:
                batch_size = storage.get_size(test=test)

            if self.kwargs['THER_use_PER']:
                sample, importanceSamplingWeights = storage.sample(batch_size=batch_size, keys=keys, test=test)
                importanceSamplingWeights = torch.from_numpy(importanceSamplingWeights)
                fulls['importanceSamplingWeights'].append(importanceSamplingWeights)
            else:
                sample = storage.sample(batch_size=batch_size, keys=keys, test=test)
            
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

        beta = self.predictor_storages[0].beta if self.kwargs['THER_use_PER'] else 1.0
        
        states = samples['s']
        actions = samples['a']
        next_states = samples['succ_s']
        rewards = samples['r']
        non_terminals = samples['non_terminal']
        goals = samples['g'] if 'g' in samples else None

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
                sampled_rnn_states = _extract_rnn_states_from_batch_indices(
                    rnn_states, 
                    batch_indices, 
                    use_cuda=self.kwargs['use_cuda'],
                )

            sampled_importanceSamplingWeights = None
            if self.kwargs['THER_use_PER']:
                sampled_importanceSamplingWeights = importanceSamplingWeights[batch_indices].cuda() if self.kwargs['use_cuda'] else importanceSamplingWeights[batch_indices]
            
            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_next_states = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]
            sampled_rewards = rewards[batch_indices].cuda() if self.kwargs['use_cuda'] else rewards[batch_indices]
            sampled_non_terminals = non_terminals[batch_indices].cuda() if self.kwargs['use_cuda'] else non_terminals[batch_indices]
            sampled_goals = None #DEPRECATED goals[batch_indices].cuda() if self.kwargs['use_cuda'] else goals[batch_indices]

            self.predictor_optimizer.zero_grad()
            
            output_dict = self.predictor_loss_fn(sampled_states, 
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
                                          summary_writer=self.algorithm.summary_writer,
                                          phase="Training")
            
            loss = output_dict['loss']
            loss_per_item = output_dict['loss_per_item']
            
            
            loss.backward(retain_graph=False)
            if self.kwargs['THER_gradient_clip'] > 1e-3:
                nn.utils.clip_grad_norm_(self.predictor.parameters(), self.kwargs['THER_gradient_clip'])
            self.predictor_optimizer.step()

            if self.kwargs['THER_use_PER']:
                sampled_losses_per_item.append(loss_per_item)
                #wandb_data = copy.deepcopy(wandb.run.history._data)
                #wandb.run.history._data = {}
                wandb.log({
                    'PerTHERPredictorUpdate/ImportanceSamplingMean':  sampled_importanceSamplingWeights.cpu().mean().item(),
                    'PerTHERPredictorUpdate/ImportanceSamplingStd':  sampled_importanceSamplingWeights.cpu().std().item(),
                    'PerTHERPredictorUpdate/PER_Beta':  beta
                }) # self.param_update_counter)
                #wandb.run.history._data = wandb_data

            self.param_predictor_update_counter += 1 

        torch.set_grad_enabled(False)
        
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

        end = time.time()
        wandb.log({'PerTHERPredictorUpdate/TimeComplexity/OptimizationLoss':  end-start}, commit=False) # self.param_update_counter)


    def test_predictor(self, minibatch_size, samples):
        training = self.predictor.training
        self.predictor.train(False)

        beta = self.predictor_storages[0].beta if self.kwargs['THER_use_PER'] else 1.0
        
        states = samples['s']
        actions = samples['a']
        next_states = samples['succ_s']
        rewards = samples['r']
        non_terminals = samples['non_terminal']
        goals = samples['g'] if 'g' in samples else None

        rnn_states = samples['rnn_states'] if 'rnn_states' in samples else None
        
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
            if self.recurrent:
                sampled_rnn_states = _extract_rnn_states_from_batch_indices(rnn_states, batch_indices, use_cuda=self.kwargs['use_cuda'])

            sampled_importanceSamplingWeights = None
            if self.kwargs['THER_use_PER']:
                sampled_importanceSamplingWeights = importanceSamplingWeights[batch_indices].cuda() if self.kwargs['use_cuda'] else importanceSamplingWeights[batch_indices]
            
            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_next_states = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]
            sampled_rewards = rewards[batch_indices].cuda() if self.kwargs['use_cuda'] else rewards[batch_indices]
            sampled_non_terminals = non_terminals[batch_indices].cuda() if self.kwargs['use_cuda'] else non_terminals[batch_indices]
            sampled_goals = None # DEPRECATED goals[batch_indices].cuda() if self.kwargs['use_cuda'] else goals[batch_indices]

            output_dict = self.predictor_loss_fn(sampled_states, 
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
                                          summary_writer=self.algorithm.summary_writer,
                                          phase="Testing")
            
            loss = output_dict['loss']
            loss_per_item = output_dict['loss_per_item']
            
            accuracy = output_dict['accuracy']
            running_acc = running_acc + accuracy

            if self.kwargs['THER_use_PER']:
                sampled_losses_per_item.append(loss_per_item)

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
                el_idx_in_storage = self.predictor_storages[storage_idx].get_test_storage().tree_indices[el_idx_in_batch]
                new_priority = self.predictor_storages[storage_idx].priority(sloss)
                self.predictor_storages[storage_idx].update(idx=el_idx_in_storage, priority=new_priority, test=True)

        self.predictor.train(training)

        running_acc = running_acc / nbr_batches
        return running_acc

    def clone(self, with_replay_buffer: bool=False, clone_proxies: bool=False, minimal: bool=False):        
        cloned_algo = THERAlgorithmWrapper2(
            algorithm=self.algorithm.clone(
                with_replay_buffer=with_replay_buffer,
                clone_proxies=clone_proxies,
                minimal=minimal
            ), 
            extra_inputs_infos=self.extra_inputs_infos,
            predictor=self.predictor, 
            predictor_loss_fn=self.predictor_loss_fn, 
            strategy=self.strategy, 
            goal_predicated_reward_fn=self.goal_predicated_reward_fn,
            _extract_goal_from_info_fn=self._extract_goal_from_info_fn,
            achieved_goal_key_from_info=self.achieved_goal_key_from_info,
            target_goal_key_from_info=self.target_goal_key_from_info,
            achieved_latent_goal_key_from_info=self.achieved_latent_goal_key_from_info,
            target_latent_goal_key_from_info=self.target_latent_goal_key_from_info,
            filtering_fn=self.filtering_fn,
            feedbacks=self.feedbacks,
        )
        return cloned_algo

