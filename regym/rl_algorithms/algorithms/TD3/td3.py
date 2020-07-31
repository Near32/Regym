import copy 
from collections import deque 
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from . import td3_actor_loss, td3_critic_loss

from ..algorithm import Algorithm
from ...replay_buffers import ReplayBuffer, PrioritizedReplayBuffer, EXP, EXPPER
from ...replay_buffers import PrioritizedReplayStorage, ReplayStorage
from ...networks import hard_update, soft_update, random_sample


summary_writer = None 


class OrnsteinUhlenbeckNoise :
    def __init__(self, dim,mu=0.0, theta=0.15, sigma=0.2) :
        self.dim = dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.X = np.ones(self.dim)*self.mu

    def setSigma(self,sigma):
        self.sigma = sigma
    
    def sample(self) :
        dx = self.theta * ( self.mu - self.X)
        dx += self.sigma *  np.random.randn( self.dim )
        self.X += dx
        return self.X

class GaussianNoise :
    def __init__(self, dim, mu=0.0, sigma=0.2) :
        self.dim = dim
        self.mu = mu
        self.sigma = sigma

    def setSigma(self,sigma):
        self.sigma = sigma
    
    def sample(self) :
        noise = self.mu+self.sigma*np.random.randn( self.dim )
        return noise

def apply_noise(action, noise_distr, action_clip=None, noise_clip=None):
    noise = torch.from_numpy(noise_distr.sample())
    if noise_clip is not None:
        noise = torch.clamp(noise, -noise_clip, noise_clip)

    action += noise.to(action.device)

    if action_clip is not None:
        action = torch.clamp(action, -action_clip, action_clip)

    return action 


class TD3Algorithm(Algorithm):
    def __init__(self, 
                 kwargs, 
                 model_actor,
                 model_critic,
                 target_model_actor=None,
                 target_model_critic=None,
                 optimizer_actor=None, 
                 optimizer_critic=None, 
                 actor_loss_fn=td3_critic_loss.compute_loss, 
                 critic_loss_fn=td3_actor_loss.compute_loss, 
                 sum_writer=None):
        '''
        :param kwargs:
            
            "path": str specifying where to save the model(s).
            "use_cuda": boolean to specify whether to use CUDA.
            
            "replay_capacity": int, capacity of the replay buffer to use.
            "min_capacity": int, minimal capacity before starting to learn.
            "batch_size": int, batch size to use [default: batch_size=256].
            
            "use_PER": boolean to specify whether to use a Prioritized Experience Replay buffer.
            "PER_alpha": float, alpha value for the Prioritized Experience Replay buffer.
            
            "lr": float, learning rate [default: lr=1e-3].
            "tau": float, soft-update rate [default: tau=1e-3].
            "gamma": float, Q-learning gamma rate [default: gamma=0.999].
            
            "preprocess": preprocessing function/transformation to apply to observations [default: preprocess=T.ToTensor()]
            "nbrTrainIteration": int, number of iteration to train the model at each new experience. [default: nbrTrainIteration=1]
        '''
        self.kwargs = copy.deepcopy(kwargs)
        self.use_cuda = kwargs["use_cuda"]

        self.actor_update_delay = int(self.kwargs["actor_update_delay"])

        self.model_actor = model_actor
        self.model_critic = model_critic
        if self.kwargs['use_cuda']:
            self.model_actor = self.model_actor.cuda()
            self.model_critic = self.model_critic.cuda()

        
        self.noisy = self.kwargs['noisy']
        self.n_step = self.kwargs['n_step'] if 'n_step' in self.kwargs else 1
        if self.n_step > 1:
            self.n_step_buffer = deque(maxlen=self.n_step)

        self.use_PER = self.kwargs['use_PER']
        
        self.goal_oriented = self.kwargs['goal_oriented'] if 'goal_oriented' in self.kwargs else False
        self.use_HER = self.kwargs['use_HER'] if 'use_HER' in self.kwargs else False
        
        self.weights_decay_lambda_actor = float(self.kwargs['weights_decay_lambda_actor'])
        self.weights_decay_lambda_critic = float(self.kwargs['weights_decay_lambda_critic'])
        
        self.nbr_actor = self.kwargs['nbr_actor']
        
        if target_model_actor is None:
            target_model_actor = copy.deepcopy(self.model_actor)
        if target_model_critic is None:
            target_model_critic = copy.deepcopy(self.model_critic)
        self.target_model_actor = target_model_actor
        self.target_model_critic = target_model_critic

        if self.kwargs['use_cuda']:
            self.target_model_actor = self.target_model_actor.cuda()
            self.target_model_critic = self.target_model_critic.cuda()
        hard_update(self.target_model_actor, self.model_actor)
        hard_update(self.target_model_critic, self.model_critic)

        for p in self.target_model_actor.parameters():
            p.requires_grad = False
        for p in self.target_model_critic.parameters():
            p.requires_grad = False

        self.target_model_actor.share_memory()
        self.target_model_critic.share_memory()

        if optimizer_actor is None:
            parameters_actor = self.model_actor.parameters()
            # Tuning learning rate with respect to the number of actors:
            # Following: https://arxiv.org/abs/1705.04862
            lr = kwargs['actor_learning_rate']
            if kwargs['lr_account_for_nbr_actor']:
                lr *= self.nbr_actor
            print(f"Learning rate::Actor: {lr}")
            self.optimizer_actor = optim.Adam(parameters_actor, lr=lr, betas=(0.9,0.999), eps=kwargs['adam_eps'])
        else: self.optimizer_actor = optimizer_actor

        if optimizer_critic is None:
            parameters_critic = self.model_critic.parameters()
            # Tuning learning rate with respect to the number of actors:
            # Following: https://arxiv.org/abs/1705.04862
            lr = kwargs['critic_learning_rate'] 
            if kwargs['lr_account_for_nbr_actor']:
                lr *= self.nbr_actor
            print(f"Learning rate::Critic: {lr}")
            self.optimizer_critic = optim.Adam(parameters_critic, lr=lr, betas=(0.9,0.999), eps=kwargs['adam_eps'])
        else: self.optimizer_critic = optimizer_critic

        self.actor_loss_fn = actor_loss_fn
        print(f"WARNING: actor_loss_fn is {self.actor_loss_fn}")
        self.critic_loss_fn = critic_loss_fn
        print(f"WARNING: critic_loss_fn is {self.critic_loss_fn}")
            
        
        self.noise = GaussianNoise(self.model_actor.action_dim, sigma=kwargs["actor_noise_std"])
        self.target_noise = GaussianNoise(self.model_actor.action_dim, sigma=kwargs["target_actor_noise_std"])
        self.noise_fn = partial(
            apply_noise,
            noise_distr=self.target_noise,
            action_clip=float(kwargs["action_scaler"]),
            noise_clip=float(kwargs["target_actor_noise_clip"]))

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
        
        global summary_writer
        if sum_writer is not None: summary_writer = sum_writer
        self.summary_writer = summary_writer
        self.param_update_counter = 0
    
    def get_models(self):
        return {
            'model_actor': self.model_actor, 
            'model_critic': self.model_critic, 
            'target_model_actor': self.target_model_actor,
            'target_model_critic': self.target_model_critic,
        }

    def get_nbr_actor(self):
        return self.nbr_actor

    def get_update_count(self):
        return self.param_update_counter

    def reset_storages(self, nbr_actor=None):
        if nbr_actor is not None:
            self.nbr_actor = nbr_actor

        if self.storages is not None:
            for storage in self.storages: storage.reset()

        self.storages = []
        #keys = ['s', 'a', 'log_pi_a', 'r', 'non_terminal']
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

    def update_targets(self):
        if (self.target_update_count//self.nbr_actor) % self.actor_update_delay == 0:
            soft_update(self.target_model_critic, self.model_critic, self.TAU)
            soft_update(self.target_model_actor, self.model_actor, self.TAU)

    def train(self, minibatch_size=None):
        if minibatch_size is None:  minibatch_size = self.batch_size

        self.target_update_count += self.nbr_actor

        samples = self.retrieve_values_from_storages(minibatch_size=minibatch_size)
        
        if self.noisy:  
            self.model_actor.reset_noise()
            self.model_critic.reset_noise()
            self.target_model_actor.reset_noise()
            self.target_model_critic.reset_noise()


        self.optimize_model(minibatch_size, samples)
        self.update_targets()

        """
        if self.target_update_count > self.target_update_interval:
            self.target_update_count = 0
            hard_update(self.target_model,self.model)
        """

    def retrieve_values_from_storages(self, minibatch_size):
        #keys=['s', 'a', 'log_pi_a', 'succ_s', 'r', 'non_terminal']
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

    def optimize_model(self, minibatch_size, samples):
        global summary_writer
        self.summary_writer = summary_writer

        beta = self.storages[0].beta if self.use_PER else 1.0
        
        states = samples['s']
        actions = samples['a']
        next_states = samples['succ_s']
        rewards = samples['r']
        non_terminals = samples['non_terminal']

        rnn_states = samples['rnn_states'] if 'rnn_states' in samples else None
        goals = samples['g'] if 'g' in samples else None

        importanceSamplingWeights = samples['importanceSamplingWeights'] if 'importanceSamplingWeights' in samples else None

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
                sampled_rnn_states = Algorithm._extract_rnn_states_from_batch_indices(rnn_states, batch_indices, use_cuda=self.kwargs['use_cuda'])

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
            
            # Critic:
            self.optimizer_critic.zero_grad()
            critic_loss, critic_loss_per_item = self.critic_loss_fn(
                sampled_states, 
                sampled_actions, 
                sampled_next_states,
                sampled_rewards,
                sampled_non_terminals,
                rnn_states=sampled_rnn_states,
                goals=sampled_goals,
                gamma=self.GAMMA,
                noise_fn=self.noise_fn,
                model_critic=self.model_critic,
                target_model_critic=self.target_model_critic,
                model_actor=self.model_actor,
                target_model_actor=self.target_model_actor,
                weights_decay_lambda=self.weights_decay_lambda_critic,
                use_PER=self.use_PER,
                PER_beta=beta,
                importanceSamplingWeights=sampled_importanceSamplingWeights,
                HER_target_clamping=self.kwargs['HER_target_clamping'],
                iteration_count=self.param_update_counter,
                summary_writer=summary_writer
            )

            critic_loss.backward(retain_graph=False)
            if self.kwargs['gradient_clip'] > 1e-3:
                nn.utils.clip_grad_norm_(self.model_critic.parameters(), self.kwargs['gradient_clip'])
            self.optimizer_critic.step()

            # Actor:
            if (self.target_update_count//self.nbr_actor) % self.actor_update_delay == 0:
                self.optimizer_actor.zero_grad()
                actor_loss, actor_loss_per_item = self.actor_loss_fn(
                    sampled_states, 
                    sampled_actions, 
                    sampled_next_states,
                    sampled_rewards,
                    sampled_non_terminals,
                    rnn_states=sampled_rnn_states,
                    goals=sampled_goals,
                    gamma=self.GAMMA,
                    model_critic=self.model_critic,
                    target_model_critic=self.target_model_critic,
                    model_actor=self.model_actor,
                    target_model_actor=self.target_model_actor,
                    weights_decay_lambda=self.weights_decay_lambda_actor,
                    use_PER=self.use_PER,
                    PER_beta=beta,
                    importanceSamplingWeights=sampled_importanceSamplingWeights,
                    HER_target_clamping=self.kwargs['HER_target_clamping'],
                    iteration_count=self.param_update_counter,
                    summary_writer=summary_writer
                )

                actor_loss.backward(retain_graph=False)
                if self.kwargs['gradient_clip'] > 1e-3:
                    nn.utils.clip_grad_norm_(self.model_actor.parameters(), self.kwargs['gradient_clip'])
                self.optimizer_actor.step()

            # Bookkeeping:
            if self.use_PER:
                sampled_losses_per_item.append(critic_loss_per_item)

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

