import copy 
from collections import deque 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from . import dqn_loss, ddqn_loss

from ..algorithm import Algorithm
from ...replay_buffers import ReplayBuffer, PrioritizedReplayBuffer, EXP, EXPPER
from ...replay_buffers import PrioritizedReplayStorage, ReplayStorage
from ...networks import hard_update, random_sample


summary_writer = None 


class DQNAlgorithm(Algorithm):
    def __init__(self, kwargs, model, target_model=None, optimizer=None, sum_writer=None):
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

        
        if optimizer is None:
            parameters = self.model.parameters()
            # Tuning learning rate with respect to the number of actors:
            # Following: https://arxiv.org/abs/1705.04862
            lr = kwargs['learning_rate'] 
            if kwargs['lr_account_for_nbr_actor']:
                lr *= self.nbr_actor
            print(f"Learning rate: {lr}")
            self.optimizer = optim.Adam(parameters, lr=lr, eps=kwargs['adam_eps'])
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

        global summary_writer
        summary_writer = sum_writer
        self.param_update_counter = 0
    
    def get_update_count(self):
        return self.param_update_counter

    def reset_storages(self, nbr_actor=None):
        if nbr_actor is not None:
            self.nbr_actor = nbr_actor

        if self.storages is not None:
            for storage in self.storages: storage.reset()

        self.storages = []
        for i in range(self.nbr_actor):
            if self.kwargs['use_PER']:
                self.storages.append(PrioritizedReplayStorage(capacity=self.kwargs['replay_capacity'],
                                                                alpha=self.kwargs['PER_alpha'],
                                                                beta=self.kwargs['PER_beta'],
                                                                circular_offsets={'succ_s':self.n_step})
                )
            else:
                self.storages.append(ReplayStorage(capacity=self.kwargs['replay_capacity'],
                                                   circular_offsets={'succ_s':self.n_step})
                )
            if self.recurrent:
                self.storages[-1].add_key('rnn_states')
                self.storages[-1].add_key('next_rnn_states')
    
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
        
        if self.use_PER:
            #init_sampling_priority = self.storages[actor_index].priority(torch.abs(current_exp_dict['r']).cpu().numpy() )
            init_sampling_priority = None 
            self.storages[actor_index].add(current_exp_dict, priority=init_sampling_priority)
        else:
            self.storages[actor_index].add(current_exp_dict)

    def train(self, minibatch_size=None):
        if minibatch_size is None:  minibatch_size = self.batch_size

        self.target_update_count += self.nbr_actor

        states, actions, next_states, rewards, non_terminals, rnn_states, importanceSamplingWeights = self.retrieve_values_from_storages(minibatch_size=minibatch_size)
        if self.recurrent: rnn_states = self.reformat_rnn_states(rnn_states)
        
        if self.noisy:  
            self.model.reset_noise()
            self.target_model.reset_noise()

        self.optimize_model(minibatch_size, states, actions, next_states, rewards, non_terminals, rnn_states, importanceSamplingWeights)
        
        if self.target_update_count > self.target_update_interval:
            self.target_update_count = 0
            hard_update(self.target_model,self.model)

    def retrieve_values_from_storages(self, minibatch_size):
        full_states = []
        full_actions = []
        full_rewards = []
        full_next_states = []
        full_non_terminals = []

        full_importanceSamplingWeights = None 
        if self.use_PER:
            full_importanceSamplingWeights = []

        full_rnn_states = None
        if self.recurrent:
            full_rnn_states = []
            
        for storage in self.storages:
            # Check that there is something in the storage 
            if len(storage) <= 1: continue
            if self.use_PER:
                sample, importanceSamplingWeights = storage.sample(batch_size=minibatch_size, keys=['s', 'a', 'succ_s', 'r', 'non_terminal', 'rnn_state'])
                importanceSamplingWeights = torch.from_numpy(importanceSamplingWeights)
                full_importanceSamplingWeights.append(importanceSamplingWeights)
            else:
                sample = storage.sample(batch_size=minibatch_size, keys=['s', 'a', 'succ_s', 'r', 'non_terminal', 'rnn_state'])
            
            states, actions, next_states, rewards, non_terminals, rnn_states = sample 
            states = torch.cat(states.tolist(), dim=0)
            actions = torch.cat(actions.tolist(), dim=0)
            next_states = torch.cat(next_states.tolist(), dim=0) 
            rewards = torch.cat(rewards.tolist(), dim=0)
            non_terminals = torch.cat(non_terminals.tolist(), dim=0)
            
            full_states.append(states)
            full_actions.append(actions)
            full_next_states.append(next_states)
            full_rewards.append(rewards)
            full_non_terminals.append(non_terminals)
            if self.recurrent:
                full_rnn_states += rnn_states[0]
        
        if self.use_PER:
            full_importanceSamplingWeights = torch.cat(full_importanceSamplingWeights, dim=0)

        full_states = torch.cat(full_states, dim=0)
        full_actions = torch.cat(full_actions, dim=0)
        full_next_states = torch.cat(full_next_states, dim=0)
        full_rewards = torch.cat(full_rewards, dim=0)
        full_non_terminals = torch.cat(full_non_terminals, dim=0)
        
        return full_states, full_actions, \
                full_next_states, full_rewards, \
                full_non_terminals, full_rnn_states, \
                full_importanceSamplingWeights

    def optimize_model(self, minibatch_size, states, actions, next_states, rewards, non_terminals, rnn_states=None, importanceSamplingWeights=None):
        global summary_writer
        beta = self.storages[0].beta if self.use_PER else 1.0
        
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

            sampled_importanceSamplingWeights = None
            if self.use_PER:
                sampled_importanceSamplingWeights = importanceSamplingWeights[batch_indices].cuda() if self.kwargs['use_cuda'] else importanceSamplingWeights[batch_indices]
            
            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_next_states = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]
            sampled_rewards = rewards[batch_indices].cuda() if self.kwargs['use_cuda'] else rewards[batch_indices]
            sampled_non_terminals = non_terminals[batch_indices].cuda() if self.kwargs['use_cuda'] else non_terminals[batch_indices]
            
            sampled_states = sampled_states.detach()
            sampled_actions = sampled_actions.detach()
            sampled_next_states = sampled_next_states.detach()
            sampled_rewards = sampled_rewards.detach()
            sampled_non_terminals = sampled_non_terminals.detach()
            
            self.optimizer.zero_grad()
            if self.double or self.dueling:
                loss, loss_per_item = ddqn_loss.compute_loss(sampled_states, 
                                              sampled_actions, 
                                              sampled_next_states,
                                              sampled_rewards,
                                              sampled_non_terminals,
                                              rnn_states=sampled_rnn_states,
                                              gamma=self.GAMMA,
                                              model=self.model,
                                              target_model=self.target_model,
                                              weights_decay_lambda=self.weights_decay_lambda,
                                              use_PER=self.use_PER,
                                              PER_beta=beta,
                                              importanceSamplingWeights=sampled_importanceSamplingWeights,
                                              iteration_count=self.param_update_counter,
                                              summary_writer=summary_writer)
            else:
                loss, loss_per_item = dqn_loss.compute_loss(sampled_states, 
                                              sampled_actions, 
                                              sampled_next_states,
                                              sampled_rewards,
                                              sampled_non_terminals,
                                              rnn_states=sampled_rnn_states,
                                              gamma=self.GAMMA,
                                              model=self.model,
                                              target_model=self.target_model,
                                              weights_decay_lambda=self.weights_decay_lambda,
                                              use_PER=self.use_PER,
                                              PER_beta=beta,
                                              importanceSamplingWeights=sampled_importanceSamplingWeights,
                                              iteration_count=self.param_update_counter,
                                              summary_writer=summary_writer)

            loss.backward(retain_graph=False)
            if self.kwargs['gradient_clip'] > 1e-3:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs['gradient_clip'])
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


class DQNAlgorithm2():
    def __init__(self, kwargs, model, target_model=None):
        """
        :param kwargs:
            "use_cuda": boolean to specify whether to use CUDA.
            "replay_capacity": int, capacity of the replay buffer to use.
            "min_capacity": int, minimal capacity before starting to learn.
            "batch_size": int, batch size to use [default: batch_size=256].
            "use_PER": boolean to specify whether to use a Prioritized Experience Replay buffer.
            "PER_alpha": float, alpha value for the Prioritized Experience Replay buffer.
            "lr": float, learning rate.
            "tau": float, target update rate.
            "gamma": float, Q-learning gamma rate.
            "preprocess": preprocessing function/transformation to apply to observations [default: preprocess=T.ToTensor()]
            "nbrTrainIteration": int, number of iteration to train the model at each new experience. [default: nbrTrainIteration=1]
            "epsstart": starting value of the epsilong for the epsilon-greedy policy.
            "epsend": asymptotic value of the epsilon for the epsilon-greedy policy.
            "epsdecay": rate at which the epsilon of the epsilon-greedy policy decays.

            "dueling": boolean specifying whether to use Dueling Deep Q-Network architecture
            "double": boolean specifying whether to use Double Deep Q-Network algorithm.
            "nbr_actions": number of dimensions in the action space.
            "actfn": activation function to use in between each layer of the neural networks.
            "state_dim": number of dimensions in the state space.
        :param model: model of the agent to use/optimize in this algorithm.

        """

        self.kwargs = kwargs
        self.use_cuda = kwargs["use_cuda"]

        self.model = model
        if self.use_cuda:
            self.model = self.model.cuda()

        if target_model is None:
            target_model = copy.deepcopy(self.model)

        self.target_model = target_model
        self.target_model.share_memory()

        hard_update(self.target_model, self.model)
        if self.use_cuda:
            self.target_model = self.target_model.cuda()

        if self.kwargs['replayBuffer'] is None:
            if kwargs["use_PER"]:
                self.replayBuffer = PrioritizedReplayBuffer(capacity=kwargs["replay_capacity"], alpha=kwargs["PER_alpha"])
            else:
                self.replayBuffer = ReplayBuffer(capacity=kwargs["replay_capacity"])
        else:
            self.replayBuffer = self.kwargs['replayBuffer']

        self.min_capacity = kwargs["min_capacity"]
        self.batch_size = kwargs["batch_size"]

        self.lr = kwargs["lr"]
        self.TAU = kwargs["tau"]
        self.target_update_interval = int(1.0/self.TAU)
        self.target_update_count = 0
        self.GAMMA = kwargs["gamma"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.preprocess = kwargs["preprocess"]

        self.epsend = kwargs['epsend']
        self.epsstart = kwargs['epsstart']
        self.epsdecay = kwargs['epsdecay']

    def clone(self):
        cloned_kwargs = self.kwargs
        cloned_model = self.model.clone()
        cloned_model.share_memory()
        cloned_target_model = self.target_model.clone()
        cloned_target_model.share_memory()
        cloned = DQNAlgorithm(kwargs=cloned_kwargs, model=cloned_model, target_model=cloned_target_model)
        return cloned

    def optimize_model(self, gradient_clamping_value=None):
        """
        1) Estimate the gradients of the loss with respect to the
        current learner model on a batch of experiences sampled
        from the replay buffer.
        2) Backward the loss.
        3) Update the weights with the optimizer.
        4) Optional: Update the Prioritized Experience Replay buffer with new priorities.

        :param gradient_clamping_value: if None, the gradient is not clamped,
                                        otherwise a positive float value is expected as a clamping value
                                        and gradients are clamped.
        :returns loss_np: numpy scalar of the estimated loss function.
        """

        if len(self.replayBuffer) < self.min_capacity:
            return None

        if self.kwargs['use_PER']:
            # Create batch with PrioritizedReplayBuffer/PER:
            transitions, importanceSamplingWeights = self.replayBuffer.sample(self.batch_size)
            batch = EXPPER(*zip(*transitions))
            importanceSamplingWeights = torch.from_numpy(importanceSamplingWeights)
        else:
            # Create Batch with replayBuffer :
            transitions = self.replayBuffer.sample(self.batch_size)
            batch = EXP(*zip(*transitions))

        '''
        next_state_batch = Variable(torch.cat(batch.next_state), requires_grad=False)
        state_batch = Variable(torch.cat(batch.state), requires_grad=False)
        action_batch = Variable(torch.cat(batch.action), requires_grad=False)
        reward_batch = Variable(torch.cat(batch.reward), requires_grad=False).view((-1, 1))
        done_batch = [0.0 if batch.done[i] else 1.0 for i in range(len(batch.done))]
        done_batch = Variable(torch.FloatTensor(done_batch), requires_grad=False).view((-1, 1))
        '''
        next_state_batch = torch.cat(batch.next_state).detach()
        state_batch = torch.cat(batch.state).detach()
        action_batch = torch.cat(batch.action).detach()
        reward_batch = torch.cat(batch.reward).detach().view((-1, 1))
        done_batch = [0.0 if batch.done[i] else 1.0 for i in range(len(batch.done))]
        done_batch = torch.FloatTensor(done_batch).view((-1, 1))

        if self.use_cuda:
            if self.kwargs['use_PER']: importanceSamplingWeights = importanceSamplingWeights.cuda()
            next_state_batch = next_state_batch.cuda()
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            done_batch = done_batch.cuda()

        self.optimizer.zero_grad()

        state_action_values = self.model(state_batch)
        state_action_values_g = state_action_values.gather(dim=1, index=action_batch)

        ############################
        targetQ_nextS_A_values = self.target_model(next_state_batch)
        argmaxA_targetQ_nextS_A_values, index_argmaxA_targetQ_nextS_A_values = targetQ_nextS_A_values.max(1)
        argmaxA_targetQ_nextS_A_values = argmaxA_targetQ_nextS_A_values.view(-1, 1)
        ############################

        # Compute the expected Q values
        gamma_next = (self.GAMMA * argmaxA_targetQ_nextS_A_values)
        expected_state_action_values = reward_batch + done_batch*gamma_next

        # Compute loss:
        diff = expected_state_action_values - state_action_values_g
        if self.kwargs['use_PER']:
            diff_squared = importanceSamplingWeights.unsqueeze(1) * diff.pow(2.0)
        else:
            diff_squared = diff.pow(2.0)
        loss_per_item = diff_squared
        loss = torch.mean(diff_squared)
        loss.backward()

        if gradient_clamping_value is not None:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), gradient_clamping_value)

        weights_decay_lambda = 1e-0
        weights_decay_loss = weights_decay_lambda * 0.5*sum([torch.mean(param*param) for param in self.model.parameters()])
        weights_decay_loss.backward()

        self.optimizer.step()

        loss_np = loss_per_item.cpu().data.numpy()
        if self.kwargs['use_PER']:
            for (idx, new_error) in zip(batch.idx, loss_np):
                new_priority = self.replayBuffer.priority(new_error)
                self.replayBuffer.update(idx, new_priority)

        return loss_np

    def handle_experience(self, experience):
        '''
        This function is only called during training.
        It stores experience in the replay buffer.

        :param experience: EXP object containing the current, relevant experience.
        '''
        if self.kwargs["use_PER"]:
            init_sampling_priority = self.replayBuffer.priority(torch.abs(experience.reward).cpu().numpy() )
            self.replayBuffer.add(experience, init_sampling_priority)
        else:
            self.replayBuffer.push(experience)

    def train(self, iteration=1):
        self.target_update_count += iteration
        for t in range(iteration):
            lossnp = self.optimize_model()

        if self.target_update_count > self.target_update_interval:
            self.target_update_count = 0
            hard_update(self.target_model,self.model)