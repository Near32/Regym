import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..networks.ppo_network_utils import layer_init

from functools import reduce

from copy import deepcopy

from ..networks import random_sample
from ..replay_buffers import Storage
from . import a2c_loss

summary_writer = None 


class A2CAlgorithm_deprecated():

    def __init__(self, policy_model_input_dim, policy_model_output_dim,
                 discount_factor, n_steps, learning_rate, adam_eps):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.n_steps = n_steps
        self.model = FullyConnectedFeedForward(policy_model_input_dim, policy_model_output_dim, hidden_units=(16,))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=adam_eps)

    def train(self, samples, bootstrapped_reward):
        rewards                  = np.array([reward for (s, a, log_a, reward, state_value, succ_s, done) in samples])
        q_values                 = self.compute_temporal_differences_targets(rewards, bootstrapped_reward)
        state_values             = torch.cat([state_value for (s, a, log_a, reward, state_value, succ_s, done) in samples])
        log_action_probabilities = torch.cat([log_a for (s, a, log_a, reward, state_value, succ_s, done) in samples])

        def closure():
            self.optimizer.zero_grad()
            policy_loss = -1. * self.compute_policy_utility_gradient(log_action_probabilities, q_values, state_values)
            value_loss  = nn.MSELoss()(state_values.squeeze(), q_values)
            (policy_loss + value_loss).backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
            return (policy_loss + value_loss)
        self.optimizer.step(closure)

    def compute_policy_utility_gradient(self, log_action_probabilities, q_values, state_values):
        advantages = (q_values - state_values.squeeze()).detach()
        return torch.mean(log_action_probabilities.squeeze() * advantages)

    def compute_temporal_differences_targets(self, rewards, bootstrapped_reward):
        discounted_rewards = np.zeros_like(rewards)
        running_add = bootstrapped_reward
        for t in reversed(range(0, len(rewards))):
            running_add = rewards[t] + self.discount_factor * running_add
            discounted_rewards[t] = running_add
        return torch.from_numpy(discounted_rewards).type(torch.FloatTensor)


class FullyConnectedFeedForward(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_units=(32,), gate=F.relu):
        super(FullyConnectedFeedForward, self).__init__()
        dimensions = (input_dim,) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out))
                                     for dim_in, dim_out in zip(dimensions[:-1], dimensions[1:])])
        self.policy_head_layer = layer_init(nn.Linear(hidden_units[-1], output_dim))
        self.value_head_layer = layer_init(nn.Linear(hidden_units[-1], 1))
        self.gate = gate

    def forward(self, x):
        x = torch.Tensor(x).unsqueeze(0).type(torch.FloatTensor).cpu()
        last_layer_output = reduce(lambda acc, layer: self.gate(layer(acc)), self.layers, x)
        # Policy head
        action, log_probability = self.policy_head(self.gate(self.policy_head_layer(last_layer_output)))
        # Value head
        state_value = self.value_head_layer(last_layer_output)
        return {'action': action,
                'action_log_probability': log_probability,
                'state_value': state_value}

    def policy_head(self, last_layer_output):
        action_probabilities = F.softmax(last_layer_output)
        distribution = torch.distributions.Categorical(probs=action_probabilities)
        action = distribution.sample(sample_shape=(action_probabilities.size(0),))
        log_probability = distribution.log_prob(action)
        return action, log_probability


class A2CAlgorithm():

    def __init__(self, kwargs, model, optimizer=None, target_intr_model=None, predict_intr_model=None, sum_writer=None):
        '''
        TODO specify which values live inside of kwargs
        Refer to original paper for further explanation: https://arxiv.org/pdf/1707.06347.pdf
        horizon: (0, infinity) Number of timesteps that will elapse in between optimization calls.
        discount: (0,1) Reward discount factor
        use_gae: Flag, wether to use Generalized Advantage Estimation (GAE) (instead of return base estimation)
        gae_tau: (0,1) GAE hyperparameter.
        use_cuda: Flag, to specify whether to use CUDA tensors in Pytorch calculations
        entropy_weight: (0,1) Coefficient for (regularatization) entropy based loss
        gradient_clip: float, Clips gradients to reduce the chance of destructive updates
        optimization_epochs: int, Number of epochs per optimization step.
        mini_batch_size: int, Mini batch size to use to calculate losses (Use power of 2 for efficciency)
        learning_rate: float, optimizer learning rate.
        adam_eps: (float), Small Epsilon value used for ADAM optimizer. Prevents numerical instability when v^{hat} (Second momentum estimator) is near 0.
        model: (Pytorch nn.Module) Used to represent BOTH policy network and value network
        '''
        self.kwargs = deepcopy(kwargs)
        self.nbr_actor = self.kwargs['nbr_actor'] if 'nbr_actor' in self.kwargs else 1
        self.use_rnd = False
        if target_intr_model is not None and predict_intr_model is not None:
            self.use_rnd = True
            self.target_intr_model = target_intr_model
            self.predict_intr_model = predict_intr_model
            self.obs_mean = 0.0
            self.obs_std = 1.0
            self.running_counter_obs = 0
            self.update_period_obs = self.kwargs['rnd_update_period_running_meanstd_obs']
            self.int_reward_mean = 0.0
            self.int_reward_std = 1.0
            self.int_return_mean = 0.0
            self.int_return_std = 1.0
            self.running_counter_intrinsic_reward = 0
            self.update_period_intrinsic_reward = self.kwargs['rnd_update_period_running_meanstd_int_reward']
            self.running_counter_intrinsic_return = 0
            self.update_period_intrinsic_return = self.kwargs['rnd_update_period_running_meanstd_int_reward']

        self.running_counter_extrinsic_reward = 0
        self.ext_reward_mean = 0.0
        self.ext_reward_std = 1.0
            
        self.use_vae = False
        if 'use_vae' in self.kwargs and kwargs['use_vae']:
            self.use_vae = True

        self.model = model
        if self.kwargs['use_cuda']:
            self.model = self.model.cuda()
            if self.use_rnd:
                self.target_intr_model = self.target_intr_model.cuda()
                self.predict_intr_model = self.predict_intr_model.cuda()
        

        if optimizer is None:
            parameters = self.model.parameters()
            if self.use_rnd: parameters = list(parameters)+list(self.predict_intr_model.parameters())
            self.optimizer = optim.RMSprop(parameters, lr=kwargs['learning_rate'], eps=kwargs['optimizer_eps'], alpha=kwargs['optimizer_alpha'])
        else: self.optimizer = optimizer

        self.recurrent = False
        # TECHNICAL DEBT: check for recurrent property by looking at the modules in the model rather than relying on the kwargs that may contain
        # elements that do not concern the model trained by this algorithm, given that it is now use-able inside I2A...
        self.recurrent_nn_submodule_names = [hyperparameter for hyperparameter, value in self.kwargs.items() if isinstance(value, str) and 'RNN' in value]
        if len(self.recurrent_nn_submodule_names): self.recurrent = True

        self.storages = None
        self.reset_storages()

        global summary_writer
        summary_writer = sum_writer
        self.param_update_counter = 0

    def reset_storages(self, nbr_actor=None):
        if nbr_actor is not None:
            self.nbr_actor = nbr_actor

        if self.storages is not None:
            for storage in self.storages: storage.reset()

        self.storages = []
        for i in range(self.nbr_actor):
            self.storages.append(Storage())
            if self.recurrent:
                self.storages[-1].add_key('rnn_states')
                self.storages[-1].add_key('next_rnn_states')
            if self.use_rnd:
                self.storages[-1].add_key('int_r')
                self.storages[-1].add_key('int_v')
                self.storages[-1].add_key('int_ret')
                self.storages[-1].add_key('int_adv')
                self.storages[-1].add_key('target_int_f')

    def train(self):
        # Compute mean and std for ext reward:
        #self.running_counter_extrinsic_reward = 0
        #for idx, storage in enumerate(self.storages):
        #    self.update_ext_reward_mean_std(storage.r)
        
        # Compute Returns and Advantages:
        for idx, storage in enumerate(self.storages): 
            if len(storage) <= 1: continue
            storage.placeholder()
            self.compute_advantages_and_returns(storage_idx=idx)
            if self.use_rnd: 
                self.compute_int_advantages_and_int_returns(storage_idx=idx, non_episodic=self.kwargs['rnd_non_episodic_int_r'])
        
        # Update observations running mean and std: 
        if self.use_rnd: 
            for idx, storage in enumerate(self.storages): 
                if len(storage) <= 1: continue
                for ob in storage.s: self.update_obs_mean_std(ob)
        
                
        states, actions, next_states, log_probs_old, returns, advantages, int_returns, int_advantages, target_random_features, rnn_states = self.retrieve_values_from_storages()

        if self.recurrent: rnn_states = self.reformat_rnn_states(rnn_states)

        for it in range(self.kwargs['optimization_epochs']):
            self.optimize_model(states, actions, next_states, log_probs_old, returns, advantages, int_returns, int_advantages, target_random_features, rnn_states)

        self.reset_storages()

    def reformat_rnn_states(self, rnn_states):
        '''
        This function reformats the :param rnn_states: into 
        a dict of dict of list of batched rnn_states.
        :param rnn_states: list of dict of dict of list: each element is an rnn_state where:
            - the first dictionnary has the name of the recurrent module in the architecture
              as keys.
            - the second dictionnary has the keys 'hidden', 'cell'.
            - the items of this second dictionnary are lists of actual hidden/cell states for the GRU/LSTMBody.
        '''
        reformated_rnn_states = {k: {'hidden': [list()], 'cell': [list()]} for k in rnn_states[0]}
        for rnn_state in rnn_states:
            for k in rnn_state:
                hstates, cstates = rnn_state[k]['hidden'], rnn_state[k]['cell']
                for idx_layer, (h, c) in enumerate(zip(hstates, cstates)):
                    reformated_rnn_states[k]['hidden'][0].append(h)
                    reformated_rnn_states[k]['cell'][0].append(c)
        for k in reformated_rnn_states:
            hstates, cstates = reformated_rnn_states[k]['hidden'], reformated_rnn_states[k]['cell']
            hstates = torch.cat(hstates[0], dim=0)
            cstates = torch.cat(cstates[0], dim=0)
            reformated_rnn_states[k] = {'hidden': [hstates], 'cell': [cstates]}
        return reformated_rnn_states

    def normalize_ext_rewards(self, storage_idx):
        normalized_ext_rewards = []
        for i in range(len(self.storages[storage_idx])):
            #normalized_ext_rewards.append(self.storages[storage_idx].r[i] / (self.ext_reward_std+1e-8))
            # Proper normalization to standard gaussian:
            normalized_ext_rewards.append( (self.storages[storage_idx].r[i]-self.ext_reward_mean) / (self.ext_reward_std+1e-8))
        return normalized_ext_rewards

    def normalize_int_rewards(self, storage_idx):
        normalized_int_rewards = []
        for i in range(len(self.storages[storage_idx])):
            # Scaling alone:
            normalized_int_rewards.append(self.storages[storage_idx].int_r[i] / (self.int_reward_std+1e-8))
            #normalized_int_rewards.append(self.storages[storage_idx].int_r[i] / (self.int_return_std+1e-8))
        return normalized_int_rewards

    def compute_advantages_and_returns(self, storage_idx, non_episodic=False):
        ext_r = self.storages[storage_idx].r
        #norm_ext_r = self.normalize_ext_rewards(storage_idx)
        advantages = torch.from_numpy(np.zeros((1, 1), dtype=np.float32)) # TODO explain (used to be number of workers)
        returns = self.storages[storage_idx].v[-1].detach()
        gae = 0.0
        for i in reversed(range(len(self.storages[storage_idx])-1)):
            if not self.kwargs['use_gae']:
                if non_episodic:    notdone = 1.0
                else:               notdone = self.storages[storage_idx].non_terminal[i]
                returns = ext_r[i] + self.kwargs['discount'] * notdone * returns
                #returns = norm_ext_r[i] + self.kwargs['discount'] * notdone * returns
                advantages = returns - self.storages[storage_idx].v[i].detach()
            else:
                if non_episodic:    notdone = 1.0
                else:               notdone = self.storages[storage_idx].non_terminal[i]
                td_error = ext_r[i]  + self.kwargs['discount'] * notdone * self.storages[storage_idx].v[i + 1].detach() - self.storages[storage_idx].v[i].detach()
                #td_error = norm_ext_r[i]  + self.kwargs['discount'] * notdone * self.storages[storage_idx].v[i + 1].detach() - self.storages[storage_idx].v[i].detach()
                advantages = gae = td_error + self.kwargs['discount'] * self.kwargs['gae_tau'] * notdone * gae 
                returns = advantages + self.storages[storage_idx].v[i].detach()
            self.storages[storage_idx].adv[i] = advantages.detach()
            self.storages[storage_idx].ret[i] = returns.detach()

    def compute_int_advantages_and_int_returns(self, storage_idx, non_episodic=True):
        '''
        Compute intrinsic returns and advantages from normalized intrinsic rewards.
        Indeed, int_r values in storages have been normalized upon computation.
        At computation-time, updates of the running mean and std are performed too.
        '''
        norm_int_r = self.normalize_int_rewards(storage_idx)
        int_advantages = torch.from_numpy(np.zeros((1, 1), dtype=np.float32))
        int_returns = self.storages[storage_idx].int_v[-1].detach()
        gae = 0.0
        for i in reversed(range(len(self.storages[storage_idx])-1)):
            if not self.kwargs['use_gae']:
                if non_episodic:    notdone = 1.0
                else:               notdone = self.storages[storage_idx].non_terminal[i]
                int_returns = norm_int_r[i] + self.kwargs['intrinsic_discount'] * notdone * int_returns
                int_advantages = int_returns - self.storages[storage_idx].int_v[i].detach()
            else:
                if non_episodic:    notdone = 1.0
                else:               notdone = self.storages[storage_idx].non_terminal[i]
                td_error = norm_int_r[i]  + self.kwargs['intrinsic_discount'] * notdone * self.storages[storage_idx].int_v[i + 1].detach() - self.storages[storage_idx].int_v[i].detach()
                int_advantages = gae = td_error + self.kwargs['intrinsic_discount'] * self.kwargs['gae_tau'] * notdone * gae 
                int_returns = int_advantages + self.storages[storage_idx].int_v[i].detach()
            self.storages[storage_idx].int_adv[i] = int_advantages.detach()
            self.storages[storage_idx].int_ret[i] = int_returns.detach()

        self.update_int_return_mean_std(int_returns.detach().cpu())

    def retrieve_values_from_storages(self):
        full_states = []
        full_actions = []
        full_log_probs_old = []
        full_returns = []
        full_advantages = []
        full_rnn_states = None
        full_next_states = None
        full_int_returns = None
        full_int_advantages = None
        full_target_random_features = None
        if self.use_rnd:
            full_next_states = []
            full_int_returns = []
            full_int_advantages = []
            full_target_random_features = []
        if self.recurrent:
            full_rnn_states = []
            
        for storage in self.storages:
            # Check that there is something in the storage 
            if len(storage) <= 1: continue
            cat = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
            states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), cat)
            full_states.append(states)
            full_actions.append(actions)
            full_log_probs_old.append(log_probs_old)
            full_returns.append(returns)
            full_advantages.append(advantages)
            if self.use_rnd:
                cat = storage.cat(['succ_s', 'int_ret', 'int_adv', 'target_int_f'])
                next_states, int_returns, int_advantages, target_random_features = map(lambda x: torch.cat(x, dim=0), cat)
                full_next_states.append(next_states)
                full_int_returns.append(int_returns)
                full_int_advantages.append(int_advantages)
                full_target_random_features.append(target_random_features)
            if self.recurrent:
                rnn_states = storage.cat(['rnn_states'])[0]
                full_rnn_states += rnn_states
            
        full_states = torch.cat(full_states, dim=0)
        full_actions = torch.cat(full_actions, dim=0)
        full_log_probs_old = torch.cat(full_log_probs_old, dim=0)
        full_returns = torch.cat(full_returns, dim=0)
        full_advantages = torch.cat(full_advantages, dim=0)
        if self.kwargs['standardized_adv']:
            full_advantages = self.standardize(full_advantages).squeeze()
        else:
            full_advantages = full_advantages.squeeze()
        if self.use_rnd:
            full_next_states = torch.cat(full_next_states, dim=0)
            full_int_returns = torch.cat(full_int_returns, dim=0)
            full_int_advantages = torch.cat(full_int_advantages, dim=0)
            full_target_random_features = torch.cat(full_target_random_features, dim=0)
            if self.kwargs['standardized_adv']:
                full_int_advantages = self.standardize(full_int_advantages).squeeze()
            else:
                full_int_advantages = full_int_advantages.squeeze()
            
        return full_states, full_actions, full_next_states, full_log_probs_old, full_returns, full_advantages, full_int_returns, full_int_advantages, full_target_random_features, full_rnn_states

    def standardize(self, x):
        return (x - x.mean()) / (x.std()+1e-8)

    def compute_intrinsic_reward(self, states):
        normalized_states = (states-self.obs_mean) / (self.obs_std+1e-8) 
        if self.kwargs['rnd_obs_clip'] > 1e-3:
          normalized_states = torch.clamp( normalized_states, -self.kwargs['rnd_obs_clip'], self.kwargs['rnd_obs_clip'])
        if self.kwargs['use_cuda']: normalized_states = normalized_states.cuda()
        
        pred_features = self.predict_intr_model(normalized_states)
        target_features = self.target_intr_model(normalized_states)
        
        # Clamping:
        #pred_features = torch.clamp(pred_features, -1e20, 1e20)
        #target_features = torch.clamp(target_features, -1e20, 1e20)
        
        # Softmax:
        #pred_features = F.softmax(pred_features)
        #softmax_target_features = F.softmax(target_features)
        if torch.isnan(pred_features).long().sum().item() or torch.isnan(target_features).long().sum().item():
            import ipdb; ipdb.set_trace()
        #int_reward = torch.nn.functional.smooth_l1_loss(target_features,pred_features)
        int_reward = torch.nn.functional.mse_loss(target_features,pred_features)
        #int_reward = torch.nn.functional.mse_loss(softmax_target_features,pred_features)
        
        # No clipping on the intrinsic reward in the original paper:
        #int_reward = torch.clamp(int_reward, -1, 1)
        int_reward = int_reward.detach().cpu()
        self.update_int_reward_mean_std(int_reward)

        # Normalization will be done upon usage...
        # Kept intact here for logging purposes...        
        #int_r = int_reward / (self.int_reward_std+1e-8)

        return int_reward, target_features.detach().cpu()

    def update_ext_reward_mean_std(self, unnormalized_er_list):
        for unnormalized_er in unnormalized_er_list:
            rmean = self.ext_reward_mean
            rstd = self.ext_reward_std
            rc = self.running_counter_extrinsic_reward

            self.running_counter_extrinsic_reward += 1
            
            self.ext_reward_mean = (self.ext_reward_mean*rc+unnormalized_er)/self.running_counter_extrinsic_reward
            self.ext_reward_std = np.sqrt( ( np.power(self.ext_reward_std,2)*rc+np.power(unnormalized_er-rmean, 2) ) / self.running_counter_extrinsic_reward )
        
    def update_int_reward_mean_std(self, unnormalized_ir):
        rmean = self.int_reward_mean
        rstd = self.int_reward_std
        rc = self.running_counter_intrinsic_reward

        self.running_counter_intrinsic_reward += 1
        
        self.int_reward_mean = (self.int_reward_mean*rc+unnormalized_ir)/self.running_counter_intrinsic_reward
        self.int_reward_std = np.sqrt( ( np.power(self.int_reward_std,2)*rc+np.power(unnormalized_ir-rmean, 2) ) / self.running_counter_intrinsic_reward )
        
        if self.running_counter_intrinsic_reward >= self.update_period_intrinsic_reward:
          self.running_counter_intrinsic_reward = 0

    def update_int_return_mean_std(self, unnormalized_ir):
        rmean = self.int_return_mean
        rstd = self.int_return_std
        rc = self.running_counter_intrinsic_return

        self.running_counter_intrinsic_return += 1
        
        self.int_return_mean = (self.int_return_mean*rc+unnormalized_ir)/self.running_counter_intrinsic_return
        self.int_return_std = np.sqrt( ( np.power(self.int_return_std,2)*rc+np.power(unnormalized_ir-rmean, 2) ) / self.running_counter_intrinsic_return )
        
        if self.running_counter_intrinsic_return >= self.update_period_intrinsic_return:
          self.running_counter_intrinsic_return = 0

    def update_obs_mean_std(self, unnormalized_obs):
        rmean = self.obs_mean
        rstd = self.obs_std
        rc = self.running_counter_obs

        self.running_counter_obs += 1
        
        self.obs_mean = (self.obs_mean*rc+unnormalized_obs)/self.running_counter_obs
        self.obs_std = np.sqrt( ( np.power(self.obs_std,2)*rc+np.power(unnormalized_obs-rmean, 2) ) / self.running_counter_obs )
        
        if self.running_counter_obs >= self.update_period_obs:
          self.running_counter_obs = 0

    def optimize_model(self, states, actions, next_states, log_probs_old, returns, advantages, int_returns, int_advantages, target_random_features, rnn_states=None):
        global summary_writer
        # What is this: create dictionary to store length of each part of the recurrent submodules of the current model
        nbr_layers_per_rnn = None
        if self.recurrent:
            nbr_layers_per_rnn = {recurrent_submodule_name: len(rnn_states[recurrent_submodule_name]['hidden'])
                                  for recurrent_submodule_name in rnn_states}

        #----------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Mid-level Old Policy Prediction:
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------
        '''
        sampler = random_sample(np.arange(advantages.size(0)), self.kwargs['mini_batch_size'])
        with torch.no_grad():
            for batch_indices in sampler:
                batch_indices = torch.from_numpy(batch_indices).long()
                
                sampled_rnn_states = None
                if self.recurrent:
                    sampled_rnn_states = self.calculate_rnn_states_from_batch_indices(rnn_states, batch_indices, nbr_layers_per_rnn)

                sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
                sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices].cuda() if self.kwargs['use_cuda'] else log_probs_old[batch_indices]
                
                sampled_states = sampled_states.detach()
                sampled_actions = sampled_actions.detach()
            
                old_prediction = self.model(sampled_states, sampled_actions, rnn_states=rnn_states)
                log_probs_old[batch_indices] = old_prediction['log_pi_a'].cpu()
        '''
        #----------------------------------------------------------------------------------------------------------------------------------------------------------------

        sampler = random_sample(np.arange(advantages.size(0)), self.kwargs['mini_batch_size'])
        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()
            
            sampled_rnn_states = None
            if self.recurrent:
                sampled_rnn_states = self.calculate_rnn_states_from_batch_indices(rnn_states, batch_indices, nbr_layers_per_rnn)

            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_log_probs_old = log_probs_old[batch_indices].cuda() if self.kwargs['use_cuda'] else log_probs_old[batch_indices]
            sampled_returns = returns[batch_indices].cuda() if self.kwargs['use_cuda'] else returns[batch_indices]
            sampled_advantages = advantages[batch_indices].cuda() if self.kwargs['use_cuda'] else advantages[batch_indices]
                
            sampled_states = sampled_states.detach()
            sampled_actions = sampled_actions.detach()
            sampled_log_probs_old = sampled_log_probs_old.detach()
            sampled_returns = sampled_returns.detach()
            sampled_advantages = sampled_advantages.detach()

            if self.use_rnd:
                sampled_next_states = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]
                sampled_next_states = sampled_next_states.detach()
                sampled_int_returns = int_returns[batch_indices].cuda() if self.kwargs['use_cuda'] else int_returns[batch_indices]
                sampled_int_advantages = int_advantages[batch_indices].cuda() if self.kwargs['use_cuda'] else int_advantages[batch_indices]
                sampled_target_random_features = target_random_features[batch_indices].cuda() if self.kwargs['use_cuda'] else target_random_features[batch_indices]
                sampled_int_returns = sampled_int_returns.detach()
                sampled_int_advantages = sampled_int_advantages.detach()
                sampled_target_random_features = sampled_target_random_features.detach()
                states_mean = self.obs_mean.cuda() if self.kwargs['use_cuda'] else self.obs_mean
                states_std = self.obs_std.cuda() if self.kwargs['use_cuda'] else self.obs_std

            self.optimizer.zero_grad()
            loss = a2c_loss.compute_loss(sampled_states, 
                                         sampled_actions, 
                                         sampled_returns, 
                                         sampled_advantages, 
                                         rnn_states=sampled_rnn_states,
                                         entropy_weight=self.kwargs['entropy_weight'],
                                         model=self.model,
                                         iteration_count=self.param_update_counter,
                                         summary_writer=summary_writer)

            loss.backward(retain_graph=False)
            if self.kwargs['gradient_clip'] > 1e-3:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs['gradient_clip'])
            self.optimizer.step()

            if summary_writer is not None:
                self.param_update_counter += 1 
                '''
                for name, param in self.model.named_parameters():
                    if hasattr(param, 'grad') and param.grad is not None:
                        summary_writer.add_histogram(f"Training/{name}", param.grad.cpu(), self.param_update_counter)
                '''
                if self.use_rnd:
                    summary_writer.add_scalar('Training/IntReturnMean', self.int_return_mean.cpu().item(), self.param_update_counter)
                    summary_writer.add_scalar('Training/IntReturnStd', self.int_return_std.cpu().item(), self.param_update_counter)
        

    def calculate_rnn_states_from_batch_indices(self, rnn_states, batch_indices, nbr_layers_per_rnn):
        sampled_rnn_states = {k: {'hidden': [None]*nbr_layers_per_rnn[k], 'cell': [None]*nbr_layers_per_rnn[k]} for k in rnn_states}
        for recurrent_submodule_name in sampled_rnn_states:
            for idx in range(nbr_layers_per_rnn[recurrent_submodule_name]):
                sampled_rnn_states[recurrent_submodule_name]['hidden'][idx] = rnn_states[recurrent_submodule_name]['hidden'][idx][batch_indices].cuda() if self.kwargs['use_cuda'] else rnn_states[recurrent_submodule_name]['hidden'][idx][batch_indices]
                sampled_rnn_states[recurrent_submodule_name]['cell'][idx]   = rnn_states[recurrent_submodule_name]['cell'][idx][batch_indices].cuda() if self.kwargs['use_cuda'] else rnn_states[recurrent_submodule_name]['cell'][idx][batch_indices]
        return sampled_rnn_states

    @staticmethod
    def check_mandatory_kwarg_arguments(kwargs: dict):
        '''
        Checks that all mandatory hyperparameters are present
        inside of dictionary :param kwargs:

        :param kwargs: Dictionary of hyperparameters
        '''
        # Future improvement: add a condition to check_kwarg (discount should be between (0:1])
        keywords = ['horizon', 'discount', 'use_gae', 'gae_tau', 'use_cuda',
                    'entropy_weight', 'gradient_clip', 'optimization_epochs',
                    'mini_batch_size', 'learning_rate', 'adam_eps']

        def check_kwarg_and_condition(keyword, kwargs):
            if keyword not in kwargs:
                raise ValueError(f"Keyword: '{keyword}' not found in kwargs")
        for keyword in keywords: check_kwarg_and_condition(keyword, kwargs)
