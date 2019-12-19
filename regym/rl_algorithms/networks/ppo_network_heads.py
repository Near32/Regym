#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ppo_network_utils import BaseNet, layer_init, tensor
from .ppo_network_bodies import DummyBody


class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.get_feature_shape(), output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        return y


class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.get_feature_shape(), 1))
        self.fc_advantage = layer_init(nn.Linear(body.get_feature_shape(), action_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return q


class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.get_feature_shape(), action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob


class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.get_feature_shape(), action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return quantiles


class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.get_feature_shape(), num_options))
        self.fc_pi = layer_init(nn.Linear(body.get_feature_shape(), num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.get_feature_shape(), num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        return q, beta, log_pi


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body, use_intrinsic_critic=False):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.get_feature_shape())
        if critic_body is None: critic_body = DummyBody(phi_body.get_feature_shape())
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.get_feature_shape(), action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.get_feature_shape(), 1), 1e0)

        self.use_intrinsic_critic = use_intrinsic_critic
        self.fc_int_critic = None
        if self.use_intrinsic_critic: self.fc_int_critic = layer_init(nn.Linear(critic_body.get_feature_shape(), 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        if self.use_intrinsic_critic: self.critic_params += list(self.fc_int_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

        print(self)
        for name, param in self.named_parameters():
            print(name, param.shape)

class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        return torch.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        return self.network.fc_critic(self.network.critic_body(phi, a))


class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 use_intrinsic_critic=False):
        super(GaussianActorCriticNet, self).__init__()
        self.use_intrinsic_critic = use_intrinsic_critic
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body,use_intrinsic_critic)
        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, action=None, rnn_states=None):
        obs = tensor(obs)
        next_rnn_states = None 
        if rnn_states is not None:
            next_rnn_states = {k: None for k in rnn_states}

        if rnn_states is not None and 'phi_arch' in rnn_states:
            phi, next_rnn_states['phi_arch'] = self.network.phi_body( (obs, rnn_states['phi_arch']) )
        else:
            phi = self.network.phi_body(obs)

        if rnn_states is not None and 'actor_arch' in rnn_states:
            phi_a, next_rnn_states['actor_arch'] = self.network.actor_body( (phi, rnn_states['actor_arch']) )
        else:
            phi_a = self.network.actor_body(phi)

        if rnn_states is not None and 'critic_arch' in rnn_states:
            phi_v, next_rnn_states['critic_arch'] = self.network.critic_body( (phi, rnn_states['critic_arch']) )
        else:
            phi_v = self.network.critic_body(phi)

        mean = torch.tanh(self.network.fc_action(phi_a))
        # batch x num_action
        v = self.network.fc_critic(phi_v)
        if self.use_intrinsic_critic:
            int_v = self.network.fc_int_critic(phi_v)
        # batch x 1
        
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        # Log likelyhood of action = sum_i dist.log_prob(action[i])
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        # batch x 1
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        # batch x 1

        prediction = {'a': action,
                    'log_pi_a': log_prob,
                    'ent': entropy,
                    'v': v}
        
        if self.use_intrinsic_critic:
            prediction['int_v'] = int_v

        if rnn_states is not None:
            prediction.update({'rnn_states': rnn_states,
                               'next_rnn_states': next_rnn_states})

        return prediction



# Categorical
'''
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
'''

class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 use_intrinsic_critic=False):
        super(CategoricalActorCriticNet, self).__init__()
        self.use_intrinsic_critic = use_intrinsic_critic
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body,use_intrinsic_critic)

    def forward(self, obs, action=None, rnn_states=None):
        obs = tensor(obs)
        next_rnn_states = None 
        if rnn_states is not None:
            next_rnn_states = {k: None for k in rnn_states}

        if rnn_states is not None and 'phi_arch' in rnn_states:
            phi, next_rnn_states['phi_arch'] = self.network.phi_body( (obs, rnn_states['phi_arch']) )
        else:
            phi = self.network.phi_body(obs)

        if rnn_states is not None and 'actor_arch' in rnn_states:
            phi_a, next_rnn_states['actor_arch'] = self.network.actor_body( (phi, rnn_states['actor_arch']) )
        else:
            phi_a = self.network.actor_body(phi)

        if rnn_states is not None and 'critic_arch' in rnn_states:
            phi_v, next_rnn_states['critic_arch'] = self.network.critic_body( (phi, rnn_states['critic_arch']) )
        else:
            phi_v = self.network.critic_body(phi)

        logits = self.network.fc_action(phi_a)
        probs = F.softmax( logits, dim=-1 )
        #https://github.com/pytorch/pytorch/issues/7014
        probs = torch.clamp(probs, -1e10, 1e10)
        
        # batch x action_dim
        v = self.network.fc_critic(phi_v)
        if self.use_intrinsic_critic:
            int_v = self.network.fc_int_critic(phi_v)
        # batch x 1

        '''
        '''

        batch_size = logits.size(0)
        
        '''
        # probs:
        dists = torch.distributions.Categorical(probs=probs)
        
        if action is None:
            #action = dists.sample()#.unsqueeze(1)
            p = probs.detach().cpu().numpy()
            axis = 1
            r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
            action = (p.cumsum(axis=axis) > r).argmax(axis=axis)
            action = torch.from_numpy(action).to(probs.device)
            # batch #x 1

        log_prob = dists.log_prob(action)
        # batch #x 1
        entropy = dists.entropy().unsqueeze(1)
        # batch #x 1

        '''
        # probs:
        log_probs = F.log_softmax(logits, dim=-1)
        #entropy = dists.entropy().unsqueeze(1)
        entropy = -(log_probs * probs).sum(1)#, keepdim=True)
        # batch #x 1
        
        if action is None:
            action = probs.multinomial(num_samples=1).squeeze(1)
            # batch #x 1
            '''
            p = probs.detach().cpu().numpy()
            axis = 1
            r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
            action = (p.cumsum(axis=axis) > r).argmax(axis=axis)
            action = torch.from_numpy(action).to(probs.device)
            # batch #x 1
            '''
            
        #log_prob = dists.log_prob(action)
        log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        # batch #x 1
        
        
        prediction = {'a': action,
                    'log_pi_a': log_probs,
                    'action_logits': logits,
                    'ent': entropy,
                    'v': v}
        
        if self.use_intrinsic_critic:
            prediction['int_v'] = int_v

        if rnn_states is not None:
            prediction.update({'rnn_states': rnn_states,
                               'next_rnn_states': next_rnn_states})

        return prediction


# class CnnActorCriticNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(CnnActorCriticNetwork, self).__init__()
#         self.conv1 = nn.Conv2d(state_dim[0], 32, 8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
#         self.fc1 = nn.Linear(7*7*64, 512)
#         self.fc2 = nn.Linear(512, 512)

#         self.actor = nn.Linear(512, action_dim)
#         self.critic = nn.Linear(512, 1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(-1, 7*7*64)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         policy = self.actor(x)
#         value = self.critic(x)
#         return policy, value

# class CategoricalActorCriticNet(nn.Module, BaseNet):
#     def __init__(self,
#                  state_dim,
#                  action_dim,
#                  phi_body=None,
#                  actor_body=None,
#                  critic_body=None,
#                  use_intrinsic_critic=False):
#         super(CategoricalActorCriticNet, self).__init__()
#         '''
#         self.use_intrinsic_critic = use_intrinsic_critic
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body,use_intrinsic_critic)
#         '''
#         self.model = CnnActorCriticNetwork(state_dim=state_dim, action_dim=action_dim)
        
#     def forward(self, obs, action=None, rnn_states=None):
#         obs = tensor(obs)
        
#         phi_a, v = self.model(obs)
#         logits = F.softmax( phi_a, dim=-1 )
#         # batch x action_dim
        
#         batch_size = logits.size(0)

#         # logits = log-odds:
#         dists = torch.distributions.Categorical(logits=logits)
        
#         if action is None:
#             action = dists.sample()#.unsqueeze(1)
#             # batch #x 1

#         log_prob = dists.log_prob(action)
#         # batch #x 1
#         entropy = dists.entropy().unsqueeze(1)
#         # batch #x 1

#         prediction = {'a': action,
#                     'log_pi_a': log_prob,
#                     'action_logits': logits,
#                     'ent': entropy,
#                     'v': v}

#         return prediction


class CategoricalActorCriticVAENet(CategoricalActorCriticNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 use_intrinsic_critic=False):
        super(CategoricalActorCriticVAENet, self).__init__(state_dim=state_dim,
                                                           action_dim=action_dim,
                                                           phi_body=phi_body,
                                                           actor_body=actor_body,
                                                           critic_body=critic_body,
                                                           use_intrinsic_critic=use_intrinsic_critic)

    def compute_vae_loss(self, states):
        return self.network.phi_body.compute_vae_loss(states)