import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import layer_init
from .bodies import DummyBody, NoisyLinear, reset_noisy_layer

import numpy as np

EPS = 1e-8


class DuelingLayer(nn.Module):
    def __init__(self, input_dim, action_dim, layer_fn=nn.Linear):
        super(DuelingLayer, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

        self.advantage = layer_init(layer_fn(self.input_dim, self.action_dim), 1e0)
        self.value = layer_init(layer_fn(self.input_dim, 1), 1e0)

    def forward(self, fx):
        v = self.value(fx)
        adv = self.advantage(fx)
        return v.expand_as(adv) + (adv - adv.mean(1, keepdim=True).expand_as(adv))

class InstructionPredictor(nn.Module):
    def __init__(self,
                 encoder,
                 decoder):
        super(InstructionPredictor, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        if next(self.encoder.parameters()).is_cuda:
            x = x.cuda()
        x = self.encoder(x)
        x = self.decoder(x)
        return x 

    def compute_loss(self, x, goal):
        x = self.encoder(x)
        output_dict = self.decoder(x, gt_sentences=goal)
        return output_dict

class CategoricalQNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 critic_body=None,
                 dueling=False,
                 noisy=False,
                 goal_oriented=False,
                 goal_shape=None,
                 goal_phi_body=None):
        super(CategoricalQNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        self.noisy = noisy 
        self.goal_oriented = goal_oriented

        if phi_body is None: phi_body = DummyBody(state_dim)
        self.phi_body = phi_body
        
        critic_input_shape = self.phi_body.get_feature_shape()
        
        if self.goal_oriented:
            self.goal_state_flattening = False
            assert(goal_shape is not None)
            if goal_phi_body is None:   
                if goal_shape == state_dim:
                    goal_phi_body = self.phi_body
                else:
                    self.goal_state_flattening = True
            self.goal_phi_body = goal_phi_body

            if not(self.goal_state_flattening):
                critic_input_shape += self.goal_phi_body.get_feature_shape()
        
        
        if critic_body is None: critic_body = DummyBody(critic_input_shape) 
        self.critic_body = critic_body

        fc_critic_input_shape = self.critic_body.get_feature_shape()
        layer_fn = nn.Linear 
        if self.noisy:  layer_fn = NoisyLinear
        if self.dueling:
            self.fc_critic = DuelingLayer(input_dim=fc_critic_input_shape, action_dim=self.action_dim, layer_fn=layer_fn)
        else:
            self.fc_critic = layer_init(layer_fn(fc_critic_input_shape, self.action_dim), 1e0)

    def reset_noise(self):
        self.apply(reset_noisy_layer)

    def forward(self, obs, action=None, rnn_states=None, goal=None):
        if not(self.goal_oriented):  assert(goal==None)
        
        if self.goal_oriented:
            if self.goal_state_flattening:
                obs = torch.cat([obs, goal], dim=1)

        next_rnn_states = None 
        if rnn_states is not None:
            next_rnn_states = {k: None for k in rnn_states}

        if rnn_states is not None and 'phi_body' in rnn_states:
            phi, next_rnn_states['phi_body'] = self.phi_body( (obs, rnn_states['phi_body']) )
        else:
            phi = self.phi_body(obs)

        gphi = None
        if self.goal_oriented and not(self.goal_state_flattening):
            if rnn_states is not None and 'goal_phi_body' in rnn_states:
                gphi, next_rnn_states['goal_phi_body'] = self.goal_phi_body( (goal, rnn_states['goal_phi_body']) )
            else:
                gphi = self.phi_body(goal)

            phi = torch.cat([phi, gphi], dim=1)


        if rnn_states is not None and 'critic_body' in rnn_states:
            phi_v, next_rnn_states['critic_body'] = self.critic_body( (phi, rnn_states['critic_body']) )
        else:
            phi_v = self.critic_body(phi)

        phi_features = phi_v
        
        # batch x action_dim
        qa = self.fc_critic(phi_features)     

        if action is None:
            action  = qa.max(dim=-1)[1]
        # batch #x 1
        
        probs = F.softmax( qa, dim=-1 )
        log_probs = torch.log(probs+EPS)
        entropy = -torch.sum(probs*log_probs, dim=-1)
        # batch #x 1
        
        prediction = {'a': action,
                    'ent': entropy,
                    'qa': qa}
        
        if rnn_states is not None:
            prediction.update({'rnn_states': rnn_states,
                               'next_rnn_states': next_rnn_states})

        return prediction


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


class GaussianActorCriticNet(nn.Module):
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
        next_rnn_states = None 
        if rnn_states is not None:
            next_rnn_states = {k: None for k in rnn_states}

        if rnn_states is not None and 'phi_body' in rnn_states:
            phi, next_rnn_states['phi_body'] = self.network.phi_body( (obs, rnn_states['phi_body']) )
        else:
            phi = self.network.phi_body(obs)

        if rnn_states is not None and 'actor_body' in rnn_states:
            phi_a, next_rnn_states['actor_body'] = self.network.actor_body( (phi, rnn_states['actor_body']) )
        else:
            phi_a = self.network.actor_body(phi)

        if rnn_states is not None and 'critic_body' in rnn_states:
            phi_v, next_rnn_states['critic_body'] = self.network.critic_body( (phi, rnn_states['critic_body']) )
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


class CategoricalActorCriticNet(nn.Module):
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
        global EPS
        next_rnn_states = None 
        if rnn_states is not None:
            next_rnn_states = {k: None for k in rnn_states}

        if rnn_states is not None and 'phi_body' in rnn_states:
            phi, next_rnn_states['phi_body'] = self.network.phi_body( (obs, rnn_states['phi_body']) )
        else:
            phi = self.network.phi_body(obs)

        if rnn_states is not None and 'actor_body' in rnn_states:
            phi_a, next_rnn_states['actor_body'] = self.network.actor_body( (phi, rnn_states['actor_body']) )
        else:
            phi_a = self.network.actor_body(phi)

        if rnn_states is not None and 'critic_body' in rnn_states:
            phi_v, next_rnn_states['critic_body'] = self.network.critic_body( (phi, rnn_states['critic_body']) )
        else:
            phi_v = self.network.critic_body(phi)

        logits = self.network.fc_action(phi_a)
        probs = F.softmax( logits, dim=-1 )
        #https://github.com/pytorch/pytorch/issues/7014
        #probs = torch.clamp(probs, -1e10, 1e10)
        
        # batch x action_dim
        v = self.network.fc_critic(phi_v)
        if self.use_intrinsic_critic:
            int_v = self.network.fc_int_critic(phi_v)
        # batch x 1

        batch_size = logits.size(0)
        
        '''
        # RND1
        # probs:
        dists = torch.distributions.categorical.Categorical(probs=probs)
        
        if action is None:
            action = dists.sample()#.unsqueeze(1)
            # batch #x 1
        log_prob = dists.log_prob(action)
        # batch #x 1
        entropy = dists.entropy().unsqueeze(1)
        # batch #x 1
        '''

        '''
        '''
        # NORMAL:
        #log_probs = F.log_softmax(logits, dim=-1)
        log_probs = torch.log(probs+EPS)
        entropy = -torch.sum(probs*log_probs, dim=-1)#, keepdim=True)
        # batch #x 1
        
        if action is None:
            #action = (probs+EPS).multinomial(num_samples=1).squeeze(1)
            action = torch.multinomial( probs, num_samples=1).squeeze(1)
            # batch #x 1
        log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        # batch #x 1
        '''
        '''

        '''
        #RND2:
        # probs:
        log_probs = F.log_softmax(logits, dim=-1)
        #entropy = dists.entropy().unsqueeze(1)
        entropy = -(log_probs * probs).sum(1)#, keepdim=True)
        # batch #x 1
        
        if action is None:
            p = probs.detach().cpu().numpy()
            axis = 1
            r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
            action = (p.cumsum(axis=axis) > r).argmax(axis=axis)
            action = torch.from_numpy(action).to(probs.device)
            # batch #x 1
        log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        # batch #x 1
        '''

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