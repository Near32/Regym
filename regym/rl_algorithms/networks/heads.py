from typing import Dict 

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import layer_init
from .bodies import DummyBody, NoisyLinear, reset_noisy_layer
from regym.rl_algorithms.utils import extract_subtree, copy_hdict

import numpy as np

EPS = 1e-8


class DuelingLayer(nn.Module):
    def __init__(self, input_dim, action_dim, layer_fn=nn.Linear, layer_init_fn=layer_init):
        super(DuelingLayer, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

        self.advantage = layer_fn(self.input_dim, self.action_dim)
        self.value = layer_fn(self.input_dim, 1)

        if layer_init_fn is not None:
            self.apply(layer_init_fn)

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
    def __init__(
        self,
        state_dim,
        action_dim,
        phi_body=None,
        critic_body=None,
        dueling=False,
        noisy=False,
        goal_oriented=False,
        goal_shape=None,
        goal_phi_body=None,
        layer_init_fn=layer_init,
        extra_inputs_infos: Dict={},
        extra_bodies: Dict={}):
        """
        :param extra_inputs_infos: Dictionnary containing the shape of the lstm-relevant extra inputs.
        """
        super(CategoricalQNet, self).__init__()

        self.greedy = True

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        self.noisy = noisy 
        self.goal_oriented = goal_oriented
        self.extra_bodies = extra_bodies

        if phi_body is None: phi_body = DummyBody(state_dim)
        self.phi_body = phi_body
        
        critic_input_shape = self.phi_body.get_feature_shape()
        if len(self.extra_bodies):
            for extra_body in self.extra_bodies.values():
                critic_input_shape += extra_body.get_feature_shape()

        self.goal_oriented = False 
        """
        # depr: goal update
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
        """
        
        if critic_body is None: critic_body = DummyBody(critic_input_shape) 
        self.critic_body = critic_body

        fc_critic_input_shape = self.critic_body.get_feature_shape()
        
        if isinstance(fc_critic_input_shape, list):
            fc_critic_input_shape = fc_critic_input_shape[-1]

        for key in extra_inputs_infos:
            shape = extra_inputs_infos[key]['shape']
            assert len(shape) == 1 
            fc_critic_input_shape += shape[-1]
        
        layer_fn = nn.Linear 
        if self.noisy:  layer_fn = NoisyLinear
        if self.dueling:
            self.fc_critic = DuelingLayer(input_dim=fc_critic_input_shape, action_dim=self.action_dim, layer_fn=layer_fn)
        else:
            self.fc_critic = layer_fn(fc_critic_input_shape, self.action_dim)
            if layer_init_fn is not None:
                self.fc_critic = layer_init_fn(self.fc_critic, 1e0)

    def reset_noise(self):
        self.apply(reset_noisy_layer)

    def forward(self, obs, action=None, rnn_states=None, goal=None):
        batch_size = obs.shape[0]

        """
        # depr : goal update
        if not(self.goal_oriented):  assert(goal==None)
        
        if self.goal_oriented:
            if self.goal_state_flattening:
                obs = torch.cat([obs, goal], dim=1)
        """

        next_rnn_states = None 
        if rnn_states is not None:
            next_rnn_states = copy_hdict(rnn_states)

        if rnn_states is not None and 'phi_body' in rnn_states:
            phi, next_rnn_states['phi_body'] = self.phi_body( (obs, rnn_states['phi_body']) )
        else:
            phi = self.phi_body(obs)

        """
        # depr: goal update
        gphi = None
        if self.goal_oriented and not(self.goal_state_flattening):
            import ipdb; ipdb.set_trace()
            if rnn_states is not None and 'goal_phi_body' in rnn_states:
                gphi, next_rnn_states['goal_phi_body'] = self.goal_phi_body( (goal, rnn_states['goal_phi_body']) )
            else:
                gphi = self.phi_body(goal)

            phi = torch.cat([phi, gphi], dim=1)
        """
        extra_outputs = {}
        import ipdb; ipdb.set_trace()
        for extra_body_id, extra_body in self.extra_bodies.items():
            if rnn_states is not None and extra_body_id in rnn_states:
                extra_outputs[extra_body_id], \
                rnn_states[extra_body_id] = extra_body((obs, rnn_states[extra_body]))
            else:
                extra_outputs[extra_body_id] = extra_body(obs)
        
        if len(extra_outputs):
            # Concatenate with phi output:
            extra_outputs = [v[0].to(phi.dtype).to(phi.device) for v in extra_outputs.values()]
            phi = torch.cat([phi]+extra_outputs, dim=-1)

        if rnn_states is not None and 'critic_body' in rnn_states:
            phi_v, next_rnn_states['critic_body'] = self.critic_body( (phi, rnn_states['critic_body']) )
        else:
            phi_v = self.critic_body(phi)

        phi_features = phi_v
        
        if 'final_critic_layer' in rnn_states:
            extra_inputs = extract_subtree(
                in_dict=rnn_states['final_critic_layer'],
                node_id='extra_inputs',
            )
            
            extra_inputs = [v[0].to(phi_features.dtype).to(phi_features.device) for v in extra_inputs.values()]
            if len(extra_inputs): phi_features = torch.cat([phi_features]+extra_inputs, dim=-1)
        
        qa = self.fc_critic(phi_features)     
        # batch x action_dim

        legal_actions = torch.ones_like(qa)
        if 'head' in rnn_states \
        and 'extra_inputs' in rnn_states['head'] \
        and 'legal_actions' in rnn_states['head']['extra_inputs']:
            legal_actions = rnn_states['head']['extra_inputs']['legal_actions'][0]
            next_rnn_states['head'] = rnn_states['head']
        legal_actions = legal_actions.to(qa.device)
        
        # The following accounts for player dimension if VDN:
        legal_qa = (1+qa-qa.min(dim=-1, keepdim=True)[0]) * legal_actions
        
        if action is None:
            if self.greedy:
                action  = legal_qa.max(dim=-1, keepdim=True)[1]
            else:
                action = torch.multinomial(legal_qa.softmax(dim=-1), num_samples=1) #.reshape((batch_size,))
        # batch #x 1
        
        # batch #x 1
        
        probs = F.softmax( qa, dim=-1 )
        log_probs = torch.log(probs+EPS)
        entropy = -torch.sum(probs*log_probs, dim=-1)
        # batch #x 1
        
        legal_probs = F.softmax( legal_qa, dim=-1 )
        legal_log_probs = torch.log(legal_probs+EPS)
        
        prediction = {
            'a': action,
            'ent': entropy,
            'qa': qa,
            'log_a': legal_log_probs,
        }
        
        prediction.update({
            'rnn_states': rnn_states,
            'next_rnn_states': next_rnn_states
        })

        return prediction

    def get_torso(self):
        def torso_forward(obs, action=None, rnn_states=None, goal=None):
            if not(self.goal_oriented):  assert(goal==None)
            
            if self.goal_oriented:
                if self.goal_state_flattening:
                    obs = torch.cat([obs, goal], dim=1)

            next_rnn_states = None 
            if rnn_states is not None:
                next_rnn_states = copy_hdict(rnn_states)

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

            return phi, next_rnn_states

        return torso_forward

    def get_head(self):
        def head_forward(phi, action=None, rnn_states=None, goal=None):
            batch_size = phi.shape[0]
            
            next_rnn_states = None 
            if rnn_states is not None:
                next_rnn_states = copy_hdict(rnn_states)

            if rnn_states is not None and 'critic_body' in rnn_states:
                phi_v, next_rnn_states['critic_body'] = self.critic_body( (phi, rnn_states['critic_body']) )
            else:
                phi_v = self.critic_body(phi)

            phi_features = phi_v
            
            if 'final_critic_layer' in rnn_states:
                extra_inputs = extract_subtree(
                    in_dict=rnn_states['final_critic_layer'],
                    node_id='extra_inputs',
                )
                
                extra_inputs = [v[0].to(phi_features.dtype).to(phi_features.device) for v in extra_inputs.values()]
                if len(extra_inputs): phi_features = torch.cat([phi_features]+extra_inputs, dim=-1)
            
            qa = self.fc_critic(phi_features)     
            # batch x action_dim
            
            legal_actions = torch.ones_like(qa)
            if 'head' in rnn_states and 'extra_inputs' in rnn_states['head'] and 'legal_actions' in rnn_states['head']['extra_inputs']:
                legal_actions = rnn_states['head']['extra_inputs']['legal_actions'][0]
                next_rnn_states['head'] = rnn_states['head']
            
            # The following accounts for player dimension if VDN:
            legal_qa = (1+qa-qa.min(dim=-1, keepdim=True)[0]) * legal_actions
            
            if action is None:
                if self.greedy:
                    action  = legal_qa.max(dim=-1, keepdim=True)[1]
                else:
                    action = torch.multinomial(legal_qa.softmax(dim=-1), num_samples=1) #.reshape((batch_size,))
            # batch #x 1
            
            probs = F.softmax( qa, dim=-1 )
            log_probs = torch.log(probs+EPS)
            entropy = -torch.sum(probs*log_probs, dim=-1)
            # batch #x 1
            
            legal_probs = F.softmax( legal_qa, dim=-1 )
            legal_log_probs = torch.log(legal_probs+EPS)
            
            prediction = {
                'a': action,
                'ent': entropy,
                'qa': qa,
                'log_a': legal_log_probs,
            }
            
            prediction.update({
                'rnn_states': rnn_states,
                'next_rnn_states': next_rnn_states
            })

            return prediction

        return head_forward

class QNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 critic_body,
                 phi_body=None,
                 action_phi_body=None,
                 noisy=False,
                 goal_oriented=False,
                 goal_shape=None,
                 goal_phi_body=None,
                 layer_init_fn=layer_init,
                 init_w=3e-3):
        super(QNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.noisy = noisy 
        self.goal_oriented = goal_oriented

        if phi_body is None: phi_body = DummyBody(state_dim)
        self.phi_body = phi_body
        
        if action_phi_body is None: action_phi_body = DummyBody(self.action_dim)
        self.action_phi_body = action_phi_body
        
        critic_input_shape = self.phi_body.get_feature_shape()+self.action_phi_body.get_feature_shape()
        
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
        
        self.critic_body = critic_body

        fc_critic_input_shape = self.critic_body.get_feature_shape()
        layer_fn = nn.Linear 
        if self.noisy:  layer_fn = NoisyLinear
        self.fc_critic = layer_fn(fc_critic_input_shape, 1)

        if layer_init_fn is not None:
            self.fc_critic = layer_init_fn(self.fc_critic, 1e0)
        else:
            self.fc_critic.weight.data.uniform_(-init_w, init_w)
            self.fc_critic.bias.data.uniform_(-init_w, init_w)

    def reset_noise(self):
        self.apply(reset_noisy_layer)

    def forward(self, obs, action, rnn_states=None, goal=None):
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

        if rnn_states is not None and 'action_phi_body' in rnn_states:
            phi, next_rnn_states['action_phi_body'] = self.action_phi_body( (action, rnn_states['action_phi_body']) )
        else:
            action_phi = self.action_phi_body(action)

        critic_input = torch.cat([phi, action_phi], dim=-1)

        gphi = None
        if self.goal_oriented and not(self.goal_state_flattening):
            if rnn_states is not None and 'goal_phi_body' in rnn_states:
                gphi, next_rnn_states['goal_phi_body'] = self.goal_phi_body( (goal, rnn_states['goal_phi_body']) )
            else:
                gphi = self.phi_body(goal)

            critic_input = torch.cat([critic_input, gphi], dim=-1)

        if rnn_states is not None and 'critic_body' in rnn_states:
            critic_output, next_rnn_states['critic_body'] = self.critic_body( (critic_input, rnn_states['critic_body']) )
        else:
            critic_output = self.critic_body(critic_input)
        
        # batch x action_dim
        qa = self.fc_critic(critic_output)

        prediction = {
            'a': action,
            'qa': qa
        }
        
        prediction.update({'rnn_states': rnn_states,
                               'next_rnn_states': next_rnn_states})

        return prediction

class EnsembleQNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 critic_body,
                 phi_body=None,
                 action_phi_body=None,
                 noisy=False,
                 goal_oriented=False,
                 goal_shape=None,
                 goal_phi_body=None,
                 nbr_models=2,
                 layer_init_fn=layer_init):
        super(EnsembleQNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.noisy = noisy 
        self.goal_oriented = goal_oriented
        self.nbr_models = nbr_models 

        self.inner_models = nn.ModuleList([
            QNet(state_dim=state_dim,
                 action_dim=action_dim,
                 critic_body=critic_body,
                 phi_body=phi_body,
                 action_phi_body=action_phi_body,
                 noisy=noisy,
                 goal_oriented=goal_oriented,
                 goal_shape=goal_shape,
                 goal_phi_body=goal_phi_body,
                 layer_init_fn=layer_init_fn,
            )
            for _ in range(self.nbr_models)
        ])
        
        if layer_init_fn is not None:
            for model in self.inner_models: 
                model.apply(layer_init_fn)
    
    def models(self):
        return self.inner_models

    def reset_noise(self):
        for model in self.inner_models:
            model.apply(reset_noisy_layer)

    def forward(self, obs, action, rnn_states=None, goal=None):
        # Retrieve Q-value from first model
        prediction = self.inner_models[0](
            obs=obs,
            action=action,
            rnn_states=rnn_states,
            goal=goal
        )

        return prediction

    def ensemble_q_values(self, obs, action, rnn_states=None, goal=None):
        predictions = []
        for model in self.inner_models:
            predictions.append(
                model(
                    obs=obs,
                    action=action,
                    rnn_states=rnn_states,
                    goal=goal
                )
            )

        q_values = torch.cat([p["qa"] for p in predictions], dim=-1) 
        output = predictions[0]
        output["qa"] = q_values
        
        return output



    def min_q_value(self, obs, action, rnn_states=None, goal=None):
        batch_size = obs.shape[0]
        pred = self.ensemble_q_values(
            obs=obs,
            action=action,
            rnn_states=rnn_states,
            goal=goal
        
        )

        q_values = pred["qa"]
        min_q_value, _ = q_values.min(dim=-1)

        pred["qa"] = min_q_value.reshape(batch_size, 1)

        return pred


class GaussianActorNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_body,
                 phi_body=None,
                 noisy=False,
                 goal_oriented=False,
                 goal_shape=None,
                 goal_phi_body=None,
                 deterministic=False,
                 action_scaler=1.0,
                 layer_init_fn=layer_init,
                 init_w=3e-3):
        super(GaussianActorNet, self).__init__()

        self.deterministic = deterministic
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.noisy = noisy 
        self.goal_oriented = goal_oriented
        self.action_scaler = action_scaler

        self.actor_body = actor_body

        if phi_body is None: phi_body = DummyBody(state_dim)
        self.phi_body = phi_body
        
        actor_input_shape = self.phi_body.get_feature_shape()
        
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
                actor_input_shape += self.goal_phi_body.get_feature_shape()
        
        fc_actor_input_shape = self.actor_body.get_feature_shape()

        layer_fn = nn.Linear 
        if self.noisy:  layer_fn = NoisyLinear
        self.fc_actor = layer_fn(fc_actor_input_shape, self.action_dim)
        if layer_init_fn is not None:
            self.fc_actor = layer_init_fn(self.fc_actor, 1e0)
        else:
            self.fc_actor.weight.data.uniform_(-init_w, init_w)
            self.fc_actor.bias.data.uniform_(-init_w, init_w)

        self.std = nn.Parameter(torch.zeros(action_dim))

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


        if rnn_states is not None and 'actor_body' in rnn_states:
            actor_output, next_rnn_states['actor_body'] = self.actor_body( (phi, rnn_states['actor_body']) )
        else:
            actor_output = self.actor_body(phi)

        # batch x action_dim
        action = self.action_scaler*torch.tanh(self.fc_actor(actor_output))
        
        prediction = {
            'a': action,
        }


        if not(self.deterministic):
            dist = torch.distributions.Normal(action, F.softplus(self.std))
            sampled_action = dist.sample()
            
            # Log likelyhood of action = sum_i dist.log_prob(action[i])
            log_prob = dist.log_prob(sampled_action).sum(-1).unsqueeze(-1)
            # batch x 1
            entropy = dist.entropy().sum(-1).unsqueeze(-1)
            # batch x 1

            prediction = {
                'a': self.action_scaler*sampled_action,
                'log_pi_a': log_prob,
                'ent': entropy,
            }
        
        prediction.update({'rnn_states': rnn_states,
                               'next_rnn_states': next_rnn_states})

        return prediction


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianActorNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_body,
                 phi_body=None,
                 noisy=False,
                 goal_oriented=False,
                 goal_shape=None,
                 goal_phi_body=None,
                 action_scaler=1.0,
                 layer_init_fn=layer_init,
                 init_w=3e-3):
        super(SquashedGaussianActorNet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.noisy = noisy 
        self.goal_oriented = goal_oriented
        self.action_scaler = action_scaler

        self.actor_body = actor_body

        if phi_body is None: phi_body = DummyBody(state_dim)
        self.phi_body = phi_body
        
        actor_input_shape = self.phi_body.get_feature_shape()
        
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
                actor_input_shape += self.goal_phi_body.get_feature_shape()
        
        fc_actor_input_shape = self.actor_body.get_feature_shape()

        layer_fn = nn.Linear 
        if self.noisy:  layer_fn = NoisyLinear
        self.fc_actor_mu = layer_fn(fc_actor_input_shape, self.action_dim)
        self.fc_actor_log_std = layer_fn(fc_actor_input_shape, self.action_dim)
        if layer_init_fn is not None:
            self.fc_actor_mu = layer_init_fn(self.fc_actor_mu)
            self.fc_actor_log_std = layer_init_fn(self.fc_actor_log_std)
        else:
            self.fc_actor_mu.weight.data.uniform_(-init_w, init_w)
            self.fc_actor_mu.bias.data.uniform_(-init_w, init_w)
            self.fc_actor_log_std.weight.data.uniform_(-init_w, init_w)
            self.fc_actor_log_std.bias.data.uniform_(-init_w, init_w)
        
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


        if rnn_states is not None and 'actor_body' in rnn_states:
            actor_output, next_rnn_states['actor_body'] = self.actor_body( (phi, rnn_states['actor_body']) )
        else:
            actor_output = self.actor_body(phi)

        # batch x action_dim
        action_mu = self.fc_actor_mu(actor_output)
        action_log_std = self.fc_actor_log_std(actor_output)

        action_std = action_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        #action_std = F.softplus(action_log_std)

        action_dist = torch.distributions.normal.Normal(action_mu, action_std)
        #action_dist = torch.distributions.normal.Normal(action_mu, 0.1*torch.ones_like(action_mu))
        
        sampled_action = action_dist.rsample()
        action = torch.tanh(sampled_action)

        # Log likelyhood of action = sum_i dist.log_prob(action[i])
        log_prob = action_dist.log_prob(sampled_action)#.sum(dim=-1).unsqueeze(-1)
        # batch x action_dim
        #extra_term = (2*(np.log(2) - sampled_action - F.softplus(-2*sampled_action)))#.sum(dim=-1).unsqueeze(-1)
        extra_term = torch.log(1-action.pow(2)+1e-6) #.sum(dim=-1).unsqueeze(-1)
        # batch x action_dim
        squashed_log_prob = (log_prob - extra_term).sum(dim=-1).unsqueeze(-1)
        # batch x 1

        entropy = action_dist.entropy()
        # batch x 1

        prediction = {
            'a': self.action_scaler*action,
            'mu': action_mu,
            'std': action_std,
            'log_pi_a': squashed_log_prob,
            'log_normal_u': log_prob,
            'extra_term_log_prob': extra_term,
            'ent': entropy,
        }
    
        prediction.update({'rnn_states': rnn_states,
                               'next_rnn_states': next_rnn_states})

        return prediction


class ActorCriticNet(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        phi_body, 
        actor_body, 
        critic_body, 
        use_intrinsic_critic=False, 
        extra_inputs_infos: Dict={},
        layer_init_fn=layer_init):
        """
        :param extra_inputs_infos: Dictionnary containing the shape of the lstm-relevant extra inputs.
        """
        super(ActorCriticNet, self).__init__()
        
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.get_feature_shape())
        if critic_body is None: critic_body = DummyBody(phi_body.get_feature_shape())
        
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        
        fc_critic_input_shape = self.critic_body.get_feature_shape()
        fc_actor_input_shape = self.actor_body.get_feature_shape()
        
        if isinstance(fc_critic_input_shape, list):
            fc_critic_input_shape = fc_critic_input_shape[-1]
        if isinstance(fc_actor_input_shape, list):
            fc_actor_input_shape = fc_actor_input_shape[-1]

        for key in extra_inputs_infos['critic']:
            shape = extra_inputs_infos[key]['shape']
            assert len(shape) == 1 
            fc_critic_input_shape += shape[-1]
        for key in extra_inputs_infos['actor']:
            shape = extra_inputs_infos[key]['shape']
            assert len(shape) == 1 
            fc_actor_input_shape += shape[-1]
        
        #self.fc_action = nn.Linear(actor_body.get_feature_shape(), action_dim)
        self.fc_action = nn.Linear(fc_actor_input_shape, action_dim)
        if layer_init_fn is not None:
            self.fc_action = layer_init_fn(self.fc_action, 1e-3)
        #self.fc_critic = nn.Linear(critic_body.get_feature_shape(), 1)
        self.fc_critic = nn.Linear(fc_critic_input_shape, 1)
        if layer_init_fn is not None:
            self.fc_critic = layer_init_fn(self.fc_critic, 1e0)

        self.use_intrinsic_critic = use_intrinsic_critic
        self.fc_int_critic = None
        if self.use_intrinsic_critic: 
            #self.fc_int_critic = nn.Linear(critic_body.get_feature_shape(), 1)
            self.fc_int_critic = nn.Linear(fc_critic_input_shape, 1)
            if layer_init_fn is not None:
                self.fc_int_critic = layer_init_fn(self.fc_int_critic, 1e-3)

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

        prediction.update({'rnn_states': rnn_states,
                               'next_rnn_states': next_rnn_states})

        return prediction


class CategoricalActorCriticNet(ActorCriticNet):
    def __init__(
        self,
        state_dim,
        action_dim,
        phi_body=None,
        actor_body=None,
        critic_body=None,
        use_intrinsic_critic=False,
        extra_inputs_infos: Dict={}):
        """
        :param extra_inputs_infos: Dictionnary containing the shape of the lstm-relevant extra inputs.
        """
        
        super(CategoricalActorCriticNet, self).__init__(
            state_dim=state_dim, 
            action_dim=action_dim, 
            phi_body=phi_body, 
            actor_body=actor_body, 
            critic_body=critic_body,
            use_intrinsic_critic=use_intrinsic_critic,
            extra_inputs_infos=extra_inputs_infos,
        )

    def forward(self, obs, action=None, rnn_states=None):
        global EPS
        batch_size = obs.shape[0]

        next_rnn_states = None 
        if rnn_states is not None:
            next_rnn_states = {k: None for k in rnn_states}

        if rnn_states is not None and 'phi_body' in rnn_states:
            phi, next_rnn_states['phi_body'] = self.phi_body( (obs, rnn_states['phi_body']) )
        else:
            phi = self.phi_body(obs)

        if rnn_states is not None and 'actor_body' in rnn_states:
            phi_a, next_rnn_states['actor_body'] = self.actor_body( (phi, rnn_states['actor_body']) )
        else:
            phi_a = self.actor_body(phi)

        if 'final_actor_layer' in rnn_states:
            extra_inputs = extract_subtree(
                in_dict=rnn_states['final_actor_layer'],
                node_id='extra_inputs',
            )
            
            extra_inputs = [v[0].to(phi_a.dtype).to(phi_a.device) for v in extra_inputs.values()]
            if len(extra_inputs): phi_a = torch.cat([phi_a]+extra_inputs, dim=-1)
        
        if rnn_states is not None and 'critic_body' in rnn_states:
            phi_v, next_rnn_states['critic_body'] = self.critic_body( (phi, rnn_states['critic_body']) )
        else:
            phi_v = self.critic_body(phi)

        if 'final_critic_layer' in rnn_states:
            extra_inputs = extract_subtree(
                in_dict=rnn_states['final_critic_layer'],
                node_id='extra_inputs',
            )
            
            extra_inputs = [v[0].to(phi_v.dtype).to(phi_v.device) for v in extra_inputs.values()]
            if len(extra_inputs): phi_v = torch.cat([phi_v]+extra_inputs, dim=-1)
        

        # batch x action_dim
        v = self.fc_critic(phi_v)
        if self.use_intrinsic_critic:
            int_v = self.fc_int_critic(phi_v)
        # batch x 1

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
        logits = self.fc_action(phi_a)
        probs = F.softmax( logits, dim=-1 )
        #https://github.com/pytorch/pytorch/issues/7014
        #probs = torch.clamp(probs, -1e10, 1e10)
        #log_probs = F.log_softmax(logits, dim=-1)
        log_probs = torch.log(probs+EPS)
        entropy = -torch.sum(probs*log_probs, dim=-1)#, keepdim=True)
        # batch #x 1
        
        legal_actions = torch.ones_like(logits)
        if 'head' in rnn_states \
        and 'extra_inputs' in rnn_states['head'] \
        and 'legal_actions' in rnn_states['head']['extra_inputs']:
            legal_actions = rnn_states['head']['extra_inputs']['legal_actions'][0]
            next_rnn_states['head'] = rnn_states['head']
        legal_actions = legal_actions.to(logits.device)
        
        # The following accounts for player dimension if VDN:
        legal_qa = (1+logits-logits.min(dim=-1, keepdim=True)[0]) * legal_actions
        
        greedy_action = legal_qa.max(dim=-1, keepdim=True)[1]
        if action is None:
            #action = (probs+EPS).multinomial(num_samples=1).squeeze(1)
            #action = torch.multinomial( probs, num_samples=1).squeeze(1)
            action = torch.multinomial(legal_qa.softmax(dim=-1), num_samples=1)#.reshape((batch_size,))
            # batch #x 1
        #log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        log_probs = log_probs.gather(1, action).squeeze(1)
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

        prediction = {
            'a': action,
            'greedy_action': greedy_action,
            'log_pi_a': log_probs,
            'action_logits': logits,
            'ent': entropy,
            'v': v
        }
        
        if self.use_intrinsic_critic:
            prediction['int_v'] = int_v

        prediction.update({
            'rnn_states': rnn_states,
            'next_rnn_states': next_rnn_states}
        )

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
