from typing import Dict, List 

import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from Archi.modules.module import Module 
from Archi.modules.utils import layer_init

from regym.rl_algorithms.networks import EPS


class RLCategoricalActorCriticHeadModule(Module):
    def __init__(
        self, 
        state_dim,   
	action_dim,
        use_intrinsic_critic=False,
        id='RLCategoricalActorCriticHeadModule_0', 
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
        layer_init_fn=layer_init,
        use_cuda=False
    ):

        super(RLCategoricalActorCriticHeadModule, self).__init__(
            id=id,
            type="RLCategoricalActorCriticHeadModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        self.greedy = False #True
        self.use_intrinsic_critic= use_intrinsic_critic
        self.state_dim = state_dim
        self.action_dim = action_dim

        layer_fn = nn.Linear 
        if False : #self.dueling:
            self.fc_critic = DuelingLayer(input_dim=self.state_dim, action_dim=self.action_dim, layer_fn=layer_fn)
        elif False:
            self.fc_action = layer_fn(self.state_dim, self.action_dim)
            self.fc_ext_critic = layer_fn(self.state_dim, 1)
            if self.use_intrinsic_critic:
                self.fc_int_critic = layer_fn(self.state_dim, 1)
            if layer_init_fn is not None:
                print(f'WARNING: using layer init fn : {layer_init_fn} in {self}')
                import ipdb; ipdb.set_trace()
                # TODO : check whether initialisation affects entropy
                self.fc_action = layer_init_fn(
                    self.fc_action, 
                    w_scale=1e0, #1.0e-2
                    init_type='ortho',
                )
                self.fc_ext_critic = layer_init_fn(
                    self.fc_ext_critic, 
                    w_scale=1e0,
                    init_type='ortho',
                )

                #self.fc_action = layer_init_fn(self.fc_action, 1e-3)
                #self.fc_ext_critic = layer_init_fn(self.fc_ext_critic, 1e0)
                if self.use_intrinsic_critic:
                    self.fc_int_critic = layer_init_fn(
                        self.fc_int_critic, 
                        w_scale=1e0,
                        init_type='ortho',
                    )
                    #self.fc_int_critic = layer_init_fn(self.fc_int_critic, 1e-3)
        else:
            self.fc_action = [
                layer_fn(self.state_dim, self.state_dim),
                nn.ReLU(),
                layer_fn(self.state_dim, self.action_dim),
            ]
            self.fc_ext_critic = [
                layer_fn(self.state_dim, self.state_dim),
                nn.ReLU(),
                layer_fn(self.state_dim, 1),
            ]
            if self.use_intrinsic_critic:
                self.fc_int_critic = layer_fn(self.state_dim, 1)
            if layer_init_fn is not None:
                print(f'WARNING: using layer init fn : {layer_init_fn} in {self}')
                # TODO : check whether initialisation affects entropy
                for lidx in range(len(self.fc_action)):
                    self.fc_action[lidx] = layer_init_fn(
                        self.fc_action[lidx], 
                        w_scale=1.0e-2,
                        init_type='ortho',
                    )
                self.fc_action = nn.Sequential(*self.fc_action)
                for lidx in range(len(self.fc_ext_critic)):
                    self.fc_ext_critic[lidx] = layer_init_fn(
                        self.fc_ext_critic[lidx], 
                        w_scale=1.0, #1e-1 if lidx==0 else 1e-2, #1e0
                        init_type='ortho',
                    )
                self.fc_ext_critic = nn.Sequential(*self.fc_ext_critic)
                # TODO: propagate for intrinsic critic ...

                #self.fc_action = layer_init_fn(self.fc_action, 1e-3)
                #self.fc_ext_critic = layer_init_fn(self.fc_ext_critic, 1e0)
                if self.use_intrinsic_critic:
                    raise NotImplementedError
                    self.fc_int_critic = layer_init_fn(
                        self.fc_int_critic, 
                        w_scale=1e0,
                        init_type='ortho',
                    )
                    #self.fc_int_critic = layer_init_fn(self.fc_int_critic, 1e-3)
        self.feature_dim = self.action_dim

        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def get_feature_shape(self):
        return self.feature_dim

    #def reset_noise(self):
    #    self.apply(reset_noisy_layer)

    def forward(self, phi_features):
        ext_v = self.fc_ext_critic(phi_features)     
        int_v = None
        if self.use_intrinsic_critic:
            int_v = self.fc_int_critic(phi_features)
        action_logit = self.fc_action(phi_features)
        # batch x 1 / action_dim
        return ext_v, int_v, action_logit
    
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        """
        outputs_stream_dict = {}

        phi_features_list = [v[0] if isinstance(v, list) else v for k,v in input_streams_dict.items() if 'input' in k]
        if self.use_cuda:   phi_features_list = [v.cuda() for v in phi_features_list]
        phi_features = torch.cat(phi_features_list, dim=-1)
        
        if self.use_cuda:   phi_features = phi_features.cuda()
        
        ext_v, int_v, action_logits = self.forward(phi_features)
        
        # PREVIOUSLY: 
        #probs = F.softmax(action_logits, dim=-1)
        #log_probs = F.log_softmax(action_logits, dim=-1)
        #NOW:
        batch_size = action_logits.shape[0]
        probs = Categorical(logits=action_logits)
        # POSSIBLE previvously : like in regym's head :
        #log_probs = torch.log(probs+1.0e-8)

        # The following leads to very different legal_ent and ent:
        # log_probs = action_logits
        #entropy = -torch.sum(probs*log_probs, dim=-1)
        entropy = probs.entropy()
        entropy = entropy.reshape(batch_size)
        # batch

        legal_actions = torch.ones_like(action_logits)
        if 'legal_actions' in input_streams_dict: 
            legal_actions = input_streams_dict['legal_actions']
            if isinstance(legal_actions, list):  legal_actions = legal_actions[0]
        legal_actions = legal_actions.to(action_logits.device)
        
        # The following accounts for player dimension if VDN:
        legal_qa = (1+action_logits-action_logits.min(dim=-1, keepdim=True)[0]) * legal_actions
        
        greedy_action = legal_qa.max(dim=-1, keepdim=True)[1]
        if 'action' in input_streams_dict:
            action = input_streams_dict['action']
            if isinstance(action, list):    action = action[0]
        if action is None:
            if self.greedy:
                action  = legal_qa.max(dim=-1, keepdim=True)[1]
            else:
                #action = torch.multinomial(legal_qa.softmax(dim=-1), num_samples=1) #.reshape((batch_size,))
                action = probs.sample()
        # batch #x 1
        
        #log_probs = torch.log(probs+EPS)
        #log_probs = log_probs.gather(1, action).squeeze(1)
        log_probs = probs.log_prob(action).reshape(batch_size)
        # batch 
        
        legal_probs = F.softmax( legal_qa, dim=-1 )
        legal_log_probs = torch.log_softmax(legal_qa+EPS, dim=-1)
        legal_entropy = -torch.sum(legal_probs*legal_log_probs, dim=-1)
        # batch

        outputs_stream_dict = {
            'a': action,
            'greedy_action': greedy_action,
            'ent': entropy,
            'legal_ent': legal_entropy,
            'v': ext_v,
            'int_v': int_v,
            'log_pi_a': log_probs,
            'legal_log_pi_a': legal_log_probs,
        }
        
        return outputs_stream_dict 

