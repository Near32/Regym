from typing import Dict

import copy
import torch
import torch.nn as nn


class ArchiRewardPredictor(nn.Module):
    def __init__(self, model, **kwargs):
        nn.Module.__init__(self)
        self.model = model
        self.archi_kwargs = kwargs
    
    def clone(self):
        return copy.deepcopy(self)
 
    def parameters(self):
        params = []
        for km, module in self.model.modules.items():
            if km in self.model.pipelines["reward_prediction"]:
                params += module.parameters()
        return params

    def forward(
        self,
        x,
        rnn_states=None,
        gt_rewards=None,
    ):
        if rnn_states is None:
            rnn_states = self.model.get_reset_states()

        input_dict = {
            'obs':x,
            'rnn_states': rnn_states,
        }
         
        '''
        if gt_rewards is None:
            return_feature_only=self.archi_kwargs["features_id"]["reward_prediction"]
        else:
            return_feature_only = None 
            input_dict['rnn_states']['gt_rewards'] = gt_rewards
        '''    
        return_feature_only=self.archi_kwargs["features_id"]["reward_prediction"]
        output = self.model.forward(
            **input_dict,
            pipelines={
                "reward_prediction":self.archi_kwargs["pipelines"]["reward_prediction"]
            },
            return_feature_only=return_feature_only,
        )

        return output
    
    def compute_loss(self, x, rnn_states, gt_rewards ):
        prediction_logits = self.forward(
            x=x,
            gt_rewards=gt_rewards,
            rnn_states=rnn_states,
        )
        
        batch_size = gt_rewards.shape[0]
        # Make the gt_rewards as labels:
        reward_labels = torch.zeros((batch_size, 3)).to(gt_rewards.device)
        for bidx in range(batch_size):
            r = gt_rewards[bidx].mean().item()
            if r < 0:
                reward_labels[bidx,0] = 1.0
            elif r == 0:
                reward_labels[bidx,1] = 1.0
            elif r > 0:
                reward_labels[bidx,2] = 1.0
        
        # Compute loss:
        prediction = prediction_logits.softmax(dim=-1)
        log_pred = prediction_logits.log_softmax(dim=-1)
        loss_per_item = -torch.sum(
            reward_labels*log_pred,
            dim=-1,
        )
        
        # Compute accuracy: 
        argmax_pred = prediction.argmax(dim=-1, keepdim=True)
        accuracy = torch.gather(
            input=reward_labels,
            index=argmax_pred,
            dim=-1,
        ).mean()*100.0

        rdict = {
            'prediction': prediction, 
            'loss_per_item':loss_per_item, 
            'accuracy':accuracy, 
        }

        return rdict


