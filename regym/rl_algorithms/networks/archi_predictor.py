from typing import Dict

import copy
import torch
import torch.nn as nn


class ArchiPredictor(nn.Module):
    def __init__(
        self, 
        model, 
        pipeline_name="instruction_generator",
        generator_name="InstructionGenerator",
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.pipeline_name = pipeline_name
        self.generator_name = generator_name
        self.model = model
        self.archi_kwargs = kwargs
        try:
            self.use_oracle = len([
                m_id for m_id in self.model.pipelines[self.pipeline_name]
                if 'oracle' in m_id.lower()
            ]) > 0
        except Exception as e:
            self.use_oracle = False
        if self.use_oracle:
            print("ARCHI PREDICTOR::WARNING: using OracleTHER.")
    
    def get_reset_states(self):
        return self.model.get_reset_states()
    
    def set_reset_states(self, new_reset_states):
        self.model.set_reset_states(new_reset_states)
    
    def clone(self):
        return copy.deepcopy(self)
 
    def parameters(self):
        params = []
        for km, module in self.model.modules.items():
            if km in self.model.pipelines[self.pipeline_name]:
                params += module.parameters()
        return params

    def forward(
        self,
        x,
        gt_sentences=None,
        rnn_states=None,
    ):
        if rnn_states is None:
            rnn_states = self.model.get_reset_states()

        input_dict = {
            'obs':x,
            'rnn_states': rnn_states,
        }
         
        if gt_sentences is None:
            return_feature_only=self.archi_kwargs["features_id"][self.pipeline_name]
        else:
            return_feature_only = None 
            input_dict['rnn_states']['gt_sentences'] = gt_sentences
            
        output = self.model.forward(
            **input_dict,
            pipelines={
                self.pipeline_name:self.archi_kwargs["pipelines"][self.pipeline_name]
            },
            return_feature_only=return_feature_only,
        )

        return output
    
    def compute_loss(self, x, rnn_states, goal=None):
        gt_sentences = rnn_states['phi_body']['extra_inputs']['desired_goal'] 
        
        output_stream_dict = self.forward(
            x=x,
            gt_sentences=gt_sentences,
            rnn_states=rnn_states,
        )
        
        if self.use_oracle:
            rdict = {
                'prediction': output_stream_dict['next_rnn_states']["input0_prediction"][0], 
                'loss_per_item':output_stream_dict['next_rnn_states']["input0_loss_per_item"][0], 
                'accuracies':output_stream_dict['next_rnn_states']["input0_accuracies"][0], 
                'sentence_accuracies':output_stream_dict['next_rnn_states']["input0_sentence_accuracies"][0],
                'bos_accuracies':output_stream_dict['next_rnn_states']["input0_bos_accuracies"][0], 
                'bos_sentence_accuracies':output_stream_dict['next_rnn_states']["input0_bos_sentence_accuracies"][0],
            }
        else:
            rdict = {
                'prediction': output_stream_dict['next_rnn_states'][self.generator_name]["input0_prediction"][0], 
                'loss_per_item':output_stream_dict['next_rnn_states'][self.generator_name]["input0_loss_per_item"][0], 
                'accuracies':output_stream_dict['next_rnn_states'][self.generator_name]["input0_accuracies"][0], 
                'sentence_accuracies':output_stream_dict['next_rnn_states'][self.generator_name]["input0_sentence_accuracies"][0],
                'bos_accuracies':output_stream_dict['next_rnn_states'][self.generator_name]["input0_bos_accuracies"][0], 
                'bos_sentence_accuracies':output_stream_dict['next_rnn_states'][self.generator_name]["input0_bos_sentence_accuracies"][0],
            }
 

        return rdict


