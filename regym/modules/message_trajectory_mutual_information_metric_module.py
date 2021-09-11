from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 
import copy 

from .module import Module

from comaze_gym.metrics import MessageTrajectoryMutualInformationMetric, RuleBasedMessagePolicy


def build_MessageTrajectoryMutualInformationMetricModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None
    ) -> Module:
    return MessageTrajectoryMutualInformationMetricModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class MessageTrajectoryMutualInformationMetricModule(Module):
    def __init__(
        self,
        id:str,
        config:Dict[str,object],
        input_stream_ids:Dict[str,str]=None
        ):
        """
        Computes multi-step CIC metric and maintains a few elements
        necessary to the computation, for 2-players alternating (not simultaneous) games.
        """
        default_input_stream_ids = {
            "logs_dict":"logs_dict",
            "losses_dict":"losses_dict",
            "epoch":"signals:epoch",
            "mode":"signals:mode",

            "vocab_size":"config:vocab_size",
            "max_sentence_length":"config:max_sentence_length",
            
            "trajectories":"modules:environment_module:trajectories",
            "filtering_signal":"modules:environment_module:new_trajectories_published",

            "current_agents":"modules:current_agents:ref",
            
            # "observations":"modules:environment_module:observations",
            # "infos":"modules:environment_module:info",
            # "actions":"modules:environment_module:actions",
            # "dones":"modules:environment_module:done",
        }
        
        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_id, default_stream in default_input_stream_ids.items():
                if default_id not in input_stream_ids:
                    input_stream_ids[default_id] = default_stream

        super(MessageTrajectoryMutualInformationMetricModule, self).__init__(
            id=id,
            type="MessageTrajectoryMutualInformationMetricModule",
            config=config,
            input_stream_ids=input_stream_ids
        )

        self.biasing = self.config.get('biasing', False)
        self.nbr_players = self.config.get('nbr_players', 2)
        self.player_id = self.config.get('player_id', 0)
        self.metric = self.config.get('metric', None)

        # inputs to the agents at each timestep
        self.observations = []
        self.infos = []
        # outputs/actions taken when info and obs were seen:
        self.actions = []
        self.dones = []

        self.iteration = 0
        self.sampling_fraction = 5
        self.sampling_period = 10.0


    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}

        if self.metric is None:
            self.agents = input_streams_dict["current_agents"].agents
            self.metric = MultiStepCIC(
                action_policy=RuleBasedMessagePolicy( 
                    wrapped_rule_based_agent=self.agents[self.player_id],
                    combined_action_space=False,
                ),
                action_policy_bar=None, #deepcopy...
            )


        logs_dict = input_streams_dict["logs_dict"]
        mode = input_streams_dict["mode"]
        epoch = input_streams_dict["epoch"]
        
        filtering_signal = input_streams_dict["filtering_signal"]
        trajectories = input_streams_dict["trajectories"]
        compute = True 

        self.iteration += 1
        #if (compute and np.random.random() < 1.0/self.sampling_period) or filtering_signal:
        if (compute and (self.iteration % self.sampling_period) == 0) or filtering_signal:
            if filtering_signal:
                self.actions = [
                    [
                        exp[1] 
                        for exp in traj[self.player_id]
                    ]
                    for traj in trajectories
                ]
                # (batch_size, timestep, (, keys if 'sad'==True) 1) 
                
                # Formatting of kwargs:
                #   - 'state': observations
                #   - 'infos': infos
                self.x = [ 
                    [
                        {
                            'state':exp[0], # for _ in range(self.nbr_players)], # see environment_module for indices...
                            'infos':[exp[6]], # for _ in range(self.nbr_players)],
                            'as_logit':True,
                        }
                        for exp in traj[self.player_id]
                    ]    
                    for traj in trajectories
                ]
                # (batch_size, timestep, keys:values) 
                
                x = self.x 
                
            else:
                if not hasattr(self, 'x'):
                    return outputs_stream_dict

            #indices = np.random.choice(list(range(len(self.x))), size=len(self.x)//10, replace=False)
            if filtering_signal:
                indices = list(range(len(self.x)))
            else:
                indices = np.random.choice(list(range(len(self.x))), size=len(self.x)//self.sampling_fraction, replace=False)
            
            x = [traj for idx, traj in enumerate(self.x) if idx in indices]

            batch_size = len(x)
            T = max([len(traj) for traj in x])
            #mask = torch.ones((batch_size, T))
            mask = torch.zeros((batch_size, T))
            for actor_id in range(batch_size):
                for t in range(len(x[actor_id])):
                    mask[actor_id][t] = (x[actor_id][t]['infos'][0]["current_player"].item()==self.player_id)
            
            ## Measure:
            rd = self.metric.compute_pos_sign_loss(
                x=x, 
                mask=mask,
                biasing=self.biasing,
            )
            L_ps = rd["L_ps"]
            L_ps_ent_term = rd["L_ps_EntTerm"]
            # batch_size 
            
            averaged_m_policy_entropy = rd["ent_pi_bar_m"]
            exp_ent_pi_m_x_it_over_x_it = rd["exp_ent_pi_m_x_it_over_x_it"]  

            mutual_info_m_x_it = averaged_m_policy_entropy - exp_ent_pi_m_x_it_over_x_it
            # (1 x 1)
            
            logs_dict[f"{mode}/{self.id}/AverageMessagePolicyEntropy/{'Eval' if filtering_signal else 'Sample'}"] = averaged_m_policy_entropy
            logs_dict[f"{mode}/{self.id}/MutualInformationMessageTrajectory/{'Eval' if filtering_signal else 'Sample'}"] = mutual_info_m_x_it
            logs_dict[f"{mode}/{self.id}/MutualInformationMessageTrajectory/ExpectedEntropyTerm/{'Eval' if filtering_signal else 'Sample'}"] = exp_ent_pi_m_x_it_over_x_it
            logs_dict[f"{mode}/{self.id}/PositiveSignallingLoss/EntropyTerm/{'Eval' if filtering_signal else 'Sample'}"] = L_ps_ent_term

            if self.biasing:
                losses_dict = input_streams_dict["losses_dict"]
                losses_dict[f"{mode}/{self.id}/PositiveSignallingLoss/{'Eval' if filtering_signal else 'Sample'}"] = [1.0, L_ps]
            else:
                logs_dict[f"{mode}/{self.id}/PositiveSignallingLoss/{'Eval' if filtering_signal else 'Sample'}"] = L_ps.cpu()
            
        return outputs_stream_dict
    
