from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 
import copy 

from .module import Module

from comaze_gym.metrics import MultiStepCIC, RuleBasedActionPolicy


def build_MultiStepCICMetricModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None
    ) -> Module:
    return MultiStepCICMetricModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class MultiStepCICMetricModule(Module):
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

        super(MultiStepCICMetricModule, self).__init__(
            id=id,
            type="MultiStepCICMetricModule",
            config=config,
            input_stream_ids=input_stream_ids
        )

        self.biasing = self.config.get('biasing', False)
        self.nbr_players = self.config.get('nbr_players', 2)
        self.player_id = self.config.get('player_id', 0)
        self.metric = self.config.get('metric', None)

        self.iteration = 0
        self.sampling_fraction = 5
        self.sampling_period = 10.0

        def message_zeroing_out_fn(
            x, 
            msg_key="communication_channel", 
            #paths_to_msg=[["infos",pidx] for pidx in range(self.nbr_players)],
            paths_to_msg=[["infos",0]],
            
            ):
            xp = copy.deepcopy(x)
            for actor_id in range(len(xp)):
                for t in range(len(xp[actor_id])):
                    pointer = xp[actor_id][t]
                    for path_to_msg in paths_to_msg:
                        for child_node in path_to_msg:
                            pointer = pointer[child_node]
                        
                        msg = pointer[msg_key]
                        if isinstance(msg, List):
                            zeroed_out_msg =  [np.zeros_like(item) for item in msg]
                        else:
                            zeroed_out_msg =  np.zeros_like(msg)
                        pointer[msg_key] = zeroed_out_msg

                        pointer = xp[actor_id][t]
            
            return xp

        self.message_zeroing_out_fn = self.config.get('message_zeroing_out_fn', message_zeroing_out_fn)

        # inputs to the agents at each timestep
        self.observations = []
        self.infos = []
        # outputs/actions taken when info and obs were seen:
        self.actions = []
        self.dones = []



    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}

        if self.metric is None:
            self.agents = input_streams_dict["current_agents"].agents
            self.metric = MultiStepCIC(
                action_policy=RuleBasedActionPolicy( 
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
                
                self.xp = self.message_zeroing_out_fn(self.x)
                self.a = self.actions

                x = self.x 
                xp = self.xp 
                a = self.a 

            else:
                if not hasattr(self, 'xp'):
                    return outputs_stream_dict

            if filtering_signal:
                indices = list(range(len(self.x)))
            else:
                indices = np.random.choice(list(range(len(self.x))), size=len(self.x)//self.sampling_fraction, replace=False)
            
            x = [traj for idx, traj in enumerate(self.x) if idx in indices]
            xp = [traj for idx, traj in enumerate(self.xp) if idx in indices]
            a = [traj for idx, traj in enumerate(self.a) if idx in indices]


            batch_size = len(x)
            T = max([len(traj) for traj in x])
            #mask = torch.ones((batch_size, T))
            mask = torch.zeros((batch_size, T))
            for actor_id in range(batch_size):
                for t in range(len(x[actor_id])):
                    mask[actor_id][t] = (x[actor_id][t]['infos'][0]["current_player"].item()==self.player_id)
            
            ## Measure:
            L_pl = self.metric.compute_pos_lis_loss(
                x=x, 
                #xp=x, #debug
                xp=xp, 
                mask=mask,
                biasing=self.biasing,
            )

            ms_cic = self.metric.compute_multi_step_cic(
                x=x, 
                #xp=x, #debug
                xp=xp, 
                mask=mask
            )

            ## Training:

            L_ce, prediction_accuracy = self.metric.train_unconditioned_policy(
                x=x, 
                #xp=x, #debug
                xp=xp, 
                mask=mask, 
                #a=a, #using actual action is risky given the exploration policy, if on training trajectories...
            )
            # batch_size 
            
            logs_dict[f"{mode}/{self.id}/multi_step_CIC/{'Eval' if filtering_signal else 'Sample'}"] = ms_cic.cpu()
            logs_dict[f"{mode}/{self.id}/UnconditionedPolicyFitting/CrossEntropyLoss/{'Eval' if filtering_signal else 'Sample'}"] = L_ce.cpu()
            logs_dict[f"{mode}/{self.id}/UnconditionedPolicyFitting/PredictionAccuracy/{'Eval' if filtering_signal else 'Sample'}"] = prediction_accuracy.cpu()
            
            if self.biasing:
                losses_dict = input_streams_dict["losses_dict"]
                losses_dict[f"{mode}/{self.id}/PositiveListeningLoss/{'Eval' if filtering_signal else 'Sample'}"] = [0.0001, L_pl]
            else:
                logs_dict[f"{mode}/{self.id}/PositiveListeningLoss/{'Eval' if filtering_signal else 'Sample'}"] = L_pl.cpu()
            
        return outputs_stream_dict
    
