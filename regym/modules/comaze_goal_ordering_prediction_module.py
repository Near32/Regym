from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 
import copy 

from .module import Module

from comaze_gym.metrics import GoalOrderingPredictionMetric


def build_CoMazeGoalOrderingPredictionModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None
    ) -> Module:
    return CoMazeGoalOrderingPredictionModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class CoMazeGoalOrderingPredictionModule(Module):
    def __init__(
        self,
        id:str,
        config:Dict[str,object],
        input_stream_ids:Dict[str,str]=None
        ):
        """
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

        super(CoMazeGoalOrderingPredictionModule, self).__init__(
            id=id,
            type="CoMazeGoalOrderingPredictionModule",
            config=config,
            input_stream_ids=input_stream_ids
        )

        self.biasing = self.config.get('biasing', False)
        self.nbr_players = self.config.get('nbr_players', 2)
        self.player_id = self.config.get('player_id', 0)
        self.metric = self.config['metric']

        self.iteration = 0
        self.sampling_fraction = 5
        self.sampling_period = 10.0

        self.secretgoalStr2id = {"RED":0, "YELLOW":1, "BLUE":2, "GREEN":3}
        #self.id2SecretgoalStr = dict(zip(self.secretgoalStr2id.values(), self.secretgoalStr2id.keys()))
        """
        self.labels = {}
        label_id = 0
        for g1 in range(5):
            for g2 in range(5):
                for g3 in range(5):
                    for g4 in range(5):
                        self.labels[[g1,g2,g3,g4]] = label_id
                        label_id += 1
        """

    def parameters(self):
        return self.metric.prediction_net.parameters()

    def build_goal_ordering_label(self, info_dict):
        reached_goals_str = info_dict['abstract_repr']['reached_goals']
        reached_goals_ids = [self.secretgoalStr2id[g] for g in reached_goals_str]
        '''
        Issues:
        1) not necessarily reaching all goals
        '''
        # Adding dummy goal if unreached:
        dummy_goal_id = 4
        while len(reached_goals_ids)<4: reached_goals_ids.append(dummy_goal_id)
        
        return torch.Tensor(reached_goals_ids).reshape((1,4))

        """label = self.labels[reached_goals_ids] 
        return label*torch.ones(1)
        """

    def build_rules_label(self, info_dict):
        rules_labels = torch.zeros(1,4)
        for pid, sgr in enumerate(info_dict['abstract_repr']["secretGoalRule"]):
            # earlier:
            rules_labels[:,pid*2] = self.secretgoalStr2id[sgr.earlierGoal.color]
            # later:
            rules_labels[:,pid*2+1] = self.secretgoalStr2id[sgr.laterGoal.color] 
        
        return rules_labels

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}

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
                
                self.goal_ordering_labels = [ 
                    self.build_goal_ordering_label(traj[self.player_id][-1][6])
                    for traj in trajectories
                ]

                self.rules_labels = [ 
                    self.build_rules_label(traj[self.player_id][-1][6])
                    for traj in trajectories
                ]

                x = self.x 
                goal_ordering_labels = self.goal_ordering_labels
                rules_labels = self.rules_labels
            else:
                if not hasattr(self, 'x'):
                    return outputs_stream_dict

            if filtering_signal:
                indices = list(range(len(self.x)))
            else:
                indices = np.random.choice(list(range(len(self.x))), size=len(self.x)//self.sampling_fraction, replace=False)
            
            x = [traj for idx, traj in enumerate(self.x) if idx in indices]
            goal_ordering_labels = [labels for idx, labels in enumerate(self.goal_ordering_labels) if idx in indices]
            rules_labels = [labels for idx, labels in enumerate(self.rules_labels) if idx in indices]

            mask = None
            
            ## Measure:
            output_dict = self.metric.compute_goal_ordering_prediction_loss(
                x=x, 
                y=goal_ordering_labels,
                yp=rules_labels,
                mask=mask,
                biasing=self.biasing,
            )
            L_gop = output_dict['l_gop']
            # batch_size
            gop_accuracy = output_dict['per_actor_gop_accuracy']
            # batch_size
            q1_correct_gop = output_dict['per_actor_acc_distr_q1']
            # batch_size 


            L_rp = output_dict['l_rp']
            # batch_size
            rp_accuracy = output_dict['per_actor_rp_accuracy']
            # batch_size
            q1_correct_rp = output_dict['per_actor_rp_acc_distr_q1']
            # batch_size 
            
            logs_dict[f"{mode}/{self.id}/GoalOrderingPredictionAccuracy/{'Eval' if filtering_signal else 'Sample'}"] = gop_accuracy.mean()
            logs_dict[f"{mode}/{self.id}/CorrectGoalOrderingPrediction-Q1/{'Eval' if filtering_signal else 'Sample'}"] = q1_correct_gop.mean()

            logs_dict[f"{mode}/{self.id}/RulesPredictionAccuracy/{'Eval' if filtering_signal else 'Sample'}"] = rp_accuracy.mean()
            logs_dict[f"{mode}/{self.id}/RulesPrediction-Q1/{'Eval' if filtering_signal else 'Sample'}"] = q1_correct_rp.mean()

            losses_dict = input_streams_dict["losses_dict"]
            losses_dict[f"{mode}/{self.id}/GoalOrderingPredictionLoss/{'Eval' if filtering_signal else 'Sample'}"] = [1.0, L_gop]
            losses_dict[f"{mode}/{self.id}/RulesPredictionLoss/{'Eval' if filtering_signal else 'Sample'}"] = [1.0, L_rp]
        
        return outputs_stream_dict
    
