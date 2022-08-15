from typing import Dict, List, Any 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 
import copy 

from .module import Module
from regym.rl_algorithms.utils import _concatenate_list_hdict
from functools import partial


def build_ReconstructionFromHiddenStateModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None
    ) -> Module:
    return ReconstructionFromHiddenStateModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class ReconstructionFromHiddenStateModule(Module):
    def __init__(
        self,
        id:str,
        config:Dict[str,object],
        input_stream_ids:Dict[str,str]=None
        ):
        """
        "build_signal_to_reconstruct_from_trajectory_fn": 
        lambda traj, player_id: return List[torch.Tensor]
            
        "signal_to_reconstruct": "None",
        """
        default_input_stream_ids = {
            "logs_dict":"logs_dict",
            "losses_dict":"losses_dict",
            "epoch":"signals:epoch",
            "mode":"signals:mode",
           
            "trajectories":"modules:marl_environment_module:trajectories",
            "filtering_signal":"modules:marl_environment_module:new_trajectories_published",

            "current_agents":"modules:current_agents:ref",
        }
        
        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_id, default_stream in default_input_stream_ids.items():
                if default_id not in input_stream_ids:
                    input_stream_ids[default_id] = default_stream

        super(ReconstructionFromHiddenStateModule, self).__init__(
            id=id,
            type="ReconstructionFromHiddenStateModule",
            config=config,
            input_stream_ids=input_stream_ids
        )

        self.rec_threshold = config.get("rec_threshold", 5e-2)

        self.biasing = self.config.get('biasing', False)
        self.nbr_players = self.config.get('nbr_players', 2)
        self.player_id = self.config.get('player_id', 0)

        self.iteration = 0
        self.sampling_fraction = 2
        self.sampling_period = self.config.get('sampling_period', 10.0) # 25.0
        
        if "build_signal_to_reconstruct_from_trajectory_fn" in self.config:
            self.build_signal_to_reconstruct_from_trajectory_fn = self.config["build_signal_to_reconstruct_from_trajectory_fn"]
        
        if "MSE" in self.config.get("reconstruction_loss", "BCE"):
            self.criterion = torch.nn.MSELoss(reduction='none')
        else: 
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        
        self.signal_to_reconstruct_dim = self.config["signal_to_reconstruct_dim"]
        self.hiddenstate_policy = self.config["hiddenstate_policy"]
        self.hidden_state_dim = self.hiddenstate_policy.get_hidden_state_dim()
        self.prediction_net = [
            nn.Linear(self.hidden_state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.signal_to_reconstruct_dim),
        ]
        self.prediction_net = nn.Sequential(*self.prediction_net)

        if self.config['use_cuda']:
            self = self.cuda()

        print(self.prediction_net)

    def parameters(self):
        return self.prediction_net.parameters()

    def compute_reconstruction_loss(
        self,
        x:List[List[Any]],
        y:List[List[torch.Tensor]],
        mask:List[List[Any]]=None,
        biasing:bool=False,
    ) -> Dict[str,torch.Tensor]:
        """
        WARNING: this function resets the :attr hiddenstate_policy:! 
        Beware of potentially erasing agent's current's internal states

        :param x: 
            List[List[object]] containing, for each actor, at each time step t an object
            representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the :attr hiddenstate_policy:.
        
        :param y: 
            List[List[torch.Tensor]] containing, for each actor, at each time step t an object
            representing the signal to reconstruct.
            Shape: signal_to_reconstruct_dim.
        
        :param mask:
            List[List[object]] containing, for each actor, at each time step t an object
            with batch_size dimensions and whose values are either
            1 or 0. For all actor b, mask[b]==1 if and only if
            the experience in x[t] is valid (e.g. episode not ended).
        """
        torch.set_grad_enabled(True)

        batch_size = len(x)
        self.iteration += 1

        nbr_actors = self.hiddenstate_policy.get_nbr_actor()

        if biasing:
            hiddenstate_policy = self.hiddenstate_policy
            self.hiddenstate_policy.save_inner_state()
        else:
            hiddenstate_policy = self.hiddenstate_policy.clone()
        
        L_rec = torch.zeros(batch_size)
        L_mse = torch.zeros(batch_size)
        per_actor_per_t_per_dim_acc = [[] for _ in range(batch_size)]
        per_actor_rec_accuracy = torch.zeros(batch_size)

        dataframe_dict = {
            'actor_id': [],
            'timestep': [],
        }

        for actor_id in range(batch_size):
            hiddenstate_policy.reset(1)
            labels_list = y[actor_id]
            # in range [0,1]
            # 1xsignal_to_reconstruct_dim

            T = len(x[actor_id])
            if mask is None:
                eff_mask = torch.ones((batch_size, T))
            else:
                eff_mask = mask 

            for t in range(T):
                m = eff_mask[actor_id][t] 
                labels = labels_list[t]
                if biasing:
                    hs_t = hiddenstate_policy(x[actor_id][t])
                    # 1 x hidden_state_dim
                else:
                    with torch.no_grad():
                        hs_t = hiddenstate_policy(x[actor_id][t]).detach()
                    # 1 x hidden_state_dim
                
                logit_pred = self.prediction_net(hs_t.reshape(1,-1))

                m = m.to(hs_t.device)
                if labels.device != logit_pred.device: labels = labels.to(logit_pred.device)    
                ###                
                pred = torch.sigmoid(logit_pred)
                # 1x dim
                if 'accuracy_pre_process_fn' not in self.config:
                    per_dim_acc_t = (((pred-self.rec_threshold<=labels).float()+(pred+self.rec_threshold>=labels)).float()>=2).float()
                else:
                    per_dim_acc_t = self.config['accuracy_pre_process_fn'](pred=pred, target=labels)
                # 1x dim
                per_actor_per_t_per_dim_acc[actor_id].append(per_dim_acc_t)
                ###

                L_rec_t = self.criterion(
                    input=logit_pred,
                    target=labels.detach(),
                ).mean()
                # 1 
                L_mse_t = 0.5*torch.pow(pred-labels, 2.0).mean()
                # 1

                if L_rec.device != L_rec_t.device:    L_rec = L_rec.to(L_rec_t.device)
                if L_mse.device != L_mse_t.device:    L_mse = L_mse.to(L_mse_t.device)
                
                L_rec[actor_id:actor_id+1] += m*L_rec_t.reshape(-1)
                # batch_size
                L_mse[actor_id:actor_id+1] += m*L_mse_t.reshape(-1)

                dataframe_dict['actor_id'].append(actor_id)
                dataframe_dict['timestep'].append(t)

            per_actor_per_t_per_dim_acc[actor_id] = torch.cat(per_actor_per_t_per_dim_acc[actor_id], dim=0)
            # timesteps x nbr_goal
            #correct_pred_indices = torch.nonzero((per_actor_per_t_per_dim_acc[actor_id].sum(dim=-1)==self.signal_to_reconstruct_dim).float())
            #per_actor_rec_accuracy[actor_id] = correct_pred_indices.shape[0]/T*100.0
            per_actor_rec_accuracy[actor_id] = per_actor_per_t_per_dim_acc[actor_id].mean()*100.0
            ###
        
        if biasing:
            self.hiddenstate_policy.reset(nbr_actors, training=True)
            self.hiddenstate_policy.restore_inner_state()
        
        torch.set_grad_enabled(False)

        output_dict = {
            'l_rec':L_rec,
            'l_mse':L_mse,
            'per_actor_rec_accuracy':per_actor_rec_accuracy, 
        }

        return output_dict
    
    def _batch(
        self,
        x:List[List[Any]],
        minT:int,
        ):
        batch_size = len(x)
        
        if isinstance(x[0][0], torch.Tensor):
            x = [torch.stack([x_a_t for x_a_t in x_a[:minT]], dim=0) for x_a in x]
            return torch.cat(x, dim=1)
            # minT x batch_size
        
        assert isinstance(x[0][0], dict)
        rx = [dict() for _ in range(minT)]
        for key in x[0][0]:
            if isinstance(x[0][0][key], torch.Tensor):
                for t in range(minT):
                    x_b_t = torch.cat([x[bidx][t][key] for bidx in range(batch_size)], dim=0)
                    # batch_size x ...
                    rx[t][key] = x_b_t
            elif isinstance(x[0][0][key], np.ndarray):
                for t in range(minT):
                    x_b_t = np.concatenate([x[bidx][t][key] for bidx in range(batch_size)], axis=0)
                    # batch_size x ...
                    rx[t][key] = x_b_t
            elif isinstance(x[0][0][key], list):
                assert isinstance(x[0][0][key][0], dict)
                for t in range(minT):
                    ld_b_t = [x[bidx][t][key][0] for bidx in range(batch_size)]
                    concat_hdict = _concatenate_list_hdict(
                        lhds=ld_b_t, 
                        concat_fn=partial(torch.cat, dim=0),
                        preprocess_fn=(lambda x:torch.from_numpy(x).float() if isinstance(x, np.ndarray) else torch.ones(1, 1).float()*x),
                    )
                    rx[t][key] = [concat_hdict]
            else:
                for t in range(minT):
                    rx[t][key] = x[0][0][key]
        return rx

    def compute_batched_reconstruction_loss(
        self,
        x:List[List[Any]],
        y:List[List[torch.Tensor]],
        mask:List[List[Any]]=None,
        biasing:bool=False,
        ) -> Dict[str,torch.Tensor]:
        """
        WARNING: this function resets the :attr hiddenstate_policy:! 
        Beware of potentially erasing agent's current's internal states
        N.B.: This should be resolved by the save_inner_state method.

        :param x: 
            List[List[object]] containing, for each actor, at each time step t an object
            representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the :attr hiddenstate_policy:.
        
        :param y: 
            List[List[torch.Tensor]] containing, for each actor, at each time step t an object
            representing the signal to reconstruct.
            Shape: signal_to_reconstruct_dim.
        
        :param mask:
            List[List[object]] containing, for each actor, at each time step t an object
            with batch_size dimensions and whose values are either
            1 or 0. For all actor b, mask[b]==1 if and only if
            the experience in x[t] is valid, i.e. message is not only made of EoS_id==0.
        """
        torch.set_grad_enabled(True)

        batch_size = len(x)
        minT = min([len(xi) for xi in x])

        batched_x, batched_y = self._batch(x, minT), self._batch(y, minT)
        # minT x batch_size : object dim

        self.iteration += 1

        nbr_actors = self.hiddenstate_policy.get_nbr_actor()

        if biasing:
            hiddenstate_policy = self.hiddenstate_policy
            self.hiddenstate_policy.save_inner_state()
        else:
            hiddenstate_policy = self.hiddenstate_policy.clone()
        
        L_rec = torch.zeros(batch_size)
        L_mse = torch.zeros(batch_size)
        per_actor_per_t_per_dim_acc = [[] for _ in range(batch_size)]
        per_actor_rec_accuracy = torch.zeros(batch_size)

        dataframe_dict = {
            'actor_id': [],
            'timestep': [],
        }
        
        hiddenstate_policy.reset(batch_size)

        labels_list = batched_y
        # in range [0,1]
        # minT x batch_size x signal_to_reconstruct_dim

        if mask is None:
            if 'masking_fn' in self.config:
                eff_mask = self.config['masking_fn'](
                    x=batched_x,
                    y=batched_y,
                )
            else:
                eff_mask = torch.ones((batch_size, minT))
        else:
            raise NotImplementedError
            # Most usecase assume mask==None...
            eff_mask = mask 

        for t in range(minT):
            m = eff_mask[:, t] 
            labels = labels_list[t]
            if biasing:
                hs_t = hiddenstate_policy(batched_x[t])
                # batch_size x hidden_state_dim
            else:
                with torch.no_grad():
                    hs_t = hiddenstate_policy(batched_x[t]).detach()
                # batch_size x hidden_state_dim
                
            logit_pred = self.prediction_net(hs_t.reshape(batch_size,-1))

            m = m.to(hs_t.device)
            if labels.device != logit_pred.device: labels = labels.to(logit_pred.device)    
            ###                
            if 'BCE' in self.config.get("reconstruction_loss", 'BCE'):
                pred = torch.sigmoid(logit_pred)
            else:
                pred = logit_pred
            # batch_size x dim
            
            if 'accuracy_pre_process_fn' not in self.config:
                per_dim_acc_t = (((pred-self.rec_threshold<=labels).float()+(pred+self.rec_threshold>=labels)).float()>=2).float()
            else:
                out_d = self.config['accuracy_pre_process_fn'](pred=pred, target=labels)
                per_dim_acc_t = out_d['acc']
                if 'mask' in out_d:
                    m = out_d['mask'].detach()
            # batch_size x dim

            for actor_id in range(batch_size):
                # Only consider acc when valid (i.e. filtered in) :
                if m[actor_id].item() > 0.5:
                    per_actor_per_t_per_dim_acc[actor_id].append(per_dim_acc_t[actor_id:actor_id+1])
            ###

            L_rec_t = self.criterion(
                input=logit_pred,
                target=labels.detach(),
            ).mean(dim=-1)
            # batch_size 
            L_mse_t = 0.5*torch.pow(pred-labels, 2.0).mean(dim=-1)
            # batch_size

            if L_rec.device != L_rec_t.device:    L_rec = L_rec.to(L_rec_t.device)
            if L_mse.device != L_mse_t.device:    L_mse = L_mse.to(L_mse_t.device)
            
            L_rec += m*L_rec_t
            L_mse += m*L_mse_t

            for actor_id in range(batch_size):
                dataframe_dict['actor_id'].append(actor_id)
                dataframe_dict['timestep'].append(t)

        for actor_id in range(batch_size):
            per_actor_per_t_per_dim_acc[actor_id] = torch.cat(per_actor_per_t_per_dim_acc[actor_id], dim=0)
            # timesteps x nbr_goal
            #correct_pred_indices = torch.nonzero((per_actor_per_t_per_dim_acc[actor_id].sum(dim=-1)==self.signal_to_reconstruct_dim).float())
            #per_actor_rec_accuracy[actor_id] = correct_pred_indices.shape[0]/T*100.0
            per_actor_rec_accuracy[actor_id] = per_actor_per_t_per_dim_acc[actor_id].mean()*100.0
            ###
        
        if biasing:
            self.hiddenstate_policy.reset(nbr_actors, training=True)
            self.hiddenstate_policy.restore_inner_state()
        
        torch.set_grad_enabled(False)

        output_dict = {
            'l_rec':L_rec,
            'l_mse':L_mse,
            'per_actor_rec_accuracy':per_actor_rec_accuracy, 
        }

        return output_dict


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
                            'training':True,
                        }
                        for exp in traj[self.player_id]
                    ]    
                    for traj in trajectories
                ]
                # (batch_size, timestep, keys:values) 
                
                if hasattr(self,'build_signal_to_reconstruct_from_trajectory_fn'):
                    self.labels = [ 
                        self.build_signal_to_reconstruct_from_trajectory_fn(
                            traj=traj,
                            player_id=self.player_id,
                        )
                        for traj in trajectories
                    ]
                else:
                    raise NotImplementedError

                x = self.x 
                labels = self.labels
            else:
                if not hasattr(self, 'x'):
                    return outputs_stream_dict

            if filtering_signal:
                indices = list(range(len(self.x)))
            else:
                indices = np.random.choice(list(range(len(self.x))), size=len(self.x)//self.sampling_fraction, replace=False)
            
            x = [traj for idx, traj in enumerate(self.x) if idx in indices]
            labels = [labels for idx, labels in enumerate(self.labels) if idx in indices]

            mask = None
            
            ## Measure:
            #output_dict = self.compute_reconstruction_loss(
            output_dict = self.compute_batched_reconstruction_loss(
                x=x, 
                y=labels,
                mask=mask,
                biasing=self.biasing,
            )
            
            L_rec = output_dict['l_rec']
            # batch_size
            rec_accuracy = output_dict['per_actor_rec_accuracy']
            # batch_size

            L_mse = output_dict['l_mse']

            logs_dict[f"{mode}/{self.id}/ReconstructionAccuracy/{'Eval' if filtering_signal else 'Sample'}"] = rec_accuracy.mean()
            
            logs_dict[f"{mode}/{self.id}/ReconstructionLoss/Log/MSE/{'Eval' if filtering_signal else 'Sample'}"] = L_mse.mean()
            #logs_dict[f"{mode}/{self.id}/ReconstructionLoss/Log/BCE/{'Eval' if filtering_signal else 'Sample'}"] = L_rec.mean()


            losses_dict = input_streams_dict["losses_dict"]
            #losses_dict[f"{mode}/{self.id}/ReconstructionLoss/{'Eval' if filtering_signal else 'Sample'}"] = [1.0, L_rec]
            #losses_dict[f"{mode}/{self.id}/ReconstructionLoss/MSE/{'Eval' if filtering_signal else 'Sample'}"] = [1.0, L_mse]
            losses_dict[f"{mode}/{self.id}/ReconstructionLoss/{self.config.get('reconstruction_loss', 'BCE')}/{'Eval' if filtering_signal else 'Sample'}"] = [1.0, L_rec]
 
        return outputs_stream_dict
    
