from typing import Dict, List 

import torch
import torch.nn as nn
import torch.optim as optim 

import numpy as np 

from .module import Module

import wandb 


def build_PerEpochLoggerModule(id:str,
                               config:Dict[str,object]=None,
                               input_stream_ids:Dict[str,str]=None) -> Module:
    return PerEpochLoggerModule(id=id,
                                config=config, 
                                input_stream_ids=input_stream_ids)


class PerEpochLoggerModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):

        if input_stream_ids is None:
            input_stream_ids = {
                #"logger":"modules:logger:ref",
                "losses_dict":"losses_dict",
                "logs_dict":"logs_dict",
                "epoch":"signals:epoch",
                "update_count":"signals:update_count",
                "agent_update_count":"signals:agent_update_count",
                "mode":"signals:mode",
                "end_of_dataset":"signals:end_of_dataset",  
                # boolean: whether the current batch/datasample is the last of the current dataset/mode.
                "global_it_datasample":"signals:it_datasample",
                "it_datasample":"signals:it_datasample",
                "end_of_repetition_sequence":"signals:end_of_repetition_sequence",
                # boolean: whether the current sample(observation from the agent of the current batch/datasample) 
                # is the last of the current sequence of repetition.
                "global_it_sample":"signals:global_it_sample",
                "it_sample":"signals:it_sample",
                # step in the sequence of repetitions of the current batch
                "end_of_communication":"signals:end_of_communication",
                # boolean: whether the current communication round is the last of 
                # the current dialog.
                "global_it_step":"signals:global_it_step",
                "it_step":"signals:it_step",
                # step in the communication round.
            }

        assert "epoch" in input_stream_ids.keys(),\
               "PerEpochLoggerModule relies on 'epoch' id.\n\
                Not found in input_stream_ids."
        
        assert "mode" in input_stream_ids.keys(),\
               "PerEpochLoggerModule relies on 'mode' id.\n\
                Not found in input_stream_ids."
        
        assert "losses_dict" in input_stream_ids.keys(),\
               "PerEpochLoggerModule relies on 'losses_dict' id.\n\
                Not found in input_stream_ids."

        assert "logs_dict" in input_stream_ids.keys(),\
               "PerEpochLoggerModule relies on 'logs_dict' id.\n\
                Not found in input_stream_ids."
        
        super(PerEpochLoggerModule, self).__init__(id=id,
                                                 type="PerEpochLoggerModule",
                                                 config=config,
                                                 input_stream_ids=input_stream_ids)
        
        self.storages = {}

        self.epoch = 0
        self.end_of_ = [key for key,value in input_stream_ids.items() if "end_of_" in key]
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_stream_dict = {}

        losses_dict = input_streams_dict["losses_dict"]
        logs_dict = input_streams_dict["logs_dict"]
        
        epoch = input_streams_dict["epoch"]
        update_count = input_streams_dict["update_count"]
        agent_update_count = input_streams_dict["agent_update_count"]
        mode = input_streams_dict["mode"]
        global_it_step = input_streams_dict["global_it_step"]
        
        # Store new data:
        for key,value in logs_dict.items():
          if key not in self.storages:
            self.storages[key] = []
          if isinstance(value, torch.Tensor):
            value = value.cpu().detach()
          self.storages[key].append(value)
        
        # Is it the end of the epoch?
        end_of_epoch = all([
          input_streams_dict[key]
          for key in self.end_of_]
        )
        
        # If so, let us average over every value and save it:
        if end_of_epoch or self.epoch != epoch:
          self.epoch = epoch
          for key, valuelist in self.storages.items():
            need_stats = False
            if isinstance(valuelist[0], torch.Tensor):# and len(valuelist[0].shape)>=1:
              values = torch.cat([vl.cpu().detach().reshape(-1) for vl in valuelist], dim=0).numpy()
              need_stats = True
            elif isinstance(valuelist[0], float) or isinstance(valuelist[0], int):
              values = np.asarray(valuelist).reshape(-1)
              if len(valuelist)>1:
                need_stats = True
            else:
              continue

            if need_stats:
              averaged_value = values.mean()
              std_value = values.std()
              wandb.log({f"PerEpoch/{key}/Mean":  averaged_value, "epoch":epoch}, commit=False)
              wandb.log({f"PerEpoch/{key}/Std":  std_value, "epoch":epoch}, commit=False)
              
              wandb.log({f"PerUpdate/{key}/Mean":  averaged_value, "update_count":update_count}, commit=False)
              wandb.log({f"PerUpdate/{key}/Std":  std_value, "update_count":update_count}, commit=False)
              wandb.log({f"PerAgentUpdate/{key}/Mean":  averaged_value, "agent_update_count":agent_update_count}, commit=False)
              wandb.log({f"PerAgentUpdate/{key}/Std":  std_value, "agent_update_count":agent_update_count}, commit=False)
              
 

              median_value = np.nanpercentile(
                values,
                q=50,
                axis=None,
                method="nearest"
              )
              q1_value = np.nanpercentile(
                values,
                q=25,
                axis=None,
                method="lower"
              )
              q3_value = np.nanpercentile(
                values,
                q=75,
                axis=None,
                method="higher"
              )
              iqr = q3_value-q1_value
              
              wandb.log({f"PerEpoch/{key}/Median":  median_value, "epoch":epoch}, commit=False)
              wandb.log({f"PerEpoch/{key}/Q1":  q1_value, "epoch":epoch}, commit=False)
              wandb.log({f"PerEpoch/{key}/Q3":  q3_value, "epoch":epoch}, commit=False)
              wandb.log({f"PerEpoch/{key}/IQR":  iqr, "epoch":epoch}, commit=False)
              
              wandb.log({f"PerUpdate/{key}/Median":  median_value, "update_count":update_count}, commit=False)
              wandb.log({f"PerUpdate/{key}/Q1":  q1_value, "update_count":update_count}, commit=False)
              wandb.log({f"PerUpdate/{key}/Q3":  q3_value, "update_count":update_count}, commit=False)
              wandb.log({f"PerUpdate/{key}/IQR":  iqr, "update_count":update_count}, commit=False)
              wandb.log({f"PerAgentUpdate/{key}/Median":  median_value, "agent_update_count":agent_update_count}, commit=False)
              wandb.log({f"PerAgentUpdate/{key}/Q1":  q1_value, "agent_update_count":agent_update_count}, commit=False)
              wandb.log({f"PerAgentUpdate/{key}/Q3":  q3_value, "agent_update_count":agent_update_count}, commit=False)
              wandb.log({f"PerAgentUpdate/{key}/IQR":  iqr, "agent_update_count":agent_update_count}, commit=False)
              
 
              #logger.add_histogram(f"PerEpoch/{key}", values, epoch)
            else:
              wandb.log({f"PerEpoch/{key}":  valuelist[-1], "epoch":epoch}, commit=False)
              wandb.log({f"PerUpdate/{key}":  valuelist[-1], "update_count":update_count}, commit=False)
              wandb.log({f"PerAgentUpdate/{key}":  valuelist[-1], "agent_update_count":agent_update_count}, commit=False)
              
              # Remove the value form the logs_dict if it is present:
              logs_dict.pop(key, None)

          # Reset epoch storages:
          self.storages = {}

        # Log new (rectified) data:
        for key,value in logs_dict.items():
          if isinstance(value, torch.Tensor): 
              value = value.mean().item()
          wandb.log({key:  value, "global_it_step":global_it_step}, commit=False)
        #wandb.log({}, commit=True)
        
        return {}
        
