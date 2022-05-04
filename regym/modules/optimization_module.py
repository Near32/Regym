from typing import Dict, List 

import os 

import torch
import torch.nn as nn
import torch.optim as optim 

from regym.modules import Module

def hasnan(tensor):
    if torch.isnan(tensor).max().item() == 1:
        return True
    return False

def reg_nan(param, verbose=False):
    if param is None or param.data is None: return param
    nan_indices = torch.isnan(param.data)
    if verbose and torch.any(nan_indices).item(): 
        print("WARNING: NaN found in {}.".format(param))
    param.data[nan_indices] = 0
    if param.grad is None: return param
    nan_indices = torch.isnan(param.grad.data)
    if verbose and torch.any(nan_indices).item(): 
        print("WARNING: NaN found in the GRADIENT of {}.".format(param))
    param.grad.data[nan_indices] = 0
    return param 

def handle_nan(layer, verbose=True):
    for name, param in layer._parameters.items():
        if param is None or param.data is None: continue
        nan_indices = torch.isnan(layer._parameters[name].data)
        if verbose and torch.any(nan_indices).item(): 
            print("WARNING: NaN found in {} of {}.".format(name, layer))
        layer._parameters[name].data[nan_indices] = 0
        if param.grad is None: continue
        nan_indices = torch.isnan(layer._parameters[name].grad.data)
        if verbose and torch.any(nan_indices).item(): 
            print("WARNING: NaN found in the GRADIENT of {} of {}.".format(name, layer))
        layer._parameters[name].grad.data[nan_indices] = 0

#TODO:
"""
1) Maybe make it possible for this module to ignore some task loss:
--> implement a mask-based policy?
"""

def build_OptimizationModule(id:str,
                             config:Dict[str,object],
                             input_stream_ids:Dict[str,str]=None) -> Module:
    return OptimizationModule(id=id,
                              config=config, 
                              input_stream_ids=input_stream_ids)


class OptimizationModule(Module):
    def __init__(self,
                 id:str,
                 config:Dict[str,object],
                 input_stream_ids:Dict[str,str]=None):

        if input_stream_ids is None:
            input_stream_ids = {
                "losses_dict":"losses_dict",
                "logs_dict":"logs_dict",
                "mode":"signals:mode",
                "it_sample":"signals:it_sample",
                # step in the sequence of repetitions of the current batch
                "it_step":"signals:it_step",
                # step in the communication round.
            }

        assert "modules" in config,\
               "OptimizationModule relies on list of modules.\n\
                Not found in config."
        
        assert "optimizer_type" in config,\
               "OptimizationModule relies on 'optimizer_type'.\n\
                Not found in config."
        
        assert "mode" in input_stream_ids.keys(),\
               "OptimizationModule relies on 'mode' id.\n\
                Not found in input_stream_ids."
        
        assert "losses_dict" in input_stream_ids.keys(),\
               "OptimizationModule relies on 'losses_dict' id.\n\
                Not found in input_stream_ids."

        assert "logs_dict" in input_stream_ids.keys(),\
               "OptimizationModule relies on 'logs_dict' id.\n\
                Not found in input_stream_ids."
        
        super(OptimizationModule, self).__init__(id=id,
                                                 type="OptimizationModule",
                                                 config=config,
                                                 input_stream_ids=input_stream_ids)
        self.update_count = 0
        parameters = []
        for k,m in self.config["modules"].items():
            parameters += m.parameters()
            print(k)
        
        if len(list(parameters)):
          if "sgd" in self.config["optimizer_type"].lower():
            self.optimizer = optim.SGD(
                parameters, 
                lr=self.config["learning_rate"],
                weight_decay=config.get("weight_decay", 0.0),
            )
          else:
            self.optimizer = optim.Adam(
                parameters, 
                lr=self.config["learning_rate"], 
                #betas=(0.9, 0.999), 
                eps=self.config["adam_eps"],
                weight_decay=config.get("weight_decay", 0.0),
            )
        else:
          self.optimizer = None 
        
    def save(self, path):
      if self.optimizer is not None:
        torch.save(self.optimizer.state_dict(), os.path.join(path, self.id+".module"))

    def load(self, path):
      if self.optimizer is not None:
        self.optimizer.load_state_dict(torch.load(os.path.join(path, self.id+".module")))

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
        torch.set_grad_enabled(True)

        outputs_stream_dict = {}

        losses_dict = input_streams_dict["losses_dict"]
        logs_dict = input_streams_dict["logs_dict"]
        mode = input_streams_dict["mode"]

        it_rep = input_streams_dict["it_sample"]
        it_comm_round = input_streams_dict["it_step"]

        for l_name, l in losses_dict.items():
            logs_dict[f"{mode}/{l_name}"] = l[-1]
        
        for l_name, l in losses_dict.items():
            losses_dict[l_name] = l[0]*l[-1]
        
        loss = sum([l.mean() for l in losses_dict.values()])

        if "train" in mode\
        and len(losses_dict)\
        and self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()

            for k,m in self.config["modules"].items():
                m.apply(handle_nan)
                if self.config["with_gradient_clip"]:
                    nn.utils.clip_grad_value_(m.parameters(), self.config["gradient_clip"])
            
            self.optimizer.step()
            self.update_count += 1

            logs_dict[f"{mode}/repetition{it_rep}/comm_round{it_comm_round}/Loss"] = loss
        
            outputs_stream_dict['signals:update_count'] = self.update_count
        
        torch.set_grad_enabled(False)

        return outputs_stream_dict
        
