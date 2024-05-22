from typing import Dict, List, Tuple, Optional

import os
import copy
from functools import partial

import torch
import torch.nn as nn

from Archi.utils import StreamHandler
from Archi.modules import Module, load_module 
from Archi.modules.utils import (
    copy_hdict, 
    apply_on_hdict,
    recursive_inplace_update,
)


class Model(Module):
    def __init__(
        self,
        module_id: str="Model_0",
        config: Dict[str,object]={},
        input_stream_ids: Dict[str,str]={},
    ):
        """
        Expected keys in :param config::
            - 'modules'     : Dict[str,object]={},
            - 'pipelines'   : Dict[str, List[str]]={},
            - 'load_path'   : str,
            - 'save_path'   : str,
        
        """
        super(Model, self).__init__(
            id=module_id,
            type="ModelModule",
            config=config,
            input_stream_ids=input_stream_ids,
        )
        
        assert 'modules' in self.config
        assert 'pipelines' in self.config
        
        self.modules = self.config['modules']
        for km, vm in self.modules.items():
            self.add_module(km, vm)

        # Register Pipelines:
        self.pipelines = self.config['pipelines']
        
        self.extra_reset_states = {}
        self.reset()
   
    #def get_reset_states(self, **kwargs):
    def get_reset_states(self, kwargs={}):
        """
        Provide a reset state without changing the current state.
        """
        batch_size = kwargs.get("repeat", 1)
        cuda = kwargs.get("cuda", False)
        rs = {}
        for k,m in self.config['modules'].items():
            if hasattr(m, 'get_reset_states'):
                reset_dict = m.get_reset_states(repeat=batch_size, cuda=cuda)
                #print(f"module {k} : reset dict:")
                #for ks, v in reset_dict.items():
                #    #print(f"--> {ks} : {type(v)}")
                rs[m.get_id()] = reset_dict
        
        def reg(x):
            outx = x.repeat(batch_size, *[1 for _ in range(len(x.shape)-1)])
            if cuda:  outx = outx.cuda()
            return outx
        for k, hdict in self.extra_reset_states.items():
            rs[k] = apply_on_hdict(
                hdict=hdict,
                fn=reg,
            )
        return rs
   
    def set_reset_states(self, new_reset_states):
        """
        Reset the reset states of every module.
        """
        self.extra_reset_states = {}
        for k,m in self.config['modules'].items():
            if hasattr(m, 'set_reset_states') \
            and k in new_reset_states.keys():
                m.set_reset_states(new_reset_states[k])
        for k, hdict in new_reset_states.items():
            if k in self.config['modules']:   continue
            self.extra_reset_states[k] = hdict
        return

    def reset(self):
        self.stream_handler = StreamHandler()
        self.stream_handler.register("logs_dict")
        self.stream_handler.register("losses_dict")
        self.stream_handler.register("signals")
        
        # Register Hyperparameters:
        for k,v in self.config.items():
            self.stream_handler.update(f"config:{k}", v)
        
        # Register Modules:
        for k,m in self.config['modules'].items():
            if hasattr(m, "reset"):  m.reset()
            self.stream_handler.update(f"modules:{m.get_id()}:ref", m)

        # Reset States:
        self.reset_states()
        
        self.output_stream_dict = None

    def reset_noise(self):
        # TODO : investiguate implementation of parameter noise ...
        pass

    def reset_states(self, batch_size=1, cuda=False):
        # WATCHOUT: reset states is usually called after inputs 
        # have been setup with obs etc.
        # Thus, it is not possible to call the following:
        # self.stream_handler.reset("inputs")
        self.batch_size = batch_size
        for k,m in self.config['modules'].items():
            if hasattr(m, 'get_reset_states'):
                reset_dict = m.get_reset_states(repeat=batch_size, cuda=cuda)
                #print(f"module {k} : reset dict:")
                for ks, v in reset_dict.items():
                    #print(f"--> {ks} : {type(v)}")
                    self.stream_handler.update(f"inputs:{m.get_id()}:{ks}",v)
        return 

    def _forward(self, pipelines=None, **kwargs):
        if pipelines is None:
            pipelines = self.pipelines
        
        self.stream_handler.reset("inputs")
        self.stream_handler.reset("outputs")
        
        batch_size = 1
        batch_size_set = False
        for k,v in kwargs.items():
            if batch_size == 1\
            and isinstance(v, torch.Tensor):
                batch_size = v.shape[0]
                batch_size_set = True
            if not isinstance(v, list)\
            and isinstance(v, torch.Tensor):    
                v = [v]
            self.stream_handler.update(f"inputs:{k}", v)
        # TODO: assert that the following lines are not necessary
        # since all the inputs that are part of the states should have
        # been resetted properly from above...
        """
        if self.batch_size != batch_size:
            assert batch_size_set, "Archi:Model : batch size was not reset properly. Need to provide an observation that is torch.Tensor, maybe?"
            self.reset_states(batch_size=batch_size)
        """

        self.stream_handler.reset("logs_dict")
        self.stream_handler.reset("losses_dict")
        self.stream_handler.reset("signals")
        
        for pipe_id in pipelines.keys():
            if pipe_id not in self.config["input_mappings"].keys(): continue
            for inp, out in self.config["input_mappings"][pipe_id].items():
                inp_v = self.stream_handler[inp]
                self.stream_handler.update(out, inp_v)

        self.stream_handler.start_recording_new_entries()

        for pipe_id, pipeline in pipelines.items():
            self.stream_handler.serve(pipeline)

        new_streams_dict = self.stream_handler.stop_recording_new_entries()

        # Output mapping:
        relevant_output_mappings = {}
        for om_pipeline, om_d in self.config['output_mappings'].items():
            if om_pipeline not in pipelines:    continue
            relevant_output_mappings.update(om_d)

        for k,v in relevant_output_mappings.items():
            value = self.stream_handler[v] 
            if value is None:
                raise NotImplementedError(f"Key {k} is not among Model's output.")
            new_streams_dict[f"outputs:{k}"] = value
        
        return new_streams_dict

    def forward(self, 
        obs: torch.Tensor, 
        action: Optional[torch.Tensor]=None, 
        rnn_states: Optional[Dict[str,object]]={}, 
        goal: Optional=None, 
        pipelines: Optional[Dict[str,List]]=None,
        return_features: Optional[bool]=False,
        return_feature_only: Optional[str]=None, 
        ):
        assert goal is None, "Deprecated goal-oriented usage ; please use frame/rnn_states."
        if pipelines is None:
            pipelines = {
                'torso':self.pipelines['torso'],
                'head':self.pipelines['head'],
            }

        batch_size = obs.shape[0]
       
        self.reset()

        self.output_stream_dict = self._forward(
	    pipelines=pipelines,
            obs=obs,
            action=action,
	    **rnn_states,
	)
        
        id2output = {
            "a":"outputs:a",
            "ent":"outputs:ent",
            "legal_ent":"outputs:legal_ent",
            "qa":"outputs:qa",
            "log_a":"outputs:log_a",
            "unlegal_log_a":"outputs:unlegal_log_a",
        
            'greedy_action': 'outputs:greedy_action',
            'v': 'outputs:v',
            'int_v': 'outputs:int_v',
            'log_pi_a': 'outputs:log_pi_a',
            'legal_log_pi_a': 'outputs:legal_log_pi_a',
        }

        prediction = {}
        for kid,key in id2output.items():
            if key in self.output_stream_dict:
                prediction[kid] = self.output_stream_dict[key]
        
        next_rnn_states = copy_hdict(rnn_states)
        if "inputs" in self.output_stream_dict:
            recursive_inplace_update(
                in_dict=next_rnn_states,
                extra_dict=self.output_stream_dict["inputs"],
            )
        
        prediction.update({
            'rnn_states': rnn_states,
            'next_rnn_states': next_rnn_states
        })
        
        if return_features \
        or return_feature_only is not None:
            features_id = []
            list_pipeline_keys2features = list(self.config["features_id"].keys())
            for pipe_id in pipelines.keys():
                if pipe_id in list_pipeline_keys2features:
                    features_id.append(self.config["features_id"][pipe_id])
            if return_feature_only is not None:
                if return_feature_only not in features_id:  features_id.append(return_feature_only)
            
            assert len(features_id) == 1
            features = self.stream_handler[features_id[0]]

            if isinstance(features, list):
                assert len(features) == 1
                features = features[0]
            if return_feature_only is not None:
                return features
            return features, prediction

        return prediction
    
    def get_torso(self):
        return partial(self.forward, pipelines={"torso":self.pipelines["torso"]}, return_features=True)

    def get_head(self):
        return partial(self.forward, pipelines={"head":self.pipelines["head"]})


def load_model(config: Dict[str, object]) -> Model:
    mcfg = {}
    
    mcfg['output_mappings'] = config.get("output_mappings", {}) 
    mcfg['input_mappings'] = config.get("input_mappings", {}) 
    mcfg['features_id'] = config.get("features_id", "input:obs") 
    mcfg['pipelines'] = config['pipelines']
    mcfg['modules'] = {}
    for mk, m_kwargs in config['modules'].items():
        if 'id' not in m_kwargs:    m_kwargs['id'] = mk
        mcfg['modules'][m_kwargs['id']] = load_module(m_kwargs.pop('type'), m_kwargs)
    
    model = Model(
        module_id = config['model_id'],
        config=mcfg,
        input_stream_ids=config['input_stream_ids'],
    )

    return model 


