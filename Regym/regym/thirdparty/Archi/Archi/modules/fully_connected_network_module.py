from typing import Dict, List 

import math 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from Archi.modules.module import Module 

from Archi.modules.utils import layer_init


class FullyConnectedNetworkModule(Module):
    def __init__(
        self, 
        state_dim, 
        hidden_units=None, 
        non_linearities=None, 
        dropout=0.0,
        id='FCNModule_0', 
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
        use_cuda=False
    ):

        super(FullyConnectedNetworkModule, self).__init__(
            id=id,
            type="FullyConnectedNetworkModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        if isinstance(state_dim,int): state_dim = [state_dim]

        if hidden_units is None:
            if config is not None and  'hidden_units' in config:
                hidden_units = config['hidden_units']

        dims = state_dim + hidden_units
        self.dropout = dropout

        if non_linearities is None:
            if config is not None and 'non_linearities' in config:
                non_linearities = config['non_linearities']

        self.non_linearities = non_linearities
        if not isinstance(non_linearities, list):
            self.non_linearities = [non_linearities]
        if len(self.non_linearities) > (len(dims)-1):
            raise ImplementationError(f"Design has too many non-linearities for {self.id}.")
        while len(self.non_linearities) < (len(dims) - 1):
            self.non_linearities.append(self.non_linearities[0])
        
        for idx, nl in enumerate(self.non_linearities):
            if nl=='None':
                nl_cls = None
            elif isinstance(nl, str):
                nl_cls = getattr(nn, nl, None)
                if nl_cls is None:
                    raise NotImplementedError
            else:
                nl_cls = nl
            self.non_linearities[idx] = nl_cls
        
        self.layers = []
        in_ch = dims[0]
        for idx, cfg in enumerate(dims[1:]):
            add_non_lin = True
            
            # No non-linearity on the output layer
            #if idx == len(dims)-2:  add_non_lin = False
            
            add_dp = (self.dropout > 0.0)
            dropout = self.dropout
            add_bn = False
            add_ln = False
            if isinstance(cfg, str) and 'NoNonLin' in cfg:
                add_non_lin = False
                cfg = cfg.replace('NoNonLin', '') 
            if isinstance(cfg, str) and '_DP' in cfg:
                add_dp = True
                cfg = cfg.split('_DP')
                dropout = float(cfg[-1])
                cfg = cfg[0] 
                # Assumes 'YX_DPZ'
                # where Y may be BN/LN/nothing
                # and X is an integer
                # and Z is the float dropout value.
            
            if isinstance(cfg, str) and 'BN' in cfg:
                add_bn = True
                cfg = int(cfg[2:])
                dims[idx] = cfg
                # Assumes 'BNX' where X is an integer...
            elif isinstance(cfg, str) and 'LN' in cfg:
                add_ln = True
                cfg = int(cfg[2:])
                dims[idx] = cfg
                # Assumes 'LNX' where X is an integer...
            elif isinstance(cfg, str):
                cfg = int(cfg)
                dims[idx] = cfg
                
            layer = nn.Linear(in_ch, cfg, bias=not(add_bn)) 
            layer = layer_init(layer, w_scale=math.sqrt(2))
            in_ch = cfg
            self.layers.append(layer)
            if add_bn:
                self.layers.append(nn.BatchNorm1d(in_ch))
            if add_ln:
                # Layer Normalization:
                # solely about the last dimension of the 4D tensor, i.e. channels...
                # TODO: It might be necessary to have the possibility to apply this 
                # normalization over the other dimensions, i.e. width x height...
                self.layers.append(nn.LayerNorm(in_ch))
            if add_dp:
                self.layers.append(nn.Dropout(p=dropout))
            if add_non_lin \
            and self.non_linearities[idx] is not None:
                self.layers.append(self.non_linearities[idx]())
        self.layers = nn.Sequential(*self.layers)

        self.feature_dim = dims[-1]

        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()
    
    def reset(self):
        self.output = None 

    def forward(self, x):
        self.output = self.layers(x)
        return self.output

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

        for key, experiences in input_streams_dict.items():
            output_key = f"processed_{key}"
            if key in self.output_stream_ids:
                output_key = self.output_stream_ids[key]

            if isinstance(experiences, list):
                assert len(experiences)==1, f"Provided too many input on id:{key}"
                experiences = experiences[0]
            batch_size = experiences.size(0)
           
            original_shape = experiences.shape
            if len(original_shape)>2 \
            and original_shape[-1] == original_shape[-2]:
                # i.e. output from CNN, just needs flattening:
                flatdim = np.prod(original_shape[-3:])
                experiences = experiences.reshape(*original_shape[:-3], flatdim)
                original_shape = experiences.shape

            if len(original_shape)>2:
                temporal_dims = original_shape[1:-1]
                product_tdims = np.prod(temporal_dims)
                experiences = experiences.view(batch_size*product_tdims, original_shape[-1])
            
            if self.use_cuda:   experiences = experiences.cuda()

            features = self.layers(experiences)

            if len(original_shape)>2:
                features = features.reshape(
                    batch_size,
                    *temporal_dims,
                    *features.shape[1:],
                )

            outputs_stream_dict[output_key] = [features]
        
        return outputs_stream_dict 

    def get_feature_shape(self):
        return self.feature_dim



