from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Archi.modules.module import Module
from Archi.modules.utils import layer_init
from Archi.modules.convolutional_network_module import addXYRhoThetaFeatures, coord4conv

class FiLMedModule(Module):
    def __init__(
        self,
        id='FiLMedModule_0',
        config:Optional[Dict[str,bool]]={
            'nbr_input_channels':32,
            'nbr_input_features':32,
            'nbr_output_channels':32,
            'kernel_sizes':[1,3],
            'strides':[1,1],
            'paddings':[0,1],
            'use_coordconv': True,
            'use_residual_connection': False,
        },
        input_stream_ids=None,
        output_stream_ids={},
        use_cuda=False,
    ):
        '''
        :param config: Dict[str, bool] with following entries:
            
            - "use_coordconv" : whether to use coord conv in conv layers. 
            - "use_residual_connection": whether to use residual connection in film block modules.
            
        '''
        super(FiLMedModule, self).__init__(
            id=id,
            type="FiLMedModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        self.cnn1 = []
        input_channels = self.config["nbr_input_channels"] 
        if self.config.get("use_coordconv", False):
            self.cnn1.append(addXYRhoThetaFeatures())
            input_channels += 4
        self.cnn1.append(layer_init(
            nn.Conv2d(
                in_channels=input_channels, 
                out_channels=self.config["nbr_output_channels"],
                kernel_size=self.config["kernel_sizes"][0],
                stride=self.config["strides"][0],
                padding=self.config["paddings"][0],
                bias=False, 
                groups=1, 
                dilation=1,
            )
        ))
        self.cnn1.append(nn.BatchNorm2d(self.config["nbr_output_channels"]))
        self.cnn1.append(nn.ReLU())
        self.cnn1 = nn.Sequential(*self.cnn1)

        self.cnn2 = nn.Conv2d(
            in_channels=self.config["nbr_output_channels"],
            out_channels=self.config["nbr_output_channels"],
            kernel_size=self.config["kernel_sizes"][1],
            stride=self.config["strides"][1],
            padding=self.config["paddings"][1],
            bias=False, 
            groups=1, 
            dilation=1,
        )

        self.weight = layer_init(
            nn.Linear(
                self.config["nbr_input_features"],
                self.config["nbr_output_channels"],
                bias=True,
            )
        )
        self.bias = layer_init(
            nn.Linear(
                self.config["nbr_input_features"],
                self.config["nbr_output_channels"],
                bias=True,
            )
        )
        
        self.postprocess = nn.Sequential(
            nn.BatchNorm2d(self.config["nbr_output_channels"]),
            nn.ReLU(),
        )

        self.feature_dim = self.config["nbr_output_channels"]

        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def reset(self):
        self.features_map = None

    def _compute_feat_map(self, x, y):
        in_feat_map = x 
        x1 = self.cnn1(in_feat_map)
        x2 = self.cnn2(x1)
        out1 = x2 * self.weight(y).unsqueeze(2).unsqueeze(3)
        out = out1+self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.postprocess(out)
        if self.config.get("use_residual_connection", False):
            out = out + x1  
        return out

    def get_feat_map(self):
        return self.features_map
    
    def forward(self, x, y):
        self.features_map = self._compute_feat_map(x, y)
        return self.features_map

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.
        
        WARNING: Experiences can be of shape [batch_size (x temporal_dim), depth_dim, h,w].

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        """
        outputs_stream_dict = {}
        
        for key, experiences_x in input_streams_dict.items():
            if "modulation" in key:  continue

            output_key = f"processed_{key}"
            if key in self.output_stream_ids:
                output_key = self.output_stream_ids[key]
            
            experiences_y = input_streams_dict.get(f"modulation_{key}", None)
            if experiences_y is None:   raise AssertionError(f"Need to provide modulation_{key} stream!")
            
            if isinstance(experiences_x, list):
                assert len(experiences_x)==1, f"Provided too many input on id:{key}"
                experiences_x = experiences_x[0]
            if isinstance(experiences_y, list):
                assert len(experiences_y)==1, f"Provided too many input on id:{key}"
                experiences_y = experiences_y[0]
            
            batch_size = experiences_x.size(0)
            original_x_shape = experiences_x.shape
            temporal_x_dim = None
            if len(original_x_shape)>4:
                temporal_x_dim = original_x_shape[1]
                experiences_x = experiences_x.reshape(batch_size*temporal_x_dim, *original_x_shape[2:])
                
                original_y_shape = experiences_y.shape
                if len(original_y_shape)>3:
                    temporal_y_dim = original_y_dim
                    assert temporal_x_dim == temporal_y_dim
                    experiences_y = experiences_y.reshape(batch_size*temporal_y_dim, *original_y_shape[1:])

            if self.use_cuda:   
                experiences_x = experiences_x.cuda()
                experiences_y = experiences_y.cuda()

            features = self.forward(experiences_x, experiences_y)

            if len(original_x_shape)>4:
                features = features.reshape(
                    batch_size,
                    temporal_dim,
                    *features.shape[1:],
                )

            outputs_stream_dict[output_key] = [features]
            
        return outputs_stream_dict 

    def get_feature_shape(self):
        return self.feature_dim

 
