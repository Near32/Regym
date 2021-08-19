from typing import Dict, List, Optional, Any 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from regym.modules import Module 

class ModuleWrapper(Module):
    def __init__(
        self, 
        id='ModuleWrapper_0', 
        config=None,
        input_stream_ids=None,
        ):
        super(ModuleWrapper, self).__init__(
            id=id,
            type="ModuleWrapper",
            config=config,
            input_stream_ids=input_stream_ids,
        )
        
        self.body = None
        # TODO implemente make arch fn...

        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

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
            batch_size = experiences.size(0)
            nbr_distractors_po = experiences.size(1)
            nbr_stimulus = experiences.size(2)

            experiences = experiences.view(batch_size*nbr_distractors_po, -1)
            if self.use_cuda:   experiences = experiences.cuda()

            features = self.layers(experiences)
            features = features.reshape(batch_size, nbr_distractors_po, nbr_stimulus, -1)
            # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, feature_dim)
            
            outputs_stream_dict[key] = features

        return outputs_stream_dict 

    def get_feature_shape(self):
        return self.feature_dim


