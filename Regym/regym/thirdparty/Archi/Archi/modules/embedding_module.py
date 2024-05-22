from typing import Dict, List 

import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

from Archi.modules.module import Module 


class EmbeddingModule(Module):
    def __init__(
        self, 
        num_embeddings, 
        embedding_dim, 
        padding_idx=None,
        max_norm=None,
        id='EModule_0', 
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
        use_cuda=False
    ):

        super(EmbeddingModule, self).__init__(
            id=id,
            type="EmbeddingModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm

        self.embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
        )

        self.feature_dim = self.embedding_dim

        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()
    
    def reset(self):
        self.output = None

    def forward(self, x):
        shape_len = len(x.shape)
        self.output = self.embed(x.long())
        while len(self.output.shape)!=shape_len:
            self.output = self.output.squeeze(1)
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

            experiences = experiences.view(batch_size, -1)
            if self.use_cuda:   experiences = experiences.cuda()

            features = self.forward(experiences)
            outputs_stream_dict[output_key] = [features]
        
        return outputs_stream_dict 

    def get_feature_shape(self):
        return self.feature_dim



