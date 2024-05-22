from typing import Dict, List 

import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

from Archi.modules.module import Module 


class ConcatenationOperationModule(Module):
    def __init__(
        self, 
        id='ConcatenationOperationModule_0', 
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
    ):

        super(ConcatenationOperationModule, self).__init__(
            id=id,
            type="ConcatenationOperationModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
    
    def get_reset_states(self, cuda=False, repeat=1):
        #h = torch.zeros((repeat, self.config.get('output_dim',1)))
        # If the output_dim is not provided then, upon concatenation
        # of experiences of a sequence_buffer of R2D2, the concat_fn
        # finds elements of different shapes and must create a np.empty
        # which is then impossible to regularise in the archi_concat_fn...
        h = torch.zeros((repeat, self.config['output_dim']))
        if cuda:
            h = h.cuda()
        output = [h]
        return {'output': output}

        
    def forward(self, **inputs):
        output = torch.cat([v.cuda() if self.config['use_cuda'] else v for k,v in inputs.items()], dim=self.config['dim'])
        return output

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
        
        element_is_list = [isinstance(v,list) for v in input_streams_dict.values()]
        list_size = None
        if any(element_is_list):
            for k,v in input_streams_dict.items():
                if isinstance(v, list): 
                    if list_size is None:
                        list_size = len(v)
                    else:
                        assert list_size == len(v)
                    continue
                assert isinstance(v, torch.Tensor) 
                input_streams_dict[k] = [v]
                list_size = 1
                if list_size is None:   
                    list_size = 1
                else:
                    assert list_size == 1

        if any(element_is_list):
            nbr_elements = len(list(input_streams_dict.values())[0])
            output_list = []
            for idx in range(nbr_elements):
                inputs = {k:v[idx] for k,v in input_streams_dict.items() if 'input' in k}
                output_list.append( self.forward(**inputs))    
            outputs_stream_dict[f'output'] = output_list
        else:
            inputs = {k:v for k,v in input_streams_dict.items() if 'input' in k}
            outputs_stream_dict[f'output'] = [self.forward(**inputs)]
        
        for k in input_streams_dict.keys():
            if 'input' not in k:    continue
            if k in self.output_stream_ids:
                outputs_stream_dict[self.output_stream_ids[k]] = outputs_stream_dict['output']
        
        # Bookkeeping:
        for k in list(outputs_stream_dict.keys()):
            outputs_stream_dict[f"inputs:{self.id}:{k}"] = outputs_stream_dict[k]
        return outputs_stream_dict 

    def get_feature_shape(self):
        return None



