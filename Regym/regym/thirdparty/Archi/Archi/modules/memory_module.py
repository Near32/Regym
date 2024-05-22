from typing import Dict, List 

import torch
from torch import nn, einsum
import torch.nn.functional as F
import copy

from Archi.modules.module import Module 

class MemoryModule(Module):
    def __init__(
        self, 
        dim=64, 
        id='MemoryModule_0', 
        config=None,
        input_stream_ids=None,
        use_cuda=False
    ):

        super(MemoryModule, self).__init__(
            id=id,
            type="MemoryModule",
            config=config,
            input_stream_ids=input_stream_ids,
        )
        
        self.dim = dim
        
        self.initialize()
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()
    
    def reset(self):
        pass

    def initialize(self):
        # Constant
        ## Null :
        self.init_mem = torch.zeros((1, 1, self.dim))

    def get_reset_states(self, cuda=False, repeat=1):
        kh = self.init_mem.clone().repeat(repeat, 1, 1)
        if cuda:    kh = kh.cuda()
        iteration = torch.zeros((repeat, 1))
        if cuda:    iteration = iteration.cuda()
        hd = {
            'memory': [kh],
            'iteration': [iteration],
        }
        return hd
    
    def forward(
        self, 
        new_element:torch.Tensor,
        memory:List[torch.Tensor],
        iteration:List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        output_dict = {}
        
        batch_size = iteration[0].shape[0]
        element_dim = memory[0].shape[-1]

        # Write values in memory:
        # AT THE CORRECT INDEX, irrespective of the actual size of the memory,
        # but with respect to the iteration count:
        iteration[0] += 1
        # Because the iteration tensor is updated first,
        # the iteration count corresponds to the slot in which the new memory elements should be inserted:
        # check that the memory is big enough:
        
        #print(f"iterations memory size:", iteration[0].transpose(0,1), key_memory[0].shape)

        """
        let us resize the memory to fit the iteration needs:
        """
        max_it = iteration[0].long().max().item()
        new_memory = memory[0][:,:max_it,...].clone()
        if self.use_cuda:
            new_memory = new_memory.cuda()
        nbr_memory_items = new_memory.shape[-2]

        new_memory = torch.cat([
            new_memory, 
            torch.zeros((
                batch_size, 
                max_it-nbr_memory_items+1, 
                element_dim,
            )).to(new_element.device),
            ],
            dim=1,
        )
        nbr_memory_items = new_memory.shape[-2]
        
        new_memory.scatter_(
            dim=-2,
            index=iteration[0].long().unsqueeze(-1).repeat(1,1,new_element.shape[-1]).to(new_memory.device),
            src=new_element.unsqueeze(1),
        )

        outputs_dict = {
            'memory': [new_memory],
            'iteration': iteration,
        }

        return outputs_dict

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object]:
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
        
        iteration = input_streams_dict['iteration']

        new_element = input_streams_dict['new_element'][0]
        memory = input_streams_dict['memory']
        
        output_dict = self.forward(
            iteration=iteration,
            memory=memory,
            new_element=new_element,
        )

        outputs_stream_dict.update(output_dict)

        # Bookkeeping:
        for k,v in output_dict.items():
            outputs_stream_dict[f"inputs:{self.id}:{k}"] = v

        return outputs_stream_dict

