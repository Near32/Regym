from typing import Dict, List 

import torch
from torch import nn, einsum
import torch.nn.functional as F
import copy

from Archi.modules.module import Module 

class ReadHeadsModule(Module):
    def __init__(
        self, 
        nbr_heads=3, 
        top_k=3,
        normalization_fn="softmax",
        normalize_output=True,
        postprocessing="self-attention+sum",
        id='ReadHeadsModule_0', 
        config=None,
        input_stream_ids=None,
        use_cuda=False
    ):
        """
        :params:
            -nbr_heads: Int, number of reading heads, and expected number of queries passed as inputs.
            -top_k: Int, number of samples to consider after nearest neighours operation.
            -normalization_fn: Str from ["softmax","inverse_dissim"] to describe how the nearest neighbours
                scores are used to create a distribution.
            -normalize_output: Bool describing whether the output is normalized with the score distribution.
            -postprocessing: Str describing the kind of post processing scheme to apply to the top-k read values.
        """
        
        super(ReadHeadsModule, self).__init__(
            id=id,
            type="ReadHeadsModule",
            config=config,
            input_stream_ids=input_stream_ids,
        )
        
        self.nbr_heads = nbr_heads
        self.top_k = top_k
        self.normalization_fn = normalization_fn
        self.normalize_output = normalize_output
        self.postprocessing = postprocessing

        if "self-attention" in self.postprocessing:
            self.postprocessings = nn.ModuleList()
            for ridx in range(self.nbr_heads):
                postprocessing = nn.MultiheadAttention(
                    embed_dim=self.config.get("value_dim", 256),
                    #embed_dim=self.config.get("postprocessing_dim", 256),
                    num_heads=self.config.get("postprocessing_num_heads",1),
                    kdim=self.config.get("value_dim", 256),
                    vdim=self.config.get("value_dim",256),
                    dropout=self.config.get("postprocessing_dropout",0.0),
                    batch_first=True,
                )
                self.postprocessings.append(postprocessing)
        else:
            raise NotImplementedError

        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()
    
    def reset(self):
        pass

    def get_reset_states(self, cuda=False, repeat=1):
        khs = [torch.zeros((repeat, self.config.get("value_dim",1))) for _ in range(self.nbr_heads)]
        if cuda:    
            khs = [kh.cuda() for kh in khs]
        hd = {
            f'{idx}_read_value': [kh] for idx, kh in enumerate(khs)
        }
        return hd
    
    def forward(
        self, 
        query:torch.Tensor, 
        key_memory:List[torch.Tensor],
        value_memory:List[torch.Tensor],
        iteration:List[torch.Tensor],
        postprocessing:torch.nn.Module=None,
    ) -> Dict[str, torch.Tensor]:

        output_dict = {}
        
        batch_size = iteration[0].shape[0]
        key_dim = key_memory[-1].shape[-1]
        value_dim = value_memory[0].shape[-1]

        # Masking memory: there can be more memory entry than the current iteration count for each batch element,
        # especially when mixing together batch elements at different timeline for update:
        nbr_memory_items = value_memory[0].shape[1]
        value_memory[0] = value_memory[0].to(query.device)
        key_memory[0] = key_memory[0].to(query.device)
        
        if nbr_memory_items < self.top_k:
            value_memory[0] = torch.cat([
                value_memory[0], 
                torch.zeros((batch_size, self.top_k-nbr_memory_items, value_dim)).to(query.device),
                ], 
                dim=1,
            )
            key_memory[0] = torch.cat([
                key_memory[0], 
                torch.zeros((batch_size, self.top_k-nbr_memory_items, key_dim)).to(query.device),
                ], 
                dim=1, 
            )
            nbr_memory_items = value_memory[0].shape[1]

        enumerate_memory_items = torch.arange(nbr_memory_items).unsqueeze(0).repeat(batch_size, 1).to(query.device)
        memory_mask = (enumerate_memory_items <= iteration[0].to(query.device)).long().unsqueeze(-1)
        # (batch_size x nbr_memory_items x 1)
        not_first_mask = (iteration[0].reshape(-1) > 0) 
        # (batch_size, )
        
        new_read_element= torch.zeros(
            (batch_size, self.top_k, value_dim),
        ).to(query.device)
        
        if not_first_mask.long().sum() > 0: 
            not_first_indices = torch.masked_select(
                input=not_first_mask.long() * torch.arange(batch_size).to(not_first_mask.device),
                mask=not_first_mask,
            )
            # (1<= dim <=batch_size, )
            # (batch_size, n, value_dim)
            km = key_memory[0][not_first_indices, ...].to(query.device)
            km = km*memory_mask[not_first_indices, ...]
            # (dim, n , key_dim)
            vm = value_memory[0][not_first_indices, ...].to(query.device)
            vm = vm*memory_mask[not_first_indices, ...]
            # (dim, n , value_dim)

            z = query[not_first_indices, ...]            
            dim = z.shape[0]
            
            # Similarities:
            #if self.normalization_fn == "softmax":
            #    sim = einsum('b n d, b d -> b n', km, z)
            #else:
            sim = F.cosine_similarity(km, z.unsqueeze(1), dim=-1)
            # (dim, n)
            
            ## Top K :
            topk_sim, topk_sim_indices = torch.topk(
                input=sim,
                k=self.top_k,
                dim=-1,
                largest=True,
                sorted=False,
            )
            # (dim, k)
            topk_vm = torch.gather(
                input=vm,
                dim=1,
                index=topk_sim_indices.unsqueeze(2).expand(
                    dim,
                    self.top_k,
                    value_dim,
                ),
            )
            # (dim, k, value_dim)
            
            ## Attention weights:
            if self.normalization_fn == "softmax":
                wv = topk_sim.softmax(dim=-1).unsqueeze(-1)
            elif self.normalization_fn =="inverse_dissim":
                wv = torch.pow(1e-8+torch.clamp(1-topk_sim, min=0.0, max=None), -1)
                wv = wv/wv.sum(dim=-1, keepdim=True)
                wv = wv.unsqueeze(-1)
            else:
                raise NotImplementedError
            # (dim, k, 1)

            if self.normalize_output:
                new_read_value = wv * topk_vm 
            else:
                new_read_value = topk_vm
            # (dim, k, value_dim)
            
            if postprocessing is not None\
            and "self-attention" in self.postprocessing:
                new_read_value, attn_weights = postprocessing(
                    query=new_read_value, 
                    key=new_read_value, 
                    value=new_read_value, 
                    key_padding_mask=None, 
                    need_weights=True, 
                    attn_mask=None, 
                    average_attn_weights=True,
                )
            elif postprocessing is not None:
                raise NotImplementedError 

            new_read_element.scatter_(
                dim=0,
                index=not_first_indices.reshape(
                    (dim, 1, 1)
                ).repeat(
                    1, 
                    self.top_k, 
                    value_dim,
                ).to(query.device),
                src=new_read_value,
            )

        if "sum" in self.postprocessing:
            new_read_element = new_read_element.sum(dim=1)
            # (batch_size, value_dim)
        else:
            raise NotImplementedError
        
        outputs_dict = {
            'read_value': [new_read_element],
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

        queries = input_streams_dict['queries'][0]
        batch_size = queries.shape[0]
        if len(queries.shape) != 3:
            queries = queries.reshape(batch_size, self.nbr_heads, -1)

        value_memory = input_streams_dict['value_memory']
        key_memory = input_streams_dict['key_memory']
        
        for ridx in range(self.nbr_heads):
            output_dict = self.forward(
                iteration=iteration,
                key_memory=key_memory,
                value_memory=value_memory,
                query=queries[:,ridx,...],
                postprocessing=self.postprocessings[ridx] if self.postprocessing is not None else None,
            )
            
            k = f"{ridx}_read_value"
            v = output_dict["read_value"]
            outputs_stream_dict[k] = v 

            # Bookkeeping:
            outputs_stream_dict[f"inputs:{self.id}:{k}"] = v

        return outputs_stream_dict

