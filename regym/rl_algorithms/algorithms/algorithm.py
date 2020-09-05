from typing import Dict, List, Any, Optional, Callable

from functools import partial

import torch
import copy


class Algorithm(object):
    def get_models(self):
        raise NotImplementedError

    def get_epsilon(self, nbr_steps, strategy='exponential'):
        raise NotImplementedError

    def get_nbr_actor(self):
        raise NotImplementedError

    def get_update_count(self):
        raise NotImplementedError

    @staticmethod
    def _extract_rnn_states_from_batch_indices(rnn_states_batched: Dict, batch_indices: torch.Tensor, use_cuda: bool=False):
        if rnn_states_batched is None:  return None 

        rnn_states = {k: {} for k in rnn_states_batched}
        for recurrent_submodule_name in rnn_states_batched:
            if 'hidden' in rnn_states_batched[recurrent_submodule_name]:
                rnn_states[recurrent_submodule_name] = {'hidden':[], 'cell':[]}
                for idx in range(len(rnn_states_batched[recurrent_submodule_name]['hidden'])):
                    hidden = rnn_states_batched[recurrent_submodule_name]['hidden'][idx][batch_indices,...]
                    if use_cuda: hidden = hidden.cuda()
                    rnn_states[recurrent_submodule_name]['hidden'].append(hidden)
                    if 'cell' in rnn_states_batched[recurrent_submodule_name]:
                        cell = rnn_states_batched[recurrent_submodule_name]['cell'][idx][batch_indices,...]
                        if use_cuda: cell = cell.cuda()
                        rnn_states[recurrent_submodule_name]['cell'].append(cell)
            else:
                rnn_states[recurrent_submodule_name] = Algorithm._extract_rnn_states_from_batch_indices(
                    rnn_states_batched=rnn_states_batched[recurrent_submodule_name], 
                    batch_indices=batch_indices, 
                    use_cuda=use_cuda
                )
        return rnn_states

    @staticmethod
    def _concatenate_hdict(hd1: Dict, 
                           hds: List, 
                           map_keys: List, 
                           concat_fn: Optional[Callable] = partial(torch.cat, dim=0),
                           preprocess_fn: Optional[Callable] = (lambda x:x) ):
        if not(isinstance(hd1, dict)):
            return Algorithm._concatenate_hdict(
                hd1=hds.pop(0), 
                hds=hds, 
                map_keys=map_keys, 
                concat_fn=concat_fn,
                preprocess_fn=preprocess_fn
            )
        
        out_hd = {}
        for key in hd1:
            out_hd[key] = {}
            map_key_not_found_at_this_level = True
            for map_key in map_keys:
                if map_key in hd1[key]:
                    map_key_not_found_at_this_level = False
                    out_hd[key][map_key] = []
                    for idx in range(len(hd1[key][map_key])):
                        concat_list = [preprocess_fn(hd1[key][map_key][idx])]
                        for hd in hds:
                            concat_list.append(preprocess_fn(hd[key][map_key][idx]))
                        out_hd[key][map_key].append(concat_fn(concat_list))
            if map_key_not_found_at_this_level:
                out_hd[key] = Algorithm._concatenate_hdict(
                    hd1=hd1[key], 
                    hds=[hd[key] for hd in hds], 
                    map_keys=map_keys, 
                    concat_fn=concat_fn,
                    preprocess_fn=preprocess_fn,
                )
        return out_hd

    def clone(self, with_replay_buffer=False):
        raise NotImplementedError