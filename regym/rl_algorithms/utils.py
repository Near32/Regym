from typing import Dict, Any, Optional, List, Callable, Union
import torch
from functools import partial 


def is_leaf(node: Dict):
    return all([ not isinstance(node[key], dict) for key in node.keys()])

# TODO
def recursive_inplace_update(odict: Dict,)

def _extract_from_rnn_states(rnn_states_batched: Dict, batch_idx: Optional[int]=None, map_keys: Optional[List]=['hidden', 'cell']):
    '''
    :param map_keys: List of keys we map the operation to.
    '''
    rnn_states = {k: {} for k in rnn_states_batched}
    for recurrent_submodule_name in rnn_states_batched:
        # It is possible that an initial rnn states dict has states for actor and critic, separately,
        # but only the actor will be operated during the take_action interface.
        # Here, we allow the critic rnn states to be skipped:
        if rnn_states_batched[recurrent_submodule_name] is None:    continue
        if is_leaf(rnn_states_batched[recurrent_submodule_name]):
            rnn_states[recurrent_submodule_name] = {key:[] for key in map_keys}
            for key in map_keys:
                for idx in range(len(rnn_states_batched[recurrent_submodule_name][key])):
                    value = rnn_states_batched[recurrent_submodule_name][key][idx]
                    if batch_idx is not None:
                        value = value[batch_idx,...].unsqueeze(0)
                    rnn_states[recurrent_submodule_name][key].append(value)
        else:
            rnn_states[recurrent_submodule_name] = _extract_from_rnn_states(rnn_states_batched=rnn_states_batched[recurrent_submodule_name], batch_idx=batch_idx)
    return rnn_states


def _extract_rnn_states_from_batch_indices(rnn_states_batched: Dict, 
                                           batch_indices: torch.Tensor, 
                                           use_cuda: bool=False,
                                           map_keys: Optional[List]=['hidden', 'cell']):
    if rnn_states_batched is None:  return None 

    rnn_states = {k: {} for k in rnn_states_batched}
    for recurrent_submodule_name in rnn_states_batched:
        if is_leaf(rnn_states_batched[recurrent_submodule_name]):
            rnn_states[recurrent_submodule_name] = {key:[] for key in map_keys}
            for key in map_keys:
                for idx in range(len(rnn_states_batched[recurrent_submodule_name][key])):
                    value = rnn_states_batched[recurrent_submodule_name][key][idx][batch_indices,...]
                    if use_cuda: hidden = value.cuda()
                    rnn_states[recurrent_submodule_name][key].append(value)
        else:
            rnn_states[recurrent_submodule_name] = _extract_rnn_states_from_batch_indices(
                rnn_states_batched=rnn_states_batched[recurrent_submodule_name], 
                batch_indices=batch_indices, 
                use_cuda=use_cuda
            )
    return rnn_states


def _concatenate_hdict(hd1: Union[Dict, List], 
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
            out_hd[key] = _concatenate_hdict(
                hd1=hd1[key], 
                hds=[hd[key] for hd in hds], 
                map_keys=map_keys, 
                concat_fn=concat_fn,
                preprocess_fn=preprocess_fn,
            )
    return out_hd