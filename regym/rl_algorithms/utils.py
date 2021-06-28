from typing import Dict, Any, Optional, List, Callable, Union
import torch
from functools import partial
import copy


def is_leaf(node: Dict):
    return any([ not isinstance(node[key], dict) for key in node.keys()])


def recursive_inplace_update(
    in_dict: Dict,
    extra_dict: Union[Dict, torch.Tensor],
    batch_mask_indices: Optional[torch.Tensor]=None,
    preprocess_fn: Optional[Callable] = None):
    '''
    Taking both :param: in_dict, extra_dict as tree structures,
    adds the nodes of extra_dict into in_dict via tree traversal.
    Extra leaf keys are created if and only if the update is over the whole batch, i.e. :param
    batch_mask_indices: is None.
    :param batch_mask_indices: torch.Tensor of shape (batch_size,), containing batch indices that
                        needs recursive inplace update. If None, everything is updated.
    '''
    if in_dict is None: return None
    if is_leaf(extra_dict):
        for leaf_key in extra_dict:
            # In order to make sure that the lack of deepcopy at this point will not endanger
            # the consistency of the data (since we are slicing at some other parts),
            # or, in other words, to make sure that this is yielding a copy rather than
            # a reference, proceed with caution:
            # WARNING: the following makes a referrence of the elements:
            # listvalue = extra_dict[node_key][leaf_key]
            # RATHER, to generate copies that lets gradient flow but do not share
            # the same data space (i.e. modifying one will leave the other intact), make
            # sure to use the clone() method, as list comprehension does not create new tensors.
            listvalue = [value.clone() for value in extra_dict[leaf_key]]
            in_dict[leaf_key] = listvalue
        return 

    for node_key in extra_dict:
        if node_key not in in_dict: in_dict[node_key] = {}
        if not is_leaf(extra_dict[node_key]):
            recursive_inplace_update(
                in_dict=in_dict[node_key], 
                extra_dict=extra_dict[node_key],
                batch_mask_indices=batch_mask_indices,
                preprocess_fn=preprocess_fn,
            )
        else:
            for leaf_key in extra_dict[node_key]:
                # In order to make sure that the lack of deepcopy at this point will not endanger
                # the consistancy of the data (since we are slicing at some other parts),
                # or, in other words, to make sure that this is yielding a copy rather than
                # a reference, proceed with caution:
                # WARNING: the following makes a referrence of the elements:
                # listvalue = extra_dict[node_key][leaf_key]
                # RATHER, to generate copies that lets gradient flow but do not share
                # the same data space (i.e. modifying one will leave the other intact), make
                # sure to use the clone() method, as list comprehension does not create new tensors.
                listvalue = [value.clone() for value in extra_dict[node_key][leaf_key]]
                if batch_mask_indices is None or batch_mask_indices==[]:
                    in_dict[node_key][leaf_key]= listvalue
                else:
                    for vidx in range(len(in_dict[node_key][leaf_key])):
                        v = listvalue[vidx]
                        if leaf_key not in in_dict[node_key]:   continue
                        new_v = v[batch_mask_indices, ...].clone()
                        if preprocess_fn is not None:   new_v = preprocess_fn(new_v)
                        in_dict[node_key][leaf_key][vidx][batch_mask_indices, ...] = new_v

def copy_hdict(in_dict: Dict):
    '''
    Makes a copy of :param in_dict:.
    '''
    if in_dict is None: return None
    
    out_dict = {key: {} for key in in_dict}
    need_reg = False
    if isinstance(in_dict, list):
        out_dict = {'dummy':{}}
        in_dict = {'dummy':in_dict}
        need_reg = True 

    recursive_inplace_update(
        in_dict=out_dict,
        extra_dict=in_dict,
    )

    if need_reg:
        out_dict = out_dict['dummy']

    return out_dict

def extract_subtree(in_dict: Dict,
                    node_id: str):
    '''
    Extracts a copy of subtree whose root is named :param node_id: from :param in_dict:.
    '''
    queue = [in_dict]
    pointer = None

    while len(queue):
        pointer = queue.pop(0)
        if not isinstance(pointer, dict): continue
        for k in pointer.keys():
            if node_id==k:
                return copy_hdict(pointer[k])
            else:
                queue.append(pointer[k])

    return {}


def _extract_from_rnn_states(rnn_states_batched: Dict,
                             batch_idx: Optional[int]=None,
                             map_keys: Optional[List]=None,
                             post_process_fn:Callable=(lambda x:x)): #['hidden', 'cell']):
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
            rnn_states[recurrent_submodule_name] = {}
            eff_map_keys = map_keys if map_keys is not None else rnn_states_batched[recurrent_submodule_name].keys()
            for key in eff_map_keys:
                if key in rnn_states_batched[recurrent_submodule_name]:
                    rnn_states[recurrent_submodule_name][key] = []
                    for idx in range(len(rnn_states_batched[recurrent_submodule_name][key])):
                        value = rnn_states_batched[recurrent_submodule_name][key][idx]
                        if batch_idx is not None:
                            value = value[batch_idx,...].unsqueeze(0)
                        rnn_states[recurrent_submodule_name][key].append(post_process_fn(value))
        else:
            rnn_states[recurrent_submodule_name] = _extract_from_rnn_states(
                rnn_states_batched=rnn_states_batched[recurrent_submodule_name], 
                batch_idx=batch_idx,
                post_process_fn=post_process_fn
            )
    return rnn_states


def _extract_rnn_states_from_batch_indices(rnn_states_batched: Dict,
                                           batch_indices: torch.Tensor,
                                           use_cuda: bool=False,
                                           map_keys: Optional[List]=None): #['hidden', 'cell']):
    if rnn_states_batched is None:  return None

    rnn_states = {k: {} for k in rnn_states_batched}
    for recurrent_submodule_name in rnn_states_batched:
        if is_leaf(rnn_states_batched[recurrent_submodule_name]):
            rnn_states[recurrent_submodule_name] = {}
            eff_map_keys = map_keys if map_keys is not None else rnn_states_batched[recurrent_submodule_name].keys()
            for key in eff_map_keys:
                if key in rnn_states_batched[recurrent_submodule_name]:
                    rnn_states[recurrent_submodule_name][key] = []
                    for idx in range(len(rnn_states_batched[recurrent_submodule_name][key])):
                        value = rnn_states_batched[recurrent_submodule_name][key][idx][batch_indices,...]
                        if use_cuda: value = value.cuda()
                        rnn_states[recurrent_submodule_name][key].append(value)
        else:
            rnn_states[recurrent_submodule_name] = _extract_rnn_states_from_batch_indices(
                rnn_states_batched=rnn_states_batched[recurrent_submodule_name],
                batch_indices=batch_indices,
                use_cuda=use_cuda,
                map_keys=map_keys
            )
    return rnn_states


def _concatenate_hdict(hd1: Union[Dict, List],
                       hds: List,
                       map_keys: List,
                       concat_fn: Optional[Callable] = partial(torch.cat, dim=0),
                       preprocess_fn: Optional[Callable] = (lambda x:x) ):
    if not(isinstance(hd1, dict)):
        return _concatenate_hdict(
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

def _concatenate_list_hdict(
    lhds: List[Dict],
    concat_fn: Optional[Callable] = partial(torch.cat, dim=0),
    preprocess_fn: Optional[Callable] = (lambda x:
        torch.from_numpy(x).unsqueeze(0) if isinstance(x, np.ndarray) else torch.ones(1, 1)*x
        )
    ):
    out_hd = {key: {} for key in lhds[0]}

    queue = [lhds]
    pointers = None

    out_queue = [out_hd]
    out_pointer = None

    while len(queue):
        pointers = [hds for hds in queue.pop(0)]
        out_pointer = out_queue.pop(0)

        if not is_leaf(pointers[0]):
            #out_pointer = {}
            # previously is taken care of at 145 upon initialization,
            # and then at 165 upon 'recurrence'.
            for k in pointers[0]:
                queue_element = [pointer[k] for pointer in pointers if k in pointer]
                queue.insert(0, queue_element)

                out_pointer[k] = {}
                out_queue.insert(0, out_pointer[k])
        else:
            for k in pointers[0]:
                try:
                    # Previously assigned as a dictionnary in 145 or 165...
                    out_pointer[k] = []
                    # Since we are at a leaf then value is
                    # either numpy or numpy.float64
                    # or list of tensors:
                    if isinstance(pointers[0][k], list):
                        for idx in range(len(pointers[0][k])):
                            concat_list = [
                                preprocess_fn(pointer[k][idx])
                                for pointer in pointers if k in pointer
                            ]
                            out_pointer[k].append(
                                concat_fn(concat_list)
                            )
                    else:
                        concat_list = [
                            preprocess_fn(pointer[k])
                            for pointer in pointers if k in pointer
                        ]
                        try:
                            out_pointer[k] = concat_fn(concat_list)
                        except Exception as e:
                            # the concat_fn may fail, silently...
                            # e.g.: shape of elements are not all the same...
                            pass
                except Exception as e:
                        # the concat_fn may fail, silently...
                        # e.g.: neither a list nor a compatible stuff....
                        pass
    return out_hd


def apply_on_hdict(
    hdict: Dict,
    fn: Optional[Callable] = lambda x: x,
    ):
    out_hd = {key: {} for key in hdict}

    queue = [hdict]
    pointer = None

    out_queue = [out_hd]
    out_pointer = None

    while len(queue):
        pointer = queue.pop(0)
        out_pointer = out_queue.pop(0)

        if not is_leaf(pointer):
            for k in pointer:
                queue_element = pointer[k]
                queue.insert(0, queue_element)

                out_pointer[k] = {}
                out_queue.insert(0, out_pointer[k])
        else:
            for k in pointer:
                out_pointer[k] = []
                # Since we are at a leaf then value is
                # either numpy or numpy.float64
                # or list of tensors:
                if isinstance(pointer[k], list):
                    for idx in range(len(pointer[k])):
                        out_pointer[k].append(
                            fn(pointer[k][idx])
                        )
                else:
                    out_pointer[k] = fn(pointer[k])
    return out_hd