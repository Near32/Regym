from typing import Dict, Any, Optional, List, Callable, Union

import torch
import torch.nn as nn

from functools import partial
import copy

import Archi 
import importlib

def load_module(module_key, module_kwargs):
    if '.' in module_key:
        module_path = module_key.split('.')
        _from = '.'.join(module_path[:-1])
        _from = importlib.import_module(_from)
        module_name = module_path[-1]
    else:
        _from = Archi.modules
        module_name = module_key
    module_cls = getattr(_from, module_name, None)
    if module_cls is None:
        raise NotImplementedError
    module = module_cls(**module_kwargs)

    print(module)
    return module 
   
def layer_init(layer, w_scale=1.0, nonlinearity='relu', init_type=None):
    for name, param in layer._parameters.items():
        if param is None or param.data is None: continue
        if 'bias' in name:
            #layer._parameters[name].data.fill_(0.0)
            if init_type=='ortho':
                nn.init.constant_(layer._parameters[name].data, 0)
            else:
                layer._parameters[name].data.uniform_(-0.08,0.08)
        else:
            if init_type=='ortho':
                nn.init.orthogonal_(layer._parameters[name].data)
                layer._parameters[name].data.mul_(w_scale)
            elif init_type=='xavier':
                nn.init_xavier_uniform_(layer._parameters[name].data)
                layer._parameters[name].data.mul_(w_scale)
            else:
                if len(layer._parameters[name].size()) > 1:
                    nn.init.kaiming_normal_(
                        layer._parameters[name], 
                        mode="fan_out", 
                        nonlinearity=nonlinearity,
                    )
                    layer._parameters[name].data.mul_(w_scale)
    return layer


def layer_init_lstm(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight_ih.data)
    nn.init.orthogonal_(layer.weight_hh.data)
    layer.weight_ih.data.mul_(w_scale)
    layer.weight_hh.data.mul_(w_scale)
    nn.init.constant_(layer.bias_ih.data, 0)
    nn.init.constant_(layer.bias_hh.data, 0)
    return layer

def layer_init_gru(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight_ih.data)
    nn.init.orthogonal_(layer.weight_hh.data)
    layer.weight_ih.data.mul_(w_scale)
    layer.weight_hh.data.mul_(w_scale)
    nn.init.constant_(layer.bias_ih.data, 0)
    nn.init.constant_(layer.bias_hh.data, 0)
    return layer


def is_leaf(node: Dict):
    return any([ not isinstance(node[key], dict) for key in node.keys()])

def get_leaf_keys(node: Dict):
    return [key for key in node.keys() if not isinstance(node[key], dict)]

def recursive_inplace_update(
    in_dict: Dict,
    extra_dict: Union[Dict, torch.Tensor],
    batch_mask_indices: Optional[torch.Tensor]=None,
    preprocess_fn: Optional[Callable] = None,
    assign_fn: Optional[Callable] = None):
    '''
    Taking both :param: in_dict, extra_dict as tree structures,
    adds the nodes of extra_dict into in_dict via tree traversal.
    Extra leaf keys are created if and only if the update is over the whole batch, i.e. :param
    batch_mask_indices: is None.
    :param batch_mask_indices: torch.Tensor of shape (batch_size,), containing batch indices that
                        needs recursive inplace update. If None, everything is updated.
    '''
    if in_dict is None: return None
    
    in_queue = [in_dict]
    extra_queue = [extra_dict]

    while len(extra_queue):
        in_pointer = in_queue.pop(0)
        extra_pointer = extra_queue.pop(0)

        leaf_keys = get_leaf_keys(extra_pointer)
        for key in extra_pointer:
            needed_init = False
            if key not in in_pointer:
                # initializing here, and preprocessing below...
                in_pointer[key] = {}
                needed_init = True 

            if key not in leaf_keys:
                in_queue.append(in_pointer[key])
                extra_queue.append(extra_pointer[key])
                continue
            
            # else : we know this key is a leaf_key:
            leaf_key = key
            if extra_pointer[leaf_key] is None \
            or (isinstance(extra_pointer[leaf_key], list) \
            and len(extra_pointer[leaf_key])==1 \
            and extra_pointer[leaf_key][0] is None):
                listvalue = None
            else:
                listvalue = [value.clone() for value in extra_pointer[leaf_key]]
            if needed_init: #leaf_key not in in_pointer:
                # initializing here, and preprocessing below...
                in_pointer[leaf_key] = listvalue
            
            if batch_mask_indices is None or batch_mask_indices==[]:
                in_pointer[leaf_key]= listvalue
            else:
                for vidx in range(len(in_pointer[leaf_key])):
                    v = listvalue[vidx]
                    
                    # SPARSE-NESS : check & record
                    sparse_v = False
                    if getattr(v, "is_sparse", False):
                        sparse_v = True
                        v = v.to_dense()
                    
                    # PREPROCESSING :
                    new_v = v[batch_mask_indices, ...].clone().to(in_pointer[leaf_key][vidx].device)
                    if preprocess_fn is not None:   new_v = preprocess_fn(new_v)
                    
                    # SPARSE-NESS : init
                    if in_pointer[leaf_key][vidx].is_sparse:
                        in_pointer[leaf_key][vidx] = in_pointer[leaf_key][vidx].to_dense()
                    
                    # ASSIGNMENT:
                    if assign_fn is not None:
                        assign_fn(
                            dest_d=in_pointer,
                            leaf_key=leaf_key,
                            vidx=vidx,
                            batch_mask_indices=batch_mask_indices,
                            new_v=new_v,
                        )
                    else:
                        in_pointer[leaf_key][vidx][batch_mask_indices, ...] = new_v
                    
                    # SPARSE-NESS / POST-PROCESSING:
                    if sparse_v:
                        v = v.to_sparse()
                        in_pointer[leaf_key][vidx] = in_pointer[leaf_key][vidx].to_sparse()
    
    return 

def DEPRECATED0_recursive_inplace_update(
    in_dict: Dict,
    extra_dict: Union[Dict, torch.Tensor],
    batch_mask_indices: Optional[torch.Tensor]=None,
    preprocess_fn: Optional[Callable] = None,
    assign_fn: Optional[Callable] = None):
    '''
    Taking both :param: in_dict, extra_dict as tree structures,
    adds the nodes of extra_dict into in_dict via tree traversal.
    Extra leaf keys are created if and only if the update is over the whole batch, i.e. :param
    batch_mask_indices: is None.
    :param batch_mask_indices: torch.Tensor of shape (batch_size,), containing batch indices that
                        needs recursive inplace update. If None, everything is updated.
    '''
    if in_dict is None: return None
    leaf_keys = get_leaf_keys(extra_dict)
    for leaf_key in leaf_keys:
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

    for node_key in extra_dict:
        if node_key in leaf_keys:   continue
        if node_key not in in_dict: in_dict[node_key] = {}
        if not is_leaf(extra_dict[node_key]):
            recursive_inplace_update(
                in_dict=in_dict[node_key], 
                extra_dict=extra_dict[node_key],
                batch_mask_indices=batch_mask_indices,
                preprocess_fn=preprocess_fn,
                assign_fn=assign_fn,
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
                # TODO: identify the issue that the following line was aiming to solve:
                #listvalue = [value.clone() for value in extra_dict[node_key][leaf_key] if value != {}]
                if leaf_key not in in_dict[node_key]:
                    # initializing here, and preprocessing below...
                    in_dict[node_key][leaf_key] = listvalue
                if batch_mask_indices is None or batch_mask_indices==[]:
                    in_dict[node_key][leaf_key]= listvalue
                else:
                    for vidx in range(len(in_dict[node_key][leaf_key])):
                        v = listvalue[vidx]
                        if leaf_key not in in_dict[node_key]:   continue
                        
                        # SPARSE-NESS : check & record
                        sparse_v = False
                        if getattr(v, "is_sparse", False):
                            sparse_v = True
                            v = v.to_dense()
                        
                        # PREPROCESSING :
                        new_v = v[batch_mask_indices, ...].clone().to(in_dict[node_key][leaf_key][vidx].device)
                        if preprocess_fn is not None:   new_v = preprocess_fn(new_v)
                        
                        # SPARSE-NESS : init
                        if in_dict[node_key][leaf_key][vidx].is_sparse:
                            in_dict[node_key][leaf_key][vidx] = in_dict[node_key][leaf_key][vidx].to_dense()
                        # ASSIGNMENT:
                        if assign_fn is not None:
                            assign_fn(
                                dest_d=in_dict,
                                node_key=node_key,
                                leaf_key=leaf_key,
                                vidx=vidx,
                                batch_mask_indices=batch_mask_indices,
                                new_v=new_v,
                            )
                        else:
                            in_dict[node_key][leaf_key][vidx][batch_mask_indices, ...] = new_v
                        
                        # SPARSE-NESS / POST-PROCESSING:
                        if sparse_v:
                            v = v.to_sparse()
                            in_dict[node_key][leaf_key][vidx] = in_dict[node_key][leaf_key][vidx].to_sparse()
    return 

def DEPRECATED_recursive_inplace_update(
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
            listvalue = [
                    value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value) 
                    for value in extra_dict[leaf_key]
            ]
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
                listvalue = [
                        value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value) 
                        for value in extra_dict[node_key][leaf_key]
                ]
                #listvalue = [value.clone() for value in extra_dict[node_key][leaf_key]]
                if leaf_key not in in_dict[node_key]:
                    # initializing here, and preprocessing below...
                    in_dict[node_key][leaf_key] = listvalue
                if batch_mask_indices is None\
                or batch_mask_indices==[]:
                    in_dict[node_key][leaf_key]= listvalue
                else:
                    for vidx in range(len(in_dict[node_key][leaf_key])):
                        v = listvalue[vidx]
                        if leaf_key not in in_dict[node_key]:   continue
                        if not isinstance(v, torch.Tensor): 
                            in_dict[node_key][leaf_key][vidx] = v
                            continue
                        new_v = v[batch_mask_indices, ...].clone().to(in_dict[node_key][leaf_key][vidx].device)
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

