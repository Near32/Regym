from typing import Dict, List, Optional, Callable

from functools import partial 
import copy
import time 

import numpy as np
import torch
import ray

from regym.rl_algorithms.algorithms import Algorithm 
from regym.rl_algorithms.utils import is_leaf, copy_hdict, _concatenate_list_hdict, recursive_inplace_update, apply_on_hdict
from regym.thirdparty.Archi.Archi.model import Model as ArchiModel

import wandb 


eps = 1e-3
study_qa_values_discrepancy = False
use_BPTT = True

use_zero_initial_states_for_target = False
"""
R2D2 paper:
"The zero start state strategy’s appeal lies in its simplicity, and it allows independent decorrelated
sampling of relatively short sequences, which is important for robust optimization of a neural net-work.
On the other hand,  it forces the RNN to learn to recover meaningful predictions from an
atypical initial recurrent state (‘initial recurrent state mismatch’), which may limit its ability to fully
rely on its recurrent state and learn to exploit long temporal correlations.  The second strategy on
the other hand avoids the problem of finding a suitable initial state, but creates a number of 
practical, computational, and algorithmic issues due to varying and potentially environment-dependent
sequence length, and higher variance of network updates because of the highly correlated nature of
states in a trajectory when compared to training on randomly sampled batches of experience tuples.

Hausknecht & Stone (2015) observed little difference between the two strategies for empirical agent
performance on a set of Atari games, and therefore opted for the simpler zero start state strategy.
One possible explanation for this is that in some cases, an RNN tends to converge to a more ‘typical’
state if allowed a certain number of ‘burn-in’ steps, and so recovers from a bad initial recurrent state
on a sufficiently long sequence.  We also hypothesize that while the zero start state strategy may
suffice in the mostly fully observable Atari domain, it prevents a recurrent network from learning
actual long-term dependencies in more memory-critical domains (e.g. on DMLab)."

"""

def archi_assign_fn(
    new_v,
    dest_d,
    node_key,
    leaf_key,
    vidx,
    batch_mask_indices=None,
    time_indices_start=None,
    time_indices_end=None,
    ):
    """
    Assumes that some memory sizes are different from dest to new_v.
    Regularise new_v to expand to the correct shape, which is likely larger...
    """
    dest = dest_d[node_key][leaf_key][vidx]

    if batch_mask_indices is None:  batch_mask_indices = torch.arange(dest.shape[0]).to(dest.device) 
    if time_indices_start is None \
    and time_indices_end is None:    
        dest = dest[batch_mask_indices, ...]
    else:
        dest = dest[batch_mask_indices, time_indices_start:time_indices_end+1, ...]
    dshape = dest.shape
    nvshape = new_v.shape
    
    if dshape == nvshape:
        if time_indices_start is None \
        and time_indices_end is None:    
            dest_d[node_key][leaf_key][vidx][batch_mask_indices, ...] = new_v#.clone()
        else:
            dest_d[node_key][leaf_key][vidx][batch_mask_indices, time_indices_start:time_indices_end+1, ...] = new_v#.clone()
        return
        #return new_v.clone()
    
    dest = dest_d[node_key][leaf_key][vidx]
    dshape = dest.shape
    
    max_shape = list(dshape)
    for sidx in range(len(dshape)):
        if max_shape[sidx] < nvshape[sidx]:
            max_shape[sidx] = nvshape[sidx]

    reshaped_new_v = torch.zeros(*max_shape).to(dest.device)
    if time_indices_start is None \
    and time_indices_end is None:    
        reshaped_new_v[:, :dshape[1], ...] = dest#.clone()
        reshaped_new_v[batch_mask_indices, :nvshape[1], ...] = new_v
    else: 
        reshaped_new_v[:, :, :dshape[2], ...] = dest#.clone()
        reshaped_new_v[batch_mask_indices, time_indices_start:time_indices_end+1, :nvshape[2], ...] = new_v
    
    dest_d[node_key][leaf_key][vidx] = reshaped_new_v 
    return
    #return reshaped_new_v

def identity_value_function_rescaling(x):
    return x 

def value_function_rescaling(x):
    '''
    Value function rescaling (table 2).
    '''
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.) - 1.) + eps * x


def inverse_value_function_rescaling(x):
    '''
    See Proposition A.2 in paper "Observe and Look Further".
    '''
    return torch.sign(x) * (
        (
            (torch.sqrt(1. + 4. * eps * (torch.abs(x) + 1. + eps)) - 1.) / (2. * eps)
        ).pow(2.0) - 1.
    )
    
    

def extract_rnn_states_from_time_indices(
    rnn_states_batched: Dict, 
    time_indices_start:int, 
    time_indices_end:int,
    preprocess_fn: Optional[Callable] = None):
    """
    If time_indices_start is out of bound, then the value is silently ommitted.
    """
    if rnn_states_batched is None:  return None 

    rnn_states = {k: {} for k in rnn_states_batched}
    for recurrent_submodule_name in rnn_states_batched:
        if is_leaf(rnn_states_batched[recurrent_submodule_name]):
            tis=time_indices_start
            tie=time_indices_end
            squeeze_needed=False
            if tis==tie:    
                tie+=1
                squeeze_needed=True
            rnn_states[recurrent_submodule_name] = {}
            for key in rnn_states_batched[recurrent_submodule_name]:
                if key not in rnn_states[recurrent_submodule_name]:
                    rnn_states[recurrent_submodule_name][key] = []
                for idx in range(len(rnn_states_batched[recurrent_submodule_name][key])):
                    if tis>=rnn_states_batched[recurrent_submodule_name][key][idx].shape[1]: 
                        continue
                    value = rnn_states_batched[recurrent_submodule_name][key][idx][:, tis:tie,...].clone()
                    if preprocess_fn is not None:   value = preprocess_fn(value)
                    if squeeze_needed:  value = value.squeeze(1) 
                    rnn_states[recurrent_submodule_name][key].append(value)
        else:
            rnn_states[recurrent_submodule_name] = extract_rnn_states_from_time_indices(
                rnn_states_batched=rnn_states_batched[recurrent_submodule_name], 
                time_indices_start=time_indices_start, 
                time_indices_end=time_indices_end,
                preprocess_fn=preprocess_fn
            )
    return rnn_states


def replace_rnn_states_at_time_indices(
        rnn_states_batched: Dict, 
        replacing_rnn_states_batched: Dict, 
        time_indices_start:int, 
        time_indices_end:int,
        assign_fn: Optional[Callable] = None,
        ):
    if rnn_states_batched is None:  return None 

    rnn_states = {k: {} for k in rnn_states_batched}
    for recurrent_submodule_name in rnn_states_batched:
        if is_leaf(rnn_states_batched[recurrent_submodule_name]):
            rnn_states[recurrent_submodule_name] = {}
            for key in rnn_states_batched[recurrent_submodule_name]:
                if key not in rnn_states[recurrent_submodule_name]: 
                    rnn_states[recurrent_submodule_name][key] = []
                for idx in range(len(rnn_states_batched[recurrent_submodule_name][key])):
                    value = rnn_states_batched[recurrent_submodule_name][key][idx] 
                    batch_size = value.shape[0]
                    unroll_size = time_indices_end+1-time_indices_start 
                    #value[:, time_indices_start:time_indices_end+1,...] = replacing_rnn_states_batched[recurrent_submodule_name][key][idx].reshape(batch_size, unroll_size, -1)
                    # reshaping is probably useless:
                    if assign_fn is None:
                        value[:, time_indices_start:time_indices_end+1,...] = replacing_rnn_states_batched[recurrent_submodule_name][key][idx].unsqueeze(1)
                        rnn_states[recurrent_submodule_name][key].append(value)
                    else:
                        # dummy assignement first...:
                        rnn_states[recurrent_submodule_name][key].append(value)#.clone())
                        assign_fn(
                            dest_d=rnn_states,
                            new_v=replacing_rnn_states_batched[recurrent_submodule_name][key][idx].unsqueeze(1),
                            node_key=recurrent_submodule_name,
                            leaf_key=key,
                            vidx=idx,
                            batch_mask_indices=None,
                            time_indices_start=time_indices_start,
                            time_indices_end=time_indices_end,
                        )
        else:
            rnn_states[recurrent_submodule_name] = replace_rnn_states_at_time_indices(
                rnn_states_batched=rnn_states_batched[recurrent_submodule_name], 
                replacing_rnn_states_batched=replacing_rnn_states_batched[recurrent_submodule_name], 
                time_indices_start=time_indices_start, 
                time_indices_end=time_indices_end, 
                assign_fn=assign_fn,
            )

    return rnn_states


def roll_sequences(unrolled_sequences:List[Dict[str, torch.Tensor]], batch_size:int=1, map_keys:List[str]=None):
    '''
    Returns a dictionnary of torch tensors from the list of dictionnaries `unrolled_sequences`. 

    :param map_keys: List of strings of keys to care about:
    '''
    keys = map_keys if map_keys is not None else unrolled_sequences[0].keys()
    d = {}
    for key in keys:
        if unrolled_sequences[0][key] is None:  continue
        if isinstance(unrolled_sequences[0][key], dict):
            values = [unrolled_sequences[i][key] for i in range(len(unrolled_sequences))]
            value = _concatenate_list_hdict(
                lhds=values, 
                concat_fn=partial(torch.cat, dim=1),   # concatenate on the unrolling dimension (axis=1).
                preprocess_fn=(lambda x: x.unsqueeze(1)), #.reshape(batch_size, 1, *x.shape[1:])),
            )
        else: 
            value = torch.cat(
                [
                    unrolled_sequences[i][key].unsqueeze(1) #.reshape(batch_size, 1, *unrolled_sequences[i][key].shape[1:])    # add unroll dim 
                    for i in range(len(unrolled_sequences)) 
                ],
                dim=1
            )
        d[key] = value
    return d


def unrolled_inferences_deprecated(model: torch.nn.Module, 
                        states: torch.Tensor, 
                        rnn_states: Dict[str, Dict[str, List[torch.Tensor]]],
                        goals: torch.Tensor=None,
                        grad_enabler: bool=False,
                        use_zero_initial_states: bool=False,
                        extras: bool=False,
                        map_keys:List[str]=None):
    """ 
    Compute feed-forward inferences on the :param model: of the :param states: with the rnn_states used as burn_in values.
    NOTE: The function also computes the inferences using the rnn states used when gathering the states, in order to 
    later allow a computation of the Q-value discrepency $$\Delta Q$$ (see end of page 4).
    
    :param model: torch.nn.Module to use for inference.
    :param states: torch.Tensor of shape (batch_size, unroll_dim, ...) to use as input for inference.
    :param rnn_states: Hierarchy of dictionnaries containing as leaf the hidden an cell states of the relevant recurrent modules.
                        The shapes are batch_first, i.e. (batch_size, unroll_dim, ...).
    :param goals: Dimension batch_size x goal shape: Goal of the agent.
    :param grad_enable: boolean specifying whether to compute gradient.
    :param use_zero_initial_states: boolean specifying whether the initial recurrent states are zeroed or sampled from the unrolled sequence.
    :param map_keys: List of strings containings the keys we want to extract and concatenate in the returned predictions.

    :return burn_in_predictions: Dict of outputs produced by the :param model: with shape (batch_size, unroll_dim, ...),
                                    when the recurrent cell states are burned in throughout the unrolled sequence, 
                                    with the exception of the first one, which can be zeroed if :param use_zero_initial_states:
                                    is `True`, otherwise it is sampled from the unrolle sequence.
    :return unrolled_predictions: Dict of outputs produced by the :param model: with shape (batch_size, unroll_dim, ...),
                                    when the recurrent cell states are sampled from the unrolled sequence.
    :return burned_in_rnn_states_inputs: Hierarchy of dictionnaries containing the final hidden and cell states of the recurrent
                                        submodules contained in :param model:, with shape (batch_size, unroll_dim=1, ...).
    """
    batch_size = states.shape[0]
    unroll_length = states.shape[1]

    init_rnn_states_inputs = None
    if not use_zero_initial_states: 
        init_rnn_states_inputs = extract_rnn_states_from_time_indices(
            rnn_states, 
            time_indices_start=0,
            time_indices_end=0,
        )

    burn_in_predictions =  []
    unrolled_predictions =  []
    
    burn_in_rnn_states_inputs = init_rnn_states_inputs
    unrolled_rnn_states_inputs = init_rnn_states_inputs

    unrolled_prediction = None
    with torch.set_grad_enabled(grad_enabler):
        for unroll_id in range(unroll_length):
            inputs = states[:, unroll_id,...]
            
            burn_in_prediction = model(inputs, rnn_states=burn_in_rnn_states_inputs)
            
            if extras:
                unrolled_prediction = model(inputs, rnn_states=unrolled_rnn_states_inputs)
            
            burn_in_predictions.append(burn_in_prediction)
            if extras:
                unrolled_predictions.append(unrolled_prediction)

            # Bookkeeping: update the rnn states:
            ## Handle next step's extra inputs:
            burn_in_rnn_states_inputs = extract_rnn_states_from_time_indices(
                rnn_states, 
                time_indices_start=unroll_id+1,  #sample for next step...
                time_indices_end=unroll_id+1,
            )
            
            if extras and unroll_id < unroll_length-1:
                unrolled_rnn_states_inputs = copy_hdict(burn_in_rnn_states_inputs)
            
            ## Update burn-in rnn states:
            # TODO : account for end of episode...
            recursive_inplace_update(
                in_dict=burn_in_rnn_states_inputs,
                extra_dict=burn_in_prediction['next_rnn_states']
            )

    burned_in_rnn_states_inputs = burn_in_rnn_states_inputs
    # (batch_size, ...)  
    burn_in_predictions = roll_sequences(
        burn_in_predictions, 
        batch_size=batch_size,
        map_keys=map_keys
    )
    if extras:
        unrolled_predictions = roll_sequences(
            unrolled_predictions, 
            batch_size=batch_size,
            map_keys=map_keys,
        )

    return burn_in_predictions, unrolled_predictions, burned_in_rnn_states_inputs


def batched_unrolled_inferences(
    unroll_length: int,
    model: torch.nn.Module, 
    states: torch.Tensor, 
    non_terminals: torch.Tensor,
    rnn_states: Dict[str, Dict[str, List[torch.Tensor]]],
    actions: torch.Tensor=None,
    goals: torch.Tensor=None,
    grad_enabler: bool=False,
    use_zero_initial_states: bool=False,
    extras: bool=False,
    map_keys:List[str]=None
    ):
    '''
    Compute feed-forward inferences on the :param model: of the :param states: with the rnn_states used as burn_in values.
    NOTE: The function also computes the inferences using the rnn states used when gathering the states, in order to 
    later allow a computation of the Q-value discrepency $$\Delta Q$$ (see end of page 4).
    
    :param model: torch.nn.Module to use for inference.
    :param states: torch.Tensor of shape (batch_size, unroll_dim, ...) to use as input for inference.
    :param non_terminals: Dimension: batch_size x unroll_length x 1: Non-terminal integers.
    :param rnn_states: Hierarchy of dictionnaries containing as leaf the hidden an cell states of the relevant recurrent modules.
                        The shapes are batch_first, i.e. (batch_size, unroll_dim, ...).
    :param goals: Dimension batch_size x goal shape: Goal of the agent.
    :param actions: torch.Tensor of shape (batch_size, unroll_dim, ...) to use as action to evaluate, unless it is None.
    :param grad_enable: boolean specifying whether to compute gradient.
    :param use_zero_initial_states: boolean specifying whether the initial recurrent states are zeroed or sampled from the unrolled sequence.
    :param map_keys: List of strings containings the keys we want to extract and concatenate in the returned predictions.

    :return burn_in_predictions: Dict of outputs produced by the :param model: with shape (batch_size, unroll_dim, ...),
                                    when the recurrent cell states are burned in throughout the unrolled sequence, 
                                    with the exception of the first one, which can be zeroed if :param use_zero_initial_states:
                                    is `True`, otherwise it is sampled from the unrolle sequence.
    :return unrolled_predictions: Dict of outputs produced by the :param model: with shape (batch_size, unroll_dim, ...),
                                    when the recurrent cell states are sampled from the unrolled sequence.
    :return burned_in_rnn_states_inputs: Hierarchy of dictionnaries containing the final hidden and cell states of the recurrent
                                        submodules contained in :param model:, with shape (batch_size, unroll_dim=1, ...).
    '''
    batch_size = states.shape[0]
    #unroll_length = states.shape[1]

    vdn = False 
    if len(states.shape)==4 or len(states.shape)==6:
        vdn = True 
        num_players = states.shape[2]
    
    recurrent_module_in_phi_body = 'phi_body' in rnn_states and ('lstm' in rnn_states['phi_body'] or 'gru' in rnn_states['phi_body']) 
    extra_inputs_in_phi_body = 'phi_body' in rnn_states and 'extra_inputs' in rnn_states['phi_body']
    torso_prediction_batched = None
    if rnn_states is None or not(recurrent_module_in_phi_body):
        # "The recurrent module must be in the critic_arch/head pipeline."
        model_torso = model.get_torso()
        model_head = model.get_head()

        eff_unroll_length = unroll_length
        if states.shape[1] > unroll_length:
            eff_unroll_length = unroll_length+1
        
        begin = time.time()
        if vdn:
            torso_input = states[:,:,:eff_unroll_length,...].reshape((batch_size*eff_unroll_length*num_players, *states.shape[3:]))
        else:
            torso_input = states[:,:eff_unroll_length,...].reshape((batch_size*eff_unroll_length, *states.shape[2:]))

        with torch.set_grad_enabled(grad_enabler):
            if extra_inputs_in_phi_body:
                if vdn:
                    #batching_time_dim_lambda_fn = (lambda x: x[:,:unroll_length,...].clone().reshape((batch_size*unroll_length*num_players, *x.shape[3:])))
                    batching_time_dim_lambda_fn = (lambda x: x[:,:eff_unroll_length,...].reshape((batch_size*eff_unroll_length*num_players, *x.shape[3:])))
                    unbatching_time_dim_lambda_fn = (lambda x: x.reshape((batch_size, eff_unroll_length, num_players, *x.shape[2:])))
                else:
                    #batching_time_dim_lambda_fn = (lambda x: x[:,:unroll_length,...].clone().reshape((batch_size*unroll_length, *x.shape[2:])))
                    batching_time_dim_lambda_fn = (lambda x: x[:,:eff_unroll_length,...].reshape((batch_size*eff_unroll_length, *x.shape[2:])))
                    unbatching_time_dim_lambda_fn = (lambda x: x.reshape((batch_size, eff_unroll_length, *x.shape[1:])))
                
                rnn_states_batched = {}
                rnn_states_batched = apply_on_hdict(
                   hdict=rnn_states,
                   fn=batching_time_dim_lambda_fn,
                )
                torso_output, torso_prediction_batched = model_torso(torso_input, rnn_states=rnn_states_batched)
            else:
                torso_output, _ = model_torso(torso_input)

        if vdn:
            head_input = torso_output.reshape((batch_size, eff_unroll_length, num_players, *torso_output.shape[1:]))
        else:
            head_input = torso_output.reshape((batch_size, eff_unroll_length, *torso_output.shape[1:]))
        
        if torso_prediction_batched is not None:
            head_input_rnn_states = apply_on_hdict(
                hdict=torso_prediction_batched['next_rnn_states'],
                fn=unbatching_time_dim_lambda_fn,
            )
            recursive_inplace_update(
                in_dict=rnn_states,
                extra_dict=head_input_rnn_states,
            )

        end = time.time()

        #print(f"Batched Forward: {end-begin} sec.")

        if rnn_states is None:
            raise NotImplementedError
            # TODO: Need to verify this head_input shape when using VDN...
            head_input = head_input.reshape((batch_size*eff_unroll_length, *head_input.shape[2:]))
            with torch.set_grad_enabled(grad_enabler):
                burn_in_prediction = model_head(head_input, rnn_states=None)
            for key in burn_in_prediction:
                if burn_in_prediction[key] is None: continue
                shape = burn_in_prediction[key]
                burn_in_prediction[key] = burn_in_prediction[key].reshape((batch_size, unroll_length, *shape[1:]))
            return burn_in_prediction, burn_in_prediction, None 
    else:
        model_head = model 
        head_input = states
    
    head_action_input = actions

    if vdn:
        head_input = head_input.transpose(1,2).reshape(-1, eff_unroll_length, *head_input.shape[3:])
        if actions is not None:
            # TODO : to test :
            import ipdb; ipdb.set_trace()
            head_action_input = head_action_input.transpose(1,2).reshape(
                (-1, eff_unroll_length, *head_action_input.shape[3:]),
            )
        batching_time_dim_lambda_fn = (
            lambda x: 
            x.transpose(1,2).reshape((batch_size*num_players, eff_unroll_length, *x.shape[3:]))
        )
        rnn_states = apply_on_hdict(
            hdict=rnn_states,
            fn=batching_time_dim_lambda_fn,
        )
            
    init_rnn_states_inputs = None
    preprocess_fn = None if use_BPTT else (lambda x:x.detach())
    if use_zero_initial_states: 
        preprocess_fn = (lambda x: torch.zeros_like(x))

    init_rnn_states_inputs = extract_rnn_states_from_time_indices(
        rnn_states, 
        time_indices_start=0,
        time_indices_end=0,
        preprocess_fn= preprocess_fn,
    )

    burn_in_predictions =  []
    unrolled_predictions =  []
    
    burn_in_rnn_states_inputs = init_rnn_states_inputs
    unrolled_rnn_states_inputs = init_rnn_states_inputs

    unrolled_prediction = None
    assign_fn = None
    if isinstance(model, ArchiModel):
        assign_fn = archi_assign_fn
    
    with torch.set_grad_enabled(grad_enabler):
        for unroll_id in range(unroll_length):
            inputs = head_input[:, unroll_id,...]
            
            action_inputs = None
            if actions is not None:
                action_inputs = head_action_input[:, unroll_id,...]
                
            non_terminals_input = non_terminals[:, unroll_id, ...].reshape(batch_size)

            burn_in_prediction = model_head(
                inputs, 
                action=action_inputs,
                rnn_states=burn_in_rnn_states_inputs,
            )
            
            if extras:
                unrolled_prediction = model_head(
                    inputs, 
                    action=action_inputs,
                    rnn_states=unrolled_rnn_states_inputs,
                )
            
            burn_in_predictions.append(burn_in_prediction)
            if extras:
                unrolled_predictions.append(unrolled_prediction)

            # Bookkeeping: update the rnn states:
            ## Handle next step's extra inputs:
            burn_in_rnn_states_inputs = extract_rnn_states_from_time_indices(
                rnn_states, 
                time_indices_start=unroll_id+1,  #sample for next step...
                time_indices_end=unroll_id+1,
                preprocess_fn= None if use_BPTT else (lambda x:x.detach()),
            )
            #WARNING: the following is bound to append upon calling the function on training loop,
            # where it cannot be padded for the very last computation:
            # len(burn_in_rnn_states_inputs['CoreLSTM']['iteration']) == 0
            # It is OK, we do not need the burn_in_rnn_states by the end of this training-purposed call.
              
            if extras and unroll_id < unroll_length-1:
                unrolled_rnn_states_inputs = copy_hdict(burn_in_rnn_states_inputs)
            
            ## Carry currently computated next rnn states into next loop iteration's 
            # burn-in rnn states inputs, unless it is the end of the episode:
            # if it is the end of the episode then we use the rnn states
            # that was stored, the one that is currently in, that should be zeroed.
            #
            non_terminals_batch_indices = torch.from_numpy(
                np.where(non_terminals_input.cpu().numpy()==1)[0]
            ).to(non_terminals_input.device)
            
            recursive_inplace_update(
                in_dict=burn_in_rnn_states_inputs,
                extra_dict=burn_in_prediction['next_rnn_states'],
                batch_mask_indices=non_terminals_batch_indices,
                preprocess_fn= None if use_BPTT else (lambda x:x.detach()),
                assign_fn=assign_fn, 
            )
            
    burned_in_rnn_states_inputs = burn_in_rnn_states_inputs
    # (batch_size, ...)  
    burn_in_predictions = roll_sequences(
        burn_in_predictions, 
        batch_size=batch_size,
        map_keys=map_keys
    )
    if extras:
        unrolled_predictions = roll_sequences(
            unrolled_predictions, 
            batch_size=batch_size,
            map_keys=map_keys,
        )

    end2 = time.time()
    #print(f"Unrolled Forward: {end2-end} sec.")
    
    # Reshaping to put the player dimension in second position:
    if vdn:
        #head_input = head_input.transpose(1,2).reshape(-1, unroll_length, *head_input.shape[3:])
        def reshape_fn(x):
            return x[:,:unroll_length,...].reshape((batch_size, num_players, unroll_length, *x.shape[2:])).transpose(1,2)
        import ipdb; ipdb.set_trace()
        #TODO : assert that the reshaping is properly done.
        burned_in_rnn_states_inputs = apply_on_hdict(
            hdict=burned_in_rnn_states_inputs,
            fn=batching_time_dim_lambda_fn,
        )

        for k, t in burn_in_predictions.items():
            if isinstance(t, torch.Tensor):
                burn_in_predictions[k] = reshape_fn(t)
            elif isinstance(t, dict):
                burn_in_predictions[k] = apply_on_hdict(
                    hdict=t,
                    fn=reshape_fn, #hdict_reshape_fn,
                )
            else:
                raise NotImplementedError

        if extras:
            for k, t in unrolled_predictions.items():
                if isinstance(t, torch.Tensor):
                    unrolled_predictions[k] = reshape_fn(t)
                elif isinstance(t, dict):
                    unrolled_predictions[k] = apply_on_hdict(
                        hdict=t,
                        fn=reshape_fn, #hdict_reshape_fn,
                    )
                else:
                    raise NotImplementedError

    return burn_in_predictions, unrolled_predictions, burned_in_rnn_states_inputs

def compute_n_step_bellman_target_depr(
    training_returns,
    training_non_terminals,
    unscaled_targetQ_Si_onlineGreedyAction, 
    gamma, 
    kwargs):
    # Mixed n-step value approach:
    unscaled_targetQ_Sipn_onlineGreedyAction = torch.cat(
        [
            unscaled_targetQ_Si_onlineGreedyAction[:, kwargs['n_step']:, ...]
        ]+[
            unscaled_targetQ_Si_onlineGreedyAction[:, -1:, ...] / (gamma**(k+1)) # it will be normalized down below when computing bellman target    
            for k in range(kwargs['n_step'])
        ],
        dim=1,
    )
    # (batch_size, training_length, 1)
    
    '''
    # Adapted from hanabi_SAD :
    targetQ_Sipn_argmaxAipn_values = torch.cat([
            targetQ_Si_argmaxAQvalue[:, kwargs['n_step']:],
            targetQ_Si_argmaxAQvalue[:, :kwargs['n_step']]
        ],
        dim=1
    )
    '''
    # Adapted from hanabi_SAD : 
    # https://github.com/facebookresearch/hanabi_SAD/blob/54a8d34f6ab192898121f8d3935339e63f1f4b35/pyhanabi/r2d2.py#L373
    # Zero-ing the value of the state we cannot compute a complete n-step sounds valuable:
    # given that the alternative is to get an unreliable target-network-based estimation..
    # unfortunately, it results in a less expressive qa function at the beginning, 
    # as opposed to using mixed n-step values... 
    #unscaled_targetQ_Sipn_onlineGreedyAction[:, -kwargs['n_step']:] = 0
    
    '''
    training_non_terminals_ipn = torch.cat(
        [   # :k+1 since k \in [0, n_step-1], and np.zeros(length)[:0] has shape (0,)...
            training_non_terminals[:, k:k+kwargs['n_step'], ...].prod(dim=1).reshape(batch_size, 1, -1)
            for k in range(kwargs['n_step'])
        ]+[
            training_non_terminals.prod(dim=1).reshape(batch_size, 1, -1)
        ]*(training_non_terminals.shape[1]-kwargs['n_step']),
        dim=1,
    )
    '''
    training_non_terminals_ipnm1 = torch.cat(
        [   
            training_non_terminals[:, k:k+kwargs['n_step'], ...].prod(dim=1).reshape(batch_size, 1, -1)
            for k in range(training_non_terminals.shape[1])
        ],
        dim=1,
    )
    
    # Compute the Bellman Target for Q values at Si,Ai: with gamma
    unscaled_bellman_target_Sipn_onlineGreedyAction = training_returns + (gamma**kwargs['n_step']) * unscaled_targetQ_Sipn_onlineGreedyAction * training_non_terminals_ipnm1
    
    return unscaled_bellman_target_Sipn_onlineGreedyAction


# Adapted from: https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/agents/r2d2/learner.py#L195
def compute_n_step_bellman_target(
    training_rewards,
    training_non_terminals,
    unscaled_targetQ_Si_onlineGreedyAction, 
    gamma, 
    kwargs):
    
    unscaled_bellman_target_Sipn_onlineGreedyAction = torch.cat(
        [
            unscaled_targetQ_Si_onlineGreedyAction[:, :kwargs['n_step'], ...]       #will be exhausted in the loop
        ]+[
            unscaled_targetQ_Si_onlineGreedyAction[:, kwargs['n_step']:, ...]
        ]+[
            unscaled_targetQ_Si_onlineGreedyAction[:, -1:, ...] # testing without normalisation ... #unscaled_targetQ_Si_onlineGreedyAction[:, -1:, ...] / (gamma**k)    #will be normalized down below when computing bellman target    
            for k in range(kwargs['n_step'])                    # not like SEED_RL (: range(1, kwargs['n_step']))
        ],
        dim=1,
    )
    # (batch_size, training_length+n_step, 1)
    
    rewards_i = torch.cat(
        [
            training_rewards
        ]+[
            torch.zeros_like(training_rewards[:, 0:1, ...])
        ]*kwargs['n_step'],     # dummy values to account for the mixed n-step approach
        dim=1,
    )
    # (batch_size, training_length+n_step, 1)
    
    notdones_i = torch.cat(
        [
            training_non_terminals
        ]+[
            torch.ones_like(training_non_terminals[:, 0:1, ...])
        ]*kwargs['n_step'],     # dummy values to account for the mixed n-step approach
        dim=1,
    )
    # (batch_size, training_length+n_step, 1)
    
    """
    # being taken care of just before the call to this function:
    if len(rewards_i.shape) > 3: #VDN
        nbr_players = rewards_i.shape[2]
        #notdones_i = notdones_i.unsqueeze(-1).repeat(1,1,nbr_players,1)
        # (batch_size, training_length+n_step, /player_dim,/ 1)
        #rewards_i = rewards_i[:,:,0,...]#
        rewards_i = rewards_i.sum(dim=2)
        unscaled_bellman_target_Sipn_onlineGreedyAction = unscaled_bellman_target_Sipn_onlineGreedyAction.sum(dim=2)
    """
    
    # Compute the Bellman Target for Q values at Si,Ai: with gamma
    for nt in range(1,kwargs["n_step"]+1):
        rewards_i = rewards_i[:, :-1]
        # (batch_size, training_length+n_step-nt, 1)
        notdones_i = notdones_i[:, :-1]
        # (batch_size, training_length+n_step-nt, 1)
        #
        # nt=1: b_0 = r_0 + gamma * notdone_0 * qTargetSA_1
        # nt=1: b_1 = r_1 + gamma * notdone_1 * qTargetSA_2
        # nt=1: b_2 = r_2 + gamma * notdone_2 * qTargetSA_3
        # nt=1: b_3 = r_3 + gamma * notdone_3 * qTargetSA_4
        #   ...
        # nt=1: b_{L-2} = r_{L-2} + gamma * notdone_{L-2} * qTargetSA_{L-1}
        # nt=1: b_{L-1} = r_{L-1} + gamma * notdone_{L-1} * qTargetSA_{L-1} 
        # nt=1: b_{L} = 0 + gamma *1* qTargetSA_{L-1}/gamma = qTargetSA_{L-1}
        # nt=1: b_{L+1} = 0 + gamma *1* qTargetSA_{L-1} = gamma * qTargetSA_{L-1}
        #
        #
        # nt=2: b'_0 = r_0 + gamma * notdone_0 * b1 (= r_1 + gamma * notdone_1 * qTargetSA_2)
        # nt=2: b'_1 = r_1 + gamma * notdone_1 * b2 (= r_2 + gamma * notdone_2 * qTargetSA_3)
        # nt=2: b'_2 = r_2 + gamma * notdone_2 * b3 (= r_3 + gamma * notdone_3 * qTargetSA_4)
        # nt=2: b'_3 = r_3 + gamma * notdone_3 * b4 (= r_4 + gamma * notdone_4 * qTargetSA_5)
        #   ...
        # nt=2: b'_{L-2} = r_{L-2} + gamma * notdone_{L-2} * b_{L-1} ( = r_{L-1} + gamma * notdone_{L-1} * qTargetSA_{L-1})
        # nt=2: b'_{L-1} = r_{L-1} + gamma * notdone_{L-1} * b_{L} ( = qTargetSA_{L-1}) = r_{L-1} + gamma * notdone_{L-1} * qTargetSA_{L-1} 
        # nt=2: b'_{L} = 0 + gamma * 1 * b_{L+1} ( = gamma * qTargetSA_{L-1}) = gamma**2 * qTargetSA_{L-1} 
        #
        #
        # nt=3: b''_0 = r_0 + gamma * notdone_0 * b'_1 (= r_1 + gamma * notdone_1 * b2 (= r_2 + gamma * notdone_2 * qTargetSA_3))
        # nt=3: b''_1 = r_1 + gamma * notdone_1 * b'_2 (= r_2 + gamma * notdone_2 * b3 (= r_3 + gamma * notdone_3 * qTargetSA_4))
        #   ...
        # nt=3: b''_{L-3} = r_{L-3} + gamma * notdone_{L-3} * b'_{L-2} ( = r_{L-2} + gamma * notdone_{L-2} * b_{L-1}) = r_{L-3} + gamma * notdone_{L-3} * r_{L-2} + gamma **2 * notdone_{L-3} * notdone_{L-2} * r_{L-1} + gamma ** 3 * notdone_{L-3} * notdone_{L-2} * notdone_{L-1} qTargetSA_{L-1})
        # nt=3: b''_{L-2} = r_{L-2} + gamma * notdone_{L-2} * b'_{L-1} ( = r_{L-1} + gamma * notdone_{L-1} * b_{L}) = r_{L-2} + gamma * notdone_{L-2} * r_{L-1} + gamma **2 * notdone_{L-2} * notdone_{L-1} qTargetSA_{L-1})
        # nt=3: b''_{L-1} = r_{L-1} + gamma * notdone_{L-1} * b'_{L} ( = gamma**2 * qTargetSA_{L-1}) = r_{L-1} + gamma * n_step * notdone_{L-1} * qTargetSA_{L-1} 
        

        # nt=3: b''_{L} = 0 + gamma * 1 * b_{L+n_step-2} ( = qTargetSA_{L-1}/gamma) ) = qTargetSA_{L-1} 
        # nt=3: b''_{L+n_step-3} = 0 + gamma * 1 * b_{L+n_step-1} ( = qTargetSA_{L-1}/gamma**2 ) = qTargetSA_{L-1}/gamma 
        unscaled_bellman_target_Sipn_onlineGreedyAction = rewards_i \
            + gamma * notdones_i * unscaled_bellman_target_Sipn_onlineGreedyAction[:, 1:] 
        # (batch_size, training_length+n_step-nt, 1)
    
    # (batch_size, training_length, 1)
    return unscaled_bellman_target_Sipn_onlineGreedyAction

def compute_n_step_bellman_target_sad(
    training_nstep_returns,
    training_non_terminals,
    unscaled_targetQ_Si_onlineGreedyAction, 
    gamma, 
    kwargs):

    batch_size = training_nstep_returns.shape[0]

    unscaled_targetQ_Si_onlineGreedyAction = torch.cat(
        [
            unscaled_targetQ_Si_onlineGreedyAction[:, kwargs['n_step']:, ...]
        ]+[
            unscaled_targetQ_Si_onlineGreedyAction[:, :kwargs['n_step'], ...]       #will be zeroed-out
        ],
        dim=1,
    )
    unscaled_targetQ_Si_onlineGreedyAction[:, -kwargs['n_step']:, ...] = 0
    # (batch_size, training_length, /player_dim,/ 1)
    
    bootstrap = torch.cat(
        [   
            training_non_terminals[:, k:k+kwargs['n_step'], ...].prod(dim=1, keepdim=True)#.unsqueeze(1) #reshape(batch_size, 1, -1)
            for k in range(training_non_terminals.shape[1])
        ],
        dim=1,
    )
    
    if len(training_nstep_returns.shape) > 3: #VDN
        bootstrap = bootstrap.unsqueeze(-1)
    # (batch_size, training_length, /player_dim,/ 1)
    
    # Compute the Bellman Target for Q values at Si,Ai: with gamma
    unscaled_bellman_target_Sipn_onlineGreedyAction = training_nstep_returns \
        + (gamma**kwargs['n_step']) * bootstrap * unscaled_targetQ_Si_onlineGreedyAction 
    
    # (batch_size, training_length, /player_dim,/ 1)
    return unscaled_bellman_target_Sipn_onlineGreedyAction

# Adapted from: https://github.com/google-research/seed_rl/blob/34fb2874d41241eb4d5a03344619fb4e34dd9be6/agents/r2d2/learner.py#L333
def compute_loss(
    samples: Dict[str, torch.Tensor],
    models: Dict[str, torch.nn.Module],
    summary_writer: object = None,
    iteration_count: int = 0,
    **kwargs:Optional[Dict[str, object]],#=None,
) -> torch.Tensor:
    '''
    :param states: Dimension: batch_size x unroll_length x state_size: States visited by the agent.
    :param actions: Dimension: batch_size x unroll_length x action_size. Actions which the agent
                    took at every state in :param states: with the same index.
    :param next_states: Dimension: batch_size x unroll_length x state_size: Next sequence of unrolled states visited by the agent.
    :param non_terminals: Dimension: batch_size x unroll_length x 1: Non-terminal integers.
    :param rewards: Dimension: batch_size x unroll_length x 1. Environment rewards, or n-step returns if using n-step returns.
    :param goals: Dimension: batch_size x unroll_length x goal shape: Goal of the agent.
    :param model: torch.nn.Module used to compute the loss.
    :param target_model: torch.nn.Module used to compute the loss.
    :param gamma: float discount factor.
    :param weights_decay_lambda: Coefficient to be used for the weight decay loss.
    :param rnn_states: The :param model: can be made up of different submodules.
                       Some of these submodules will feature an LSTM architecture.
                       This parameter is a dictionary which maps recurrent submodule names
                       to a dictionary which contains 2 lists of tensors, each list
                       corresponding to the 'hidden' and 'cell' states of
                       the LSTM submodules. These tensors are used by the
                       :param model: when calculating the policy probability ratio.
    :param next_rnn_states: Resulting 'hidden' and 'cell' states of the LSTM submodules after
                            feedforwarding :param states: in :param model:. See :param rnn_states:
                            for further details on type and shape.
    '''
    states = samples['states']
    actions = samples['actions']
    next_states = samples['next_states']
    rewards = samples['rewards']
    non_terminals = samples['non_terminals']
    goals = None #samples['goals']
    rnn_states = samples['rnn_states']
    next_rnn_states = samples['next_rnn_states']
    importanceSamplingWeights = samples['importanceSamplingWeights']
    
    model = models['model']
    target_model = models['target_model']
    
    gamma = kwargs['gamma']
    weights_decay_lambda = float(kwargs.get('weights_decay_lambda', 0.0))
    weights_entropy_lambda = float(kwargs.get('weights_entropy_lambda', 0.0))
    weights_entropy_reg_alpha = float(kwargs.get('weights_entropy_reg_alpha', 0.0))
    use_PER = kwargs['use_PER']
    PER_beta = kwargs['PER_running_beta']
    HER_target_clamping = kwargs['HER_target_clamping']
    
    #torch.autograd.set_detect_anomaly(True)
    batch_size = states.shape[0]
    unroll_length = states.shape[1]
    map_keys=['qa', 'a', 'ent', 'legal_ent']

    """
    if len(rewards.shape) > 3:
        states = states[:,:,0,...]
        actions = actions[:,:,0,...]
        next_states = next_states[:,:,0,...]
        rewards = rewards[:,:,0,...]
        def vdn_fn(x): 
            return x[:,:,0,...] #.reshape((batch_size*unroll_length*num_players, *x.shape[3:])))
        rnn_states = apply_on_hdict(
            hdict=rnn_states,
            fn=vdn_fn,
        )
    """
                
    if kwargs['r2d2_use_value_function_rescaling']:
        inv_vfr = inverse_value_function_rescaling
        vfr = value_function_rescaling
    else:
        inv_vfr = identity_value_function_rescaling
        vfr = identity_value_function_rescaling

    start = time.time()
    assign_fn = None
    if isinstance(model, ArchiModel):
        assign_fn = archi_assign_fn

    if kwargs['burn_in']:
        burn_in_length = kwargs['sequence_replay_burn_in_length']
        training_length = kwargs['sequence_replay_unroll_length']-burn_in_length

        burn_in_states, training_states = torch.split(
            states, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )
        """
        _burn_in_rnn_states = extract_rnn_states_from_time_indices(
            rnn_states, 
            time_indices_start=0,
            time_indices_end=kwargs['sequence_replay_burn_in_length'],
            preprocess_fn= (lambda x:x.detach()),
        )
        """
        _training_rnn_states = extract_rnn_states_from_time_indices(
            rnn_states, 
            time_indices_start=kwargs['sequence_replay_burn_in_length'],
            time_indices_end=kwargs['sequence_replay_unroll_length'],
            preprocess_fn= None if use_BPTT else (lambda x:x.detach()), # not performing BPTT
        )
        _, training_rewards = torch.split(
            rewards, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )
        burn_in_non_terminals, training_non_terminals = torch.split(
            non_terminals, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )
        _, training_actions = torch.split(
            actions, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )

        # SEED RL does put a stop_gradient:
        # https://github.com/google-research/seed_rl/blob/5f07ba2a072c7a562070b5a0b3574b86cd72980f/agents/r2d2/learner.py#L368
        # No BPTT on the subsequent rnn_states:
        burned_in_predictions, \
        unrolled_predictions, \
        burned_in_rnn_states_inputs = batched_unrolled_inferences(
            unroll_length=burn_in_length,
            model=model, 
            states=states, #burn_in_states, 
            non_terminals=burn_in_non_terminals,
            rnn_states=rnn_states,
            grad_enabler=False,
            use_zero_initial_states=kwargs['sequence_replay_use_zero_initial_states'],
            extras=False,
            map_keys=map_keys,
        )

        target_model.reset_noise()

        burned_in_target_predictions, \
        unrolled_target_predictions, \
        burned_in_rnn_states_target_inputs = batched_unrolled_inferences(
            unroll_length=burn_in_length,
            model=target_model, 
            states=states, #burn_in_states, 
            non_terminals=burn_in_non_terminals,
            rnn_states=rnn_states,
            grad_enabler=False,
            use_zero_initial_states=kwargs['sequence_replay_use_zero_initial_states'],
            extras=False,
            map_keys=map_keys,
        )

        # Replace the burned in rnn states in the training rnn states:
        training_rnn_states = replace_rnn_states_at_time_indices(
            rnn_states_batched=_training_rnn_states, 
            replacing_rnn_states_batched=burned_in_rnn_states_inputs, 
            time_indices_start=0, 
            time_indices_end=0,
            assign_fn=assign_fn,
        )

        training_target_rnn_states = replace_rnn_states_at_time_indices(
            rnn_states_batched=_training_rnn_states, 
            replacing_rnn_states_batched=burned_in_rnn_states_target_inputs, 
            time_indices_start=0, 
            time_indices_end=0,
            assign_fn=assign_fn,
        )
    else:
        training_length = unroll_length
        training_states = states 
        training_actions = actions 
        training_rewards = rewards
        training_non_terminals = non_terminals
        training_rnn_states = rnn_states
        training_target_rnn_states = rnn_states

    training_next_states = next_states

    # Unrolled predictions is using the stored RNN states.
    # burned_in_predictions is using the online RNN states computed in the function loop.
    training_burned_in_predictions, \
    training_unrolled_predictions, _ = batched_unrolled_inferences(
        unroll_length=training_length,
        model=model, 
        states=training_states, 
        non_terminals=training_non_terminals,
        rnn_states=training_rnn_states,
        grad_enabler=True,
        use_zero_initial_states=kwargs['sequence_replay_use_zero_initial_states'] if not(kwargs['burn_in']) else False,
        extras=not(kwargs['burn_in']) or study_qa_values_discrepancy or not(kwargs['sequence_replay_use_online_states']),
        map_keys=map_keys,
    )

    target_model.reset_noise()

    training_burned_in_target_predictions, \
    training_unrolled_target_predictions, _ = batched_unrolled_inferences(
        unroll_length=training_length,
        model=target_model, 
        states=training_states, 
        non_terminals=training_non_terminals,
        rnn_states=training_target_rnn_states,
        grad_enabler=False,
        use_zero_initial_states=kwargs['sequence_replay_use_zero_initial_states']\
        if kwargs['sequence_replay_use_zero_initial_states'] else use_zero_initial_states_for_target,
        extras=not(kwargs['burn_in']) or not(kwargs['sequence_replay_use_online_states']),
        map_keys=map_keys,
    )

    if kwargs['burn_in'] or kwargs['sequence_replay_use_online_states']:
        training_predictions = training_burned_in_predictions
        training_target_predictions = training_burned_in_target_predictions
    else:
        training_predictions = training_unrolled_predictions
        training_target_predictions = training_unrolled_target_predictions
    
    qa_values_key = "qa"
    
    Q_Si_values = training_predictions[qa_values_key]
    # (batch_size, unroll_dim, ...)
    online_greedy_action = Q_Si_values.max(dim=-1, keepdim=True)[1]#.reshape(batch_size, training_length, Q_Si_values.shape[])
    # (batch_size, unroll_dim, ...)
    
    # Stable training: crucial: cf loss equation of Ape-X paper in section 3.1 Ape-X DQN:
    Q_Si_Ai_value = Q_Si_values.gather(
        dim=-1, 
        index=training_actions
    )
    # (batch_size, unroll_dim, /player_dim,/ 1)
    
    unscaled_targetQ_Si_A_values = inv_vfr(training_target_predictions[qa_values_key])
    # (batch_size, training_length, /player_dim,/ num_actions)
    
    # Double Q learning target:
    unscaled_targetQ_Si_onlineGreedyAction = unscaled_targetQ_Si_A_values.gather(
        dim=-1, 
        index=online_greedy_action
    )
    # (batch_size, training_length, /player_dim,/ 1)
    
    if weights_entropy_reg_alpha > 1.0e-12:
        # Adding entropy regularisation term for soft-DQN:
        online_target_entropy = training_target_predictions["legal_ent"]
        # Naive:
        #unscaled_targetQ_Si_onlineGreedyAction += weights_entropy_reg_alpha*online_target_entropy.unsqueeze(-1)
        # Legendre-Fenchel:
        unscaled_targetQ_Si_onlineGreedyAction = weights_entropy_reg_alpha*torch.log(
            torch.exp(Q_Si_values/weights_entropy_reg_alpha).sum(dim=-1)
        ).unsqueeze(-1)
    
    """
    # Assumes training_rewards is actually n-step returns...
    unscaled_bellman_target_Sipn_onlineGreedyAction = compute_n_step_bellman_target(
        training_returns=training_rewards,
        training_non_terminals=training_non_terminals,
        unscaled_targetQ_Si_onlineGreedyAction=unscaled_targetQ_Si_onlineGreedyAction,
        gamma=gamma,
        kwargs=kwargs
    )
    """

    unscaled_Q_Si_Ai_value = inv_vfr(Q_Si_Ai_value)

    if False:
        if len(training_rewards.shape) > 3:
            if False:
                assert ("vdn" in kwargs and kwargs["vdn"])
                # Summing on the player dimension:
                unscaled_Q_Si_Ai_value = unscaled_Q_Si_Ai_value.sum(dim=2)
                training_rewards = training_rewards.sum(dim=2)
                unscaled_targetQ_Si_onlineGreedyAction = unscaled_targetQ_Si_onlineGreedyAction.sum(dim=2)
            else:
                assert ("vdn" in kwargs and kwargs["vdn"]), "debugging in progress..."
                # only take one of the player values to debug whether the above VDN combination at the loss level is the 
                # issue or not...
                # So far, it is the main issue: the following trains without interruption 
                # whereas the above either plateaus too early or has catastrophic forgetting...
                unscaled_Q_Si_Ai_value = unscaled_Q_Si_Ai_value[:,:,0,...]
                training_rewards = training_rewards[:,:,0,...]
                unscaled_targetQ_Si_onlineGreedyAction = unscaled_targetQ_Si_onlineGreedyAction[:,:,0,...]
                
        if kwargs["r2d2_bellman_target_SAD"]:
            raise NotImplementedError
            # Need to implement n-step return computation 
            nstep_returns : torch.Tensor 
            unscaled_bellman_target_Sipn_onlineGreedyAction = compute_n_step_bellman_target_sad(
                trianing_nstep_returns=nstep_returns,
                training_non_terminals=training_non_terminals,
                unscaled_targetQ_Si_onlineGreedyAction=unscaled_targetQ_Si_onlineGreedyAction,
                gamma=gamma,
                kwargs=kwargs
            )
        else:    
            unscaled_bellman_target_Sipn_onlineGreedyAction = compute_n_step_bellman_target(
                training_rewards=training_rewards,
                training_non_terminals=training_non_terminals,
                unscaled_targetQ_Si_onlineGreedyAction=unscaled_targetQ_Si_onlineGreedyAction,
                gamma=gamma,
                kwargs=kwargs
            )
        # (batch_size, training_length, ...)
    else:
        if len(training_rewards.shape) > 3:
            assert ("vdn" in kwargs and kwargs["vdn"])
            unscaled_bellman_target_Sipn_onlineGreedyAction = torch.zeros_like(training_rewards[:,:,0])
            for pidx in range(kwargs['vdn_nbr_players']):
                unscaled_bellman_target_Sipn_onlineGreedyAction += compute_n_step_bellman_target(
                    training_rewards=training_rewards[:,:,pidx],
                    training_non_terminals=training_non_terminals,
                    unscaled_targetQ_Si_onlineGreedyAction=unscaled_targetQ_Si_onlineGreedyAction[:,:,pidx],
                    gamma=gamma,
                    kwargs=kwargs
                )
            unscaled_Q_Si_Ai_value = unscaled_Q_Si_Ai_value.sum(dim=2)
        else:
            unscaled_bellman_target_Sipn_onlineGreedyAction = compute_n_step_bellman_target(
                training_rewards=training_rewards,
                training_non_terminals=training_non_terminals,
                unscaled_targetQ_Si_onlineGreedyAction=unscaled_targetQ_Si_onlineGreedyAction,
                gamma=gamma,
                kwargs=kwargs
            )
        # (batch_size, training_length, ...)
        
        #unscaled_bellman_target_Sipn_onlineGreedyAction = unscaled_bellman_target_Sipn_onlineGreedyAction.sum(dim=2)
        assert len(unscaled_bellman_target_Sipn_onlineGreedyAction.shape) == 3  
    
    Q_Si_Ai_value = vfr(unscaled_Q_Si_Ai_value)    
    scaled_bellman_target_Sipn_onlineGreedyAction = vfr(unscaled_bellman_target_Sipn_onlineGreedyAction)
    
    # Compute loss:
    # MSE ?
    """
    td_error = 0.5*(unscaled_bellman_target_Sipn_onlineGreedyAction.detach() - unscaled_Q_Si_Ai_value)**2
    scaled_td_error = 0.5*(scaled_bellman_target_Sipn_onlineGreedyAction.detach() - Q_Si_Ai_value)**2
    """

    # Abs:
    if HER_target_clamping:
        # clip the unscaled target to [-50,0]
        unscaled_bellman_target_Sipn_onlineGreedyAction = torch.clamp(
            unscaled_bellman_target_Sipn_onlineGreedyAction, 
            -1. / (1 - gamma),
            0.0
        )
    td_error = torch.abs(unscaled_bellman_target_Sipn_onlineGreedyAction.detach() - unscaled_Q_Si_Ai_value)
    scaled_td_error = torch.abs(scaled_bellman_target_Sipn_onlineGreedyAction.detach() - Q_Si_Ai_value)
    assert list(td_error.shape) == [batch_size, training_length, 1]

    """
    if len(training_rewards.shape) > 3:
        assert ("vdn" in kwargs and kwargs["vdn"])
        td_error = td_error.sum(dim=2)
        scaled_td_error = scaled_td_error.sum(dim=2)
    """
    
    # Hanabi_SAD repo does not use the scaled values:
    loss_per_item = td_error
    diff_squared = td_error.pow(2.0)
    # SEED RL repo uses the scaled td error for priorities:
    #loss_per_item = scaled_td_error
    #diff_squared = scaled_td_error.pow(2.0)

    if use_PER and importanceSamplingWeights is not None:
      diff_squared = importanceSamplingWeights.reshape((batch_size, 1, 1)) * diff_squared
      assert list(diff_squared.shape) == [batch_size, training_length, 1]

    # not sure where this masking strategy comes from, maybe forget about it
    # since the distribution of qa values is more expressive without it...
    # the initial rational for it was to allow training on the last value only if terminal...
    assert kwargs["r2d2_loss_masking"], "r2d2_loss_masking must be True for this test."
    if kwargs["r2d2_loss_masking"]:
        mask = torch.ones_like(diff_squared)
        """
        assert kwargs['r2d2_loss_masking_n_step_regularisation'], "debugging in progress"
        if kwargs['r2d2_loss_masking_n_step_regularisation']:
            mask[:, -kwargs["n_step"]:, ...] = 0

        # maybe but 1 back:
        mask[:,-1, ...] = (1-training_non_terminals[:,-1,...])
        """
        # Combined:
        assert kwargs['r2d2_loss_masking_n_step_regularisation'], "debugging in progress"
        if kwargs['r2d2_loss_masking_n_step_regularisation']:
            mask[:, -kwargs["n_step"]:, ...] = (1-training_non_terminals[:,-kwargs['n_step']:,...])

        loss_per_item = loss_per_item*mask
        loss = 0.5*torch.mean(diff_squared*mask)-weights_entropy_lambda*training_predictions['legal_ent'].mean()
    else:
        mask = torch.ones_like(diff_squared)
        loss_per_item = loss_per_item*mask
        loss = 0.5*torch.mean(diff_squared*mask)-weights_entropy_lambda*training_predictions['legal_ent'].mean()
        #loss = 0.5*torch.mean(diff_squared)-weights_entropy_lambda*training_predictions['ent'].mean()
    
    end = time.time()

    #wandb_data = copy.deepcopy(wandb.run.history._data)
    #wandb.run.history._data = {}
    wandb.log({'Training/TimeComplexity':  end-start, "training_step":iteration_count}, commit=False)
    
    if study_qa_values_discrepancy:
        denominator = eps+torch.abs(training_burned_in_predictions['qa'].reshape(batch_size, -1).max(dim=-1)[0])
        # (batch_size, )
        initial_diff = training_burned_in_predictions['qa'][:,0,...]-training_unrolled_predictions['qa'][:,0,...]
        # (batch_size, num_actions)
        final_diff = training_burned_in_predictions['qa'][:,-1,...]-training_unrolled_predictions['qa'][:,-1,...]
        # (batch_size, num_actions)
        initial_discrepancy_qa = initial_diff.pow(2).sum(-1).sqrt() / denominator
        # (batch_size,)
        final_discrepancy_qa = final_diff.pow(2).sum(-1).sqrt() / denominator
        # (batch_size, )
        
        wandb.log({'Training/DiscrepancyQAValues/Initial':  initial_discrepancy_qa.cpu().mean().item(), "training_step":iteration_count}, commit=False)
        wandb.log({'Training/DiscrepancyQAValues/Final':  final_discrepancy_qa.cpu().mean().item(), "training_step":iteration_count}, commit=False)
    
    if kwargs.get("logging", False):
        columns = ["stimulus_(t)", "stimulus_(t-1)"]
        #columns += [f"a_(t-{v})" for v in range(4)]
        sample_table = wandb.Table(columns=columns) 
    
        for bidx in range(batch_size//4):
            nbr_states = states.shape[1]
            nbr_frames = states[bidx].shape[1]//4
            stimulus_t = [next_states[bidx,s].reshape(nbr_frames,4,56,56)[-1:,:3] for s in range(nbr_states)]#.numpy()[:,:3]*255
            stimulus_t = torch.cat(stimulus_t, dim=0).cpu().numpy()*255
            stimulus_t = stimulus_t.astype(np.uint8)
            stimulus_t = wandb.Video(stimulus_t, fps=2, format="mp4")
            #stimulus_tm = s[bidx].cpu().reshape(nbr_frames,4,56,56).numpy()[:,:3]*255
            stimulus_tm = [states[bidx,s].reshape(nbr_frames,4,56,56)[-1:,:3] for s in range(nbr_states)]#.numpy()[:,:3]*255
            stimulus_tm = torch.cat(stimulus_tm, dim=0).cpu().numpy()*255
            stimulus_tm = stimulus_tm.astype(np.uint8)
            stimulus_tm = wandb.Video(stimulus_tm, fps=2, format="mp4")
            '''
            previous_action_int = [
                self.episode_buffer[actor_index][aidx]["rnn_states"]['critic_body']['extra_inputs']['previous_action_int'][0][bidx].cpu().item()
                for aidx in [idx, idx-1, idx-2, idx-3]
            ]
            '''
            sample_table.add_data(*[
                #*gt_word_sentence,
                stimulus_t,
                stimulus_tm,
                #*previous_action_int
                ]
            )

        wandb.log({f"Training/R2D2StimuliTable":sample_table}, commit=False)

    # wandb.log({'Training/MeanTrainingNStepReturn':  training_rewards.cpu().mean().item(), "training_step":iteration_count}, commit=False)
    # wandb.log({'Training/MinTrainingNStepReturn':  training_rewards.cpu().min().item(), "training_step":iteration_count}, commit=False)
    # wandb.log({'Training/MaxTrainingNStepReturn':  training_rewards.cpu().max().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MeanTrainingReward':  training_rewards.cpu().mean().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MinTrainingReward':  training_rewards.cpu().min().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MaxTrainingReward':  training_rewards.cpu().max().item(), "training_step":iteration_count}, commit=False)

    #wandb.log({'Training/MeanTargetQSipn_ArgmaxAOnlineQSipn_A':  unscaled_targetQ_Sipn_onlineGreedyAction.cpu().mean().item(), "training_step":iteration_count}, commit=False)
    #wandb.log({'Training/MinTargetQSipn_ArgmaxAOnlineQSipn_A':  unscaled_targetQ_Sipn_onlineGreedyAction.cpu().min().item(), "training_step":iteration_count}, commit=False)
    #wandb.log({'Training/MaxTargetQSipn_ArgmaxAOnlineQSipn_A':  unscaled_targetQ_Sipn_onlineGreedyAction.cpu().max().item(), "training_step":iteration_count}, commit=False)
    
    wandb.log({'Training/MeanTargetQsi':  unscaled_targetQ_Si_A_values.cpu().mean().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MinTargetQsi':  unscaled_targetQ_Si_A_values.cpu().min().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MaxTargetQsi':  unscaled_targetQ_Si_A_values.cpu().max().item(), "training_step":iteration_count}, commit=False)
    
    wandb.log({'Training/MeanBellmanTarget':  unscaled_bellman_target_Sipn_onlineGreedyAction.cpu().mean().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MinBellmanTarget':  unscaled_bellman_target_Sipn_onlineGreedyAction.cpu().min().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MaxBellmanTarget':  unscaled_bellman_target_Sipn_onlineGreedyAction.cpu().max().item(), "training_step":iteration_count}, commit=False)
    
    wandb.log({'Training/MeanQAValues':  training_predictions['qa'].cpu().mean().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MinQAValues':  training_predictions['qa'].cpu().min().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MaxQAValues':  training_predictions['qa'].cpu().max().item(), "training_step":iteration_count}, commit=False)
    
    wandb.log({'Training/StdQAValues':  training_predictions['qa'].cpu().std().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/QAValueLoss':  loss.cpu().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/LegalEntropyVal':  training_predictions['legal_ent'].mean().cpu().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/EntropyVal':  training_predictions['ent'].mean().cpu().item(), "training_step":iteration_count}, commit=False)
    #wandb.log({'Training/TotalLoss':  loss.cpu().item(), "training_step":iteration_count}, commit=False)
    if use_PER:
        wandb.log({'Training/ImportanceSamplingMean':  importanceSamplingWeights.cpu().mean().item(), "training_step":iteration_count}, commit=False)
        wandb.log({'Training/ImportanceSamplingStd':  importanceSamplingWeights.cpu().std().item(), "training_step":iteration_count}, commit=False)
        wandb.log({'Training/PER_Beta':  PER_beta, "training_step":iteration_count}, commit=False)
    
    wandb.log({}, commit=True)
    #wandb.run.history._data = wandb_data

    return loss, loss_per_item



unrolled_inferences_ray = ray.remote(unrolled_inferences_deprecated)
