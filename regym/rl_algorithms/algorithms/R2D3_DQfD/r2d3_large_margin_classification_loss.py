from typing import Dict, List, Optional

from functools import partial 
import copy
import time 

import torch
import ray

from regym.rl_algorithms.algorithms import Algorithm 
from regym.rl_algorithms.algorithms.R2D2.r2d2_loss import  value_function_rescaling, inverse_value_function_rescaling
from regym.rl_algorithms.algorithms.R2D2.r2d2_loss import  extract_rnn_states_from_time_indices, replace_rnn_states_at_time_indices
from regym.rl_algorithms.algorithms.R2D2.r2d2_loss import  roll_sequences, unrolled_inferences, unrolled_inferences_ray
from regym.rl_algorithms.utils import is_leaf, copy_hdict, _concatenate_list_hdict, recursive_inplace_update


eps = 1e-3
study_qa_values_discrepancy = True
soft = False


# Adapted from: https://github.com/google-research/seed_rl/blob/34fb2874d41241eb4d5a03344619fb4e34dd9be6/agents/r2d2/learner.py#L333
def compute_loss(states: torch.Tensor,
                 actions: torch.Tensor,
                 next_states: torch.Tensor,
                 rewards: torch.Tensor,
                 non_terminals: torch.Tensor,
                 goals: torch.Tensor,
                 demo_transition_mask: torch.Tensor,
                 model: torch.nn.Module,
                 target_model: torch.nn.Module,
                 gamma: float = 0.99,
                 weights_decay_lambda: float = 1.0,
                 weights_entropy_lambda: float = 0.1,
                 use_PER: bool = False,
                 PER_beta: float = 1.0,
                 importanceSamplingWeights: torch.Tensor = None,
                 HER_target_clamping: bool = False,
                 summary_writer: object = None,
                 iteration_count: int = 0,
                 rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None,
                 next_rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None,
                 kwargs:Optional[Dict]=None) -> torch.Tensor:
    '''
    :param states: Dimension: batch_size x unroll_length x state_size: States visited by the agent.
    :param actions: Dimension: batch_size x unroll_length x action_size. Actions which the agent
                    took at every state in :param states: with the same index.
    :param next_states: Dimension: batch_size x unroll_length x state_size: Next sequence of unrolled states visited by the agent.
    :param non_terminals: Dimension: batch_size x unroll_length x 1: Non-terminal integers.
    :param rewards: Dimension: batch_size x unroll_length x 1. Environment rewards, or n-step returns if using n-step returns.
    :param goals: Dimension: batch_size x unroll_length x goal shape: Goal of the agent.
    :param demo_transition_mask: Dimension: batch_size : maks to specify whether the current batch element is from the policy or from the exper replay buffer.
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
    batch_size = states.shape[0]
    unroll_length = states.shape[1]
    map_keys=['qa', 'a', 'ent']

    start = time.time()

    if kwargs['burn_in']:
        burn_in_length = kwargs['sequence_replay_burn_in_length']
        training_length = kwargs['sequence_replay_unroll_length']-burn_in_length

        burn_in_states, training_states = torch.split(
            states, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )

        training_rnn_states = extract_rnn_states_from_time_indices(
            rnn_states, 
            time_indices_start=kwargs['sequence_replay_burn_in_length'],
            time_indices_end=kwargs['sequence_replay_unroll_length'],
        )
        _, training_rewards = torch.split(
            rewards, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )
        _, training_non_terminals = torch.split(
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
        burned_in_rnn_states_inputs = unrolled_inferences(
            model=model, 
            states=burn_in_states, 
            rnn_states=rnn_states,
            grad_enabler=False,
            use_zero_initial_states=kwargs['sequence_replay_use_zero_initial_states'],
            extras=False,
            map_keys=map_keys,
        )

        target_model.reset_noise()

        burned_in_target_predictions, \
        unrolled_target_predictions, \
        burned_in_rnn_states_target_inputs = unrolled_inferences(
            model=target_model, 
            states=burn_in_states, 
            rnn_states=rnn_states,
            grad_enabler=False,
            use_zero_initial_states=kwargs['sequence_replay_use_zero_initial_states'],
            extras=False,
            map_keys=map_keys,
        )

        # Replace the bruned in rnn states in the training rnn states:
        training_rnn_states = replace_rnn_states_at_time_indices(
            rnn_states_batched=training_rnn_states, 
            replacing_rnn_states_batched=burned_in_rnn_states_inputs, 
            time_indices_start=0, 
            time_indices_end=0
        )

        training_target_rnn_states = replace_rnn_states_at_time_indices(
            rnn_states_batched=training_rnn_states, 
            replacing_rnn_states_batched=burned_in_rnn_states_target_inputs, 
            time_indices_start=0, 
            time_indices_end=0
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

    training_burned_in_predictions, \
    training_unrolled_predictions, _ = unrolled_inferences(
        model=model, 
        states=training_states, 
        rnn_states=training_rnn_states,
        grad_enabler=True,
        use_zero_initial_states=False,
        extras=not(kwargs['burn_in']) or study_qa_values_discrepancy,
        map_keys=map_keys,
    )

    target_model.reset_noise()

    training_burned_in_target_predictions, \
    training_unrolled_target_predictions, _ = unrolled_inferences(
        model=target_model, 
        states=training_states, 
        rnn_states=training_target_rnn_states,
        grad_enabler=False,
        use_zero_initial_states=False,
        extras=not(kwargs['burn_in']),
        map_keys=map_keys,
    )

    if kwargs['burn_in']:
        training_predictions = training_burned_in_predictions
        training_target_predictions = training_burned_in_target_predictions
    else:
        training_predictions = training_unrolled_predictions
        training_target_predictions = training_unrolled_target_predictions
    
    
    state_action_values = training_predictions["qa"]
    # (batch_size, unroll_dim, nbr_actions)
    current_actions = training_predictions["a"].reshape(batch_size, training_length, -1)
    # (batch_size, unroll_dim, ...)
    
    #---------------------------------------------------------------------------------------
    # R2D3 Loss:
    #---------------------------------------------------------------------------------------
    
    # Although it is the approach in some other repo, this is unstable (unless maybe the greater n_step values help regularize):
    #state_action_values_g = state_action_values.gather(dim=-1, index=current_actions).reshape(batch_size, training_length, -1)
    # Stable training: crucial:
    state_action_values_g = state_action_values.gather(dim=-1, index=training_actions.reshape(batch_size, training_length,-1)).reshape(batch_size, training_length, -1)
    # (batch_size, unroll_dim, 1)
    
    if soft:
        targetQ_Si_Ai_values = inverse_value_function_rescaling(training_target_predictions['qa']+training_target_predictions['ent'])
    else:
        targetQ_Si_Ai_values = inverse_value_function_rescaling(training_target_predictions['qa'])
    # (batch_size, training_length, num_actions)
    argmaxA_Q_Si_A_values = state_action_values.max(dim=-1)[1].unsqueeze(-1)
    # (batch_size, training_length, 1)
    
    # Non-greedy:
    #targetQ_Si_argmaxAQvalue = targetQ_Si_Ai_values.gather(dim=-1, index=current_actions).reshape(batch_size, training_length, -1)
    # Greedy:
    targetQ_Si_argmaxAQvalue = targetQ_Si_Ai_values.gather(dim=-1, index=argmaxA_Q_Si_A_values).reshape(batch_size, training_length, -1)
    # (batch_size, training_length, 1)

    targetQ_Sipn_argmaxAipn_values = torch.cat(
        [
            targetQ_Si_argmaxAQvalue[:, kwargs['n_step']:, ...]
        ]+[
            targetQ_Si_argmaxAQvalue[:, -1:, ...] / gamma**(k+1) # it will be normalized down below when computing bellman target    
            for k in range(kwargs['n_step'])
        ],
        dim=1,
    )
    # (batch_size, training_length, 1)
    
    training_non_terminals_ipnm1 = torch.cat(
        [   
            training_non_terminals[:, k:k+kwargs['n_step'], ...].prod(dim=1).reshape(batch_size, 1, -1)
            for k in range(training_non_terminals.shape[1])
        ],
        dim=1,
    )
    
    # Compute the Bellman Target for Q values at Si,Ai: with gamma
    #bellman_target_Sipn_Aipn = training_rewards + (gamma**kwargs['n_step']) * targetQ_Sipn_argmaxAipn_values * training_non_terminals
    #bellman_target_Sipn_Aipn = training_rewards + (gamma**kwargs['n_step']) * targetQ_Sipn_argmaxAipn_values * training_non_terminals_ipn
    bellman_target_Sipn_Aipn = training_rewards + (gamma**kwargs['n_step']) * targetQ_Sipn_argmaxAipn_values * training_non_terminals_ipnm1
    # (batch_size, training_length, 1)
    
    # (batch_size, training_length, ...)
    scaled_bellman_target_Sipn_Aipn = value_function_rescaling(bellman_target_Sipn_Aipn)
    # (batch_size, training_length, 1)
    
    # Compute loss:
    #scaled_bellman_target_Sipn_Aipn = scaled_bellman_target_Sipn_Aipn[:,:-1,...]
    #state_action_values_g = state_action_values_g[:,:-1,...]

    state_action_values_g = state_action_values_g.reshape(scaled_bellman_target_Sipn_Aipn.shape)
    unscaled_state_action_values_g = inverse_value_function_rescaling(state_action_values_g)

    #td_error = torch.abs(scaled_bellman_target_Sipn_Aipn.detach() - state_action_values_g)
    td_error = torch.abs(bellman_target_Sipn_Aipn.detach() - unscaled_state_action_values_g)
    scaled_td_error = torch.abs(scaled_bellman_target_Sipn_Aipn.detach() - state_action_values_g)
    # (batch_size, training_length, 1)
    
    #loss_per_item = td_error
    # SEED RL repo uses the scaled td error for priorities:
    loss_per_item = scaled_td_error
    # (batch_size, training_length, 1)
    
    diff_squared = scaled_td_error.pow(2.0)
    # (batch_size, training_length, 1)
    
    if use_PER:
      diff_squared = importanceSamplingWeights * diff_squared

    mask = torch.ones_like(diff_squared)
    mask[:,-1, ...] = (1-training_non_terminals[:,-1,...])

    r2d3_loss = 0.5*torch.mean(diff_squared*mask)-weights_entropy_lambda*training_predictions['ent'].mean()

    # DQfD Loss:
    nbr_actions = state_action_values.shape[-1]
    expert_margin_value = 0.8
    expert_actions = training_actions.reshape(batch_size, training_length, -1).repeat(1,1, nbr_actions)
    #(batch_size, training_length, nbr_actions)
    
    expert_margin_actions = torch.arange(nbr_actions).reshape(1,1, -1).repeat( batch_size, training_length, 1).to(expert_actions.device)
    #(batch_size, training_length, nbr_actions)
    expert_margin = expert_margin_value*(expert_margin_actions != expert_actions).float()

    maxA_Q_Si_A_margin_values = (state_action_values+expert_margin).max(dim=-1)[0].unsqueeze(-1)
    #(batch_size, training_length, 1)
    
    expert_state_action_values = state_action_values.gather(
        dim=-1, 
        index=training_actions.reshape(batch_size, training_length,-1)
    ).reshape(batch_size, training_length, -1)
    # (batch_size, unroll_dim, 1)
    
    per_item_dqfd_loss = (maxA_Q_Si_A_margin_values - expert_state_action_values)*demo_transition_mask.reshape(batch_size, 1, 1)
    # (batch_size, unroll_dim, 1)
    dqfd_loss = torch.mean(per_item_dqfd_loss)
    # (1)
    
    loss_per_item = loss_per_item + per_item_dqfd_loss

    loss = r2d3_loss + dqfd_loss
    # (1)
    
    end = time.time()

    if summary_writer is not None:
        summary_writer.add_scalar('Training/TimeComplexity', end-start, iteration_count)
        
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
            
            summary_writer.add_scalar('Training/DiscrepancyQAValues/Initial', initial_discrepancy_qa.cpu().mean().item(), iteration_count)
            summary_writer.add_scalar('Training/DiscrepancyQAValues/Final', final_discrepancy_qa.cpu().mean().item(), iteration_count)
            
        summary_writer.add_scalar('Training/MeanBellmanTarget', bellman_target_Sipn_Aipn.cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/MinBellmanTarget', bellman_target_Sipn_Aipn.cpu().min().item(), iteration_count)
        summary_writer.add_scalar('Training/MaxBellmanTarget', bellman_target_Sipn_Aipn.cpu().max().item(), iteration_count)
        
        summary_writer.add_scalar('Training/MeanQAValues', training_predictions['qa'].cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/MinQAValues', training_predictions['qa'].cpu().min().item(), iteration_count)
        summary_writer.add_scalar('Training/MaxQAValues', training_predictions['qa'].cpu().max().item(), iteration_count)
        
        summary_writer.add_scalar('Training/StdQAValues', training_predictions['qa'].cpu().std().item(), iteration_count)
        summary_writer.add_scalar('Training/QA_R2D3Loss', r2d3_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/QA_DQfDLoss', dqfd_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/QAValueLoss', loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/EntropyVal', training_predictions['ent'].mean().cpu().item(), iteration_count)
        #summary_writer.add_scalar('Training/TotalLoss', loss.cpu().item(), iteration_count)
        if use_PER:
            summary_writer.add_scalar('Training/ImportanceSamplingMean', importanceSamplingWeights.cpu().mean().item(), iteration_count)
            summary_writer.add_scalar('Training/ImportanceSamplingStd', importanceSamplingWeights.cpu().std().item(), iteration_count)
            summary_writer.add_scalar('Training/PER_Beta', PER_beta, iteration_count)

    return loss, loss_per_item
