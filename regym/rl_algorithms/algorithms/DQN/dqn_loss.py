from typing import Dict, List, Optional
import torch


def compute_loss(states: torch.Tensor, 
                 actions: torch.Tensor,
                 next_states: torch.Tensor, 
                 rewards: torch.Tensor,
                 non_terminals: torch.Tensor,
                 goals: torch.Tensor,
                 model: torch.nn.Module,
                 target_model: torch.nn.Module,
                 gamma: float = 0.99,
                 weights_decay_lambda: float = 1.0,
                 use_PER: bool = False,
                 PER_beta: float = 1.0,
                 importanceSamplingWeights: torch.Tensor = None,
                 HER_target_clamping: bool = False,
                 summary_writer: object = None,
                 iteration_count: int = 0,
                 rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None,
                 kwargs:Optional[Dict]=None) -> torch.Tensor:
    '''
    :param states: Dimension: batch_size x state_size: States visited by the agent.
    :param actions: Dimension: batch_size x action_size. Actions which the agent
                    took at every state in :param states: with the same index.
    :param next_states: Dimension: batch_size x state_size: Next states visited by the agent.
    :param non_terminals: Dimension: batch_size x 1: Non-terminal integers.
    :param rewards: Dimension: batch_size x 1. Environment rewards.
    :param goals: Dimension: batch_size x goal shape: Goal of the agent.
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
    '''
    prediction = model(states, action=actions, rnn_states=rnn_states, goal=goals)

    state_action_values = prediction["qa"]
    state_action_values_g = state_action_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

    ############################
    with torch.no_grad():
      next_rnn_states = None
      if rnn_states is not None:
        target_prediction = target_model(states, action=actions, rnn_states=rnn_states, goal=goals)
        next_rnn_states = target_prediction['next_rnn_states']
      
      next_target_prediction = target_model(next_states, rnn_states=next_rnn_states, goal=goals)
      targetQ_nextS_A_values = next_target_prediction['qa']
      
      maxA_targetQ_nextS_A_values = targetQ_nextS_A_values.max(dim=1)[0]
    
      # Compute the expected Q values
      expected_state_action_values = rewards + non_terminals*(gamma**kwargs['n_step']) * maxA_targetQ_nextS_A_values
      if HER_target_clamping:
            # clip the target to [-50,0]
            expected_state_action_values = torch.clamp(expected_state_action_values, -1. / (1 - gamma), 0)
    ############################

    # Compute loss:
    #diff_squared = torch.abs(expected_state_action_values.detach() - state_action_values_g) 
    diff_squared = (expected_state_action_values.detach() - state_action_values_g).pow(2.0)
    loss_per_item = diff_squared
    
    if use_PER:
      diff_squared = importanceSamplingWeights * diff_squared
    
    loss = 0.5*torch.mean(diff_squared)

    if summary_writer is not None:
        summary_writer.add_scalar('Training/MeanQAValues', prediction['qa'].cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/StdQAValues', prediction['qa'].cpu().std().item(), iteration_count)
        summary_writer.add_scalar('Training/QAValueLoss', loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/EntropyVal', prediction['ent'].mean().cpu().item(), iteration_count)
        #summary_writer.add_scalar('Training/TotalLoss', loss.cpu().item(), iteration_count)
        if use_PER:
            summary_writer.add_scalar('Training/ImportanceSamplingMean', importanceSamplingWeights.cpu().mean().item(), iteration_count)
            summary_writer.add_scalar('Training/ImportanceSamplingStd', importanceSamplingWeights.cpu().std().item(), iteration_count)
            summary_writer.add_scalar('Training/PER_Beta', PER_beta, iteration_count)

    return loss, loss_per_item
