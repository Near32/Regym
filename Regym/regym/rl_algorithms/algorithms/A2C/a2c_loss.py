from typing import Dict, List
import torch


def compute_loss(states: torch.Tensor, 
                 actions: torch.Tensor,
                 returns: torch.Tensor,
                 advantages: torch.Tensor, 
                 model: torch.nn.Module,
                 entropy_weight: float,
                 value_weight: float,
                 summary_writer: object = None,
                 iteration_count: int = 0,
                 rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None) -> torch.Tensor:
    '''
    Computes the loss of an actor critic model using the
    loss function from equation (9) in the paper:
    Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347

    :param states: Dimension: batch_size x state_size: States visited by the agent.
    :param actions: Dimension: batch_size x action_size. Actions which the agent
                    took at every state in :param states: with the same index.
    :param returns: Dimension: batch_size x 1. Empirical returns obtained via
                    calculating the discounted return from the environment's rewards
    :param advantages: Dimension: batch_size. Estimated advantage function
                       for every state and action in :param states: and
                       :param actions: (respectively) with the same index.
    :param model: torch.nn.Module used to compute the policy probability ratio
                  as specified in equation (6) of original paper.
    :param entropy_weight: Coefficient to be used for the entropy bonus
                           for the loss function. Refer to original paper eq (9)
    :param rnn_states: The :param model: can be made up of different submodules.
                       Some of these submodules will feature an LSTM architecture.
                       This parameter is a dictionary which maps recurrent submodule names
                       to a dictionary which contains 2 lists of tensors, each list
                       corresponding to the 'hidden' and 'cell' states of
                       the LSTM submodules. These tensors are used by the
                       :param model: when calculating the policy probability ratio.
    '''
    prediction = model(states, actions, rnn_states=rnn_states)
    
    policy_val = -(advantages.detach() * prediction['log_pi_a']).mean()
    entropy_val = prediction['ent'].mean()
    policy_loss = policy_val - entropy_weight * entropy_val
    
    value_loss = value_weight * torch.nn.functional.mse_loss(input=prediction['v'], target=returns)
    total_loss = (policy_loss + value_loss)

    if summary_writer is not None:
        summary_writer.add_scalar('Training/AdvantageMean', advantages.mean().cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/MeanVValues', prediction['v'].cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/MeanReturns', returns.cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/StdVValues', prediction['v'].cpu().std().item(), iteration_count)
        summary_writer.add_scalar('Training/StdReturns', returns.cpu().std().item(), iteration_count)
        summary_writer.add_scalar('Training/ValueLoss', value_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/PolicyVal', policy_val.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/EntropyVal', entropy_val.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/PolicyLoss', policy_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/TotalLoss', total_loss.cpu().item(), iteration_count)
        
    return total_loss
