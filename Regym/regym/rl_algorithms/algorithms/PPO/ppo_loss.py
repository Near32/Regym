from typing import Dict, List
import torch

import wandb 


def compute_loss(states: torch.Tensor, 
                 actions: torch.Tensor,
                 log_probs_old: torch.Tensor, 
                 returns: torch.Tensor,
                 advantages: torch.Tensor, 
                 std_advantages: torch.Tensor, 
                 model: torch.nn.Module,
                 use_std_adv: bool = True,
                 ratio_clip: float = 0.1, 
                 entropy_weight: float = 0.01,
                 value_weight: float = 1.0,
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
    :param log_probs_old: Dimension: batch_size. Log probability of taking
                          the action with the same index in :param actions:.
                          Used to compute the policy probability ratio.
                          Refer to original paper equation (6)
    :param returns: Dimension: batch_size x 1. Empirical returns obtained via
                    calculating the discounted return from the environment's rewards
    :param advantages: Dimension: batch_size. Estimated advantage function
                       for every state and action in :param states: and
                       :param actions: (respectively) with the same index.
    :param std_advantages: Dimension: batch_size. Estimated standardized advantage function
                       for every state and action in :param states: and
                       :param actions: (respectively) with the same index.
    :param model: torch.nn.Module used to compute the policy probability ratio
                  as specified in equation (6) of original paper.
    :param use_std_adv: bool deciding whether to use a standardized advantage or not.
    :param ratio_clip: Epsilon value used to clip the policy ratio's value.
                       This parameter acts as the radius of the Trust Region.
                       Refer to original paper equation (7).
    :param entropy_weight: Coefficient to be used for the entropy bonus
                           for the loss function. Refer to original paper eq (9)
    :param value_weight: Coefficient to be used for the value loss
                           for the loss function. Refer to original paper eq (9)
    :param rnn_states: The :param model: can be made up of different submodules.
                       Some of these submodules will feature an LSTM architecture.
                       This parameter is a dictionary which maps recurrent submodule names
                       to a dictionary which contains 2 lists of tensors, each list
                       corresponding to the 'hidden' and 'cell' states of
                       the LSTM submodules. These tensors are used by the
                       :param model: when calculating the policy probability ratio.
    '''
    prediction = model(obs=states, action=actions, rnn_states=rnn_states)
    
    '''
    with torch.no_grad():
      old_prediction = model(states, actions, rnn_states=rnn_states)
    
    ratio = torch.exp((prediction['log_pi_a'] - old_prediction['log_pi_a']))
    '''
    
    ratio = torch.exp((prediction['log_pi_a'] - log_probs_old))
    
    if use_std_adv:
      adv = std_advantages
    else:
      adv = advantages

    obj = ratio * adv
    obj_clipped = torch.clamp(ratio,
                              1.0 - ratio_clip,
                              1.0 + ratio_clip) * adv
    
    policy_val = -torch.min(obj, obj_clipped).mean()
    entropy_val = prediction['ent'].mean()
    policy_loss = policy_val - entropy_weight * entropy_val # L^{clip} and L^{S} from original paper
    #policy_loss = -torch.min(obj, obj_clipped).mean() - entropy_weight * prediction['ent'].mean() # L^{clip} and L^{S} from original paper
    
    value_loss = value_weight * torch.nn.functional.mse_loss(input=prediction['v'], target=returns)
    total_loss = policy_loss + value_loss

    wandb.log({'Training/RatioMean': ratio.mean().cpu().item(), "training_step": iteration_count}, commit=False)
    #summary_writer.add_histogram('Training/Ratio', ratio.cpu(), iteration_count)
    wandb.log({'Training/AdvantageMean': advantages.mean().cpu().item(), "training_step": iteration_count}, commit=False)
    #summary_writer.add_histogram('Training/Advantage', advantages.cpu(), iteration_count)
    wandb.log({'Training/MeanVValues': prediction['v'].cpu().mean().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/MeanReturns': returns.cpu().mean().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/StdVValues': prediction['v'].cpu().std().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/StdReturns': returns.cpu().std().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/ValueLoss': value_loss.cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/PolicyVal': policy_val.cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/EntropyVal': entropy_val.cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/PolicyLoss': policy_loss.cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/TotalLoss': total_loss.cpu().item(), "training_step": iteration_count}, commit=False)
        
    return total_loss
