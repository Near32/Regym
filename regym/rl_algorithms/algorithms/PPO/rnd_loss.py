from typing import Dict, List
import torch
import torch.nn.functional as F 

import wandb 


def compute_loss(states: torch.Tensor, 
                 actions: torch.Tensor,
                 next_states: torch.Tensor,
                 log_probs_old: torch.Tensor, 
                 ext_returns: torch.Tensor,
                 ext_advantages: torch.Tensor,
                 std_ext_advantages: torch.Tensor,
                 int_returns: torch.Tensor,
                 int_advantages: torch.Tensor, 
                 std_int_advantages: torch.Tensor,
                 target_random_features: torch.Tensor,
                 states_mean: torch.Tensor, 
                 states_std: torch.Tensor,
                 model: torch.nn.Module,
                 pred_intr_model: torch.nn.Module,
                 intrinsic_reward_ratio: float,
                 ratio_clip: float, 
                 entropy_weight: float,
                 value_weight: float,
                 rnd_weight: float,
                 rnd_obs_clip: float,
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
    :param log_probs_old: Dimension: batch_size x 1. Log probability of taking
                          the action with the same index in :param actions:.
                          Used to compute the policy probability ratio.
                          Refer to original paper equation (6)
    :param ext_returns: Dimension: batch_size x 1. Empirical returns obtained via
                    calculating the discounted return from the environment's rewards
    :param ext_advantages: Dimension: batch_size x 1. Estimated advantage function
                       for every state and action in :param states: and
                       :param actions: (respectively) with the same index.
    :param std_ext_advantages: Dimension: batch_size x 1. Estimated standardized advantage function
                       for every state and action in :param states: and
                       :param actions: (respectively) with the same index.
    :param int_returns: Dimension: batch_size x 1. Empirical intrinsic returns obtained via
                        calculating the discounted intrinsic return from the intrinsic rewards.
    :param int_advantages: Dimension: batch_size x 1. Estimated intrisinc advantage function
                           for every state and action in :param states: and
                           :param actions: (respectively) with the same index.
    :param std_int_advantages: Dimension: batch_size x 1. Estimated standardized intrinsic advantage function
                       for every state and action in :param states: and
                       :param actions: (respectively) with the same index.
    :param target_random_features: target random features used to compute the intrinsic rewards.
    :param states_mean: mean over the previous training step's states.
    :param states_std: standard deviation over the previous training step's states.
    :param model: torch.nn.Module used to compute the policy probability ratio
                  as specified in equation (6) of original paper.
    :param predict_intr_model: intrinsic reward prediction model.
    :param intrinsic_reward_ratio: ratio of intrinsic reward to extrinsic reward.
    :param ratio_clip: Epsilon value used to clip the policy ratio's value.
                       This parameter acts as the radius of the Trust Region.
                       Refer to original paper equation (7).
    :param entropy_weight: Coefficient to be used for the entropy bonus
                           for the loss function. Refer to original paper eq (9)
    :param value_weight: Coefficient to be used for the value loss
                           for the loss function. Refer to original paper eq (9)
    :param rnd_weight: Coefficient to be used for the rnd loss
                           for the loss function.
    :param rnn_states: The :param model: can be made up of different submodules.
                       Some of these submodules will feature an LSTM architecture.
                       This parameter is a dictionary which maps recurrent submodule names
                       to a dictionary which contains 2 lists of tensors, each list
                       corresponding to the 'hidden' and 'cell' states of
                       the LSTM submodules. These tensors are used by the
                       :param model: when calculating the policy probability ratio.
    '''
    advantages = ext_advantages + intrinsic_reward_ratio*int_advantages
    std_advantages = std_ext_advantages + intrinsic_reward_ratio*std_int_advantages
    
    prediction = model(states, actions, rnn_states=rnn_states)
    
    ratio = torch.exp((prediction['log_pi_a'] - log_probs_old))
    
    obj = ratio * std_advantages
    obj_clipped = torch.clamp(ratio,
                              1.0 - ratio_clip,
                              1.0 + ratio_clip) * std_advantages
    
    policy_val = -torch.min(obj, obj_clipped).mean()
    entropy_val = prediction['ent'].mean()
    policy_loss = policy_val - entropy_weight * entropy_val # L^{clip} and L^{S} from original paper
    #policy_loss = -torch.min(obj, obj_clipped).mean() - entropy_weight * prediction['ent'].mean() # L^{clip} and L^{S} from original paper
    
    # Random Network Distillation loss:
    with torch.no_grad():
        norm_next_states = (next_states-states_mean) / (states_std+1e-8)
    if rnd_obs_clip > 1e-1:
      norm_next_states = torch.clamp( norm_next_states, -rnd_obs_clip, rnd_obs_clip)
    pred_random_features = pred_intr_model(norm_next_states)
    
    # Clamping:
    #pred_random_features = torch.clamp(pred_random_features, -1e20, 1e20)
    #target_random_features = torch.clamp(target_random_features, -1e20, 1e20)
    
    # Softmax:
    #pred_random_features = F.softmax(pred_random_features)
    
    # Losses:
    #int_reward_loss = torch.nn.functional.smooth_l1_loss(target_random_features.detach(), pred_random_features)
    int_reward_loss = torch.nn.functional.mse_loss( pred_random_features, target_random_features.detach())
    
    #ext_returns = torch.clamp(ext_returns, -1e10, 1e10)
    #int_returns = torch.clamp(int_returns, -1e10, 1e10)
    #prediction['v'] = torch.clamp(prediction['v'], -1e10, 1e10)
    #prediction['int_v'] = torch.clamp(prediction['int_v'], -1e10, 1e10)
    
    #ext_v_loss = torch.nn.functional.smooth_l1_loss(ext_returns, prediction['v']) 
    #int_v_loss = torch.nn.functional.smooth_l1_loss(int_returns, prediction['int_v']) 
    ext_v_loss = torch.nn.functional.mse_loss(input=prediction['v'], target=ext_returns) 
    int_v_loss = torch.nn.functional.mse_loss(input=prediction['int_v'], target=int_returns) 
    
    value_loss = (ext_v_loss + int_v_loss)
    #value_loss = ext_v_loss
    rnd_loss = int_reward_loss 

    total_loss = policy_loss + rnd_weight * rnd_loss + value_weight * value_loss
    #total_loss = policy_loss + value_weight * value_loss

    wandb.log({'Training/RatioMean': ratio.mean().cpu().item(), "training_step": iteration_count}, commit=False)
    #summary_writer.add_histogram('Training/Ratio', ratio.cpu(), iteration_count)
    wandb.log({'Training/ExtAdvantageMean': ext_advantages.mean().cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/IntAdvantageMean': int_advantages.mean().cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/AdvantageMean': advantages.mean().cpu().item(), "training_step": iteration_count}, commit=False)
    #summary_writer.add_histogram('Training/ExtAdvantage', ext_advantages.cpu(), iteration_count)
    #summary_writer.add_histogram('Training/IntAdvantage', int_advantages.cpu(), iteration_count)
    #summary_writer.add_histogram('Training/Advantage', advantages.cpu(), iteration_count)
    wandb.log({'Training/RNDLoss': int_reward_loss.cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/ExtVLoss': ext_v_loss.cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/IntVLoss': int_v_loss.cpu().item(), "training_step": iteration_count}, commit=False)
    
    wandb.log({'Training/MeanVValues': prediction['v'].cpu().mean().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/MeanReturns': ext_returns.cpu().mean().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/StdVValues': prediction['v'].cpu().std().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/StdReturns': ext_returns.cpu().std().item(), "training_step": iteration_count}, commit=False)
    
    wandb.log({'Training/MeanIntVValues': prediction['int_v'].cpu().mean().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/MeanIntReturns': int_returns.cpu().mean().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/StdIntVValues': prediction['int_v'].cpu().std().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/StdIntReturns': int_returns.cpu().std().item(), "training_step": iteration_count}, commit=False)
    
    wandb.log({'Training/ValueLoss': value_loss.cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/PolicyVal': policy_val.cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/EntropyVal': entropy_val.cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/PolicyLoss': policy_loss.cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/TotalLoss': total_loss.cpu().item(), "training_step": iteration_count}, commit=False)
        
    return total_loss
