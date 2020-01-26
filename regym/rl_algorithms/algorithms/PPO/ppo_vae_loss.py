from typing import Dict, List
import torch


def compute_loss(states: torch.Tensor, 
                 actions: torch.Tensor,
                 log_probs_old: torch.Tensor, 
                 returns: torch.Tensor,
                 advantages: torch.Tensor, 
                 std_advantages: torch.Tensor, 
                 model: torch.nn.Module,
                 ratio_clip: float, 
                 entropy_weight: float,
                 value_weight: float,
                 vae_weight: float,
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
    :param returns: Dimension: batch_size x 1. Empirical returns obtained via
                    calculating the discounted return from the environment's rewards
    :param advantages: Dimension: batch_size x 1. Estimated advantage function
                       for every state and action in :param states: and
                       :param actions: (respectively) with the same index.
    :param std_advantages: Dimension: batch_size. Estimated standardized advantage function
                       for every state and action in :param states: and
                       :param actions: (respectively) with the same index.
    :param model: torch.nn.Module used to compute the policy probability ratio
                  as specified in equation (6) of original paper.
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
    prediction = model(states, actions, rnn_states=rnn_states)
    
    ratio = torch.exp((prediction['log_pi_a'] - log_probs_old.detach()))
    obj = ratio * std_advantages
    obj_clipped = ratio.clamp(1.0 - ratio_clip,
                              1.0 + ratio_clip) * std_advantages
    
    policy_val = -torch.min(obj, obj_clipped).mean()
    entropy_val = -prediction['ent'].mean()
    policy_loss = policy_val + entropy_weight * entropy_val # L^{clip} and L^{S} from original paper
    #policy_loss = -torch.min(obj, obj_clipped).mean() - entropy_weight * prediction['ent'].mean() # L^{clip} and L^{S} from original paper
    
    value_loss = value_weight * torch.nn.functional.mse_loss(input=prediction['v'], target=returns)
    total_loss = (policy_loss + value_loss)

    if summary_writer is not None:
        summary_writer.add_scalar('Training/RatioMean', ratio.mean().cpu().item(), iteration_count)
        #summary_writer.add_histogram('Training/Ratio', ratio.cpu(), iteration_count)
        summary_writer.add_scalar('Training/AdvantageMean', advantages.mean().cpu().item(), iteration_count)
        #summary_writer.add_histogram('Training/Advantage', advantages.cpu(), iteration_count)
        summary_writer.add_histogram('Training/VValues', prediction['v'].cpu(), iteration_count)
        summary_writer.add_scalar('Training/MeanVValues', prediction['v'].cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/MeanReturns', returns.cpu().mean().item(), iteration_count)
        summary_writer.add_histogram('Training/Returns', returns.cpu(), iteration_count)
        summary_writer.add_scalar('Training/StdVValues', prediction['v'].cpu().std().item(), iteration_count)
        summary_writer.add_scalar('Training/StdReturns', returns.cpu().std().item(), iteration_count)
        summary_writer.add_scalar('Training/ValueLoss', value_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/PolicyVal', policy_val.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/EntropyVal', entropy_val.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/PolicyLoss', policy_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/TotalLoss', total_loss.cpu().item(), iteration_count)


    VAE_loss, \
    neg_log_lik, \
    kl_div_reg, \
    kl_div, \
    tc_loss, \
    modularity = model.compute_vae_loss(states)
    
    total_loss += vae_weight*VAE_loss.mean()+tc_loss.mean()


    if summary_writer is not None:
        summary_writer.add_scalar('Training/VAE/loss', VAE_loss.mean().cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/VAE/neg_log_lik', neg_log_lik.mean().cpu(), iteration_count)
        summary_writer.add_scalar('Training/VAE/kl_div_reg', kl_div_reg.mean().cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/VAE/kl_div', kl_div.mean().cpu(), iteration_count)

        summary_writer.add_scalar('Training/VAE/tc_loss', tc_loss.mean().cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/VAE/modularity', modularity.mean().cpu().item(), iteration_count)
        
    return total_loss
