from typing import Dict, List
import torch
import torch.nn.functional as F


def compute_loss(states: torch.Tensor, 
                 actions: torch.Tensor,
                 next_states: torch.Tensor, 
                 rewards: torch.Tensor,
                 non_terminals: torch.Tensor,
                 goals: torch.Tensor,
                 model_critic: torch.nn.Module,
                 target_model_critic: torch.nn.Module,
                 model_actor: torch.nn.Module,
                 target_model_actor: torch.nn.Module,
                 gamma: float = 0.99,
                 weights_decay_lambda: float = 1.0,
                 use_PER: bool = False,
                 PER_beta: float = 1.0,
                 importanceSamplingWeights: torch.Tensor = None,
                 HER_target_clamping: bool = False,
                 summary_writer: object = None,
                 iteration_count: int = 0,
                 rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None) -> torch.Tensor:
    '''
    :param states: Dimension: batch_size x state_size: States visited by the agent.
    :param actions: Dimension: batch_size x action_size. Actions which the agent
                    took at every state in :param states: with the same index.
    :param next_states: Dimension: batch_size x state_size: Next states visited by the agent.
    :param non_terminals: Dimension: batch_size x 1: Non-terminal integers.
    :param rewards: Dimension: batch_size x 1. Environment rewards.
    :param goals: Dimension: batch_size x goal shape: Goal of the agent.
    :param model_critic: torch.nn.Module used to compute the critic loss, critic network.
    :param target_model_critic: torch.nn.Module used to compute the loss, target critic network.
    :param model_actor: torch.nn.Module used to compute the loss, actor network.
    :param target_model_actor: torch.nn.Module used to compute the loss, target actor network.
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
    prediction = model_critic.ensemble_q_values(
      obs=states, 
      action=actions, 
      rnn_states=rnn_states, 
      goal=goals
    )
    predictionQA = prediction["qa"]

    ############################
    # Compute target y:
    with torch.no_grad():
      next_rnn_states = None
      # Compute next_state's target actor action:
      ## Compute next rnn state if needs be:
      if rnn_states is not None:
        target_prediction = target_model_actor(states, action=actions, rnn_states=rnn_states, goal=goals)
        next_rnn_states = target_prediction['next_rnn_states']
      ##
      next_state_target_actor_prediction = target_model_actor(next_states, rnn_states=next_rnn_states, goal=goals)
      nextS_target_actor_action = next_state_target_actor_prediction['a']
      
      #Compute next_state's target QA value using target actor action:
      next_critic_rnn_states = None
      ## Compute next rnn state if needs be:
      if rnn_states is not None:
        target_critic_prediction = target_model_critic(
          obs=states, 
          action=actions, 
          rnn_states=rnn_states, 
          goal=goals
        )
        next_critic_rnn_states = target_critic_prediction['next_rnn_states']
      ##
      nextS_target_critic_prediction = target_model_critic.min_q_value(
        obs=next_states,
        action=nextS_target_actor_action,
        rnn_states=next_critic_rnn_states,
        goal=goals
      )
      nextS_target_critic_QA = nextS_target_critic_prediction['qa']
    
      # Compute the target Q values
      targetQA = rewards + non_terminals*(gamma * nextS_target_critic_QA)

      if HER_target_clamping:
            # clip the target to [-50,0]
            targetQA = torch.clamp(targetQA, -1. / (1 - gamma), 0)
    ############################

    # Compute loss:
    #loss_per_item = 0.5*(targetQA.detach() - predictionQA).pow(2.0)
    loss_per_item = torch.zeros_like(targetQA)
    for model_idx in range(model_critic.nbr_models):
        loss_per_item += F.smooth_l1_loss(predictionQA[..., model_idx], targetQA, reduction='none')
    
    if use_PER:
      loss_per_item = importanceSamplingWeights * loss_per_item
    
    loss = torch.mean(loss_per_item)

    #weight decay :
    weights_decay_loss = weights_decay_lambda * 0.5*sum( [torch.mean(param*param) for param in model_critic.parameters()])
    
    total_loss = loss + weights_decay_loss

    if summary_writer is not None:
        summary_writer.add_scalar('Training/MeanQAValues', prediction['qa'].cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/StdQAValues', prediction['qa'].cpu().std().item(), iteration_count)
        summary_writer.add_scalar('Training/QAValueLoss', loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/WeightsDecayLoss', weights_decay_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/TotalCriticLoss', total_loss.cpu().item(), iteration_count)
        if use_PER:
            summary_writer.add_scalar('Training/ImportanceSamplingMean', importanceSamplingWeights.cpu().mean().item(), iteration_count)
            summary_writer.add_scalar('Training/ImportanceSamplingStd', importanceSamplingWeights.cpu().std().item(), iteration_count)
            summary_writer.add_scalar('Training/PER_Beta', PER_beta, iteration_count)

    return total_loss, loss_per_item
