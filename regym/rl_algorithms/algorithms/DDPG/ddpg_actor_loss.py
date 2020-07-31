from typing import Dict, List
import torch


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
    actor_prediction = model_actor(
      obs=states,
      rnn_states=rnn_states,
      goal=goals
    )

    for p in model_critic.parameters():
      p.requires_grad = False

    critic_prediction = model_critic(
      obs=states, 
      action=actor_prediction["a"], 
      rnn_states=rnn_states, 
      goal=goals
    )
    
    predictionQA = critic_prediction["qa"]
    
    # Compute loss:
    loss_per_item = -predictionQA
    
    if use_PER:
      loss_per_item = importanceSamplingWeights * loss_per_item
    
    loss = loss_per_item.mean()

    #weight decay :
    weights_decay_loss = weights_decay_lambda * 0.5*sum( [torch.mean(param*param) for param in model_actor.parameters()])
    
    total_loss = loss + weights_decay_loss

    for p in model_critic.parameters():
      p.requires_grad = True

    
    if summary_writer is not None:
        summary_writer.add_scalar('Training/MeanQAValues', critic_prediction['qa'].cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/StdQAValues', critic_prediction['qa'].cpu().std().item(), iteration_count)
        summary_writer.add_scalar('Training/QAValueLoss', loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/WeightsDecayLoss', weights_decay_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/TotalActorLoss', total_loss.cpu().item(), iteration_count)
        ## PER logs are handled by the critic loss...

    return total_loss, loss_per_item
