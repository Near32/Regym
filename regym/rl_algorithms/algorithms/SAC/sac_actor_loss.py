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
                 gamma: float = 0.99,
                 alpha: float = 0.2,
                 alpha_tuning: bool = False,
                 log_alpha: object = None,
                 target_expected_entropy: torch.Tensor= None,
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
    :param gamma: float discount factor.
    :param alpha: float entropy regularization coefficient.
    :param alpha_tuning: boolean specifying whether the entropy regularization coefficient is optimized.
    :param log_alpha: torch.Tensor from which entropy regularization coefficient is derived and optimized.
    :param target_expected_entropy: float target expected entropy used for entropy regularization coefficient optimization.
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

    #for p in model_critic.parameters():
    #  p.requires_grad = False

    # Unlike TD3, SAC updates the policy with 
    # respect to the min of the ensemble of critic:
    critic_prediction = model_critic.min_q_value(
      obs=states, 
      action=actor_prediction["a"], 
      rnn_states=rnn_states, 
      goal=goals
    )
    
    predictionQA = critic_prediction["qa"]
    log_pi_A = actor_prediction["log_pi_a"]
    unsquashed_policy_entropy = actor_prediction["ent"]

    # Compute loss:
    # SAC incorporates the entropy term to the loss:
    loss_per_item = -(predictionQA-alpha*log_pi_A)
    
    if use_PER:
      loss_per_item = importanceSamplingWeights.reshape(loss_per_item.shape) * loss_per_item
    
    loss = loss_per_item.mean()

    #weight decay :
    weights_decay_loss = weights_decay_lambda * 0.5*sum( [torch.mean(param*param) for param in model_actor.parameters()])
    
    total_loss = loss + weights_decay_loss

    #for p in model_critic.parameters():
    #  p.requires_grad = True

    alpha_tuning_loss = torch.zeros_like(total_loss)
    if alpha_tuning:
      alpha_tuning_loss_per_item = -log_alpha * (target_expected_entropy + log_pi_A.detach())
      if use_PER:
        alpha_tuning_loss_per_item = importanceSamplingWeights.reshape(alpha_tuning_loss_per_item.shape) * alpha_tuning_loss_per_item      
      alpha_tuning_loss = alpha_tuning_loss_per_item.mean()

    
    if summary_writer is not None:
        #summary_writer.add_scalar('Training/ActorLoss/MeanLogProbU', actor_prediction["log_normal_u"].cpu().mean().item(), iteration_count)
        #summary_writer.add_scalar('Training/ActorLoss/StdLogProbU', actor_prediction["log_normal_u"].cpu().std().item(), iteration_count)
        
        #summary_writer.add_scalar('Training/ActorLoss/MeanUMu', actor_prediction["mu"].cpu().mean().item(), iteration_count)
        #summary_writer.add_scalar('Training/ActorLoss/StdUMu', actor_prediction["mu"].cpu().std().item(), iteration_count)
        
        #summary_writer.add_scalar('Training/ActorLoss/MeanUStd', actor_prediction["std"].cpu().mean().item(), iteration_count)
        #summary_writer.add_scalar('Training/ActorLoss/StdUStd', actor_prediction["std"].cpu().std().item(), iteration_count)
        
        #summary_writer.add_scalar('Training/ActorLoss/MeanExtraTermLogProb', actor_prediction["extra_term_log_prob"].cpu().mean().item(), iteration_count)
        #summary_writer.add_scalar('Training/ActorLoss/StdExtraTermLogProb', actor_prediction["extra_term_log_prob"].cpu().std().item(), iteration_count)
        
        #summary_writer.add_scalar('Training/ActorLoss/MeanQAValues', predictionQA.cpu().mean().item(), iteration_count)
        #summary_writer.add_scalar('Training/ActorLoss/StdQAValues', predictionQA.cpu().std().item(), iteration_count)
        
        summary_writer.add_scalar('Training/ActorLoss/MeanUnsquashedPolicyEntropy', unsquashed_policy_entropy.cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/ActorLoss/StdUnsquashedPolicyEntropy', unsquashed_policy_entropy.cpu().std().item(), iteration_count)
        
        summary_writer.add_scalar('Training/ActorLoss/MeanLogPiAction', log_pi_A.cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/ActorLoss/StdLogPiAction', log_pi_A.cpu().std().item(), iteration_count)
        
        summary_writer.add_scalar('Training/ActorLoss/Loss', loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/ActorLoss/WeightsDecayLoss', weights_decay_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/TotalActorLoss', total_loss.cpu().item(), iteration_count)
        ## PER logs are handled by the critic loss...

        if alpha_tuning:
          #summary_writer.add_scalar('Training/LogAlpha', log_alpha.cpu().item(), iteration_count)
          summary_writer.add_scalar('Training/Alpha', log_alpha.cpu().exp().item(), iteration_count)
          summary_writer.add_scalar('Training/MeanAlphaTuningLoss', alpha_tuning_loss_per_item.cpu().mean().item(), iteration_count)
          summary_writer.add_scalar('Training/StdAlphaTuningLoss', alpha_tuning_loss_per_item.cpu().std().item(), iteration_count)


    return total_loss, loss_per_item, alpha_tuning_loss
