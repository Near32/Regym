from typing import Dict, List
import torch


def compute_loss(states: torch.Tensor, 
                 actions: torch.Tensor,
                 next_states: torch.Tensor, 
                 rewards: torch.Tensor,
                 non_terminals: torch.Tensor,
                 goals: torch.Tensor,
                 predictor: torch.nn.Module,
                 weights_decay_lambda: float = 1.0,
                 use_PER: bool = False,
                 PER_beta: float = 1.0,
                 importanceSamplingWeights: torch.Tensor = None,
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
    output_dict= predictor.compute_loss(states, goal=goals)
    prediction = output_dict['prediction']
    loss_per_item = output_dict['loss_per_item']
    accuracies = output_dict['accuracies']
    sentence_accuracies = output_dict['sentence_accuracies']
    accuracy = sentence_accuracies.cpu().mean().item()

    if use_PER:
      loss_per_item = importanceSamplingWeights * loss_per_item
    
    loss = torch.mean(loss_per_item)
    

    if summary_writer is not None:
        summary_writer.add_scalar('Training/THER_Predictor/Loss', loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/THER_Predictor/Accuracy', accuracies.cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/THER_Predictor/SentenceAccuracy', sentence_accuracies.cpu().item(), iteration_count)
        for idx in range(accuracies.shape[-1]):
          summary_writer.add_scalar(f'Training/THER_Predictor/Accuracy_{idx}', accuracies[..., idx].cpu().item(), iteration_count)
        
        if use_PER:
            summary_writer.add_scalar('Training/THER_Predictor/ImportanceSamplingMean', importanceSamplingWeights.cpu().mean().item(), iteration_count)
            summary_writer.add_scalar('Training/THER_Predictor/ImportanceSamplingStd', importanceSamplingWeights.cpu().std().item(), iteration_count)
            summary_writer.add_scalar('Training/THER_Predictor/PER_Beta', PER_beta, iteration_count)

    return {'loss':loss, 
            'loss_per_item':loss_per_item,
            'accuracy':accuracy
            }
