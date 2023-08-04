from typing import Dict, List
import numpy as np
import torch
import wandb


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
                 rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None,
                 next_rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None,
                 phase: str = "Training",
    ) -> torch.Tensor:
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
    :param next_rnn_states: The :param model: can be made up of different submodules.
                       Some of these submodules will feature an LSTM architecture.
                       This parameter is a dictionary which maps recurrent submodule names
                       to a dictionary which contains 2 lists of tensors, each list
                       corresponding to the 'hidden' and 'cell' states of
                       the LSTM submodules. These tensors are used by the
                       :param model: when calculating the policy probability ratio.
    '''
    
    # Prediction must be done from current state as it is aimed
    # at predicting immediate value of a state (i.e. gamma=0).
    output_dict= predictor.compute_loss(
        states,  
        gt_rewards=rewards,
        rnn_states=rnn_states,
    )
    prediction = output_dict['prediction']
    loss_per_item = output_dict['loss_per_item']
    accuracy = output_dict['accuracy']

    if importanceSamplingWeights is not None:
      loss_per_item = importanceSamplingWeights * loss_per_item
    
    loss = torch.mean(loss_per_item)
    
    wandb.log({f'{phase}/RP_Predictor/Loss': loss.cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({f'{phase}/RP_Predictor/Accuracy': accuracy.cpu().mean().item(), "training_step": iteration_count}, commit=False)
    
    if importanceSamplingWeights is not None:
        wandb.log({f'{phase}/RP_Predictor/ImportanceSamplingMean': importanceSamplingWeights.cpu().mean().item(), "training_step": iteration_count}, commit=False)
        wandb.log({f'{phase}/RP_Predictor/ImportanceSamplingStd': importanceSamplingWeights.cpu().std().item(), "training_step": iteration_count}, commit=False)
        wandb.log({f'{phase}/RP_Predictor/PER_Beta': PER_beta, "training_step": iteration_count}, commit=False)

    out_dict = {
        'loss':loss, 
        'loss_per_item':loss_per_item,
        'accuracy':accuracy
    }

    return out_dict
