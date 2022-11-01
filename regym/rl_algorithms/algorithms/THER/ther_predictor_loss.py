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
    '''
    output_dict= predictor.compute_loss(
        next_states, #states, 
        rnn_states=rnn_states,
        goal=goals,
    )
    prediction = output_dict['prediction']
    loss_per_item = output_dict['loss_per_item']
    accuracies = output_dict['accuracies']
    sentence_accuracies = output_dict['sentence_accuracies']
    accuracy = sentence_accuracies.cpu().mean().item()

    if importanceSamplingWeights is not None:
      loss_per_item = importanceSamplingWeights * loss_per_item
    
    loss = torch.mean(loss_per_item)
    
    # Logging:
    if iteration_count % 128 == 0:
        if goals is None:
            goals = rnn_states['gt_sentences'][0]
            idx2w = predictor.model.modules['InstructionGenerator'].idx2w
        
        columns = [f"token{idx}" for idx in range(prediction.shape[1])]
        columns += [f"gt_token{idx}" for idx in range(goals.shape[1])]
        columns += ["loss", "stimulus", "a_{t-1}",]
        text_table = wandb.Table(columns=columns)
        for bidx in range(prediction.shape[0]):
            word_sentence = [idx2w[token.item()] for token in prediction[bidx]]
            gt_word_sentence = [idx2w[token.item()] for token in goals[bidx]] 
            nbr_frames = next_states[bidx].shape[0]//4
            stimulus = next_states[bidx].cpu().reshape(nbr_frames,4,56,56).numpy()[:,:3]*255
            stimulus = stimulus.astype(np.uint8)
            stimulus = wandb.Video(stimulus, fps=2, format="gif")
            previous_action_int = rnn_states['critic_body']['extra_inputs']['previous_action_int'][0][bidx].cpu().item() 
            text_table.add_data(*[
                *word_sentence, 
                *gt_word_sentence,
                loss_per_item[bidx], 
                stimulus,
                previous_action_int
                ]
            )
        
        wandb.log({f"{phase}/THER_Predictor/SampleTable":text_table, "training_step": iteration_count}, commit=False)
            
    wandb.log({f'{phase}/THER_Predictor/Loss': loss.cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({f'{phase}/THER_Predictor/Accuracy': accuracies.cpu().mean().item(), "training_step": iteration_count}, commit=False)
    wandb.log({f'{phase}/THER_Predictor/SentenceAccuracy': sentence_accuracies.cpu().item(), "training_step": iteration_count}, commit=False)
    for idx in range(accuracies.shape[-1]):
        wandb.log({f'{phase}/THER_Predictor/Accuracy_{idx}': accuracies[..., idx].cpu().item(), "training_step": iteration_count}, commit=False)
    
    if importanceSamplingWeights is not None:
        wandb.log({f'{phase}/THER_Predictor/ImportanceSamplingMean': importanceSamplingWeights.cpu().mean().item(), "training_step": iteration_count}, commit=False)
        wandb.log({f'{phase}/THER_Predictor/ImportanceSamplingStd': importanceSamplingWeights.cpu().std().item(), "training_step": iteration_count}, commit=False)
        wandb.log({f'{phase}/THER_Predictor/PER_Beta': PER_beta, "training_step": iteration_count}, commit=False)

    return {'loss':loss, 
            'loss_per_item':loss_per_item,
            'accuracy':accuracy
            }
