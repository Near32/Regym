from typing import Dict, List, Optional, Callable

from functools import partial 
import copy
import time 

import numpy as np
import torch
import ray

from regym.rl_algorithms.utils import is_leaf, copy_hdict, _concatenate_list_hdict, recursive_inplace_update, apply_on_hdict
from regym.thirdparty.Archi.Archi.model import Model as ArchiModel

import wandb 

from regym.rl_algorithms.algorithms.R2D2.r2d2_loss import archi_assign_fn
from regym.rl_algorithms.algorithms.R2D2.r2d2_loss import extract_rnn_states_from_time_indices
from regym.rl_algorithms.algorithms.R2D2.r2d2_loss import replace_rnn_states_at_time_indices
from regym.rl_algorithms.algorithms.R2D2.r2d2_loss import roll_sequences, batched_unrolled_inferences

use_BPTT = True

def compute_loss(states: torch.Tensor,
                 actions: torch.Tensor,
                 next_states: torch.Tensor,
                 rewards: torch.Tensor,
                 non_terminals: torch.Tensor,
                 model: torch.nn.Module,
                 target_model: torch.nn.Module,
                 gamma: float = 0.99,
                 weights_decay_lambda: float = 0.0,
                 weights_entropy_lambda: float = 0.0,
                 weights_entropy_reg_alpha: float = 0.0,
                 HER_target_clamping: bool = False,
                 summary_writer: object = None,
                 iteration_count: int = 0,
                 rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None,
                 next_rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None,
                 kwargs:Optional[Dict]=None) -> torch.Tensor:
    '''
    :param states: Dimension: batch_size x unroll_length x state_size: States visited by the agent.
    :param actions: Dimension: batch_size x unroll_length x action_size. Actions which the agent
                    took at every state in :param states: with the same index.
    :param next_states: Dimension: batch_size x unroll_length x state_size: Next sequence of unrolled states visited by the agent.
    :param non_terminals: Dimension: batch_size x unroll_length x 1: Non-terminal integers.
    :param rewards: Dimension: batch_size x unroll_length x 1. Environment rewards, or n-step returns if using n-step returns.
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
    :param next_rnn_states: Resulting 'hidden' and 'cell' states of the LSTM submodules after
                            feedforwarding :param states: in :param model:. See :param rnn_states:
                            for further details on type and shape.
    '''
    #torch.autograd.set_detect_anomaly(True)
    batch_size = states.shape[0]
    unroll_length = states.shape[1]
    map_keys=['qa', 'a', 'ent', 'legal_ent']

    start = time.time()
    assign_fn = None
    if isinstance(model, ArchiModel):
        assign_fn = archi_assign_fn

    if kwargs['burn_in']:
        burn_in_length = kwargs['sequence_replay_burn_in_length']
        training_length = kwargs['sequence_replay_unroll_length']-burn_in_length

        burn_in_states, training_states = torch.split(
            states, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )
        _training_rnn_states = extract_rnn_states_from_time_indices(
            rnn_states, 
            time_indices_start=kwargs['sequence_replay_burn_in_length'],
            time_indices_end=kwargs['sequence_replay_unroll_length'],
            preprocess_fn= None if use_BPTT else (lambda x:x.detach()), # not performing BPTT
        )
        _, training_rewards = torch.split(
            rewards, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )
        burn_in_non_terminals, training_non_terminals = torch.split(
            non_terminals, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )
        _, training_actions = torch.split(
            actions, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )

        # SEED RL does put a stop_gradient:
        # https://github.com/google-research/seed_rl/blob/5f07ba2a072c7a562070b5a0b3574b86cd72980f/agents/r2d2/learner.py#L368
        # No BPTT on the subsequent rnn_states:
        burned_in_predictions, \
        unrolled_predictions, \
        burned_in_rnn_states_inputs = batched_unrolled_inferences(
            unroll_length=burn_in_length,
            model=model, 
            states=states, #burn_in_states, 
            non_terminals=burn_in_non_terminals,
            rnn_states=rnn_states,
            grad_enabler=False,
            use_zero_initial_states=kwargs['sequence_replay_use_zero_initial_states'],
            extras=False,
            map_keys=map_keys,
        )

        #target_model.reset_noise()

        burned_in_target_predictions, \
        unrolled_target_predictions, \
        burned_in_rnn_states_target_inputs = batched_unrolled_inferences(
            unroll_length=burn_in_length,
            model=target_model, 
            states=states, #burn_in_states, 
            non_terminals=burn_in_non_terminals,
            rnn_states=rnn_states,
            grad_enabler=False,
            use_zero_initial_states=kwargs['sequence_replay_use_zero_initial_states'],
            extras=False,
            map_keys=map_keys,
        )

        # Replace the burned in rnn states in the training rnn states:
        training_rnn_states = replace_rnn_states_at_time_indices(
            rnn_states_batched=_training_rnn_states, 
            replacing_rnn_states_batched=burned_in_rnn_states_inputs, 
            time_indices_start=0, 
            time_indices_end=0,
            assign_fn=assign_fn,
        )

        training_target_rnn_states = replace_rnn_states_at_time_indices(
            rnn_states_batched=_training_rnn_states, 
            replacing_rnn_states_batched=burned_in_rnn_states_target_inputs, 
            time_indices_start=0, 
            time_indices_end=0,
            assign_fn=assign_fn,
        )
    else:
        training_length = unroll_length
        training_states = states 
        training_actions = actions 
        training_rewards = rewards
        training_non_terminals = non_terminals
        training_rnn_states = rnn_states
        training_target_rnn_states = rnn_states

    training_next_states = next_states

    # Unrolled predictions is using the stored RNN states.
    # burned_in_predictions is using the online RNN states computed in the function loop.
    training_burned_in_predictions, \
    training_unrolled_predictions, _ = batched_unrolled_inferences(
        unroll_length=training_length,
        model=model, 
        states=training_states, 
        non_terminals=training_non_terminals,
        rnn_states=training_rnn_states,
        grad_enabler=True,
        use_zero_initial_states=kwargs['sequence_replay_use_zero_initial_states'] if not(kwargs['burn_in']) else False,
        extras=not(kwargs['burn_in']) or not(kwargs['sequence_replay_use_online_states']),
        map_keys=map_keys,
    )

    #target_model.reset_noise()

    training_burned_in_target_predictions, \
    training_unrolled_target_predictions, _ = batched_unrolled_inferences(
        unroll_length=training_length,
        model=target_model, 
        states=training_states, 
        non_terminals=training_non_terminals,
        rnn_states=training_target_rnn_states,
        grad_enabler=False,
        use_zero_initial_states=kwargs['sequence_replay_use_zero_initial_states'],
        extras=not(kwargs['burn_in']) or not(kwargs['sequence_replay_use_online_states']),
        map_keys=map_keys,
    )

    if kwargs['burn_in'] or kwargs['sequence_replay_use_online_states']:
        training_predictions = training_burned_in_predictions
        training_target_predictions = training_burned_in_target_predictions
    else:
        training_predictions = training_unrolled_predictions
        training_target_predictions = training_unrolled_target_predictions
    
    qa_values_key = "qa"
    
    Q_Si_values = training_predictions[qa_values_key]
    # (batch_size, unroll_dim, ...)
    online_greedy_action = Q_Si_values.max(dim=-1, keepdim=True)[1]#.reshape(batch_size, training_length, Q_Si_values.shape[])
    # (batch_size, unroll_dim, ...)
    
    # Stable training: crucial: cf loss equation of Ape-X paper in section 3.1 Ape-X DQN:
    Q_Si_Ai_value = Q_Si_values.gather(
        dim=-1, 
        index=training_actions
    )
    # (batch_size, unroll_dim, /player_dim,/ 1)
    
    unscaled_targetQ_Si_A_values = inv_vfr(training_target_predictions[qa_values_key])
    # (batch_size, training_length, /player_dim,/ num_actions)
    
    # Double Q learning target:
    unscaled_targetQ_Si_onlineGreedyAction = unscaled_targetQ_Si_A_values.gather(
        dim=-1, 
        index=online_greedy_action
    )
    # (batch_size, training_length, /player_dim,/ 1)
    
    if weights_entropy_reg_alpha > 1.0e-12:
        # Adding entropy regularisation term for soft-DQN:
        online_target_entropy = training_target_predictions["legal_ent"]
        # Naive:
        #unscaled_targetQ_Si_onlineGreedyAction += weights_entropy_reg_alpha*online_target_entropy.unsqueeze(-1)
        # Legendre-Fenchel:
        unscaled_targetQ_Si_onlineGreedyAction = weights_entropy_reg_alpha*torch.log(
            torch.exp(Q_Si_values/weights_entropy_reg_alpha).sum(dim=-1)
        ).unsqueeze(-1)
    
    unscaled_Q_Si_Ai_value = inv_vfr(Q_Si_Ai_value)

    if len(training_rewards.shape) > 3:
        assert ("vdn" in kwargs and kwargs["vdn"])
        unscaled_bellman_target_Sipn_onlineGreedyAction = torch.zeros_like(training_rewards[:,:,0])
        for pidx in range(kwargs['vdn_nbr_players']):
            unscaled_bellman_target_Sipn_onlineGreedyAction += compute_n_step_bellman_target(
                training_rewards=training_rewards[:,:,pidx],
                training_non_terminals=training_non_terminals,
                unscaled_targetQ_Si_onlineGreedyAction=unscaled_targetQ_Si_onlineGreedyAction[:,:,pidx],
                gamma=gamma,
                kwargs=kwargs
            )
        unscaled_Q_Si_Ai_value = unscaled_Q_Si_Ai_value.sum(dim=2)
    else:
        unscaled_bellman_target_Sipn_onlineGreedyAction = compute_n_step_bellman_target(
            training_rewards=training_rewards,
            training_non_terminals=training_non_terminals,
            unscaled_targetQ_Si_onlineGreedyAction=unscaled_targetQ_Si_onlineGreedyAction,
            gamma=gamma,
            kwargs=kwargs
        )
    # (batch_size, training_length, ...)
    
    #unscaled_bellman_target_Sipn_onlineGreedyAction = unscaled_bellman_target_Sipn_onlineGreedyAction.sum(dim=2)
    assert len(unscaled_bellman_target_Sipn_onlineGreedyAction.shape) == 3  
    
    Q_Si_Ai_value = vfr(unscaled_Q_Si_Ai_value)    
    scaled_bellman_target_Sipn_onlineGreedyAction = vfr(unscaled_bellman_target_Sipn_onlineGreedyAction)
    
    # Compute loss:
    # Abs:
    if HER_target_clamping:
        # clip the unscaled target to [-50,0]
        unscaled_bellman_target_Sipn_onlineGreedyAction = torch.clamp(
            unscaled_bellman_target_Sipn_onlineGreedyAction, 
            -1. / (1 - gamma),
            0.0
        )
    td_error = torch.abs(unscaled_bellman_target_Sipn_onlineGreedyAction.detach() - unscaled_Q_Si_Ai_value)
    scaled_td_error = torch.abs(scaled_bellman_target_Sipn_onlineGreedyAction.detach() - Q_Si_Ai_value)
    assert list(td_error.shape) == [batch_size, training_length, 1]

    # Hanabi_SAD repo does not use the scaled values:
    loss_per_item = td_error
    diff_squared = td_error.pow(2.0)
    # SEED RL repo uses the scaled td error for priorities:
    #loss_per_item = scaled_td_error
    #diff_squared = scaled_td_error.pow(2.0)

    if use_PER and importanceSamplingWeights is not None:
      diff_squared = importanceSamplingWeights.reshape((batch_size, 1, 1)) * diff_squared
      assert list(diff_squared.shape) == [batch_size, training_length, 1]

    # not sure where this masking strategy comes from, maybe forget about it
    # since the distribution of qa values is more expressive without it...
    # the initial rational for it was to allow training on the last value only if terminal...
    assert kwargs["r2d2_loss_masking"], "r2d2_loss_masking must be True for this test."
    if kwargs["r2d2_loss_masking"]:
        mask = torch.ones_like(diff_squared)
        # Combined:
        assert kwargs['r2d2_loss_masking_n_step_regularisation'], "debugging in progress"
        if kwargs['r2d2_loss_masking_n_step_regularisation']:
            mask[:, -kwargs["n_step"]:, ...] = (1-training_non_terminals[:,-kwargs['n_step']:,...])

        loss_per_item = loss_per_item*mask
        loss = 0.5*torch.mean(diff_squared*mask)-weights_entropy_lambda*training_predictions['legal_ent'].mean()
    else:
        mask = torch.ones_like(diff_squared)
        loss_per_item = loss_per_item*mask
        loss = 0.5*torch.mean(diff_squared*mask)-weights_entropy_lambda*training_predictions['legal_ent'].mean()
    
    end = time.time()

    #wandb_data = copy.deepcopy(wandb.run.history._data)
    #wandb.run.history._data = {}
    wandb.log({'Training/TimeComplexity':  end-start, "training_step":iteration_count}, commit=False)
    
    if kwargs.get("logging", False):
        columns = ["stimulus_(t)", "stimulus_(t-1)"]
        #columns += [f"a_(t-{v})" for v in range(4)]
        sample_table = wandb.Table(columns=columns) 
    
        for bidx in range(batch_size//4):
            nbr_states = states.shape[1]
            nbr_frames = states[bidx].shape[1]//4
            stimulus_t = [
                next_states[bidx,s].reshape(nbr_frames,4,56,56)[-1:,:3] 
                for s in range(nbr_states)
            ]#.numpy()[:,:3]*255
            stimulus_t = torch.cat(stimulus_t, dim=0).cpu().numpy()*255
            stimulus_t = stimulus_t.astype(np.uint8)
            stimulus_t = wandb.Video(stimulus_t, fps=2, format="mp4")
            #stimulus_tm = s[bidx].cpu().reshape(nbr_frames,4,56,56).numpy()[:,:3]*255
            stimulus_tm = [
                states[bidx,s].reshape(nbr_frames,4,56,56)[-1:,:3] 
                for s in range(nbr_states)
            ]#.numpy()[:,:3]*255
            stimulus_tm = torch.cat(stimulus_tm, dim=0).cpu().numpy()*255
            stimulus_tm = stimulus_tm.astype(np.uint8)
            stimulus_tm = wandb.Video(stimulus_tm, fps=2, format="mp4")
            
            sample_table.add_data(*[
                #*gt_word_sentence,
                stimulus_t,
                stimulus_tm,
                #*previous_action_int
                ]
            )

        wandb.log({f"Training/R2D2StimuliTable":sample_table}, commit=False)

    wandb.log({'Training/MeanTrainingReward':  training_rewards.cpu().mean().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MinTrainingReward':  training_rewards.cpu().min().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MaxTrainingReward':  training_rewards.cpu().max().item(), "training_step":iteration_count}, commit=False)

    wandb.log({'Training/MeanTargetQsi':  unscaled_targetQ_Si_A_values.cpu().mean().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MinTargetQsi':  unscaled_targetQ_Si_A_values.cpu().min().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MaxTargetQsi':  unscaled_targetQ_Si_A_values.cpu().max().item(), "training_step":iteration_count}, commit=False)
    
    wandb.log({'Training/MeanBellmanTarget':  unscaled_bellman_target_Sipn_onlineGreedyAction.cpu().mean().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MinBellmanTarget':  unscaled_bellman_target_Sipn_onlineGreedyAction.cpu().min().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MaxBellmanTarget':  unscaled_bellman_target_Sipn_onlineGreedyAction.cpu().max().item(), "training_step":iteration_count}, commit=False)
    
    wandb.log({'Training/MeanQAValues':  training_predictions['qa'].cpu().mean().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MinQAValues':  training_predictions['qa'].cpu().min().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/MaxQAValues':  training_predictions['qa'].cpu().max().item(), "training_step":iteration_count}, commit=False)
    
    wandb.log({'Training/StdQAValues':  training_predictions['qa'].cpu().std().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/QAValueLoss':  loss.cpu().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/LegalEntropyVal':  training_predictions['legal_ent'].mean().cpu().item(), "training_step":iteration_count}, commit=False)
    wandb.log({'Training/EntropyVal':  training_predictions['ent'].mean().cpu().item(), "training_step":iteration_count}, commit=False)
    
    wandb.log({}, commit=True)
    #wandb.run.history._data = wandb_data

    return loss, loss_per_item


