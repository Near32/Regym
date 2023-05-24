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

def compute_loss(
    samples: Dict[str, torch.Tensor],
    models: Dict[str, torch.nn.Module],
    summary_writer: object = None,
    iteration_count: int = 0,
    **kwargs:Optional[Dict[str, object]],#=None,
) -> torch.Tensor:
    '''
    Computes the loss of an actor critic model using the
    loss function from equation (9) in the paper:
    Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347

    :param samples: Dictionnary of many different possible elements:
        :param states: Dimension: batch_size x unroll_length x state_size: States visited by the agent.
        :param actions: Dimension: batch_size x unroll_length x action_size. Actions which the agent
                    took at every state in :param states: with the same index.
        :param next_states: Dimension: batch_size x unroll_length x state_size: Next sequence of unrolled states visited by the agent.
        :param non_terminals: Dimension: batch_size x unroll_length x 1: Non-terminal integers.
        :param rewards: Dimension: batch_size x unroll_length x 1. Environment rewards, or n-step returns if using n-step returns.
        :param log_probs_old: Dimension: batch_size. Log probability of taking the action with the same index in :param actions:. Used to compute the policy probability ratio. Refer to original paper equation (6)
        :param returns: Dimension: batch_size x 1. Empirical returns obtained via calculating the discounted return from the environment's rewards.
        :param advantages: Dimension: batch_size. Estimated advantage function for every state and action in :param states: and :param actions: (respectively) with the same index.
        :param std_advantages: Dimension: batch_size. Estimated standardized advantage function for every state and action in :param states: and :param actions: (respectively) with the same index.
        :param rnn_states: The :param model: can be made up of different submodules. Some of these submodules will feature an LSTM architecture. This parameter is a dictionary which maps recurrent submodule names to a dictionary which contains 2 lists of tensors, each list corresponding to the 'hidden' and 'cell' states of the LSTM submodules. These tensors are used by the :param model: when calculating the policy probability ratio.
        :param next_rnn_states: Resulting 'hidden' and 'cell' states of the LSTM submodules after feedforwarding :param states: in :param model:. See :param rnn_states: for further details on type and shape.
    :param models: Dictionnary of all the necessary models, e.g. training model and target model : torch.nn.Module used to compute the loss.
    :param kwargs: Dictionnary of different hyperparameters such as :
        :param gamma: float discount factor.
        :param weights_decay_lambda: Coefficient to be used for the weight decay loss.
        :param use_std_adv: bool deciding whether to use a standardized advantage or not.
        :param ratio_clip: Epsilon value used to clip the policy ratio's value. This parameter acts as the radius of the Trust Region. Refer to original paper equation (7).
        :param entropy_weight: Coefficient to be used for the entropy bonus for the loss function. Refer to original paper eq (9).
        :param value_weight: Coefficient to be used for the value loss for the loss function. Refer to original paper eq (9).
    '''
    #torch.autograd.set_detect_anomaly(True)
    states = samples['states']
    actions = samples['actions']
    non_terminals = samples['non_terminals']
    rnn_states = samples['rnn_states']
    returns = samples['returns']
    advantages = samples['advantages']
    std_advantages = samples['std_advantages']
    log_probs_old = samples['log_probs_old']

    model = models['model']

    batch_size = states.shape[0]
    unroll_length = states.shape[1]
    map_keys=['v','log_pi_a', 'a', 'ent',]
    # TODO: if using 'legal_ent', then need to make it
    # an output of the relevant RLHead too.
    
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
        _, training_returns = torch.split(
            returns, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )
        _, training_advantages = torch.split(
            advantages, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )
        _, training_std_advantages = torch.split(
            std_advantages, 
            split_size_or_sections=[burn_in_length, training_length],
            dim=1
        )
        
        _, training_log_probs_old = torch.split(
            log_probs_old, 
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

        # Replace the burned in rnn states in the training rnn states:
        training_rnn_states = replace_rnn_states_at_time_indices(
            rnn_states_batched=_training_rnn_states, 
            replacing_rnn_states_batched=burned_in_rnn_states_inputs, 
            time_indices_start=0, 
            time_indices_end=0,
            assign_fn=assign_fn,
        )

    else:
        training_length = unroll_length
        training_states = states 
        training_actions = actions 
        training_non_terminals = non_terminals
        training_rnn_states = rnn_states
        training_returns = returns
        training_advantages = advantages
        training_std_advantages = std_advantages
        training_log_probs_old = log_probs_old

    # Unrolled predictions is using the stored RNN states.
    # burned_in_predictions is using the online RNN states computed in the function loop.
    training_burned_in_predictions, \
    training_unrolled_predictions, _ = batched_unrolled_inferences(
        unroll_length=training_length,
        model=model, 
        states=training_states, 
        ##############################################################
        # WARNING : in R2D2, the evaluation is relying on the current 
        # estimate of the best action. 
        # but in PPO, we use the offline estime:
        actions=training_actions,
        ##############################################################
        non_terminals=training_non_terminals,
        rnn_states=training_rnn_states,
        grad_enabler=True,
        use_zero_initial_states=kwargs['sequence_replay_use_zero_initial_states'] if not(kwargs['burn_in']) else False,
        extras=not(kwargs['burn_in']) or not(kwargs['sequence_replay_use_online_states']),
        map_keys=map_keys,
    )

    if kwargs['burn_in'] or kwargs['sequence_replay_use_online_states']:
        training_predictions = training_burned_in_predictions
    else:
        training_predictions = training_unrolled_predictions
    
    #prediction = model(obs=states, action=actions, rnn_states=rnn_states)
    
    ratio = torch.exp((training_predictions['log_pi_a'] - training_log_probs_old.detach()))
    
    if kwargs['standardized_adv']:
      #adv = training_std_advantages
      # Standardize on minibatch:
      def standardize(x):
          stable_eps = 1.0e-8
          std_x = (x-x.mean())/(x.std()+stable_eps)
          return std_x
      adv = standardize(training_advantages)
    else:
      adv = training_advantages.detach()

    adv = adv.reshape((batch_size, training_length))
    obj = ratio * adv.detach()
    obj_clipped = torch.clamp(ratio,
                              1.0 - kwargs['ppo_ratio_clip'],
                              1.0 + kwargs['ppo_ratio_clip']) * adv.detach()
    
    policy_val = policy_loss = -torch.min(obj, obj_clipped) #.mean()
    entropy_val = training_predictions['ent'] #.mean()

    #policy_loss = policy_val - kwargs['entropy_weight'] * entropy_val 
    # L^{clip} and L^{S} from original paper
    #policy_loss = -torch.min(obj, obj_clipped).mean() - entropy_weight * prediction['ent'].mean() # L^{clip} and L^{S} from original paper
    
    value_loss = kwargs['value_weight'] * torch.nn.functional.mse_loss(
        input=training_predictions['v'], 
        target=training_returns,
        reduction='none',
    ).reshape((batch_size, training_length))

    # TODO: Testing in progress : trying mean then addition to check if it affects anything:
    #total_loss = policy_loss + value_loss
    total_loss = policy_loss.mean(-1) + value_loss.mean(-1)
    total_loss = policy_loss.mean(-1) + value_loss.mean(-1) - kwargs['entropy_weight']*entropy_val.mean(-1)
    # Mean over unroll_length :
    #total_loss = total_loss.mean(-1)

    wandb.log({'Training/RatioMean': ratio.mean().cpu().item(), "training_step": iteration_count}, commit=False)
    #summary_writer.add_histogram('Training/Ratio', ratio.cpu(), iteration_count)
    wandb.log({'Training/AdvantageMean': training_advantages.mean().cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/AdvantageStd': training_advantages.std().cpu().item(), "training_step": iteration_count}, commit=False)
    #summary_writer.add_histogram('Training/Advantage', advantages.cpu(), iteration_count)
    wandb.log({'Training/MeanVValues': training_predictions['v'].cpu().mean().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/MeanReturns': returns.cpu().mean().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/StdVValues': training_predictions['v'].cpu().std().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/StdReturns': training_returns.cpu().std().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/ValueLoss': value_loss.mean().cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/PolicyVal': policy_val.mean().cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/EntropyVal': entropy_val.mean().cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/PolicyLoss': policy_loss.mean().cpu().item(), "training_step": iteration_count}, commit=False)
    wandb.log({'Training/TotalLoss': total_loss.mean().cpu().item(), "training_step": iteration_count}, commit=False)
        
    '''
    if weights_entropy_reg_alpha > 1.0e-12:
        # Adding entropy regularisation term for soft-DQN:
        online_target_entropy = training_target_predictions["legal_ent"]
        # Naive:
        #unscaled_targetQ_Si_onlineGreedyAction += weights_entropy_reg_alpha*online_target_entropy.unsqueeze(-1)
        # Legendre-Fenchel:
        unscaled_targetQ_Si_onlineGreedyAction = weights_entropy_reg_alpha*torch.log(
            torch.exp(Q_Si_values/weights_entropy_reg_alpha).sum(dim=-1)
        ).unsqueeze(-1)
    '''

    #TODO
    '''
    if HER_target_clamping:
        # clip the unscaled target to [-50,0]
        unscaled_bellman_target_Sipn_onlineGreedyAction = torch.clamp(
            unscaled_bellman_target_Sipn_onlineGreedyAction, 
            -1. / (1 - gamma),
            0.0
        )
    '''

    '''
    if use_PER and importanceSamplingWeights is not None:
      diff_squared = importanceSamplingWeights.reshape((batch_size, 1, 1)) * diff_squared
      assert list(diff_squared.shape) == [batch_size, training_length, 1]
    '''

    if kwargs.get("logging", False):
        raise NotImplementedError
        # TODO : deal with next_states not being defined...
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

        wandb.log({f"Training/RecurrentPPOStimuliTable":sample_table}, commit=False)

    wandb.log({}, commit=True)
    #wandb.run.history._data = wandb_data

    return total_loss.mean(), total_loss


