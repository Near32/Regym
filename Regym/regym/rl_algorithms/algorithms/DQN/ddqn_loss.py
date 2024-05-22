from typing import Dict, List, Optional
import time
import torch


def compute_loss(states: torch.Tensor,
                 actions: torch.Tensor,
                 next_states: torch.Tensor,
                 rewards: torch.Tensor,
                 non_terminals: torch.Tensor,
                 goals: torch.Tensor,
                 model: torch.nn.Module,
                 target_model: torch.nn.Module,
                 gamma: torch.Tensor, #float = 0.99,
                 weights_decay_lambda: float = 1.0,
                 weights_entropy_lambda: float = 0.1,
                 use_PER: bool = False,
                 PER_beta: float = 1.0,
                 importanceSamplingWeights: torch.Tensor = None,
                 HER_target_clamping: bool = False,
                 summary_writer: object = None,
                 iteration_count: int = 0,
                 rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None,
                 next_rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None,
                 kwargs:Optional[Dict]=None) -> torch.Tensor:
    '''
    :param states: Dimension: batch_size x state_size: States visited by the agent.
    :param actions: Dimension: batch_size x action_size. Actions which the agent
                    took at every state in :param states: with the same index.
    :param next_states: Dimension: batch_size x state_size: Next states visited by the agent.
                        NOTE: if n-step returns are used, then it is the n-th subsequent states.
                        It may breach the episode barrier... 
    :param non_terminals: Dimension: batch_size x 1: Non-terminal integers.
    :param rewards: Dimension: batch_size x 1. Environment rewards, or n-step returns if using n-step returns.
    :param goals: Dimension: batch_size x goal shape: Goal of the agent.
    :param model: torch.nn.Module used to compute the loss.
    :param target_model: torch.nn.Module used to compute the loss.
    :param gamma: Previously a float for the discount factor, but now a torch.Tensor matching the :param rewards: in a shape batch_size x 1.
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
    start = time.time()

    batch_size = states.shape[0]

    prediction = model(states, action=actions, rnn_states=rnn_states, goal=goals)

    state_action_values = prediction["qa"]
    
    '''
    # Sample actions from the replay buffer:
    state_action_values_g = state_action_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
    
    '''
    # Sample actions from the current model outputs: which is actually the training batch action 'actions' (cf bodies.py CategoricalQNet)
    current_actions = prediction["a"]
    if len(current_actions.shape) != 2:
        current_actions = current_actions.unsqueeze(1)
    state_action_values_g = state_action_values.gather(dim=1, index=current_actions).squeeze(1)
    

    ############################
    with torch.no_grad():
        next_rnn_states = None
        next_target_rnn_states = None
        if rnn_states is not None:
            next_rnn_states = prediction['next_rnn_states']

            target_prediction = target_model(states, action=actions, rnn_states=rnn_states, goal=goals)
            next_target_rnn_states = target_prediction['next_rnn_states']

        target_model.reset_noise()

        next_prediction = model(next_states, rnn_states=next_rnn_states, goal=goals)
        next_target_prediction = target_model(next_states, rnn_states=next_target_rnn_states, goal=goals)

        Q_nextS_A_values = next_prediction['qa']
        argmaxA_Q_nextS_A_values = Q_nextS_A_values.max(dim=1)[1].unsqueeze(1)

        targetQ_nextS_A_values = next_target_prediction['qa']
        targetQ_nextS_argmaxA_Q_value = targetQ_nextS_A_values.gather(1, argmaxA_Q_nextS_A_values).reshape(batch_size, -1)

        # Compute the expected Q values:
        import ipdb; ipdb.set_trace()
        # TODO: check dimension changes due to gamma :
        expected_state_action_values = rewards + (gamma**kwargs['n_step']) * targetQ_nextS_argmaxA_Q_value * non_terminals

        if HER_target_clamping:
            # clip the target to [-50,0]
            import ipdb; ipdb.set_trace()
            # TODO: check dimension changes due to gamma:
            expected_state_action_values = torch.clamp(expected_state_action_values, -1. / (1 - gamma), 0)
    ############################

    # Compute loss:
    #diff_squared = torch.abs(expected_state_action_values.detach() - state_action_values_g)
    diff_squared = (expected_state_action_values.detach() - state_action_values_g.reshape(expected_state_action_values.shape)).pow(2.0)
    loss_per_item = diff_squared

    if use_PER:
      diff_squared = importanceSamplingWeights * diff_squared

    loss = 0.5*torch.mean(diff_squared)-weights_entropy_lambda*prediction['ent'].mean()

    end = time.time()

    if summary_writer is not None:
        summary_writer.add_scalar('Training/TimeComplexity', end-start, iteration_count)
        
        summary_writer.add_scalar('Training/MeanTrainingNStepReturn', rewards.cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/MinTrainingNStepReturn', rewards.cpu().min().item(), iteration_count)
        summary_writer.add_scalar('Training/MaxTrainingNStepReturn', rewards.cpu().max().item(), iteration_count)
        
        summary_writer.add_scalar('Training/MeanTargetQSipn_ArgmaxAOnlineQSipn_A', targetQ_nextS_argmaxA_Q_value.cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/MinTargetQSipn_ArgmaxAOnlineQSipn_A', targetQ_nextS_argmaxA_Q_value.cpu().min().item(), iteration_count)
        summary_writer.add_scalar('Training/MaxTargetQSipn_ArgmaxAOnlineQSipn_A', targetQ_nextS_argmaxA_Q_value.cpu().max().item(), iteration_count)
        
        summary_writer.add_scalar('Training/MeanTargetQsi', targetQ_nextS_A_values.cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/MinTargetQsi', targetQ_nextS_A_values.cpu().min().item(), iteration_count)
        summary_writer.add_scalar('Training/MaxTargetQsi', targetQ_nextS_A_values.cpu().max().item(), iteration_count)
        
        summary_writer.add_scalar('Training/MeanBellmanTarget', expected_state_action_values.cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/MinBellmanTarget', expected_state_action_values.cpu().min().item(), iteration_count)
        summary_writer.add_scalar('Training/MaxBellmanTarget', expected_state_action_values.cpu().max().item(), iteration_count)
        
        summary_writer.add_scalar('Training/MeanQAValues', prediction['qa'].cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/MinQAValues', prediction['qa'].cpu().min().item(), iteration_count)
        summary_writer.add_scalar('Training/MaxQAValues', prediction['qa'].cpu().max().item(), iteration_count)
        
        summary_writer.add_scalar('Training/StdQAValues', prediction['qa'].cpu().std().item(), iteration_count)
        summary_writer.add_scalar('Training/QAValueLoss', loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/EntropyVal', prediction['ent'].mean().cpu().item(), iteration_count)
        #summary_writer.add_scalar('Training/TotalLoss', loss.cpu().item(), iteration_count)
        if use_PER:
            summary_writer.add_scalar('Training/ImportanceSamplingMean', importanceSamplingWeights.cpu().mean().item(), iteration_count)
            summary_writer.add_scalar('Training/ImportanceSamplingStd', importanceSamplingWeights.cpu().std().item(), iteration_count)
            summary_writer.add_scalar('Training/PER_Beta', PER_beta, iteration_count)

    return loss, loss_per_item
