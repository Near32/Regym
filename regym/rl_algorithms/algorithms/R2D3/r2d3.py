from typing import Dict, Any, Optional, Callable

import scipy
from scipy import stats
import torch.nn as nn

from regym.rl_algorithms.algorithms.DQN import dqn_loss, ddqn_loss
from regym.rl_algorithms.algorithms.DQN import DQNAlgorithm
from regym.rl_algorithms.replay_buffers import ReplayStorage

sum_writer = None

class R2D3Algorithm(DQNAlgorithm):

    def __init__(self, kwargs: Dict[str, Any], model: nn.Module,
                 target_model: Optional[nn.Module] = None,
                 optimizer=None,
                 expert_demonstrations: ReplayStorage = None,
                 loss_fn: Callable = dqn_loss.compute_loss,
                 sum_writer=None):
        super().__init__(
            kwargs, model, target_model, optimizer, loss_fn, sum_writer)
        self.sequence_length = kwargs['sequence_length']  # Not doing anything with this so far.
        self.demo_ratio = kwargs['demo_ratio']  # Should be small (around: 1 / 256)

        self.expert_demonstrations = expert_demonstrations  # Putting the 3 in R2D3


    # NOTE: we are overriding this function from DQNAlgorithm
    def retrieve_values_from_storages(self, minibatch_size: int):
        '''
        We sample from both replay buffers (expert_demonstrations and agent
        collected experiences) according to property self.demo_ratio
        '''
        coin_flips = scipy.stats.bernoulli.rvs(p=self.demo_ratio, size=minibatch_size)
        num_demonstrations_samples = coin_flips.tolist().count(1)
        num_replay_buffer_samples = minibatch_size - num_demonstrations_samples
        # TODO: Sample num_samples_expert_demonstrations from self.expert_demonstrations
        #       and num_replay_buffer from self.storages
        raise NotImplementedError()

    # NOTE: we are overriding this function from DQNAlgorithm
    def update_replay_buffer_priorities(self):
        '''
        Separately updates priorities for both replay buffers
        (expert_demonstrations and agent collected experiences)
        '''
        # TODO: figure out a way of storing which indices we sampled both
        #       from self.expert_demonstrations and self.storages so that
        #       we can update these accordingly
        raise NotImplementedError()
