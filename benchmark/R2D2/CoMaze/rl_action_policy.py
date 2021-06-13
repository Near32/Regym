from typing import List, Dict

import torch

from regym.rl_algorithms.agents.agent import Agent
from comaze_gym.metrics import ActionPolicy

class RLActionPolicy(ActionPolicy):
    def __init__(
        self, 
        agent:Agent,
        combined_action_space:bool = False):
        """
        
        :param combined_action_space:
            If True, then the message and actions performed
            by the current agent are treated as belonging to
            the same OpenAI's Discrete action space of size 
            n= #messages * #actions.
            Else, n = # actions : directional actions.
        """
        super(RLActionPolicy, self).__init__(
            model=agent
        )
        self.combined_action_space = combined_action_space
    
    def clone(self, training=False):
        return RLActionPolicy(
            agent=self.model.clone(training=training), 
            combined_action_space=self.combined_action_space
        )

    def reset(self, batch_size:int):
        self.model.set_nbr_actor(batch_size)

    def get_nbr_actor(self):
        return self.model.get_nbr_actor()

    def forward(self, x:object):
        """
        :param x:
            Object representing the observation of the current agent.
            e.g.: the object can be a kwargs argument containing
            expected argument to the model.

            Here, x:Dict containing the keys:
            -'state': torch.Tensor containing the environment state.
            -'infos': Dict containing the entry 'abstract_repr' that is
                actually used by the :param model:RuleBasedAgentWrapper.
        
        :return log_a:
            torch.Tensor of logits over actions 
            (as a Discrete OpenAI's action space).

            Here, depending on :attr combined_action_space:,
            we either marginalized over possible messages or not.
        """

        log_p_a = self.model.take_action(**x)
        # batch_size x action_space_dim

        batch_size = log_p_a.shape[0]
        action_space_dim = log_p_a.shape[-1]

        if self.combined_action_space:
            return log_p_a

        # Otherwise, we sum over the messages dimension (excluding the NOOP action):
        self.vocab_size = (action_space_dim-1)//5
        # There are 5 possible directional actions:
        log_p_a = log_p_a[...,:-1].reshape((batch_size, 5, self.vocab_size)).sum(dim=-1).log_softmax(dim=1)
        # batch_size x 5

        return log_p_a
