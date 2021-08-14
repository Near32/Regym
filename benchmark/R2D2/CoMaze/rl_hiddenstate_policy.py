from typing import List, Dict, Optional

import torch

from regym.rl_algorithms.agents.agent import Agent
from comaze_gym.metrics import MessagePolicy

from regym.rl_algorithms.utils import copy_hdict

def extract_subtrees(
    in_dict: Dict,
    node_id: str):
    '''
    Extracts a copy of subtree whose root is named :param node_id: from :param in_dict:.
    '''
    queue = [in_dict]
    pointer = None

    subtrees = []

    while len(queue):
        pointer = queue.pop(0)
        if not isinstance(pointer, dict): continue
        for k in pointer.keys():
            if node_id==k:
                subtrees.append(
                    copy_hdict(pointer[k])
                )
            else:
                queue.append(pointer[k])

    return subtrees

class RLHiddenStatePolicy(MessagePolicy):
    def __init__(
        self, 
        agent:Agent,
        augmented:bool=False):
        """
        
        """
        super(RLHiddenStatePolicy, self).__init__(
            model=agent,
        )
        self.player_idx = 0
        self.augmented = augmented
        
        self.secretgoalStr2id = {"RED":0, "YELLOW":1, "BLUE":2, "GREEN":3}

    
    def get_hiddens(self, info=None, from_pred=None):
        if from_pred is None:
            rnn_states = self.model.get_rnn_states()
        else:
            rnn_states = from_pred['next_rnn_states']
        # Extract 'hidden''s list:
        hiddens = extract_subtrees(in_dict=rnn_states, node_id='hidden')
        # List[List[Tensor]]
        
        vdn = self.model.kwargs.get('vdn', False)
        vdn_nbr_players = self.model.kwargs.get('vdn_nbr_players', 2)
        
        nbr_rnn_modules = len(hiddens[0])
        batch_size = hiddens[0][0].shape[0]

        mult = 0
        if vdn and batch_size!=1: 
            batch_size = batch_size // vdn_nbr_players
            mult = self.player_idx
        
        hiddens = torch.stack(
            [
                torch.cat(
                    [hiddens[0][part_id][mult*batch_size+actor_id].reshape(-1) for part_id in range(nbr_rnn_modules)],
                    dim=0,
                )
                for actor_id in range(batch_size)
            ],
            dim=0,
        )
        # batch_size x nbr_parts*hidden_dims
        if self.augmented:
            extras = []
            for actor_id in range(batch_size):
                if info is not None:
                    abs_repr = info[actor_id]['abstract_repr']
                    
                    reached_goals_str = abs_repr['reached_goals']
                    rg_hs = torch.zeros((4*3))
                    startidx = 0
                    for goal_str in reached_goals_str:
                        rg_hs[startidx+self.secretgoalStr2id[goal_str]] = 1.0
                        startidx += 4
                    
                    player_id = abs_repr['player_id']
                    secretGoalRule = abs_repr['secretGoalRule'][player_id]
                    sgr_hs = torch.zeros((4*2))
                    startidx = 0
                    sgr_hs[ startidx+self.secretgoalStr2id[secretGoalRule.earlierGoal.color] ] = 1.0
                    startidx += 4
                    sgr_hs[ startidx+self.secretgoalStr2id[secretGoalRule.laterGoal.color] ] = 1.0

                    extra = torch.cat([rg_hs, sgr_hs], dim=0).reshape((1,-1))
                else:
                    extra = torch.zeros((1,4*3+4*2))
                extras.append(extra)
            extras = torch.cat(extras, dim=0).to(hiddens.device)
            hiddens = torch.cat([hiddens, extras], dim=1)
            # batch_size x (nbr_parts*hidden_dims + extra_dim)

        return hiddens 

    def get_hidden_state_dim(self):
        hiddens = self.get_hiddens()
        return hiddens.shape[-1]

    def clone(self, training=False):
        return RLHiddenStatePolicy(
            agent=self.model.clone(training=training), 
            augmented=self.augmented,
        )

    def reset(self, batch_size:int, training:Optional[bool]=False):
        self.model.set_nbr_actor(batch_size, vdn=False, training=training)

    def save_inner_state(self):
        self.saved_inner_state = self.model.get_rnn_states()

    def restore_inner_state(self):
        self.model.set_rnn_states(self.saved_inner_state)

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
        
        :return log_m:
            torch.Tensor of logits over messages 
            (as a Discrete OpenAI's action space).

            Here, depending on :attr combined_action_space:,
            we either marginalized over possible actions or not.
        """

        #log_p_a = self.model.query_action(**x)
        pred_dict = self.model.query_action(**x)
        # batch_size x action_space_dim

        hiddens = self.get_hiddens(info=x.get('infos', None), from_pred=pred_dict)
        # batch_size x nbr_parts*hidden_dims + extra_dim if self.augmented

        return hiddens
