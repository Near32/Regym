from typing import List, Dict, Optional

import torch
import torch.nn as nn

from regym.rl_algorithms.agents.agent import Agent
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

class RLHiddenStatePolicy(nn.Module):
    def __init__(
        self, 
        agent:Agent,
        node_id_to_extract:Optional[str]="hidden",
        ):
        """
        
        """
        super(RLHiddenStatePolicy, self).__init__()
        self.model = agent
        self.node_id_to_extract = node_id_to_extract.split(",")
        
        # TODO remove or update the following as it does not matter
        # since we are only ever using one such actor...
        self.player_idx = 0
    
    def get_hiddens(self, info=None, from_pred=None):
        if from_pred is None:
            rnn_states = self.model.get_rnn_states()
        else:
            rnn_states = from_pred['next_rnn_states']
        
                # Extract 'hidden''s list:
        
        nodes = []
        for node_id in self.node_id_to_extract:
            node = extract_subtrees(in_dict=rnn_states, node_id=node_id)
            # List[List[Tensor]]
            
            # TODO: provide an attention scheme to breakdown the series of vectors in the case of
            # non-fixed size memory into a single predictable-size memory:
            if len(node[0][0].shape) > 2:
                for idx1 in range(len(node)):
                    for idx2 in range(len(node[0])):
                        node[idx1][idx2] = node[idx1][idx2][...,-3:,:]
                        if node[idx1][idx2].shape[-2] < 3:
                            hshape = list(node[idx1][idx2].shape)
                            hshape[-2] = 3-hshape[-2]
                            padding = torch.zeros(*hshape).to(node[idx1][idx2].device)
                            node[idx1][idx2] = torch.cat([padding, node[idx1][idx2]], dim=-2)

            for nidx in range(len(node)):
                nodes.append(node[nidx])

        vdn = self.model.kwargs.get('vdn', False)
        vdn_nbr_players = self.model.kwargs.get('vdn_nbr_players', 2)
        
        nbr_nodes = len(nodes)
        nbr_rnn_modules = len(nodes[0])
        batch_size = nodes[0][0].shape[0]

        mult = 0
        if vdn and batch_size!=1: 
            batch_size = batch_size // vdn_nbr_players
            mult = self.player_idx
        
        tnodes = []
        for hiddens in nodes:
            hiddens = torch.stack(
                [
                    torch.cat(
                        [hiddens[part_id][mult*batch_size+actor_id].reshape(-1) for part_id in range(nbr_rnn_modules)],
                        dim=0,
                    )
                    for actor_id in range(batch_size)
                ],
                dim=0,
            )
            # batch_size x (nbr_parts*hidden_dims)
            tnodes.append(hiddens)
        nodes = torch.cat(tnodes, dim=-1)
        # batch_size x (nbr_parts*hidden_dims)*nbr_nodes

        return nodes 

    def get_hidden_state_dim(self):
        hiddens = self.get_hiddens()
        return hiddens.shape[-1]

    def clone(self, training=False):
        return RLHiddenStatePolicy(
            agent=self.model.clone(training=training), 
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
        
        :return hiddens:
            torch.Tensor concatenations of hidden states.
        """

        #log_p_a = self.model.query_action(**x)
        pred_dict = self.model.query_action(**x)
        # batch_size x action_space_dim

        hiddens = self.get_hiddens(info=x.get('infos', None), from_pred=pred_dict)
        # batch_size x nbr_parts*hidden_dims*nbr_nodes + extra_dim if self.augmented

        return hiddens
