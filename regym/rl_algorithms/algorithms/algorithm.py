import torch
import copy


class Algorithm(object):
    def get_models(self):
        raise NotImplementedError

    def get_epsilon(self, nbr_steps, strategy='exponential'):
        raise NotImplementedError

    def get_nbr_actor(self):
        raise NotImplementedError

    def get_update_count(self):
        raise NotImplementedError

    @staticmethod
    def _extract_rnn_states_from_batch_indices(rnn_states_batched: dict, batch_indices: torch.Tensor, use_cuda: bool=False):
        if rnn_states_batched is None:  return None 

        rnn_states = {k: {} for k in rnn_states_batched}
        for recurrent_submodule_name in rnn_states_batched:
            if 'hidden' in rnn_states_batched[recurrent_submodule_name]:
                rnn_states[recurrent_submodule_name] = {'hidden':[], 'cell':[]}
                for idx in range(len(rnn_states_batched[recurrent_submodule_name]['hidden'])):
                    hidden = rnn_states_batched[recurrent_submodule_name]['hidden'][idx][batch_indices,...].cuda() if use_cuda else rnn_states_batched[recurrent_submodule_name]['hidden'][idx][batch_indices,...]
                    rnn_states[recurrent_submodule_name]['hidden'].append(hidden)
                    cell = rnn_states_batched[recurrent_submodule_name]['cell'][idx][batch_indices,...].cuda() if use_cuda else rnn_states_batched[recurrent_submodule_name]['cell'][idx][batch_indices,...]
                    rnn_states[recurrent_submodule_name]['cell'].append(cell)
            else:
                rnn_states[recurrent_submodule_name] = Algorithm._extract_rnn_states_from_batch_indices(rnn_states_batched=rnn_states_batched[recurrent_submodule_name], batch_indices=batch_indices, use_cuda=use_cuda)
        return rnn_states

    @staticmethod
    def _concatenate_hdict(hd1: dict, hds: list, map_keys: list, dim:int=0):
        if not(isinstance(hd1, dict)):
            return Algorithm._concatenate_hdict(hd1=hds.pop(0), hds=hds, map_keys=map_keys, dim=dim)
        
        out_hd = copy.deepcopy(hd1)

        for key in hd1:
            for map_key in map_keys:
                if map_key in hd1[key]:
                    for idx in range(len(hd1[key][map_key])):
                        out_hd[key][map_key][idx] = torch.cat([hd1[key][map_key][idx], 
                                                            *[hd[key][map_key][idx] for hd in hds]
                                                            ], dim=dim)
                else:
                    out_hd[key] = Algorithm._concatenate_hdict(hd1=hd1[key], hds=[hd[key] for hd in hds], map_keys=map_keys, dim=dim)
        return out_hd

    def clone(self):
        raise NotImplementedError