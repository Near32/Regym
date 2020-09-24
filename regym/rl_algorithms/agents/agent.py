from typing import Dict, Any, Optional, List, Callable
import torch
import numpy as np

from functools import partial
from regym.rl_algorithms.utils import is_leaf, _extract_from_rnn_states, recursive_inplace_update, _concatenate_list_hdict


def named_children(cm):
    for name, m in cm._modules.items():
        if m is not None:
            yield name, m


def look_for_keys_and_apply(cm, keys, prefix='', accum: Optional[Dict]=dict(), apply_fn: Optional[Callable]=None, kwargs: Optional[Dict]={}):
    for name, m in named_children(cm):
        accum[name] = {}
        look_for_keys_and_apply(m, keys=keys, prefix=prefix+'.'+name, accum=accum[name], apply_fn=apply_fn, kwargs=kwargs)
        if any( [key in m._get_name() for key in keys]):
            if isinstance(apply_fn, str):   apply_fn = getattr(m, apply_fn, None)
            if apply_fn is not None:    accum[name] = apply_fn(**kwargs)
        elif accum[name]=={}:
            del accum[name]


class Agent(object):
    def __init__(self, name, algorithm):
        self.name = name
        self.algorithm = algorithm

        self.handled_experiences = 0
        self.save_path = None
        self.episode_count = 0

        self.training = True
        self.state_preprocessing = self.algorithm.kwargs['state_preprocess']

        self.goal_oriented = self.algorithm.kwargs['goal_oriented'] if 'goal_oriented' in self.algorithm.kwargs else False
        self.goals = None
        if 'goal_preprocess' in self.algorithm.kwargs:
            self.goal_preprocessing = self.algorithm.kwargs['goal_preprocess']
        elif self.goal_oriented:
            raise NotImplementedError

        self.nbr_actor = self.algorithm.get_nbr_actor()
        self.previously_done_actors = [False]*self.nbr_actor
        # Holds model output from last observation
        self.current_prediction: Dict[str, Any] = None

        self.recurrent = False
        self.rnn_states = None
        self.rnn_keys, self.rnn_states = self._reset_rnn_states(self.algorithm, self.nbr_actor)
        if len(self.rnn_keys):
            self.recurrent = True

    def get_experience_count(self):
        return self.handled_experiences

    def get_update_count(self):
        raise NotImplementedError

    def set_nbr_actor(self, nbr_actor:int):
        if nbr_actor != self.nbr_actor:
            self.nbr_actor = nbr_actor
            self.reset_actors(init=True)
            self.algorithm.reset_storages(nbr_actor=self.nbr_actor)

    def reset_actors(self, indices:Optional[List]=None, init:Optional[bool]=False):
        '''
        In case of a multi-actor process, this function is called to reset
        the actors' internal values.
        '''
        self.current_prediction: Dict[str, Any] = None

        if indices is None: indices = range(self.nbr_actor)

        if init:
            self.previously_done_actors = [False]*self.nbr_actor
        else:
            for idx in indices: self.previously_done_actors[idx] = False

        if self.recurrent:
            _, self.rnn_states = self._reset_rnn_states(self.algorithm, self.nbr_actor)

    def update_actors(self, batch_idx:int):
        '''
        In case of a multi-actor process, this function is called to handle
        the (dynamic) number of environment that are being used.
        More specifically, it regularizes the rnn_states when
        an actor's episode ends.
        It is assumed that update can only go by removing stuffs...
        Indeed, actors all start at the same time, and thus self.reset_actors()
        ought to be called at that time.
        Note: since the number of environment that are running changes,
        the size of the rnn_states on the batch dimension will change too.
        Therefore, we cannot identify an rnn_state by the actor/environment index.
        Thus, it is a batch index that is requested, that would truly be in touch
        with the current batch dimension.
        :param batch_idx: index of the actor whose episode just finished.
        '''
        if self.recurrent:
            self.remove_from_rnn_states(batch_idx=batch_idx)

        if self.goal_oriented:
            self.remove_from_goals(batch_idx=batch_idx)

    def update_goals(self, goals:torch.Tensor):
        assert(self.goal_oriented)
        self.goals = goals

    def remove_from_goals(self, batch_idx:int):
        self.goals = np.concatenate(
                    [self.goals[:batch_idx,...],
                     self.goals[batch_idx+1:,...]],
                     axis=0)

    def _reset_rnn_states(self, algorithm: object, nbr_actor: int):
        # TODO: account for the indices in rnn states:
        lookedup_keys = ['LSTM', 'GRU']
        rnn_states = {}
        kwargs = {'cuda': algorithm.kwargs['use_cuda'], 'repeat':nbr_actor}
        #look_for_keys_and_apply(algorithm.get_models()['model'], keys=lookedup_keys, accum=rnn_states, apply_fn='get_reset_states', kwargs=kwargs)
        for name, model in algorithm.get_models().items():
            if "model" in name:
                look_for_keys_and_apply( model, keys=lookedup_keys, accum=rnn_states, apply_fn='get_reset_states', kwargs=kwargs)
        rnn_keys = list(rnn_states.keys())
        return rnn_keys, rnn_states

    def remove_from_rnn_states(self, batch_idx:int, rnn_states_dict:Optional[Dict]=None, map_keys: Optional[List]=None):
        '''
        Remove a row(=batch) of data from the rnn_states.
        :param batch_idx: Integer index on the batch dimension that specifies which row to remove.
        :param map_keys: List of keys we map the operation to.
        '''
        if rnn_states_dict is None: rnn_states_dict = self.rnn_states
        for recurrent_submodule_name in rnn_states_dict:
            if not is_leaf(rnn_states_dict[recurrent_submodule_name]):
                self.remove_from_rnn_states(batch_idx=batch_idx, rnn_states_dict=rnn_states_dict[recurrent_submodule_name])
            else:
                eff_map_keys = map_keys if map_keys is not None else rnn_states_dict[recurrent_submodule_name].keys()
                for key in eff_map_keys:
                    for idx in range(len(rnn_states_dict[recurrent_submodule_name][key])):
                        rnn_states_dict[recurrent_submodule_name][key][idx] = torch.cat(
                            [rnn_states_dict[recurrent_submodule_name][key][idx][:batch_idx,...],
                             rnn_states_dict[recurrent_submodule_name][key][idx][batch_idx+1:,...]],
                             dim=0
                        )

    def _pre_process_rnn_states(self, rnn_states_dict: Optional[Dict]=None, map_keys: Optional[List]=None):
        '''
        :param map_keys: List of keys we map the operation to.
        '''
        if rnn_states_dict is None:
            if self.rnn_states is None:
                _, self.rnn_states = self._reset_rnn_states(self.algorithm, self.nbr_actor)
            rnn_states_dict = self.rnn_states

        if self.algorithm.kwargs['use_cuda']:
            for recurrent_submodule_name in rnn_states_dict:
                if not is_leaf(rnn_states_dict[recurrent_submodule_name]):
                    self._pre_process_rnn_states(rnn_states_dict=rnn_states_dict[recurrent_submodule_name])
                else:
                    eff_map_keys = map_keys if map_keys is not None else rnn_states_dict[recurrent_submodule_name].keys()
                    for key in eff_map_keys:
                        if key in rnn_states_dict[recurrent_submodule_name]:
                            for idx in range(len(rnn_states_dict[recurrent_submodule_name][key])):
                                rnn_states_dict[recurrent_submodule_name][key][idx] = rnn_states_dict[recurrent_submodule_name][key][idx].cuda()
                                rnn_states_dict[recurrent_submodule_name][key][idx]   = rnn_states_dict[recurrent_submodule_name][key][idx].cuda()

    @staticmethod
    def _post_process_and_update_rnn_states(next_rnn_states_dict: Dict, rnn_states_dict: Dict, map_keys: Optional[List]=None):
        '''
        Update the rnn_state to the values of next_rnn_states, when present in both.
        Otherwise, simply detach+cpu the values. 

        :param next_rnn_states_dict: Dict with a hierarchical structure.
        :param rnn_states_dict: Dict with a hierarchical structure, ends up being update when possible.
        :param map_keys: List of keys we map the operation to.
        '''
        for recurrent_submodule_name in rnn_states_dict:
            if not is_leaf(rnn_states_dict[recurrent_submodule_name]):
                Agent._post_process_and_update_rnn_states(
                    next_rnn_states_dict=next_rnn_states_dict[recurrent_submodule_name],
                    rnn_states_dict=rnn_states_dict[recurrent_submodule_name]
                )
            else:
                eff_map_keys = map_keys if map_keys is not None else rnn_states_dict[recurrent_submodule_name].keys()
                for key in eff_map_keys:
                    updateable = False
                    if key in next_rnn_states_dict[recurrent_submodule_name]:
                        updateable = True
                        for idx in range(len(next_rnn_states_dict[recurrent_submodule_name][key])):
                            # Post-process:
                            next_rnn_states_dict[recurrent_submodule_name][key][idx] = next_rnn_states_dict[recurrent_submodule_name][key][idx].detach().cpu()
                    if key in rnn_states_dict[recurrent_submodule_name]:
                        for idx in range(len(rnn_states_dict[recurrent_submodule_name][key])):
                            if updateable:
                                # Updating rnn_states:
                                rnn_states_dict[recurrent_submodule_name][key][idx] = next_rnn_states_dict[recurrent_submodule_name][key][idx].detach().cpu()
                            else:
                                # only post-process:
                                rnn_states_dict[recurrent_submodule_name][key][idx] = rnn_states_dict[recurrent_submodule_name][key][idx].detach().cpu()
                            
    def _post_process(self, prediction: Dict[str, Any]):
        """
        Post-process a prediction by detaching-cpuing the tensors.
        Note: if there are some recurrent components, then the agent's 
        recurrent states are being updated from the prediction's 
        `"next_rnn_states"` entry.
        """
        if self.recurrent:
            Agent._post_process_and_update_rnn_states(
                next_rnn_states_dict=prediction['next_rnn_states'],
                rnn_states_dict=self.rnn_states
            )

            for k, v in prediction.items():
                if isinstance(v, torch.Tensor):
                    prediction[k] = v.detach().cpu()
        else:
            prediction = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                             for k, v in prediction.items()
                            }

        return prediction

    @staticmethod
    def _extract_from_prediction(prediction: Dict, batch_idx: int):
        out_pred = dict()
        for k, v in prediction.items():
            if v is None or isinstance(v, dict):
                continue
            out_pred[k] = v[batch_idx,...].unsqueeze(0)
        return out_pred

    @staticmethod
    def _extract_from_hdict(hdict: Dict, batch_idx: int, goal_preprocessing_fn:Optional[Callable]=None):
        out_hdict = dict()
        for k, v in hdict.items():
            if isinstance(v, dict):
                v = Agent._extract_from_hdict(hdict=v, batch_idx=batch_idx, goal_preprocessing_fn=goal_preprocessing_fn)
            else:
                if isinstance(v, torch.Tensor):
                    v = v[batch_idx, ...].unsqueeze(0)
                elif isinstance(v, np.ndarray):
                    v = np.expand_dims(v[batch_idx, ...], axis=0)
                else:
                    raise NotImplementedError
                if goal_preprocessing_fn is not None:
                    v = goal_preprocessing_fn(v)
            out_hdict[k] = v
        return out_hdict

    def preprocess_environment_signals(self, state, reward, succ_state, done):
        non_terminal = torch.from_numpy(1 - np.array(done)).reshape(-1,1).type(torch.FloatTensor)
        state = self.state_preprocessing(state, use_cuda=False)
        succ_state = self.state_preprocessing(succ_state, use_cuda=False)
        if isinstance(reward, np.ndarray): r = torch.from_numpy(reward).reshape(-1,1).type(torch.FloatTensor)
        else: r = torch.ones(1,1).type(torch.FloatTensor)*reward
        return state, r, succ_state, non_terminal

    def handle_experience(self, s, a, r, succ_s, done, goals=None, infos=None):
        '''
        Note: the batch size may differ from the nbr_actor as soon as some
        actors' episodes end before the others...

        :param s: numpy tensor of states of shape batch x state_shape.
        :param a: numpy tensor of actions of shape batch x action_shape.
        :param r: numpy tensor of rewards of shape batch x reward_shape.
        :param succ_s: numpy tensor of successive states of shape batch x state_shape.
        :param done: list of boolean (batch=nbr_actor) x state_shape.
        :param goals: Dictionnary of goals 'achieved_goal' and 'desired_goal' for each state 's' and 'succ_s'.
        :param infos: Dictionnary of information from the environment.
        '''
        raise NotImplementedError

    def take_action(self, state):
        raise NotImplementedError

    def clone(self, training=None, with_replay_buffer=False):
        raise NotImplementedError

    def save(self, with_replay_buffer=False):
        assert(self.save_path is not None)
        torch.save(self.clone(with_replay_buffer=with_replay_buffer), self.save_path)



class ExtraInputsHandlingAgent(Agent):
    def __init__(self, name, algorithm, extra_inputs_infos):
        self.extra_inputs_infos = extra_inputs_infos
        self.dummies = {
            key: torch.zeros(size=(1, *extra_inputs_infos[key]['shape'])) 
            for key in self.extra_inputs_infos
        }

        Agent.__init__(
            self,
            name=name, 
            algorithm=algorithm
        )

    def _reset_rnn_states(self, algorithm: object, nbr_actor: int):
        self.rnn_keys, self.rnn_states = super()._reset_rnn_states(algorithm=algorithm, nbr_actor=nbr_actor)
        
        # Resetting extra inputs:
        hdict = self._init_hdict()
        recursive_inplace_update(self.rnn_states, hdict)
        
        return self.rnn_keys, self.rnn_states

    def _init_hdict(self, init:Optional[Dict]={}):
        hdict = {}
        for key in self.extra_inputs_infos:
            value = init.get(key, torch.cat([self.dummies[key]]*self.nbr_actor, dim=0))
            pointer = hdict
            for child_node in self.extra_inputs_infos[key]['target_location']:
                if child_node not in pointer:
                    pointer[child_node] = {}
                pointer = pointer[child_node]
            pointer[key] = [value]
        return hdict
    
    def _build_dict_from(self, lhdict: Dict):
        concat_hdict = _concatenate_list_hdict(
            lhds=lhdict, 
            concat_fn=partial(torch.cat, dim=0),
            preprocess_fn=(lambda x:torch.from_numpy(x).float() if isinstance(x, np.ndarray) else torch.ones(1, 1).float()*x),
        )

        out_hdict = self._init_hdict(init=concat_hdict)

        return out_hdict

    def handle_experience(self, s, a, r, succ_s, done, goals=None, infos=None):
        '''
        Wrapper around the actual function now living in _handle_experience.
        It prepares the rnn_states.

        Note: the batch size may differ from the nbr_actor as soon as some
        actors' episodes end before the others...

        :param s: numpy tensor of states of shape batch x state_shape.
        :param a: numpy tensor of actions of shape batch x action_shape.
        :param r: numpy tensor of rewards of shape batch x reward_shape.
        :param succ_s: numpy tensor of successive states of shape batch x state_shape.
        :param done: list of boolean (batch=nbr_actor) x state_shape.
        :param goals: Dictionnary of goals 'achieved_goal' and 'desired_goal' for each state 's' and 'succ_s'.
        :param infos: List of Dictionnaries of information from the environment.
        '''
        hdict = self._build_dict_from(lhdict=infos)
        
        recursive_inplace_update(self.rnn_states, hdict)
        
        self._handle_experience(s, a, r, succ_s, done, goals=goals, infos=infos)
    
    def _handle_experience(self, s, a, r, succ_s, done, goals=None, infos=None):
        '''
        Note: the batch size may differ from the nbr_actor as soon as some
        actors' episodes end before the others...

        :param s: numpy tensor of states of shape batch x state_shape.
        :param a: numpy tensor of actions of shape batch x action_shape.
        :param r: numpy tensor of rewards of shape batch x reward_shape.
        :param succ_s: numpy tensor of successive states of shape batch x state_shape.
        :param done: list of boolean (batch=nbr_actor) x state_shape.
        :param goals: Dictionnary of goals 'achieved_goal' and 'desired_goal' for each state 's' and 'succ_s'.
        :param infos: Dictionnary of information from the environment.
        '''
        raise NotImplementedError
