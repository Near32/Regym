from typing import Dict, Any, Optional, List, Callable
import torch
import numpy as np

from functools import partial
import regym
from regym.rl_algorithms.utils import is_leaf, _extract_from_rnn_states, recursive_inplace_update, _concatenate_list_hdict

import ray

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
        self.nbr_actor = self.algorithm.get_nbr_actor()
        self.previously_done_actors = [False]*self.nbr_actor
        
        self.async_actor = False
        self.async_actor_idx: int = -1
        self.async_learner = False 
        actor_learner_shared_dict = {"models_update_required":[], "models": None}
        if regym.RegymManager is not None:
            from regym import RaySharedVariable
            try:
                self.actor_learner_shared_dict = ray.get_actor(f"{self.name}.actor_learner_shared_dict")
            except ValueError:  # Name is not taken.
                self.actor_learner_shared_dict = RaySharedVariable.options(name=f"{self.name}.actor_learner_shared_dict").remote(actor_learner_shared_dict)
        else:
            from regym import SharedVariable
            self.actor_learner_shared_dict = SharedVariable(actor_learner_shared_dict)

        self.actor_models_update_steps_interval = 32
        if "actor_models_update_steps_interval" in self.algorithm.kwargs:
            self.actor_models_update_steps_interval = self.algorithm.kwargs["actor_models_update_steps_interval"]
        
        # Accounting for the number of actors:
        self.actor_models_update_steps_interval *= self.nbr_actor
        self.previous_actor_models_update_quotient = -1

        if regym.RegymManager is not None:
            from regym import RaySharedVariable
            try:
                self._handled_experiences = ray.get_actor(f"{self.name}.handled_experiences")
            except ValueError:  # Name is not taken.
                self._handled_experiences = RaySharedVariable.options(name=f"{self.name}.handled_experiences").remote(0)
        else:
            from regym import SharedVariable
            self._handled_experiences = SharedVariable(0)

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

        # Holds model output from last observation
        self.current_prediction: Dict[str, Any] = None

        # DEPRECATED in order to allow extra_inputs infos 
        # stored in the rnn_states that acts as frame_states...
        #self.recurrent = False
        self.recurrent = True
        self.rnn_states = None
        self.rnn_keys, self.rnn_states = self._reset_rnn_states(self.algorithm, self.nbr_actor)
        """
        if len(self.rnn_keys):
            self.recurrent = True
        """

    def parameters(self):
        return self.algorithm.parameters()
        
    @property
    def handled_experiences(self):
        if isinstance(self._handled_experiences, ray.actor.ActorHandle):
            return ray.get(self._handled_experiences.get.remote())
        else:
            return self._handled_experiences.get()

    @handled_experiences.setter
    def handled_experiences(self, val):
        if isinstance(self._handled_experiences, ray.actor.ActorHandle):
            self._handled_experiences.set.remote(val)
        else:
            self._handled_experiences.set(val)

    def get_experience_count(self):
        return self.handled_experiences

    def get_update_count(self):
        raise NotImplementedError

    def get_nbr_actor(self):
        return self.nbr_actor

    def set_nbr_actor(self, nbr_actor:int, vdn:Optional[bool]=None, training:Optional[bool]=None):
        if nbr_actor != self.nbr_actor:
            self.nbr_actor = nbr_actor
            self.algorithm.set_nbr_actor(nbr_actor=self.nbr_actor)
            if training is None:
                self.algorithm.reset_storages(nbr_actor=self.nbr_actor)
            else:
                self.training = training
        self.reset_actors(init=True, vdn=vdn)

    def get_rnn_states(self):
        return self.rnn_states 

    def set_rnn_states(self, rnn_states):
        self.rnn_states = rnn_states 
        
    def reset_actors(self, indices:Optional[List]=[], init:Optional[bool]=False, vdn=None):
        '''
        In case of a multi-actor process, this function is called to reset
        the actors' internal values.
        '''
        # the following is interfering with rl_agent_module
        # that operates on a delay with MARLEnvironmentModule
        # when it comes to the time prediction is made
        # and then the time when an experience is handled.
        # TODO: make sure that disabling it is not affecting other behaviours...
        #self.current_prediction: Dict[str, Any] = None

        if init:
            self.previously_done_actors = [False]*self.nbr_actor
        else:
            for idx in indices: self.previously_done_actors[idx] = False

        if self.recurrent:
            _, self.rnn_states = self._reset_rnn_states(self.algorithm, self.nbr_actor, actor_indices=indices, vdn=vdn)

    def _reset_rnn_states(self, algorithm: object, nbr_actor: int, actor_indices: Optional[List[int]]=[], vdn:Optional[bool]=None):
        # TODO: account for the indices in rnn states:
        if ((vdn is not None and vdn) or (vdn is None))\
        and self.algorithm.kwargs.get("vdn", False):
            nbr_players = self.algorithm.kwargs["vdn_nbr_players"]
            nbr_envs = nbr_actor
            nbr_actor *= nbr_players
            if isinstance(actor_indices,list):
                new_actor_indices = []
                for aidx in actor_indices:
                    for pidx in range(nbr_players):
                        new_actor_indices.append(aidx+nbr_envs*pidx)
                actor_indices = new_actor_indices

        lookedup_keys = ['LSTM', 'GRU', 'NTM', 'DNC']
        new_rnn_states = {}
        kwargs = {'cuda': False, 'repeat':nbr_actor}
        for name, model in algorithm.get_models().items():
            if "model" in name and model is not None:
                look_for_keys_and_apply( 
                    model, 
                    keys=lookedup_keys, 
                    accum=new_rnn_states, 
                    apply_fn='get_reset_states', 
                    kwargs=kwargs
                )
        rnn_keys = list(new_rnn_states.keys())
        
        if self.rnn_states is None or actor_indices==[]:    
            return rnn_keys, new_rnn_states

        # Reset batch element only: 
        batch_indices_to_update = torch.Tensor(actor_indices).long()
        
        recursive_inplace_update(
            in_dict=self.rnn_states,
            extra_dict=new_rnn_states,
            batch_mask_indices=batch_indices_to_update
        )

        return rnn_keys, self.rnn_states
        
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

    def _pre_process_rnn_states(self, rnn_states_dict: Optional[Dict]=None, vdn:Optional[bool]=None):
        '''
        :param map_keys: List of keys we map the operation to.
        '''
        if rnn_states_dict is None:
            if self.rnn_states is None:
                _, self.rnn_states = self._reset_rnn_states(self.algorithm, self.nbr_actor, vdn=vdn)
            rnn_states_dict = self.rnn_states

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
                            
    @staticmethod
    def _keep_grad_update_rnn_states(next_rnn_states_dict: Dict, rnn_states_dict: Dict, map_keys: Optional[List]=None):
        '''
        Update the rnn_state to the values of next_rnn_states, when present in both.
        Otherwise, simply detach+cpu the values. 

        :param next_rnn_states_dict: Dict with a hierarchical structure.
        :param rnn_states_dict: Dict with a hierarchical structure, ends up being update when possible.
        :param map_keys: List of keys we map the operation to.
        '''
        for recurrent_submodule_name in rnn_states_dict:
            if not is_leaf(rnn_states_dict[recurrent_submodule_name]):
                Agent._keep_grad_update_rnn_states(
                    next_rnn_states_dict=next_rnn_states_dict[recurrent_submodule_name],
                    rnn_states_dict=rnn_states_dict[recurrent_submodule_name]
                )
            else:
                eff_map_keys = map_keys if map_keys is not None else rnn_states_dict[recurrent_submodule_name].keys()
                for key in eff_map_keys:
                    updateable = False
                    """
                    if key in next_rnn_states_dict[recurrent_submodule_name]:
                        updateable = True
                        for idx in range(len(next_rnn_states_dict[recurrent_submodule_name][key])):
                            # Post-process:
                            next_rnn_states_dict[recurrent_submodule_name][key][idx] = next_rnn_states_dict[recurrent_submodule_name][key][idx].detach().cpu()
                    """
                    if key in rnn_states_dict[recurrent_submodule_name]:
                        for idx in range(len(rnn_states_dict[recurrent_submodule_name][key])):
                            if updateable:
                                # Updating rnn_states:
                                rnn_states_dict[recurrent_submodule_name][key][idx] = next_rnn_states_dict[recurrent_submodule_name][key][idx]#.detach().cpu()
                            else:
                                # only post-process:
                                rnn_states_dict[recurrent_submodule_name][key][idx] = rnn_states_dict[recurrent_submodule_name][key][idx]#.detach().cpu()
    
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

    def train(self):
        raise NotImplementedError

    def take_action(self, state, as_logit=False):
        raise NotImplementedError

    def clone(self, training=None, with_replay_buffer=False, clone_proxies=False, minimal=False):
        raise NotImplementedError

    def get_async_actor(self, training=None, with_replay_buffer=False):
        """
        Returns an asynchronous actor agent (i.e. attribute async_actor
        of the return agent must be set to True).
        RegymManager's value must be reference back from original to clone!
        """
        self.async_learner = True 
        self.async_actor = False 
        # self.async_actor_idx needs to be implemented in the resulting actor.
        # and the actor_learner_shared_dict's toggle boolean needs to be increased.

        return 

    def save(self, with_replay_buffer=False, minimal=False):
        assert(self.save_path is not None)
        torch.save(
            self.clone(
                with_replay_buffer=with_replay_buffer, 
                clone_proxies=False,
                minimal=minimal), 
            self.save_path
        )



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

    def _reset_rnn_states(self, algorithm: object, nbr_actor: int, actor_indices: Optional[List[int]]=None, vdn:Optional[bool]=None):
        self.rnn_keys, self.rnn_states = super()._reset_rnn_states(
            algorithm=algorithm, 
            nbr_actor=nbr_actor,
            actor_indices=actor_indices,
            vdn=vdn
        )
        
        
        # Resetting extra inputs:
        hdict = self._init_hdict()
        recursive_inplace_update(
            in_dict=self.rnn_states, 
            extra_dict=hdict,
            batch_mask_indices=actor_indices,
        )
        
        return self.rnn_keys, self.rnn_states

    def _init_hdict(self, init:Optional[Dict]={}):
        hdict = {}
        for key in self.extra_inputs_infos:
            value = init.get(key, torch.cat([self.dummies[key]]*self.nbr_actor, dim=0))
            if not isinstance(self.extra_inputs_infos[key]['target_location'][0], list):
                self.extra_inputs_infos[key]['target_location'] = [self.extra_inputs_infos[key]['target_location']]
            for tl in self.extra_inputs_infos[key]['target_location']:
                pointer = hdict
                for child_node in tl:
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

    def take_action(self, state, infos=None, as_logit=False):
        hdict = None
        if infos:# and not self.training:
            agent_infos = [info for info in infos if info is not None]
            hdict = self._build_dict_from(lhdict=agent_infos)
            recursive_inplace_update(self.rnn_states, hdict)
        return self._take_action(state, infos=hdict, as_logit=as_logit)

    def query_action(self, state, infos=None, as_logit=False):
        hdict = None
        if infos:# and not self.training:
            agent_infos = [info for info in infos if info is not None]
            hdict = self._build_dict_from(lhdict=agent_infos)
            recursive_inplace_update(self.rnn_states, hdict)
        return self._query_action(state, infos=hdict, as_logit=as_logit)

    def _take_action(self, state, infos=None):
        raise NotImplementedError

    def _query_action(self, state, infos=None):
        raise NotImplementedError

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

        """
        #TODO: make sure that it is not important to update rnn states from this info...
        
        agent_infos = [info for info in infos if info is not None]
        hdict = self._build_dict_from(lhdict=agent_infos)
        
        recursive_inplace_update(self.rnn_states, hdict)
        """

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
