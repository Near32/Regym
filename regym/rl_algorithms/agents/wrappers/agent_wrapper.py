from typing import Dict, Optional, List, Any
import torch
import numpy as np
from ..agent import Agent
from regym.rl_algorithms.utils import _extract_from_rnn_states, recursive_inplace_update


class AgentWrapper(Agent):
    def __init__(self, agent):
        self.agent = agent
        super(AgentWrapper, self).__init__(name=agent.name, algorithm=agent.algorithm)

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
        self.agent.handle_experience(self, s, a, r, succ_s, done, goals, infos)

    def _reset_rnn_states(self, algorithm: object, nbr_actor: int):
        self.agent._reset_rnn_states(algorithm, nbr_actor)

    def remove_from_rnn_states(self, batch_idx:int, rnn_states_dict:Optional[Dict]=None, map_keys: Optional[List]=['hidden', 'cell']):
        self.agent.remove_from_rnn_states(self, batch_idx, rnn_states_dict, map_keys)

    def _pre_process_rnn_states(self, rnn_states_dict: Optional[Dict]=None, map_keys: Optional[List]=['hidden', 'cell']):
        self.agent._pre_process_rnn_states(self, rnn_states_dict, map_keys)

    def preprocess_environment_signals(self, state, reward, succ_state, done):
        self.agent.preprocess_environment_signals(self, state, reward, succ_state, done)

    def _post_process(self, prediction: Dict[str, Any]):
        self.agent_post_process(self, prediction)

    def take_action(self, state):
        self.agent.take_action(state=state)

    def clone(self, training=None, with_replay_buffer=False):
        return AgentWrapper(agent=self.agent.clone(training=training, with_replay_buffer=with_replay_buffer))

    """
    def save(self, with_replay_buffer=False):
        torch.save(self.clone(with_replay_buffer=with_replay_buffer), self.save_path)
    """
    def save(self, with_replay_buffer=False, minimal=False):
        assert(self.save_path is not None)
        torch.save(
            self.clone(
                with_replay_buffer=with_replay_buffer, 
                clone_proxies=False,
                minimal=minimal), 
            self.save_path
        )


class ExtraInputsHandlingAgentWrapper(AgentWrapper):
    def __init__(self, agent, extra_inputs_infos):
        self.extra_inputs_infos = extra_inputs_infos
        self.dummies = {
            key: torch.zeros(size=extra_inputs_infos[key]['shape']) 
            for key in self.extra_inputs_infos
        }

        super(ExtraInputsHandlingAgentWrapper, self).__init__(agent=agent)

    def _reset_rnn_states(self, algorithm: object, nbr_actor: int):
        rnn_keys, rnn_states = self.agent._reset_rnn_states(algorithm=algorithm, nbr_actor=nbr_actor)
        # Resetting extra inputs:
        hdict = self._build_dict_from(fhdict={})
        
        recursive_inplace_update(rnn_states, hdict)
        self.agent.rnn_keys, self.agent.rnn_states = rnn_keys, rnn_states
        return rnn_keys, rnn_states

    def _build_dict_from(self, fhdict: Dict):
        hdict = {}
        for key in self.extra_inputs_infos:
            value = fhdict.get(key, torch.stack([self.dummies[key]]*self.nbr_actor, dim=0))
            import ipdb; ipdb.set_trace()
            if not isinstance(value, torch.Tensor): 
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                else:
                    raise NotImplementedError 
            pointer = hdict
            for child_node in self.extra_inputs_infos[key]['target_location']:
                if child_node not in pointer:
                    pointer[child_node] = {}
                pointer = pointer[child_node]
            
            pointer[key] = [value]
        return hdict 

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
        hdict = self._build_dict_from(fhdict=infos)
        
        recursive_inplace_update(self.agent.rnn_states, hdict)

        self.agent.handle_experience(s, a, r, succ_s, done, goals=goals, infos=infos)
        
    def clone(self, training=None, with_replay_buffer=False):
        return AgentWrapper(agent=self.agent.clone(training=training, with_replay_buffer=with_replay_buffer))

    def save(self, with_replay_buffer=False):
        torch.save(self.clone(with_replay_buffer=with_replay_buffer), self.save_path)


class DictHandlingAgentWrapper(AgentWrapper):
    def __init__(self, agent, use_achieved_goal=True):
        super(DictHandlingAgentWrapper, self).__init__(agent=agent)
        self.use_achieved_goal = use_achieved_goal

    def _build_obs_dict(self, s):
        obs_dict = {}
        for lidx in range(s.shape[0]):
            d = s[lidx]
            if not isinstance(d, dict): d = d[0]
            for key, value in d.items():
                if key not in obs_dict: obs_dict[key] = []
                obs_dict[key].append(np.expand_dims(value, axis=0))

        for key in obs_dict:
            obs_dict[key] = np.concatenate(obs_dict[key], axis=0)

        return obs_dict
    
    def handle_experience(self, s, a, r, succ_s, done, goals=None, infos=None):
        obs_dict = self._build_obs_dict(s)
        next_obs_dict = self._build_obs_dict(succ_s)

        state = obs_dict['observation']
        succ_state = next_obs_dict['observation']

        goals=None
        if 'desired_goal' in obs_dict:
            goals = {'desired_goals': {'s': obs_dict['desired_goal'], 'succ_s': next_obs_dict['desired_goal']}}

        if self.use_achieved_goal and 'achieved_goal' in obs_dict:
            goals.update({'achieved_goals': {'s': obs_dict['achieved_goal'], 'succ_s': next_obs_dict['achieved_goal']}})
        
        self.agent.handle_experience(state, a, r, succ_state, done, goals=goals, infos=infos)


    def take_action(self, s):
        '''
        Assumes `param s` to be a `numpy.ndarray` of shape (nbr_actors(-), 1)
        where each line element is a dictionnary.
        This function firstly separates this `numpy.ndarray` into a dictionnary
        of `numpy.ndarray`.
        Then, if there is a `"goal"` item, it feeds it to the agent.
        Finaly, it feeds the relevant `"state"`'s value to 
        the agent for action selection.
        '''
        obs_dict = self._build_obs_dict(s)
        state = obs_dict["observation"]
        if "desired_goal" in obs_dict:
            self.agent.update_goals(goals=obs_dict["desired_goal"])
        return self.agent.take_action(state=state)

    def clone(self, training=None):
        return DictHandlingAgentWrapper(agent=self.agent.clone(training=training))


class SADAgentWrapper(AgentWrapper):
    def __init__(self, agent):
        super(SADAgentWrapper, self).__init__(agent=agent)
        
    def take_action(self, s):
        '''
        Assumes `param s` to be a `numpy.ndarray` of shape (nbr_actors(-), 1)
        where each line element is a dictionnary.
        This function firstly separates this `numpy.ndarray` into a dictionnary
        of `numpy.ndarray`.
        Then, if there is a `"goal"` item, it feeds it to the agent.
        Finaly, it feeds the relevant `"state"`'s value to 
        the agent for action selection.
        '''
        """
        obs_dict = self._build_obs_dict(s)
        state = obs_dict["observation"]
        if "desired_goal" in obs_dict:
            self.agent.update_goals(goals=obs_dict["desired_goal"])
        return self.agent.take_action(state=state)

        if self.async_actor:
            # Update the algorithm's model if needs be:
            if isinstance(self.actor_learner_shared_dict, ray.actor.ActorHandle):
                actor_learner_shared_dict = ray.get(self.actor_learner_shared_dict.get.remote())
            else:
                actor_learner_shared_dict = self.actor_learner_shared_dict.get()
            if actor_learner_shared_dict["models_update_required"][self.async_actor_idx]:
                actor_learner_shared_dict["models_update_required"][self.async_actor_idx] = False
                
                if isinstance(self.actor_learner_shared_dict, ray.actor.ActorHandle):
                    self.actor_learner_shared_dict.set.remote(actor_learner_shared_dict)
                else:
                    self.actor_learner_shared_dict.set(actor_learner_shared_dict)
                
                if "models" in actor_learner_shared_dict.keys():
                    new_models = actor_learner_shared_dict["models"]
                    self.algorithm.set_models(new_models)
                else:
                    raise NotImplementedError 

        if self.training:
            self.nbr_steps += state.shape[0]
        self.eps = self.algorithm.get_epsilon(nbr_steps=self.nbr_steps, strategy=self.epsdecay_strategy)

        state = self.state_preprocessing(state, use_cuda=self.algorithm.kwargs['use_cuda'])
        goal = None
        if self.goal_oriented:
            goal = self.goal_preprocessing(self.goals, use_cuda=self.algorithm.kwargs['use_cuda'])

        model = self.algorithm.get_models()['model']
        if 'use_target_to_gather_data' in self.kwargs and self.kwargs['use_target_to_gather_data']:
            model = self.algorithm.get_models()['target_model']
        model = model.train(mode=self.training)

        self.current_prediction = self.query_model(model, state, goal)
        
        # Post-process and update the rnn_states from the current prediction:
        # self.rnn_states <-- self.current_prediction['next_rnn_states']
        # WARNING: _post_process affects self.rnn_states. It is imperative to
        # manipulate a copy of it outside of the agent's manipulation, e.g.
        # when feeding it to the models.
        self.current_prediction = self._post_process(self.current_prediction)

        greedy_action = self.current_prediction['a'].reshape((-1,1)).numpy()
        if self.noisy or not(self.training):
            return greedy_action

        legal_actions = torch.ones_like(self.current_prediction['qa'])
        if infos is not None\
        and 'head' in infos\
        and 'extra_inputs' in infos['head']\
        and 'legal_actions' in infos['head']['extra_inputs']:
            legal_actions = infos['head']['extra_inputs']['legal_actions'][0]
            # in case there are no legal actions for this agent in this current turn:
            for actor_idx in range(legal_actions.shape[0]):
                if legal_actions[actor_idx].sum() == 0: 
                    legal_actions[actor_idx, ...] = 1
        sample = np.random.random(size=self.eps.shape)
        greedy = (sample > self.eps)
        greedy = np.reshape(greedy[:state.shape[0]], (state.shape[0],1))

        #random_actions = [random.randrange(model.action_dim) for _ in range(state.shape[0])]
        random_actions = [
            legal_actions[actor_idx].multinomial(num_samples=1).item() 
            for actor_idx in range(legal_actions.shape[0])
        ]
        random_actions = np.reshape(np.array(random_actions), (state.shape[0],1))
        
        actions = greedy*greedy_action + (1-greedy)*random_actions
        
        return actions
        """
        pass

    def clone(self, training=None, with_replay_buffer=False, clone_proxies=False, minimal=False):
        return SADAgentWrapper(agent=self.agent.clone(
            training=training,
            with_replay_buffer=with_replay_buffer,
            clone_proxies=clone_proxies,
            minimal=minimal)
        )
    
    def get_async_actor(self, training=None, with_replay_buffer=False):
        """
        Returns an asynchronous actor agent (i.e. attribute async_actor
        of the return agent must be set to True).
        RegymManager's value must be reference back from original to clone!
        """
        pass
