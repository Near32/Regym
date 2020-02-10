import torch
import numpy as np
from ..agent import Agent


class AgentWrapper(Agent):
    def __init__(self, agent):
        super(AgentWrapper, self).__init__(name=agent.name, algorithm=agent.algorithm)
        self.agent = agent
        
    def get_experience_count(self):
        return self.agent.get_experience_count()

    def get_update_count(self):
        return self.agent.get_update_count()

    def set_nbr_actor(self, nbr_actor):
        self.agent.set_nbr_actor(nbr_actor)

    def reset_actors(self, indices=None, init=False):
        self.agent.reset_actors(indices, init)

    def update_actors(self, batch_idx):
        self.agent.update_actors(batch_idx)

    def update_goals(self, goals):
        self.agent.update_goals(goals)

    def remove_from_rnn_states(self, batch_idx):
        self.agent.remove_from_rnn_states(batch_idx)

    def _pre_process_rnn_states(self):
        self.agent._pre_process_rnn_states()

    @staticmethod
    def _extract_from_rnn_states(rnn_states_batched: dict, batch_idx: int):
        return self.agent._extract_from_rnn_states(rnn_states_batched, batch_idx)

    def _post_process(self, prediction):
        return self.agent._post_process(prediction)

    @staticmethod
    def _extract_from_prediction(prediction: dict, batch_idx: int):
        return self.agent._extract_from_prediction(prediction, batch_idx)

    def preprocess_environment_signals(self, state, reward, succ_state, done):
        return self.agent.preprocess_environment_signals(state, reward, succ_state, done)
    
    def handle_experience(self, s, a, r, succ_s, done):
        '''
        Note: the batch size may differ from the nbr_actor as soon as some
        actors' episodes end before the others...

        :param s: numpy tensor of states of shape batch x state_shape.
        :param a: numpy tensor of actions of shape batch x action_shape.
        :param r: numpy tensor of rewards of shape batch x reward_shape.
        :param succ_s: numpy tensor of successive states of shape batch x state_shape.
        :param done: list of boolean (batch=nbr_actor) x state_shape.
        '''
        raise NotImplementedError

    def take_action(self, state):
        raise NotImplementedError

    def clone(self, training=None):
        return AgentWrapper(agent=self.agent.clone(training=training))

    def save(self):
        torch.save(self.clone(), self.save_path)


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
    
    def handle_experience(self, s, a, r, succ_s, done, goals=None):
        obs_dict = self._build_obs_dict(s)
        next_obs_dict = self._build_obs_dict(succ_s)

        state = obs_dict['observation']
        succ_state = next_obs_dict['observation']

        goals=None
        if 'desired_goal' in obs_dict:
            goals = {'desired_goals': {'s': obs_dict['desired_goal'], 'succ_s': next_obs_dict['desired_goal']}}

        if self.use_achieved_goal and 'achieved_goal' in obs_dict:
            goals.update({'achieved_goals': {'s': obs_dict['achieved_goal'], 'succ_s': next_obs_dict['achieved_goal']}})
        
        self.agent.handle_experience(state, a, r, succ_state, done, goals=goals)


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

