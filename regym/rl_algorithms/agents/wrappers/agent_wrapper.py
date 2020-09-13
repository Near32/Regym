import torch
import numpy as np
from ..agent import Agent
from regym.rl_algorithms.utils import _extract_from_rnn_states


class AgentWrapper(Agent):
    def __init__(self, agent):
        super(AgentWrapper, self).__init__(name=agent.name, algorithm=agent.algorithm)
        self.agent = agent
        
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

