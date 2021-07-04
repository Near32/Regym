from typing import Dict, Optional

import torch
import numpy as np
import copy
from regym.rl_algorithms.algorithms.wrappers.algorithm_wrapper import AlgorithmWrapper
from regym.rl_algorithms.utils import copy_hdict


def state_eq_goal_reward_fn2(state, goal, epsilon=1e-3):
    if torch.abs(state-goal).mean() < epsilon:
        return torch.zeros(1), state
    else:
        return -torch.ones(1), state


def state_eq_goal_reward_fn2(achieved_exp, desired_exp, _extract_goals_from_info_fn, epsilon=1e-3):
    state = achieved_exp['succ_s']
    #goal = desired_exp['goals']['achieved_goals']['s']
    #goal = _extract_goals_from_rnn_states_fn(desired_exp['info'], goal_key="achieved_goal")
    #goal = torch.from_numpy(desired_exp['info']['achieved_goal']).float()
    goal = _extract_goals_from_info_fn(desired_exp['info'], goal_key="achieved_goal")
    if torch.abs(state-goal).mean() < epsilon:
        return torch.zeros(1,1), goal
    else:
        return -torch.ones(1,1), goal


def latent_based_goal_predicated_reward_fn2(achieved_exp, desired_exp, epsilon=1e-3):
    raise NotImplementedError
    state = achieved_exp['info']['latents']['succ_s']
    goal = desired_exp['info']['latents']['achieved_goal']
    abs_fn = torch.abs 
    if not(isinstance(state, torch.Tensor)):    abs_fn = np.abs
    if abs_fn(state-goal).mean() < epsilon:
        return torch.zeros(1), desired_exp['goals']['achieved_goals']['s']
    else:
        return -torch.ones(1), desired_exp['goals']['achieved_goals']['s']

class HERAlgorithmWrapper2(AlgorithmWrapper):
    def __init__(self, algorithm, extra_inputs_infos, strategy="future-4", goal_predicated_reward_fn=None):
        super(HERAlgorithmWrapper2, self).__init__(algorithm=algorithm)
        if goal_predicated_reward_fn is None:   goal_predicated_reward_fn = state_eq_goal_reward_fn2

        self.extra_inputs_infos = extra_inputs_infos

        self.episode_buffer = [[] for i in range(self.algorithm.get_nbr_actor())]
        self.strategy = strategy
        assert( ('future' in self.strategy or 'final' in self.strategy) and '-' in self.strategy)
        self.k = int(self.strategy.split('-')[-1])    
        self.goal_predicated_reward_fn = goal_predicated_reward_fn
        self.episode_count = 0

    def _update_goals_in_rnn_states(self, hdict:Dict, goal_value:torch.Tensor, goal_key:Optional[str]='desired_goal'):
        if goal_key in self.extra_inputs_infos:
            if not isinstance(self.extra_inputs_infos[goal_key]['target_location'][0], list):
                self.extra_inputs_infos[goal_key]['target_location'] = [self.extra_inputs_infos[goal_key]['target_location']]
            for tl in self.extra_inputs_infos[goal_key]['target_location']:
                pointer = hdict
                for child_node in tl:
                    if child_node not in pointer:
                        pointer[child_node] = {}
                    pointer = pointer[child_node]
                pointer[goal_key] = [goal_value]
        return hdict

    def _extract_goals_from_rnn_states(self, hdict:Dict, goal_key:Optional[str]='desired_goal'):
        import ipdb; ipdb.set_trace()
        assert goal_key in self.extra_inputs_infos
        tl = self.extra_inputs_infos[goal_key]['target_location'][-1]
        pointer = hdict
        for child_node in tl:
            if child_node not in pointer:
                pointer[child_node] = {}
            pointer = pointer[child_node]
        return pointer[goal_key]

    def _extract_goals_from_info(self, hdict:Dict, goal_key:Optional[str]='desired_goal'):
        assert goal_key in hdict
        value = hdict[goal_key]
        postprocess_fn=(lambda x:torch.from_numpy(x).float() if isinstance(x, np.ndarray) else torch.ones(1, 1).float()*x)
        return postprocess_fn(value)

    def store(self, exp_dict, actor_index=0):
        self.episode_buffer[actor_index].append(exp_dict)
        
        if not(exp_dict['non_terminal']):
            episode_length = len(self.episode_buffer[actor_index])
            per_episode_d2store = {}

            for idx in range(episode_length):
                s = self.episode_buffer[actor_index][idx]['s']
                a = self.episode_buffer[actor_index][idx]['a']
                r = self.episode_buffer[actor_index][idx]['r']
                succ_s = self.episode_buffer[actor_index][idx]['succ_s']
                non_terminal = self.episode_buffer[actor_index][idx]['non_terminal']

                info = self.episode_buffer[actor_index][idx]['info']
                rnn_states = self.episode_buffer[actor_index][idx]['rnn_states']
                
                #desired_goal = self.episode_buffer[actor_index][idx]['goals']['desired_goals']['s']
                #desired_goal = info['desired_goal']
                
                d2store = {
                    's':s, 
                    'a':a, 
                    'r':r, 
                    'succ_s':succ_s, 
                    'non_terminal':non_terminal, 
                    
                    'rnn_states': copy_hdict(rnn_states),
                    'info': info,
                    #'g':desired_goal
                }
                #self.algorithm.store(d2store, actor_index=actor_index)
                if -1 not in per_episode_d2store: per_episode_d2store[-1] = []
                per_episode_d2store[-1].append(d2store)

                if self.algorithm.summary_writer is not None and all(non_terminal<=0.5):
                    self.episode_count += 1
                    self.algorithm.summary_writer.add_scalar('PerEpisode/Success', 1+r.mean().item(), self.episode_count)
                

                for k in range(self.k):
                    d2store = {}
                    if 'final' in self.strategy:
                        raise NotImplementedError
                        #achieved_goal = self.episode_buffer[actor_index][-1]['goals']['achieved_goals']['s']
                        
                        achieved_exp = self.episode_buffer[actor_index][idx]
                        desired_exp = self.episode_buffer[actor_index][-1]
                        new_r, achieved_goal = self.goal_predicated_reward_fn(
                            achieved_exp=achieved_exp, 
                            desired_exp=desired_exp,
                            _extract_goals_from_info_fn=self._extract_goals_from_info
                        )
                        
                        new_non_terminal = torch.zeros(1) if all(new_r>-0.5) else torch.ones(1)
                        
                        d2store = {'s':s, 'a':a, 'r':new_r, 'succ_s':succ_s, 'non_terminal':new_non_terminal, 'g':achieved_goal}
                        
                        if self.algorithm.summary_writer is not None:
                            self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_final', new_r.mean().item(), self.algorithm.get_update_count())
                    
                    if 'future' in self.strategy:
                        future_idx = np.random.randint(idx, episode_length)
                        #achieved_goal = self.episode_buffer[actor_index][future_idx]['goals']['achieved_goals']['s']
                        
                        achieved_exp = self.episode_buffer[actor_index][idx]
                        desired_exp = self.episode_buffer[actor_index][future_idx]
                        new_r, achieved_goal = self.goal_predicated_reward_fn(
                            achieved_exp=achieved_exp, 
                            desired_exp=desired_exp,
                            _extract_goals_from_info_fn=self._extract_goals_from_info
                        )

                        new_non_terminal = torch.zeros_like(non_terminal) if all(new_r>-0.5) else torch.ones_like(non_terminal) 
                        d2store = {
                            's':s, 
                            'a':a, 
                            'r':new_r, 
                            'succ_s':succ_s, 
                            'non_terminal':new_non_terminal, 

                            'rnn_states': copy_hdict(
                                self._update_goals_in_rnn_states(
                                    hdict=rnn_states,
                                    goal_value=achieved_goal,
                                    goal_key='desired_goal',
                                )
                            ),
                            'info': info,
                            #'g':achieved_goal
                        }
                        
                        if self.algorithm.summary_writer is not None:
                            self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_future', new_r.mean().item(), self.algorithm.get_update_count())
                        
                    #self.algorithm.store(d2store, actor_index=actor_index)
                    if k not in per_episode_d2store: per_episode_d2store[k] = []
                    per_episode_d2store[k].append(d2store)
            
            for key in per_episode_d2store:
                for d2store in per_episode_d2store[key]:
                    self.algorithm.store(d2store, actor_index=actor_index)
                
            
            # Reset episode buffer:
            self.episode_buffer[actor_index] = []

    def clone(self):
        return HERAlgorithmWrapper2(
            algorithm=self.algorithm.clone(), 
            extra_inputs_infos=copy.deepcopy(self.extra_inputs_infos),
            strategy=self.strategy, 
            goal_predicated_reward_fn=self.goal_predicated_reward_fn,
        )