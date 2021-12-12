from typing import Dict, Optional

import torch
import numpy as np
import copy
from regym.rl_algorithms.algorithms.wrappers.algorithm_wrapper import AlgorithmWrapper
from regym.rl_algorithms.utils import copy_hdict

import wandb 


def state_eq_goal_reward_fn2(state, goal, epsilon=1e-3):
    if torch.abs(state-goal).mean() < epsilon:
        return torch.zeros(1), state
    else:
        return -torch.ones(1), state


def state_eq_goal_reward_fn2(
        achieved_exp, 
        target_exp, 
        _extract_goal_from_info_fn,
        goal_key="achieved_goal",
        epsilon=1e-3):
    """
    :param goal_key: Str that is the key to the goal value from the info dict.
        E.g. it is possible to use any key from the info dict as the goal.
    """
    state = achieved_exp['succ_s']
    #goal = desired_exp['goals']['achieved_goals']['s']
    #goal = _extract_goals_from_rnn_states_fn(desired_exp['info'], goal_key="achieved_goal")
    #goal = torch.from_numpy(desired_exp['info']['achieved_goal']).float()
    goal = _extract_goal_from_info_fn(target_exp['info'], goal_key=goal_key)
    if torch.abs(state-goal).mean() < epsilon:
        return torch.zeros((1,1)), goal
    else:
        return -torch.ones((1,1)), goal


def latent_based_goal_predicated_reward_fn2(
    achieved_exp, 
    target_exp, 
    _extract_goal_from_info_fn,
    goal_key="achieved_goal",
    epsilon=1e-3):
    
    achieved_goal = _extract_goal_from_info_fn(
        achieved_exp['succ_info'],
        goal_key=goal_key,
    )
    target_goal = _extract_goal_from_info_fn(
        target_exp['succ_info'],
        goal_key=goal_key,
    )
    abs_fn = torch.abs 
    if not(isinstance(achieved_goal, torch.Tensor)):    abs_fn = np.abs
    if abs_fn(achieved_goal-target_goal).mean() < epsilon:
        return torch.zeros((1,1)), target_goal
    else:
        return -torch.ones((1,1)), target_goal

class HERAlgorithmWrapper2(AlgorithmWrapper):
    def __init__(
        self, 
        algorithm, 
        extra_inputs_infos, 
        strategy="future-4", 
        goal_predicated_reward_fn=None,
        _extract_goal_from_info_fn=None,
        achieved_goal_key_from_info="achieved_goal",
        target_goal_key_from_info="target_goal",
        achieved_latent_goal_key_from_info=None,
        target_latent_goal_key_from_info=None,
        filtering_fn="None",
        ):
        """
        :param achieved_goal_key_from_info: Str of the key from the info dict
            used to retrieve the *achieved* goal from the *desired*/target
            experience's info dict.
        :param target_goal_key_from_info: Str of the key from the info dict
            used to replace the *target* goal into the HER-modified rnn/frame_states. 
        """
        super(HERAlgorithmWrapper2, self).__init__(algorithm=algorithm)
        if goal_predicated_reward_fn is None:   goal_predicated_reward_fn = state_eq_goal_reward_fn2
        if _extract_goal_from_info_fn is None:  _extract_goal_from_info_fn = self._extract_goal_from_info_default_fn

        self.extra_inputs_infos = extra_inputs_infos
        self.filtering_fn = filtering_fn 

        self.episode_buffer = [[] for i in range(self.algorithm.get_nbr_actor())]
        self.strategy = strategy
        assert( ('future' in self.strategy or 'final' in self.strategy) and '-' in self.strategy)
        self.k = int(self.strategy.split('-')[-1])    
        self.goal_predicated_reward_fn = goal_predicated_reward_fn
        self._extract_goal_from_info_fn = _extract_goal_from_info_fn
        self.achieved_goal_key_from_info = achieved_goal_key_from_info
        self.target_goal_key_from_info = target_goal_key_from_info
        self.achieved_latent_goal_key_from_info = achieved_latent_goal_key_from_info
        self.target_latent_goal_key_from_info = target_latent_goal_key_from_info

        self.episode_count = 0

    def _update_goals_in_rnn_states(
        self, 
        hdict:Dict, 
        goal_value:torch.Tensor, 
        latent_goal_value:Optional[torch.Tensor]=None,
        goal_key:Optional[str]='target_goal',
        latent_goal_key:Optional[str]=None,
        ):
        goals = {goal_key:goal_value}
        if latent_goal_key is not None: goals[latent_goal_key] = latent_goal_value
        for gkey, gvalue in goals.items():
            if gkey in self.extra_inputs_infos:
                if not isinstance(self.extra_inputs_infos[gkey]['target_location'][0], list):
                    self.extra_inputs_infos[gkey]['target_location'] = [self.extra_inputs_infos[gkey]['target_location']]
                for tl in self.extra_inputs_infos[gkey]['target_location']:
                    pointer = hdict
                    for child_node in tl:
                        if child_node not in pointer:
                            pointer[child_node] = {}
                        pointer = pointer[child_node]
                    pointer[gkey] = [gvalue]
        return hdict

    """
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
    """

    def _extract_goal_from_info_default_fn(
        self, 
        hdict:Dict, 
        goal_key:Optional[str]='achieved_goal',
        ):
        assert goal_key in hdict
        value = hdict[goal_key]
        postprocess_fn=(lambda x:torch.from_numpy(x).float() if isinstance(x, np.ndarray) else torch.ones(1, 1).float()*x)
        return postprocess_fn(value)

    def store(self, exp_dict, actor_index=0):
        self.episode_buffer[actor_index].append(exp_dict)
        
        if not(exp_dict['non_terminal']):
            self.episode_count += 1
            episode_length = len(self.episode_buffer[actor_index])
            per_episode_d2store = {}

            for idx in range(episode_length):
                s = self.episode_buffer[actor_index][idx]['s']
                a = self.episode_buffer[actor_index][idx]['a']
                r = self.episode_buffer[actor_index][idx]['r']
                succ_s = self.episode_buffer[actor_index][idx]['succ_s']
                non_terminal = self.episode_buffer[actor_index][idx]['non_terminal']

                info = self.episode_buffer[actor_index][idx]['info']
                succ_info = self.episode_buffer[actor_index][idx]['succ_info']
                rnn_states = self.episode_buffer[actor_index][idx]['rnn_states']
                # TODO: maybe consider handling the next_rnn_states ?

                d2store = {
                    's':s, 
                    'a':a, 
                    'r':r, 
                    'succ_s':succ_s, 
                    'non_terminal':non_terminal, 
                    
                    'rnn_states': copy_hdict(rnn_states),
                    'info': info,
                    'succ_info': succ_info,
                }
                
                # Not storing now:
                # we need to send whole trajectoriess all at once,
                # without HER-modified experiences in between...
                # previously: self.algorithm.store(d2store, actor_index=actor_index)
                
                # What is this -1 key?
                # per_episode_d2store is collecting for each k values (of the HER) algo
                # the HER-modified trajectories, where k is positive.
                # So, we also need to store the normal trajectory: this is done
                # at index k=-1 below:
                valid_exp = True
                if self.filtering_fn != "None":
                    achieved_exp = self.episode_buffer[actor_index][idx]
                    target_exp = self.episode_buffer[actor_index][-1]
                    _ , achieved_goal_from_target_exp, \
                    achieved_latent_goal_from_target_exp = self.goal_predicated_reward_fn(
                        achieved_exp=achieved_exp, 
                        target_exp=target_exp,
                        _extract_goal_from_info_fn=self._extract_goal_from_info_fn,
                        goal_key=self.achieved_goal_key_from_info,
                        latent_goal_key=self.achieved_latent_goal_key_from_info,
                    )
                    kwargs = {
                        "d2store":d2store,
                        "episode_buffer":self.episode_buffer[actor_index],
                        "achieved_goal_from_target_exp":achieved_goal_from_target_exp,
                        "achieved_latent_goal_from_target_exp":achieved_latent_goal_from_target_exp,
                    }
                    valid_exp = self.filtering_fn(**kwargs)
                if valid_exp:                    
                    if -1 not in per_episode_d2store: per_episode_d2store[-1] = []
                    per_episode_d2store[-1].append(d2store)

                if self.algorithm.summary_writer is not None and all(non_terminal<=0.5):
                    self.algorithm.summary_writer.add_scalar('PerEpisode/Success', 1+r.mean().item(), self.episode_count)
                if all(non_terminal<=0.5):
                    wandb.log({'PerEpisode/HER_Success': 1+r.mean().item()}, commit=False)
                

                for k in range(self.k):
                    d2store = {}
                    if 'final' in self.strategy:
                        #achieved_goal = self.episode_buffer[actor_index][-1]['goals']['achieved_goals']['s']
                        achieved_exp = self.episode_buffer[actor_index][idx]
                        target_exp = self.episode_buffer[actor_index][-1]
                        new_r, achieved_goal_from_target_exp, \
                        achieved_latent_goal_from_target_exp = self.goal_predicated_reward_fn(
                            achieved_exp=achieved_exp, 
                            target_exp=target_exp,
                            _extract_goal_from_info_fn=self._extract_goal_from_info_fn,
                            goal_key=self.achieved_goal_key_from_info,
                            latent_goal_key=self.achieved_latent_goal_key_from_info,
                        )
                        
                        #new_non_terminal = torch.zeros(1) if all(new_r>-0.5) else torch.ones(1)
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
                                    goal_value=achieved_goal_from_target_exp,
                                    latent_goal_value=achieved_latent_goal_from_target_exp,
                                    goal_key=self.target_goal_key_from_info,
                                    latent_goal_key=self.target_latent_goal_key_from_info,
                                )
                            ),
                            'info': info,
                            'succ_info': succ_info,
                            #'g':achieved_goal
                        }

                        if self.algorithm.summary_writer is not None:
                            self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_final', new_r.mean().item(), self.algorithm.get_update_count())
                        wandb.log({'PerUpdate/HER_reward_final': new_r.mean().item()}, commit=True)
                    
                    if 'future' in self.strategy:
                        raise NotImplementedError
                        future_idx = np.random.randint(idx, episode_length)
                        #achieved_goal = self.episode_buffer[actor_index][future_idx]['goals']['achieved_goals']['s']
                        
                        achieved_exp = self.episode_buffer[actor_index][idx]
                        target_exp = self.episode_buffer[actor_index][future_idx]
                        # TODO:
                        new_r, achieved_goal_from_target_exp = self.goal_predicated_reward_fn(
                            achieved_exp=achieved_exp, 
                            target_exp=target_exp,
                            _extract_goal_from_info_fn=self._extract_goal_from_info_fn,
                            goal_key=self.achieved_goal_key_from_info,
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
                                    goal_value=achieved_goal_from_target_exp,
                                    latent_goal_value=achieved_latent_goal_from_target_exp,
                                    goal_key=self.target_goal_key_from_info,
                                    latent_goal_key=self.target_latent_goal_key_from_info,
                                )
                            ),
                            'info': info,
                            'succ_info': succ_info,
                        }
                        
                        if self.algorithm.summary_writer is not None:
                            self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_future', new_r.mean().item(), self.algorithm.get_update_count())
                        wandb.log({'PerUpdate/HER_reward_future': new_r.mean().item()}, commit=True)
                        
                    #self.algorithm.store(d2store, actor_index=actor_index)
                    valid_exp = True
                    if self.filtering_fn != "None":
                        kwargs = {
                            "d2store":d2store,
                            "episode_buffer":self.episode_buffer[actor_index],
                            "achieved_goal_from_target_exp":achieved_goal_from_target_exp,
                            "achieved_latent_goal_from_target_exp":achieved_latent_goal_from_target_exp,
                        }
                        valid_exp = self.filtering_fn(**kwargs)
                    if not valid_exp:   continue
                    
                    if k not in per_episode_d2store: per_episode_d2store[k] = []
                    per_episode_d2store[k].append(d2store)
            
            # Now that we have all the different trajectories,
            # we can send them to the main algorithm as complete
            # whole trajectories, one experience at a time.
            for key in per_episode_d2store:
                for d2store in per_episode_d2store[key]:
                    self.algorithm.store(d2store, actor_index=actor_index)
                wandb.log({f'PerEpisode/HER_traj_length/{key}': len(per_episode_d2store[key])}, commit=False)
                # TODO: implement callback/hooks ...
                if key>=0:
                    wandb.log({'PerEpisode/HER_IGLU_nbr_blocks': (achieved_goal_from_target_exp>0).sum().item()})
            
            # Reset episode buffer:
            self.episode_buffer[actor_index] = []

    def clone(self, with_replay_buffer: bool=False, clone_proxies: bool=False, minimal=False):        
        return HERAlgorithmWrapper2(
            algorithm=self.algorithm.clone(
                with_replay_buffer=with_replay_buffer,
                clone_proxies=clone_proxies,
                minimal=minimal
            ), 
            extra_inputs_infos=copy.deepcopy(self.extra_inputs_infos),
            strategy=self.strategy, 
            goal_predicated_reward_fn=self.goal_predicated_reward_fn,
            _extract_goal_from_info_fn=self._extract_goal_from_info_fn,
            achieved_goal_key_from_info=self.achieved_goal_key_from_info,
            target_goal_key_from_info=self.target_goal_key_from_info,
        )
