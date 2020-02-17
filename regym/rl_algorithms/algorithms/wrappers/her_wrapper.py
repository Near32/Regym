import torch
import numpy as np
from .algorithm_wrapper import AlgorithmWrapper


def state_eq_goal_reward_fn(state, goal, epsilon=1e-3):
    if torch.abs(state-goal).mean() < epsilon:
        return torch.zeros(1), state
    else:
        return -torch.ones(1), state


def state_eq_goal_reward_fn(achieved_exp, desired_exp, epsilon=1e-3):
    state = achieved_exp['succ_s']
    goal = desired_exp['goals']['achieved_goals']['s']
    if torch.abs(state-goal).mean() < epsilon:
        return torch.zeros(1), goal
    else:
        return -torch.ones(1), goal


def latent_based_goal_predicated_reward_fn(achieved_exp, desired_exp, epsilon=1e-3):
    state = achieved_exp['info']['latents']['succ_s']
    goal = desired_exp['info']['latents']['achieved_goal']
    abs_fn = torch.abs 
    if not(isinstance(state, torch.Tensor)):    abs_fn = np.abs
    if abs_fn(state-goal).mean() < epsilon:
        return torch.zeros(1), desired_exp['goals']['achieved_goals']['s']
    else:
        return -torch.ones(1), desired_exp['goals']['achieved_goals']['s']


class HERAlgorithmWrapper(AlgorithmWrapper):
    def __init__(self, algorithm, strategy="future-4", goal_predicated_reward_fn=None):
        super(HERAlgorithmWrapper, self).__init__(algorithm=algorithm)
        if goal_predicated_reward_fn is None:   goal_predicated_reward_fn = state_eq_goal_reward_fn

        self.episode_buffer = [[] for i in range(self.algorithm.get_nbr_actor())]
        self.strategy = strategy
        assert( ('future' in self.strategy or 'final' in self.strategy) and '-' in self.strategy)
        self.k = int(self.strategy.split('-')[-1])    
        self.goal_predicated_reward_fn = goal_predicated_reward_fn
        self.episode_count = 0

    def store(self, exp_dict, actor_index=0):
        self.episode_buffer[actor_index].append(exp_dict)
        
        if not(exp_dict['non_terminal']):
            episode_length = len(self.episode_buffer[actor_index])
            for idx in range(episode_length):
                s = self.episode_buffer[actor_index][idx]['s']
                a = self.episode_buffer[actor_index][idx]['a']
                r = self.episode_buffer[actor_index][idx]['r']
                succ_s = self.episode_buffer[actor_index][idx]['succ_s']
                non_terminal = self.episode_buffer[actor_index][idx]['non_terminal']

                desired_goal = self.episode_buffer[actor_index][idx]['goals']['desired_goals']['s']
                
                d2store = {'s':s, 'a':a, 'r':r, 'succ_s':succ_s, 'non_terminal':non_terminal, 'g':desired_goal}
                self.algorithm.store(d2store, actor_index=actor_index)
                
                if self.algorithm.summary_writer is not None and all(non_terminal<=0.5):
                    self.episode_count += 1
                    self.algorithm.summary_writer.add_scalar('PerEpisode/Success', 1+r.mean().item(), self.episode_count)
                

                for k in range(self.k):
                    d2store = {}
                    if 'final' in self.strategy:
                        #achieved_goal = self.episode_buffer[actor_index][-1]['goals']['achieved_goals']['s']
                        
                        achieved_exp = self.episode_buffer[actor_index][idx]
                        desired_exp = self.episode_buffer[actor_index][-1]
                        new_r, achieved_goal = self.goal_predicated_reward_fn(achieved_exp=achieved_exp, desired_exp=desired_exp)
                        
                        new_non_terminal = torch.zeros(1) if all(new_r>-0.5) else torch.ones(1)
                        
                        d2store = {'s':s, 'a':a, 'r':new_r, 'succ_s':succ_s, 'non_terminal':new_non_terminal, 'g':achieved_goal}
                        
                        if self.algorithm.summary_writer is not None:
                            self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_final', new_r.mean().item(), self.algorithm.get_update_count())
                    
                    if 'future' in self.strategy:
                        future_idx = np.random.randint(idx, episode_length)
                        #achieved_goal = self.episode_buffer[actor_index][future_idx]['goals']['achieved_goals']['s']
                        
                        achieved_exp = self.episode_buffer[actor_index][idx]
                        desired_exp = self.episode_buffer[actor_index][future_idx]
                        new_r, achieved_goal = self.goal_predicated_reward_fn(achieved_exp=achieved_exp, desired_exp=desired_exp)
                        
                        new_non_terminal = torch.zeros(1) if all(new_r>-0.5) else torch.ones(1) 
                        d2store = {'s':s, 'a':a, 'r':new_r, 'succ_s':succ_s, 'non_terminal':new_non_terminal, 'g':achieved_goal}
                        
                        if self.algorithm.summary_writer is not None:
                            self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_future', new_r.mean().item(), self.algorithm.get_update_count())
                        
                    self.algorithm.store(d2store, actor_index=actor_index)

            # Reset episode buffer:
            self.episode_buffer[actor_index] = []

    def clone(self):
        return HERAlgorithmWrapper(algorithm=self.algorithm.clone(), 
                                   strategy=self.strategy, 
                                   goal_predicated_reward_fn=self.goal_predicated_reward_fn)