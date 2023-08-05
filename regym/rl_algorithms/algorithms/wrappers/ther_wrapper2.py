from typing import Dict, Optional, List 

import time
from functools import partial
import copy

import torch
import torch.optim as optim 
import torch.nn as nn 

import numpy as np
from regym.rl_algorithms.algorithms.algorithm import Algorithm 
from regym.rl_algorithms.networks import random_sample

from regym.rl_algorithms.algorithms.wrappers.algorithm_wrapper import AlgorithmWrapper

from regym.rl_algorithms.algorithms.wrappers.her_wrapper2 import state_eq_goal_reward_fn2

from regym.rl_algorithms.replay_buffers import PrioritizedReplayStorage, SplitReplayStorage, SplitPrioritizedReplayStorage
from regym.rl_algorithms.utils import archi_concat_fn, _extract_rnn_states_from_batch_indices, _concatenate_hdict, _concatenate_list_hdict, copy_hdict

import wandb 
import pandas as pd


def predictor_based_goal_predicated_reward_fn2(
    predictor, 
    achieved_exp, 
    target_exp, 
    _extract_goal_from_info_fn=None, 
    goal_key="achieved_goal",
    latent_goal_key=None,
    epsilon=1e0,
    feedbacks={"failure":-1, "success":0},
    reward_shape=[1,1],
    ):
    '''
    Relabelling an unsuccessful trajectory, so the desired_exp's goal is not interesting.
    We want to know the goal that is achieved on the desired_exp succ_s / desired_state.
    
    Comparison between the predicted goal of the achieved state and the desired state
    tells us whether the achieved state is achieving the relabelling goal.

    Returns -1 for failure and 0 for success
    '''
    target_latent_goal = None 

    state = achieved_exp['succ_s']
    target_state = target_exp['succ_s']
    with torch.no_grad():
        training = predictor.training
        predictor.train(False)
        achieved_pred_goal = predictor(x=state).cpu()
        target_pred_goal = predictor(x=target_state).cpu()
        predictor.train(training)
    abs_fn = torch.abs
    dist = abs_fn(achieved_pred_goal-target_pred_goal).float().mean()
    if dist < epsilon:
        return feedbacks["success"]*torch.ones(reward_shape), target_pred_goal, target_latent_goal
    else:
        return feedbacks["failure"]*torch.ones(reward_shape), target_pred_goal, target_latent_goal


def batched_predictor_based_goal_predicated_reward_fn2(
    predictor, 
    achieved_exp:List[Dict[str,object]], 
    target_exp:List[Dict[str,object]], 
    _extract_goal_from_info_fn=None, 
    goal_key:str="achieved_goal",
    latent_goal_key:str=None,
    epsilon:float=1e0,
    feedbacks:Dict[str,float]={"failure":-1, "success":0},
    reward_shape:List[int]=[1,1],
    **kwargs:Dict[str,object],
    ):
    '''
    Relabelling an unsuccessful trajectory, so the desired_exp's goal is not interesting.
    We want to know the goal that is achieved on the desired_exp succ_s / desired_state.
    
    Comparison between the predicted goal of the achieved state and the desired state
    tells us whether the achieved state is achieving the relabelling goal.

    Returns -1 for failure and 0 for success
    '''
    target_latent_goal = None 

    state = torch.stack(
        [exp['succ_s'] for exp in achieved_exp],
        dim=0,
    )
    target_state = torch.stack(
        [exp['succ_s'] for exp in target_exp],
        dim=0,
    )
    
    rnn_states = _concatenate_list_hdict(
        lhds=[exp['next_rnn_states'] for exp in achieved_exp], 
        concat_fn=archi_concat_fn,
        preprocess_fn=(lambda x:x),
    )
    
    target_rnn_states = _concatenate_list_hdict(
        lhds=[exp['next_rnn_states'] for exp in target_exp], 
        concat_fn=archi_concat_fn,
        preprocess_fn=(lambda x:x),
    )
    
    with torch.no_grad():
        training = predictor.training
        predictor.train(False)
        achieved_pred_goal = predictor(x=state, rnn_states=rnn_states).cpu()
        target_pred_goal = predictor(x=target_state, rnn_states=target_rnn_states).cpu()
        predictor.train(training)
    abs_fn = torch.abs
    dist = abs_fn(achieved_pred_goal-target_pred_goal).float()
    while len(dist.shape) >1:
        dist = dist.mean(-1)
    reward_mask = dist < epsilon
    reward = reward_mask.unsqueeze(-1)*feedbacks["success"]*torch.ones(reward_shape)
    reward += (~reward_mask.unsqueeze(-1))*feedbacks["failure"]*torch.ones(reward_shape)
    return reward, target_pred_goal, target_latent_goal


class THERAlgorithmWrapper2(AlgorithmWrapper):
    def __init__(
        self, 
        algorithm, 
        extra_inputs_infos,
        predictor, 
        predictor_loss_fn, 
        strategy="future-4", 
        goal_predicated_reward_fn=None,
        _extract_goal_from_info_fn=None,
        achieved_goal_key_from_info="achieved_goal",
        target_goal_key_from_info="target_goal",
        achieved_latent_goal_key_from_info=None,
        target_latent_goal_key_from_info=None,
        filtering_fn="None",
        #rewards={'failure':-1, 'success':0}
        feedbacks={"failure":-1, "success":0},
        #rewards={'failure':0, 'success':1},
        relabel_terminal:Optional[bool]=True,
        filter_predicate_fn:Optional[bool]=False,
        filter_out_timed_out_episode:Optional[bool]=False,
        timing_out_episode_length_threshold:Optional[int]=40,
        episode_length_reward_shaping:Optional[bool]=False,
        train_contrastively:Optional[bool]=False,
        contrastive_training_nbr_neg_examples:Optional[int]=0,
        ):
        """
        :param achieved_goal_key_from_info: Str of the key from the info dict
            used to retrieve the *achieved* goal from the *desired*/target
            experience's info dict.
        :param target_goal_key_from_info: Str of the key from the info dict
            used to replace the *target* goal into the HER-modified rnn/frame_states. 
        """
        
        super(THERAlgorithmWrapper2, self).__init__(algorithm=algorithm)
        self.hook_fns = []
        self.semantic_cooccurrence_test = False 

        self.nbr_episode_success_range = 256

        if goal_predicated_reward_fn is None:   goal_predicated_reward_fn = state_eq_goal_reward_fn2
        if _extract_goal_from_info_fn is None:  _extract_goal_from_info_fn = self._extract_goal_from_info_default_fn

        self.goal_predicated_reward_fn_kwargs = {}

        self.extra_inputs_infos = extra_inputs_infos
        self.filtering_fn = filtering_fn 
        self.relabel_terminal = relabel_terminal
        self.filter_predicate_fn = filter_predicate_fn
        self.filter_out_timed_out_episode = filter_out_timed_out_episode
        self.timing_out_episode_length_threshold = timing_out_episode_length_threshold
        self.episode_length_reward_shaping = episode_length_reward_shaping
        self.train_contrastively = train_contrastively
        self.contrastive_training_nbr_neg_examples = contrastive_training_nbr_neg_examples
        self.contrastive_goal_value = None 
        # To be initialized with the first usable state encountered.

        #self.rewards = rewards 
        self.feedbacks = feedbacks 
        self.test_acc = 0.0

        self.predictor = predictor 
        
        self.use_oracle = self.predictor.use_oracle
        if self.kwargs['use_cuda']:
            self.predictor = self.predictor.cuda()
        self.best_predictor = self.predictor.clone()

        self.predictor_loss_fn = predictor_loss_fn
        #print(f"WARNING: THER loss_fn is {self.predictor_loss_fn}")
        
        # Tuning learning rate with respect to the number of actors:
        # Following: https://arxiv.org/abs/1705.04862
        lr = self.kwargs['THER_predictor_learning_rate'] 
        if isinstance(lr, str): lr = float(lr)
        if self.kwargs['lr_account_for_nbr_actor']:
            lr *= self.nbr_actor
        #print(f"THER Predictor Learning rate: {lr}")
        
        self.predictor_optimizer = optim.Adam(
            self.predictor.parameters(), 
            lr=lr, betas=(0.9,0.999), 
            eps=float(self.kwargs.get('ther_adam_eps')),
            weight_decay=float(self.kwargs.get("ther_adam_weight_decay", 0.0)),
        )
        self.best_predictor_optimizer_sd = self.predictor_optimizer.state_dict()

        self.predictor_storages = None 
        self._reset_predictor_storages()

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

        self.per_goal_episode_counts = {}
        columns = ["color_goal", "shape_goal", "semantic", "color", "shape"]
        self.co_occurrence_table = wandb.Table(columns=columns)
        self.idx2w = copy.deepcopy(self.predictor.model.modules['InstructionGenerator'].idx2w)

        self.episode_count = 0
        self.param_predictor_update_counter = 0

        self.nbr_buffered_predictor_experience = 0
        self.nbr_handled_predictor_experience = 0
        self.batch_size = self.kwargs['THER_predictor_batch_size']
        self.nbr_minibatches = self.kwargs['THER_predictor_nbr_minibatches']
        
        self.nbr_relabelled_traj = 0 
        self.nbr_successfull_traj = 0

    def _reset_predictor_storages(self):
        if self.predictor_storages is not None:
            for storage in self.predictor_storages: storage.reset()
       
        nbr_storages = 1  

        self.predictor_storages = []
        keys = ['s', 'a', 'r', 'non_terminal']
        if self.recurrent:  keys += ['rnn_states', 'next_rnn_states']
        
        '''
        circular_keys= {} #{'succ_s':'s'}
        circular_offsets= {} #{'succ_s':1}
        if self.recurrent:
            circular_keys.update({'next_rnn_states':'rnn_states'})
            #circular_offsets.update({'next_rnn_states':1})
        '''
        circular_keys= {} #{'succ_s':'s'}
        circular_offsets= {} #{'succ_s':1}
        keys.append('succ_s')
        
        beta_increase_interval = None
        if 'PER_beta_increase_interval' in self.kwargs and self.kwargs['PER_beta_increase_interval']!='None':
            beta_increase_interval = float(self.kwargs['PER_beta_increase_interval'])  

        for i in range(nbr_storages):
            if self.kwargs['THER_use_PER']:
                self.predictor_storages.append(
                    SplitPrioritizedReplayStorage(
                        capacity=int(self.kwargs['THER_replay_capacity']),
                        alpha=self.kwargs['THER_PER_alpha'],
                        beta=self.kwargs['THER_PER_beta'],
                        keys=keys,
                        circular_keys=circular_keys,
                        circular_offsets=circular_offsets,
                        test_train_split_interval=self.kwargs['THER_predictor_test_train_split_interval'],
                        test_capacity=int(self.kwargs['THER_test_replay_capacity']),
                        lock_test_storage=self.kwargs['THER_lock_test_storage'],
                    )
                )
            else:
                self.predictor_storages.append(
                    SplitReplayStorage(
                        capacity=int(self.kwargs['THER_replay_capacity']),
                        keys=keys,
                        circular_keys=circular_keys,
                        circular_offsets=circular_offsets,
                        test_train_split_interval=self.kwargs['THER_predictor_test_train_split_interval'],
                        test_capacity=int(self.kwargs['THER_test_replay_capacity']),
                        lock_test_storage=self.kwargs['THER_lock_test_storage'],
                    )
                )

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
        #################
        #################
        # Vocabulary logging:
        if not hasattr(self, "w2idx"):
            self.w2idx = self.predictor.model.modules['InstructionGenerator'].w2idx
            vocab_data = {"token_idx": list(self.w2idx.values()), "token": list(self.w2idx.keys())}
            vocab_df = pd.DataFrame(vocab_data)
            wandb.log({"VocabularyTable":wandb.Table(data=vocab_df),}, commit=True)
         
        # Semantic Co-Occurrence Data logging :
        if self.semantic_cooccurrence_test:
            COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
            IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

            OBJECT_TO_IDX = {
                "unseen": 0,
                "empty": 1,
                "wall": 2,
                "floor": 3,
                "door": 4,
                "key": 5,
                "ball": 6,
                "box": 7,
                "goal": 8,
                "lava": 9,
                "agent": 10,
            }
            IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))
        
            if 'symbolic_image' in exp_dict['info']:
                rnn_states = exp_dict['rnn_states']
                goal = rnn_states['phi_body']['extra_inputs']['desired_goal'][0][0]
                lang_goal = [self.idx2w[token.item()] for token in goal] 
                str_goal = ' '.join(lang_goal)
                color_goal = [word for word in lang_goal if word in COLOR_TO_IDX.keys()]
                if len(color_goal)==0:
                    color_goal = 'NA'
                else:
                    color_goal = color_goal[0]
                shape_goal = [word for word in lang_goal if word in OBJECT_TO_IDX.keys()]
                if len(shape_goal)==0:
                    shape_goal = 'object'
                else:
                    shape_goal = shape_goal[0]
                
                if str_goal not in self.per_goal_episode_counts:
                    self.per_goal_episode_counts[str_goal] = 0
                elif exp_dict['non_terminal'].item() == 0:
                    self.per_goal_episode_counts[str_goal] += 1

                symb_image = exp_dict['info']['symbolic_image']
                for i in range(symb_image.shape[0]):
                    for j in range(symb_image.shape[1]):
                        if symb_image[i,j,0] <= 3 : continue
                        color_idx = symb_image[i,j,1]
                        shape_idx = symb_image[i,j,0]
                        color = IDX_TO_COLOR[color_idx]
                        shape = IDX_TO_OBJECT[shape_idx]
                        data = [
                            color_goal,
                            shape_goal,
                            color,
                            color,
                            shape,
                        ]
                        self.co_occurrence_table.add_data(*data)
                        data = [
                            color_goal,
                            shape_goal,
                            shape,
                            color,
                            shape,
                        ]
                        self.co_occurrence_table.add_data(*data)
            
                count_threshold = 8
                if all([count >= count_threshold for lg, count in self.per_goal_episode_counts.items()]):
                    wandb.log({f"PerEpisode/SemanticCoOccurrenceTable": self.co_occurrence_table}, commit=False)
                    self.per_goal_episode_counts = {}
                    columns = ["color_goal", "shape_goal", "semantic", "color", "shape"]
                    self.co_occurrence_table = wandb.Table(columns=columns)
            
                wandb.log({
                    f"PerEpisode/SemanticCoOccurrenceCounts/{str_goal}":value
                    for str_goal, value in self.per_goal_episode_counts.items()
                    },
                    commit=False,
                )
        #################
        #################
        #################

        self.episode_buffer[actor_index].append(exp_dict)
        self.nbr_buffered_predictor_experience += 1

        successful_traj = False

        if not(exp_dict['non_terminal']):
            self.episode_count += 1
            episode_length = len(self.episode_buffer[actor_index])

            # Assumes non-successful rewards are non-positive:
            successful_traj = all(self.episode_buffer[actor_index][-1]['r']>0)
            if successful_traj: self.nbr_successfull_traj += 1

            # Relabelling if unsuccessfull trajectory:
            relabelling = not successful_traj
            
            episode_rewards = []
            her_rs = []
            per_episode_d2store = {}
            previous_d2stores = [] 

            for idx in range(episode_length):
                s = self.episode_buffer[actor_index][idx]['s']
                a = self.episode_buffer[actor_index][idx]['a']
                r = self.episode_buffer[actor_index][idx]['r']
                
                # Assumes failure rewards are non-positive:
                self.reward_shape = r.shape
                her_r = self.feedbacks['success']*torch.ones_like(r) if r.item()>0 else self.feedbacks['failure']*torch.ones_like(r)
                if self.episode_length_reward_shaping:
                    if her_r > 0:
                        her_r *= (1.0-float(idx)/self.timing_out_episode_length_threshold)

                succ_s = self.episode_buffer[actor_index][idx]['succ_s']
                non_terminal = self.episode_buffer[actor_index][idx]['non_terminal']

                info = self.episode_buffer[actor_index][idx]['info']
                succ_info = self.episode_buffer[actor_index][idx]['succ_info']
                rnn_states = self.episode_buffer[actor_index][idx]['rnn_states']
                next_rnn_states = self.episode_buffer[actor_index][idx]['next_rnn_states']
                
                episode_rewards.append(r)
                her_rs.append(her_r)

                d2store = {
                    's':s, 
                    'a':a, 
                    'r':her_r if self.kwargs['THER_use_THER'] else r, 
                    'succ_s':succ_s, 
                    'non_terminal':non_terminal, 
                    'rnn_states':copy_hdict(rnn_states),
                    'next_rnn_states':copy_hdict(next_rnn_states),
                    'info': info,
                    'succ_info': succ_info,
                }

                """
                if not(relabelling):
                    # Only insert this experience that way if successfull:
                    #self.algorithm.store(d2store, actor_index=actor_index)
                    if -1 not in per_episode_d2store: per_episode_d2store[-1] = []
                    per_episode_d2store[-1].append(d2store)
                """
                if -1 not in per_episode_d2store: per_episode_d2store[-1] = []
                per_episode_d2store[-1].append(d2store)
                
                for hook_fn in self.hook_fns:
                    hook_fn(
                        exp_dict=d2store,
                        actor_index=actor_index,
                        negative=False,
                        self=self,
                    )

                # Store data in predictor storages if successfull:
                if self.kwargs['THER_use_THER'] and r.item()>0:
                    if self.train_contrastively:
                        if self.contrastive_goal_value is None:
                            target_state = succ_s
                            with torch.no_grad():
                                training = self.predictor.training
                                self.predictor.train(False)
                                target_pred_goal = self.predictor(x=target_state).cpu()
                                self.predictor.train(training)
                            w2idx = self.predictor.model.modules['InstructionGenerator'].w2idx
                            # PADDING with EoS:
                            self.contrastive_goal_value = w2idx["EoS"]+0*target_pred_goal
                            self.contrastive_goal_value[..., 0] = w2idx["EoS"]
                        
                        for ctr_example_idx in range(self.contrastive_training_nbr_neg_examples):
                            if -ctr_example_idx-1 < -len(previous_d2stores) :    break
                            previous_d2stores[-ctr_example_idx-1]['rnn_states'] = copy_hdict(
                                self._update_goals_in_rnn_states(
                                    hdict=previous_d2stores[-ctr_example_idx-1]["rnn_states"],
                                    goal_value=self.contrastive_goal_value,
                                    latent_goal_value=None,
                                    goal_key=self.target_goal_key_from_info,
                                    latent_goal_key=self.target_latent_goal_key_from_info,
                                )
                            )
                            previous_d2stores[-ctr_example_idx-1]['next_rnn_states'] = copy_hdict(
                                self._update_goals_in_rnn_states(
                                    hdict=previous_d2stores[-ctr_example_idx-1]["next_rnn_states"],
                                    goal_value=self.contrastive_goal_value,
                                    latent_goal_value=None,
                                    goal_key=self.target_goal_key_from_info,
                                    latent_goal_key=self.target_latent_goal_key_from_info,
                                )
                            )
                            self.predictor_store(
                                previous_d2stores[-ctr_example_idx-1], 
                                actor_index=actor_index,
                                negative=True,
                            )

                    self.predictor_store(
                        d2store, 
                        actor_index=actor_index, 
                        negative=False,
                    )
                   
                    goals = rnn_states['phi_body']['extra_inputs']['desired_goal'][0]
                    idx2w = self.predictor.model.modules['InstructionGenerator'].idx2w
                    
                    if self.kwargs.get('THER_log_samples', False):
                        if not hasattr(self, "sample_table"):
                            columns = [f"gt_token{idx}" for idx in range(goals.shape[1])]
                            columns += ["stimulus_(t)", "stimulus_(t-1)"]
                            columns += [f"a_(t-{v})" for v in range(4)]
                            self.sample_table = wandb.Table(columns=columns) 
                    
                        for bidx in range(1):
                            if self.nbr_handled_predictor_experience % 16 != 0: continue
                            gt_word_sentence = [idx2w[token.item()] for token in goals[bidx]] 
                            nbr_frames = self.kwargs['task_config']['nbr_frame_stacking'] #succ_s[bidx].shape[0]//4
                            frame_depth = self.kwargs['task_config']['frame_depth']
                            stimulus_t = succ_s[bidx].cpu().reshape(nbr_frames,frame_depth,56,56).numpy()[:,:3]*255
                            stimulus_t = stimulus_t.astype(np.uint8)
                            stimulus_t = wandb.Video(stimulus_t, fps=1, format="gif")
                            stimulus_tm = s[bidx].cpu().reshape(nbr_frames,frame_depth,56,56).numpy()[:,:3]*255
                            stimulus_tm = stimulus_tm.astype(np.uint8)
                            stimulus_tm = wandb.Video(stimulus_tm, fps=1, format="gif")
                            previous_action_int = [
                                self.episode_buffer[actor_index][aidx]["rnn_states"]['critic_body']['extra_inputs']['previous_action_int'][0][bidx].cpu().item()
                                for aidx in [idx, idx-1, idx-2, idx-3]
                            ]
                            self.sample_table.add_data(*[
                                *gt_word_sentence,
                                stimulus_t,
                                stimulus_tm,
                                *previous_action_int
                                ]
                            )
    
                        if self.nbr_handled_predictor_experience % 128 == 0:
                            wandb.log({f"PerEpisode/SampleTable":self.sample_table}, commit=False)
                            columns = [f"gt_token{idx}" for idx in range(goals.shape[1])]
                            columns += ["stimulus_(t)", "stimulus_(t-1)"]
                            columns += [f"a_(t-{v})" for v in range(4)]
                            self.sample_table = wandb.Table(columns=columns) 

                    wandb.log({'Training/THER_Predictor/DatasetSize': self.nbr_handled_predictor_experience}, commit=False) # self.param_predictor_update_counter)
                    if self.algorithm.summary_writer is not None:
                        self.algorithm.summary_writer.add_scalar('Training/THER_Predictor/DatasetSize', self.nbr_handled_predictor_experience, self.param_predictor_update_counter)
                    
                    ####################################################################################
                    # Verification on Successful Episodes whether the Predictor can have high Recall:
                    ####################################################################################
                    batched_target_exp = [self.episode_buffer[actor_index][-1]]
                    batched_achieved_exp = self.episode_buffer[actor_index]
                    batched_new_r, batched_achieved_goal_from_target_exp, \
                    batched_achieved_latent_goal_from_target_exp = self.goal_predicated_reward_fn(
                        achieved_exp=batched_achieved_exp, 
                        target_exp=batched_target_exp,
                        _extract_goal_from_info_fn=self._extract_goal_from_info_fn,
                        goal_key=self.achieved_goal_key_from_info,
                        latent_goal_key=self.achieved_latent_goal_key_from_info,
                        epsilon=1e-1,
                        feedbacks=self.feedbacks,
                        reward_shape=self.reward_shape,
                        **self.goal_predicated_reward_fn_kwargs,
                    )
                    
                    positive_new_r_mask = (batched_new_r.detach() == self.feedbacks['success']).cpu().reshape(-1)
                    positive_new_r_step_positions = torch.arange(episode_length).masked_select(positive_new_r_mask)
                    positive_new_r_step_histogram = wandb.Histogram(positive_new_r_step_positions)

                    valid_hist_index = self.nbr_successfull_traj
                    wandb.log({
                        "PerEpisode/THER_Predicate/SuccessfulEpisodeValidationStepHistogram":positive_new_r_step_histogram, 
                        "PerEpisode/THER_Predicate/SuccessfulEpisodeGoalSimilarityRatioOverEpisode": positive_new_r_mask.float().sum()/episode_length,
                        "PerEpisode/THER_Predicate/SuccessfulEpisodeGoalSimilarityCount": positive_new_r_mask.float().sum(),
                        "PerEpisode/THER_Predicate/SuccessfulEpisodeLength": episode_length,
                        "PerEpisode/THER_Predicate/SuccessfulEpisodeValidationStepHistogramIndex": valid_hist_index,
                        }, 
                        commit=False,
                    )
                    ####################################################################################
                    ####################################################################################
                else:
                    previous_d2stores.append(d2store)
                    while len(previous_d2stores) > self.contrastive_training_nbr_neg_examples:
                        del previous_d2stores[0]

                #if all(non_terminal<=0.5) 
                if idx==(episode_length-1):
                    wandb.log({'PerEpisode/EpisodeLength': episode_length}, commit=False)
                    
                    wandb.log({
                        'PerEpisode/HER_Success': float(r.item()>0), #1+her_r.mean().item(),
                    }, commit=False) 
                    wandb.log({'PerEpisode/HER_FinalReward': her_r.mean().item()}, commit=False) 
                    wandb.log({'PerEpisode/HER_Return': sum(her_rs)}, commit=False)
                    wandb.log({'PerEpisode/HER_NormalizedReturn': sum(her_rs)/episode_length}, commit=False)
                    wandb.log({'PerEpisode/OriginalFinalReward': r.mean().item()}, commit=False)
                    wandb.log({'PerEpisode/OriginalReturn': sum(episode_rewards)}, commit=False)
                    wandb.log({'PerEpisode/OriginalNormalizedReturn': sum(episode_rewards)/episode_length}, commit=False) # self.episode_count)
                    if not hasattr(self, "nbr_success"):  self.nbr_success = 0
                    if successful_traj: self.nbr_success += 1
                    if self.episode_count % self.nbr_episode_success_range == 0:
                        wandb.log({
                            'PerEpisode/SuccessRatio': float(self.nbr_success)/self.nbr_episode_success_range,
                            'PerEpisode/SuccessRatioIndex': int(self.episode_count//self.nbr_episode_success_range),
                            },
                            commit=False,
                        ) # self.episode_count)
                        self.nbr_success = 0

                    if self.algorithm.summary_writer is not None:
                        self.algorithm.summary_writer.add_scalar('PerEpisode/Success', (self.rewards['success']==her_r).float().mean().item(), self.episode_count)
                        self.algorithm.summary_writer.add_histogram('PerEpisode/Rewards', episode_rewards, self.episode_count)

                
            # Are we relabelling?
            # Is it safe to use the predictor:
            safe_relabelling = self.test_acc >= self.kwargs['THER_predictor_accuracy_safe_to_relabel_threshold']
            # Is it a timed out episode that we should filter:
            timed_out_episode = episode_length >= self.timing_out_episode_length_threshold
            if self.filter_out_timed_out_episode:
                safe_relabelling = safe_relabelling and not(timed_out_episode)
            wandb.log({'PerEpisode/THER_Predicate/UnsuccessfulTraj': int(relabelling)}, commit=False)
            wandb.log({'PerEpisode/THER_Predicate/SafeRelabelling': int(safe_relabelling)}, commit=False)
            wandb.log({'PerEpisode/THER_Predicate/TimedOutEpisodeFiltering': int(timed_out_episode)}, commit=False)
            wandb.log({'PerEpisode/THER_Predicate/UnsuccessfulTraj+SafeRelabelling': int(relabelling and safe_relabelling)}, commit=False)
            wandb.log({'PerEpisode/THER_Predicate/UnsuccessfulTraj+NotTimedOut': int(relabelling and not(timed_out_episode))}, commit=False)
            wandb.log({'PerEpisode/THER_Predicate/PerformingRelabelling': int(relabelling and safe_relabelling)}, commit=False)
                

            if self.kwargs['THER_use_THER'] \
            and relabelling \
            and safe_relabelling:
                self.nbr_relabelled_traj += 1
                # Relabelling everything with the hindsight_goal computed on the fly, and set the reward accordingly:
                if 'final' in self.strategy:
                    batched_target_exp = [self.episode_buffer[actor_index][-1]]
                    batched_achieved_exp = self.episode_buffer[actor_index]
                    batched_new_r, batched_achieved_goal_from_target_exp, \
                    batched_achieved_latent_goal_from_target_exp = self.goal_predicated_reward_fn(
                        achieved_exp=batched_achieved_exp, 
                        target_exp=batched_target_exp,
                        _extract_goal_from_info_fn=self._extract_goal_from_info_fn,
                        goal_key=self.achieved_goal_key_from_info,
                        latent_goal_key=self.achieved_latent_goal_key_from_info,
                        epsilon=1e-1,
                        feedbacks=self.feedbacks,
                        reward_shape=self.reward_shape,
                        **self.goal_predicated_reward_fn_kwargs,
                    )
                    
                    positive_new_r_mask = (batched_new_r.detach() == self.feedbacks['success']).cpu().reshape(-1)
                    positive_new_r_step_positions = torch.arange(episode_length).masked_select(positive_new_r_mask)
                    positive_new_r_step_histogram = wandb.Histogram(positive_new_r_step_positions)
                    
                    hist_index = self.nbr_relabelled_traj
                    wandb.log({
                        "PerEpisode/THER_Predicate/StepHistogram": positive_new_r_step_histogram,
                        "PerEpisode/THER_Predicate/RelabelledEpisodeGoalSimilarityRatioOverEpisode": positive_new_r_mask.float().sum()/episode_length,
                        "PerEpisode/THER_Predicate/RelabelledEpisodeGoalSimilarityCount": positive_new_r_mask.float().sum(),
                        "PerEpisode/THER_Predicate/RelabelledEpisodeLength": episode_length,
                        "PerEpisode/THER_Predicate/StepHistogramIndex": hist_index,
                        }, 
                        commit=False,
                    )
                    
                    achieved_goal_from_target_exp = batched_achieved_goal_from_target_exp[0:1]
                    achieved_latent_goal_from_target_exp = batched_achieved_latent_goal_from_target_exp
                    if achieved_latent_goal_from_target_exp is not None:
                        achieved_latent_goal_from_target_exp = achieved_latent_goal_from_target_exp[0:1]
                    last_terminal_idx = 0
                    for idx in range(episode_length):    
                        s = self.episode_buffer[actor_index][idx]['s']
                        a = self.episode_buffer[actor_index][idx]['a']
                        r = self.episode_buffer[actor_index][idx]['r']
                        
                        new_r = batched_new_r[idx:idx+1]

                        succ_s = self.episode_buffer[actor_index][idx]['succ_s']
                        non_terminal = self.episode_buffer[actor_index][idx]['non_terminal']

                        info = self.episode_buffer[actor_index][idx]['info']
                        succ_info = self.episode_buffer[actor_index][idx]['succ_info']
                        rnn_states = self.episode_buffer[actor_index][idx]['rnn_states']
                        next_rnn_states = self.episode_buffer[actor_index][idx]['next_rnn_states']
                        
                        for k in range(self.k):
                            if self.filter_predicate_fn:
                                new_her_r = self.feedbacks['success'] if idx==(episode_length-1) else self.feedbacks['failure']
                            else:
                                new_her_r = new_r.item() #self.feedbacks['success']*torch.ones_like(r) if all(new_r>-0.5) else self.feedbacks['failure']*torch.ones_like(r)
                            if self.episode_length_reward_shaping:
                                if new_her_r > 0:
                                    reshaping_idx = idx-last_terminal_idx
                                    new_her_r *= (1.0-float(reshaping_idx)/self.timing_out_episode_length_threshold)
                            new_her_r = new_her_r*torch.ones_like(r)

                            if self.relabel_terminal:
                                if all(new_her_r>self.feedbacks['failure']):
                                    last_terminal_idx = idx
                                    new_non_terminal = torch.zeros_like(non_terminal)
                                else:
                                    new_non_terminal = torch.ones_like(non_terminal)
                            else:
                                new_non_terminal = non_terminal

                            d2store_her = {
                                's':s, 
                                'a':a, 
                                'r':new_her_r, 
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
                                'next_rnn_states': copy_hdict(
                                    self._update_goals_in_rnn_states(
                                        hdict=next_rnn_states,
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
                                self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_final', new_her_r.mean().item(), self.algorithm.get_update_count())
                                #self.algorithm.summary_writer.add_scalar('PerUpdate/HER_reward_dist', dist.mean().item(), self.algorithm.get_update_count())
                            #wandb.log({'PerUpdate/HER_AfterRelabellingReward': new_her_r.mean().item()}, commit=False)
                    
                            # Adding this relabelled experience to the replay buffer with 'proper' goal...
                            #self.algorithm.store(d2store_her, actor_index=actor_index)
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
                            per_episode_d2store[k].append(d2store_her)
                
                if 'future' in self.strategy:
                    raise NotImplementedError
                       
            # Now that we have all the different trajectories,
            # we can send them to the main algorithm as complete
            # whole trajectories, one experience at a time.
            for key in per_episode_d2store:
                for didx, d2store in enumerate(per_episode_d2store[key]):
                    self.algorithm.store(d2store, actor_index=actor_index)
                wandb.log({f'PerEpisode/HER_traj_length/{key}': len(per_episode_d2store[key])}, commit=False)
            # Reset the relevant episode buffer:
            self.episode_buffer[actor_index] = []

        self.update_predictor(successful_traj=successful_traj)
	   
    def predictor_store(self, exp_dict, actor_index=0, negative=False):
        # WARNING : multi storage is deprecated!
        actor_index = 0
        self.nbr_handled_predictor_experience += 1
        test_set = None
        if negative:    test_set = False
        if self.kwargs['THER_use_PER']:
            init_sampling_priority = None 
            self.predictor_storages[actor_index].add(exp_dict, priority=init_sampling_priority, test_set=test_set)
        else:
            self.predictor_storages[actor_index].add(exp_dict, test_set=test_set)

    def update_predictor(self, successful_traj=False):
        period_check = self.kwargs['THER_replay_period']
        period_count_check = self.nbr_buffered_predictor_experience
        
        # Update predictor:
        if not self.kwargs.get('THER_use_THER', False):
            return
        if not(self.nbr_handled_predictor_experience >= self.kwargs['THER_min_capacity']):
            return
        
        if not((period_count_check % period_check == 0) or (self.kwargs['THER_train_on_success'] and successful_traj)):
            return 
        
        full_update = True
        for it in range(self.kwargs['THER_nbr_training_iteration_per_update']):
            self.test_acc = self.train_predictor()
            if self.test_acc >= self.kwargs['THER_predictor_accuracy_threshold']:
                full_update = False
                break
        wandb.log({f"Training/THER_Predictor/FullUpdate":int(full_update)}, commit=False)
         
    def train_predictor(self, minibatch_size=None):
        if minibatch_size is None:  minibatch_size = self.batch_size

        start = time.time()
        #samples = self.retrieve_values_from_predictor_storages(minibatch_size=minibatch_size)
        samples = self.retrieve_values_from_predictor_storages(minibatch_size=self.nbr_minibatches*minibatch_size)
        end = time.time()
        
        wandb.log({'PerTHERPredictorUpdate/TimeComplexity/RetrieveValuesFn':  end-start}, commit=False) # self.param_update_counter)
        
        start = time.time()
        self.optimize_predictor(minibatch_size, samples)
        end = time.time()
        
        wandb.log({'PerTHERPredictorUpdate/TimeComplexity/OptimizeModelFn':  end-start}, commit=False) # self.param_update_counter)
        
        test_storage_size = self.predictor_storages[0].get_size(test=True) #.test_storage.current_size['s']  
        train_storage_size = self.predictor_storages[0].get_size(test=False) #test_storage.current_size['s']  
        wandb.log({'PerTHERPredictorUpdate/TestStorageSize':  test_storage_size}, commit=False)
        wandb.log({'PerTHERPredictorUpdate/TrainStorageSize':  train_storage_size}, commit=False)
        if test_storage_size > self.kwargs['THER_test_min_capacity']:
            #test_samples = self.retrieve_values_from_predictor_storages(minibatch_size=minibatch_size, test=True)
            test_samples = self.retrieve_values_from_predictor_storages(minibatch_size=self.nbr_minibatches*minibatch_size, test=True)
            with torch.no_grad():
                updated_acc = self.test_predictor( self.predictor, minibatch_size, test_samples)
                best_acc = self.test_predictor( self.best_predictor, minibatch_size, test_samples)
        else:
            updated_acc = 0.0
            best_acc = 0.0
        
        successful_update = int(updated_acc >= best_acc)
        wandb.log({f"Training/THER_Predictor/SuccessfulUpdate":successful_update}, commit=False)
        if not successful_update:
            self.predictor.load_state_dict(self.best_predictor.state_dict(), strict=False)
            self.predictor_optimizer.load_state_dict(self.best_predictor_optimizer_sd)
            acc = best_acc
        else:
            self.best_predictor.load_state_dict(self.predictor.state_dict(), strict=False)
            self.best_predictor_optimizer_sd = self.predictor_optimizer.state_dict()
            acc = updated_acc 

        wandb.log({'PerTHERPredictorUpdate/TestSentenceAccuracy': acc, "ther_predictor_update_count":self.param_predictor_update_counter}, commit=True)
        
        return acc 

    def retrieve_values_from_predictor_storages(self, minibatch_size, test=False):
        torch.set_grad_enabled(False)
        keys=['s', 'a', 'succ_s', 'r', 'non_terminal']

        fulls = {}
        
        if self.kwargs['THER_use_PER'] and not test:
            fulls['importanceSamplingWeights'] = []

        if self.recurrent:
            keys += ['rnn_states', 'next_rnn_states']
        
        for key in keys:    fulls[key] = []

        for storage in self.predictor_storages:
            # Check that there is something in the storage 
            if len(storage) <= 1: continue
            batch_size = minibatch_size
            if batch_size is None:
                batch_size = storage.get_size(test=test)

            if self.kwargs['THER_use_PER'] and not test:
                sample, importanceSamplingWeights = storage.sample(batch_size=batch_size, keys=keys, test=test)
                importanceSamplingWeights = torch.from_numpy(importanceSamplingWeights)
                fulls['importanceSamplingWeights'].append(importanceSamplingWeights)
            else:
                sample = storage.sample(
                    batch_size=batch_size, 
                    keys=keys, 
                    test=test,
                    replace=test,
                )
            
            values = {}
            for key, value in zip(keys, sample):
                value = value.tolist()
                if isinstance(value[0], dict):   
                    value = _concatenate_list_hdict(
                        lhds=value, 
                        concat_fn=archi_concat_fn,
                        preprocess_fn=(lambda x:x),
                        #map_keys=['hidden', 'cell']
                    )
                else:
                    value = torch.cat(value, dim=0)
                values[key] = value 

            for key, value in values.items():
                fulls[key].append(value)
        
        for key, value in fulls.items():
            if len(value) > 1:
                if isinstance(value[0], dict):
                    value = _concatenate_list_hdict(
                        lhds=value,
                        concat_fn=partial(torch.cat, dim=0),
                        preprocess_fn=(lambda x:x),
                    )
                else:
                    value = torch.cat(value, dim=0)
            else:
                value = value[0]
            fulls[key] = value

        return fulls

    def optimize_predictor(self, minibatch_size, samples):
        start = time.time()
        torch.set_grad_enabled(True)
        self.predictor.train(True)

        beta = self.predictor_storages[0].beta if self.kwargs['THER_use_PER'] else 1.0
        
        states = samples['s']
        actions = samples['a']
        next_states = samples['succ_s']
        rewards = samples['r']
        non_terminals = samples['non_terminal']
        goals = samples['g'] if 'g' in samples else None

        rnn_states = samples['rnn_states'] if 'rnn_states' in samples else None
        next_rnn_states = samples['next_rnn_states'] if 'next_rnn_states' in samples else None
        
        importanceSamplingWeights = samples['importanceSamplingWeights'] if 'importanceSamplingWeights' in samples else None

        # For each actor, there is one mini_batch update:
        sampler = list(random_sample(np.arange(states.size(0)), minibatch_size))
        nbr_minibatches = len(sampler)
        nbr_sampled_element_per_storage = self.nbr_minibatches*minibatch_size
        list_batch_indices = [storage_idx*nbr_sampled_element_per_storage+np.arange(nbr_sampled_element_per_storage) \
                                for storage_idx, storage in enumerate(self.predictor_storages)]
        '''
        list_batch_indices = [storage_idx*minibatch_size+np.arange(minibatch_size) \
                                for storage_idx, storage in enumerate(self.predictor_storages)]
        '''
        array_batch_indices = np.concatenate(list_batch_indices, axis=0)
        sampled_batch_indices = []
        sampled_losses_per_item = []
        
        self.predictor_optimizer.zero_grad()
        
        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()
            sampled_batch_indices.append(batch_indices)

            sampled_rnn_states = None
            sampled_next_rnn_states = None
            if self.recurrent:
                sampled_rnn_states = _extract_rnn_states_from_batch_indices(
                    rnn_states, 
                    batch_indices, 
                    use_cuda=self.kwargs['use_cuda'],
                )
                sampled_next_rnn_states = _extract_rnn_states_from_batch_indices(
                    next_rnn_states, 
                    batch_indices, 
                    use_cuda=self.kwargs['use_cuda'],
                )

            sampled_importanceSamplingWeights = None
            if importanceSamplingWeights is not None:
                sampled_importanceSamplingWeights = importanceSamplingWeights[batch_indices].cuda() if self.kwargs['use_cuda'] else importanceSamplingWeights[batch_indices]
            
            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_next_states = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]
            sampled_rewards = rewards[batch_indices].cuda() if self.kwargs['use_cuda'] else rewards[batch_indices]
            sampled_non_terminals = non_terminals[batch_indices].cuda() if self.kwargs['use_cuda'] else non_terminals[batch_indices]
            sampled_goals = None #DEPRECATED goals[batch_indices].cuda() if self.kwargs['use_cuda'] else goals[batch_indices]

            #self.predictor_optimizer.zero_grad()
            
            output_dict = self.predictor_loss_fn(sampled_states, 
                                          sampled_actions, 
                                          sampled_next_states,
                                          sampled_rewards,
                                          sampled_non_terminals,
                                          goals=sampled_goals,
                                          rnn_states=sampled_rnn_states,
                                          next_rnn_states=sampled_next_rnn_states,
                                          predictor=self.predictor,
                                          weights_decay_lambda=self.kwargs['THER_weights_decay_lambda'],
                                          use_PER=self.kwargs['THER_use_PER'],
                                          PER_beta=beta,
                                          importanceSamplingWeights=sampled_importanceSamplingWeights,
                                          iteration_count=self.param_predictor_update_counter,
                                          summary_writer=self.algorithm.summary_writer,
                                          phase="Training")
            
            loss = output_dict['loss']
            #loss_per_item = output_dict['loss_per_item']
            loss_per_item = output_dict['loss_per_item'].detach()
            
            
            
            if not self.use_oracle:
                (loss/nbr_minibatches).backward(retain_graph=False)
            '''
            loss.backward(retain_graph=False)
            if self.kwargs['THER_gradient_clip'] > 1e-3:
                nn.utils.clip_grad_norm_(self.predictor.parameters(), self.kwargs['THER_gradient_clip'])
            self.predictor_optimizer.step()
            '''

            if importanceSamplingWeights is not None:
                sampled_losses_per_item.append(loss_per_item)
                #wandb_data = copy.deepcopy(wandb.run.history._data)
                #wandb.run.history._data = {}
                wandb.log({
                    'PerTHERPredictorUpdate/ImportanceSamplingMean':  sampled_importanceSamplingWeights.cpu().mean().item(),
                    'PerTHERPredictorUpdate/ImportanceSamplingStd':  sampled_importanceSamplingWeights.cpu().std().item(),
                    'PerTHERPredictorUpdate/PER_Beta':  beta
                }) # self.param_update_counter)
                #wandb.run.history._data = wandb_data

            self.param_predictor_update_counter += 1 

        if self.kwargs['THER_gradient_clip'] > 1e-3:
            nn.utils.clip_grad_norm_(self.predictor.parameters(), self.kwargs['THER_gradient_clip'])
        self.predictor_optimizer.step()
        
        torch.set_grad_enabled(False)
        self.predictor.train(False)

        if importanceSamplingWeights is not None:
            # losses corresponding to sampled batch indices: 
            sampled_losses_per_item = torch.cat(sampled_losses_per_item, dim=0).cpu().detach().numpy()
            sampled_batch_indices = np.concatenate(sampled_batch_indices, axis=0)
            # let us align the batch indices with the losses:
            array_batch_indices = array_batch_indices[sampled_batch_indices]
            # Now we can iterate through the losses and retrieve what 
            # storage and what batch index they were associated with:
            for sloss, arr_bidx in zip(sampled_losses_per_item, array_batch_indices):
                storage_idx = arr_bidx//nbr_sampled_element_per_storage
                el_idx_in_batch = arr_bidx%nbr_sampled_element_per_storage
                #storage_idx = arr_bidx//minibatch_size
                #el_idx_in_batch = arr_bidx%minibatch_size
                el_idx_in_storage = self.predictor_storages[storage_idx].tree_indices[el_idx_in_batch]
                new_priority = self.predictor_storages[storage_idx].priority(sloss)
                self.predictor_storages[storage_idx].update(idx=el_idx_in_storage, priority=new_priority)

        end = time.time()
        wandb.log({'PerTHERPredictorUpdate/TimeComplexity/OptimizationLoss':  end-start}, commit=False) # self.param_update_counter)

    def test_predictor(self, predictor, minibatch_size, samples):
        training = predictor.training
        predictor.train(False)

        torch.set_grad_enabled(False)
        
        beta = self.predictor_storages[0].beta if self.kwargs['THER_use_PER'] else 1.0
        
        states = samples['s']
        actions = samples['a']
        next_states = samples['succ_s']
        rewards = samples['r']
        non_terminals = samples['non_terminal']
        goals = samples['g'] if 'g' in samples else None

        rnn_states = samples['rnn_states'] if 'rnn_states' in samples else None
        next_rnn_states = samples['next_rnn_states'] if 'next_rnn_states' in samples else None
        
        importanceSamplingWeights = samples['importanceSamplingWeights'] if 'importanceSamplingWeights' in samples else None

        # For each actor, there is one mini_batch update:
        sampler = random_sample(np.arange(states.size(0)), minibatch_size)
        list_batch_indices = [storage_idx*minibatch_size+np.arange(minibatch_size) \
                                for storage_idx, storage in enumerate(self.predictor_storages)]
        array_batch_indices = np.concatenate(list_batch_indices, axis=0)
        sampled_batch_indices = []
        sampled_losses_per_item = []

        running_acc = 0
        nbr_batches = 0
        for batch_indices in sampler:
            nbr_batches += 1
            batch_indices = torch.from_numpy(batch_indices).long()
            sampled_batch_indices.append(batch_indices)

            sampled_rnn_states = None
            sampled_next_rnn_states = None
            if self.recurrent:
                sampled_rnn_states = _extract_rnn_states_from_batch_indices(rnn_states, batch_indices, use_cuda=self.kwargs['use_cuda'])
                sampled_next_rnn_states = _extract_rnn_states_from_batch_indices(next_rnn_states, batch_indices, use_cuda=self.kwargs['use_cuda'])

            sampled_importanceSamplingWeights = None
            if importanceSamplingWeights is not None:
                sampled_importanceSamplingWeights = importanceSamplingWeights[batch_indices].cuda() if self.kwargs['use_cuda'] else importanceSamplingWeights[batch_indices]
            
            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_next_states = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]
            sampled_rewards = rewards[batch_indices].cuda() if self.kwargs['use_cuda'] else rewards[batch_indices]
            sampled_non_terminals = non_terminals[batch_indices].cuda() if self.kwargs['use_cuda'] else non_terminals[batch_indices]
            sampled_goals = None # DEPRECATED goals[batch_indices].cuda() if self.kwargs['use_cuda'] else goals[batch_indices]

            output_dict = self.predictor_loss_fn(sampled_states, 
                                          sampled_actions, 
                                          sampled_next_states,
                                          sampled_rewards,
                                          sampled_non_terminals,
                                          goals=sampled_goals,
                                          rnn_states=sampled_rnn_states,
                                          next_rnn_states=sampled_next_rnn_states,
                                          predictor=predictor,
                                          weights_decay_lambda=self.kwargs['THER_weights_decay_lambda'],
                                          use_PER=self.kwargs['THER_use_PER'],
                                          PER_beta=beta,
                                          importanceSamplingWeights=sampled_importanceSamplingWeights,
                                          iteration_count=self.param_predictor_update_counter,
                                          summary_writer=self.algorithm.summary_writer,
                                          phase="Testing")
            
            loss = output_dict['loss']
            loss_per_item = output_dict['loss_per_item']
            
            accuracy = output_dict['accuracy']
            running_acc = running_acc + accuracy

            if self.kwargs['THER_use_PER']:
                sampled_losses_per_item.append(loss_per_item)

        '''
        if importanceSamplingWeights is not None:
            # losses corresponding to sampled batch indices: 
            sampled_losses_per_item = torch.cat(sampled_losses_per_item, dim=0).cpu().detach().numpy()
            sampled_batch_indices = np.concatenate(sampled_batch_indices, axis=0)
            # let us align the batch indices with the losses:
            array_batch_indices = array_batch_indices[sampled_batch_indices]
            # Now we can iterate through the losses and retrieve what 
            # storage and what batch index they were associated with:
            for sloss, arr_bidx in zip(sampled_losses_per_item, array_batch_indices):
                storage_idx = arr_bidx//minibatch_size
                el_idx_in_batch = arr_bidx%minibatch_size
                el_idx_in_storage = self.predictor_storages[storage_idx].get_test_storage().tree_indices[el_idx_in_batch]
                new_priority = self.predictor_storages[storage_idx].priority(sloss)
                self.predictor_storages[storage_idx].update(idx=el_idx_in_storage, priority=new_priority, test=True)
        '''

        predictor.train(training)

        running_acc = running_acc / nbr_batches
        return running_acc

    def clone(self, with_replay_buffer: bool=False, clone_proxies: bool=False, minimal: bool=False):        
        cloned_algo = THERAlgorithmWrapper2(
            algorithm=self.algorithm.clone(
                with_replay_buffer=with_replay_buffer,
                clone_proxies=clone_proxies,
                minimal=minimal
            ), 
            extra_inputs_infos=self.extra_inputs_infos,
            predictor=self.predictor, 
            predictor_loss_fn=self.predictor_loss_fn, 
            strategy=self.strategy, 
            goal_predicated_reward_fn=self.goal_predicated_reward_fn,
            _extract_goal_from_info_fn=self._extract_goal_from_info_fn,
            achieved_goal_key_from_info=self.achieved_goal_key_from_info,
            target_goal_key_from_info=self.target_goal_key_from_info,
            achieved_latent_goal_key_from_info=self.achieved_latent_goal_key_from_info,
            target_latent_goal_key_from_info=self.target_latent_goal_key_from_info,
            filtering_fn=self.filtering_fn,
            feedbacks=self.feedbacks,
            relabel_terminal=self.relabel_terminal,
            filter_predicate_fn=self.filter_predicate_fn,
            filter_out_timed_out_episode=self.filter_out_timed_out_episode,
            timing_out_episode_length_threshold=self.timing_out_episode_length_threshold,
            episode_length_reward_shaping=self.episode_length_reward_shaping,
            train_contrastively=self.train_contrastively,
            contrastive_training_nbr_neg_examples=self.contrastive_training_nbr_neg_examples,
        )
        return cloned_algo

