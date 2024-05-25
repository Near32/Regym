from typing import Dict, Optional, List 

import os
import time
import copy
import wandb 
from functools import partial

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
import numpy as np
import pandas as pd

from regym.rl_algorithms.algorithms.wrappers.algorithm_wrapper import AlgorithmWrapper
from regym.rl_algorithms.algorithms.wrappers.ther_wrapper2 import batched_predictor_based_goal_predicated_reward_fn2
from regym.rl_algorithms.algorithms.wrappers.ether_wrapper import GaussianBlur, SplitImg

from regym.rl_algorithms.replay_buffers import PrioritizedReplayStorage, SplitReplayStorage, SplitPrioritizedReplayStorage
from regym.rl_algorithms.utils import archi_concat_fn, _extract_rnn_states_from_batch_indices, _concatenate_hdict, _concatenate_list_hdict, copy_hdict

import ReferentialGym
from ReferentialGym.datasets import DemonstrationDataset
from ReferentialGym.agents import DiscriminativeListener, LSTMCNNListener


# Adapted from: 
# https://github.com/facebookresearch/EGG/blob/424c9aa2d56f9d5cc17e78f0ba94e1b7a9810add/egg/zoo/language_bottleneck/intervention.py#L37
def ht(t):
    if isinstance(t, tuple):
        return t
    if isinstance(t, int):
        return t
    if isinstance(t, list):
        return tuple(t)

    try:
        t = t.item()
    except:
        t = tuple(t.reshape(-1).tolist())
    
    return t


class ELAAlgorithmWrapper(AlgorithmWrapper):
    def __init__(
        self, 
        algorithm, 
        predictor, 
        extrinsic_weight=10.0,
        intrinsic_weight=0.5,
        feedbacks={"failure":0, "success":1},
    ):
        """
        """
        
        super(ELAAlgorithmWrapper, self).__init__(
            algorithm=algorithm,
        )
        
        self.hook_fns = []
        self.nbr_episode_success_range = 32 #256
        self.feedbacks = feedbacks 
        
        self.extrinsic_weight = extrinsic_weight
        self.intrinsic_weight = intrinsic_weight

        self.test_acc = 0.0
        self.predictor = predictor 
        if self.kwargs['use_cuda']:
            self.predictor = self.predictor.cuda()
        self.best_predictor = self.predictor.clone()

        self.episode_buffer = [[] for i in range(self.algorithm.unwrapped.get_nbr_actor())]

        self.idx2w = copy.deepcopy(self.predictor.model.modules['CaptionGenerator'].idx2w)

        self.episode_count = 0
        self.param_predictor_update_counter = 0

        self.nbr_buffered_predictor_experience = 0
        self.nbr_handled_predictor_experience = 0
        
        self.nbr_relabelled_traj = 0 
        self.nbr_successfull_traj = 0

        self.rg_storages = None
        self._reset_rg_storages()
        self.hook_fns.append(
            ELAAlgorithmWrapper.referential_game_store,
        )
        self.non_unique_data = 0
        self.nbr_data = 0 

        self.rg_iteration = 0
        
        self.init_referential_game()
        
    def _reset_rg_storages(self):
        if self.rg_storages is not None:
            for storage in self.rg_storages: storage.reset()
       
        nbr_storages = 1  

        self.rg_storages = []
        keys = ['s', 'a', 'r', 'non_terminal']
        keys += ['info']
        if self.recurrent:  keys += ['rnn_states', 'next_rnn_states']
        
        circular_keys= {} #{'succ_s':'s'}
        circular_offsets= {} #{'succ_s':1}
        keys.append('succ_s')
        
        beta_increase_interval = None
        if 'ELA_rg_PER_beta_increase_interval' in self.kwargs and self.kwargs['ELA_rg_PER_beta_increase_interval']!='None':
            beta_increase_interval = float(self.kwargs['ELA_rg_PER_beta_increase_interval'])  

        for i in range(nbr_storages):
            if self.kwargs.get('ELA_use_PER', False):
                raise NotImplementedError
            else:
                self.rg_storages.append(
                    SplitReplayStorage(
                        capacity=int(self.kwargs['ELA_replay_capacity']),
                        keys=keys,
                        circular_keys=circular_keys,
                        circular_offsets=circular_offsets,
                        test_train_split_interval=self.kwargs['ELA_test_train_split_interval'],
                        test_capacity=int(self.kwargs['ELA_test_replay_capacity']),
                        lock_test_storage=self.kwargs['ELA_lock_test_storage'],
                    )
                )

    def referential_game_store(self, exp_dict, actor_index=0, negative=False):
        # TODO: implement multi storage approach
        # No, it is not necessary, because the function is called on consecutive states,
        # all coming from the actor, until end is reached, and then one episode at a time
        actor_index = 0
        
        if not hasattr(self, "nbr_handled_rg_experience"):
            self.nbr_handled_rg_experience = 0
        self.nbr_handled_rg_experience += 1
        
        '''
        No longer putting any data into the split test set as it is
        messing about with the compactness-ambiguity metric visualisation.
        '''
        '''
        test_set = None
        if negative:    test_set = False
        '''
        test_set = False 

        '''
        if self.kwargs['ELA_use_PER']:
            init_sampling_priority = None 
            self.rg_storages[actor_index].add(exp_dict, priority=init_sampling_priority, test_set=test_set)
        else:
        '''
        self.nbr_data += 1 
        wandb.log({f"Training/ELA/NbrData":self.nbr_data}, commit=False)
         
        if "symbolic_image" in exp_dict['info']:
            if not hasattr(self, 'uniqueObs2Hist'): self.uniqueObs2Hist = {}
            unique = True
            symbolicRepr = ht(exp_dict['info']['symbolic_image'])
            currentReprPosition = self.rg_storages[actor_index].position['info']
            lastSymbolicRepr = self.rg_storages[actor_index].info[0][currentReprPosition]
            currentTestReprPosition = self.rg_storages[actor_index].test_storage.position['info']
            lastTestSymbolicRepr = self.rg_storages[actor_index].test_storage.info[0][currentTestReprPosition]
            if isinstance(lastSymbolicRepr, dict):
                lastSymbolicRepr = lastSymbolicRepr.get('symbolic_image', None)
            else:
                lastSymbolicRepr = None
            if isinstance(lastTestSymbolicRepr, dict):
                lastTestSymbolicRepr = lastTestSymbolicRepr.get('symbolic_image', None)
            else:
                lastTestSymbolicRepr = None
            #for idx in range(len(self.rg_storages[actor_index])):
            for otherRepr in self.uniqueObs2Hist.keys():
                #if all((self.rg_storages[actor_index].info[0][idx]['symbolic_image'] == symbolicRepr).reshape(-1)):
                if otherRepr == symbolicRepr:
                    self.non_unique_data += 1
                    #self.nbr_data = self.rg_storages[actor_index].get_size()+self.rg_storages[actor_index].get_size(test=True)
                    unique = False
                    break
            
            if unique:
                self.uniqueObs2Hist[symbolicRepr] = 1
            elif not self.kwargs['ELA_rg_filter_out_non_unique']:
                self.uniqueObs2Hist[symbolicRepr] += 1
            
            nonNullOcc = [val for val in self.uniqueObs2Hist.values() if val > 0]
            values = np.asarray(nonNullOcc)
            max_value = values.max()
            min_value = values.min()
            averaged_value = values.mean()
            std_value = values.std()
            median_value = np.nanpercentile(
              values,
              q=50,
              axis=None,
              method="nearest"
            )
            q1_value = np.nanpercentile(
              values,
              q=25,
              axis=None,
              method="lower"
            )
            q3_value = np.nanpercentile(
              values,
              q=75,
              axis=None,
              method="higher"
            )
            iqr = q3_value-q1_value
            
            totalData = sum(values)
            medianFreq = float(median_value)/totalData 
            meanFreq = float(averaged_value)/totalData
            maxFreq = float(max_value)/totalData
            q3Freq = float(q3_value)/totalData
            wandb.log({f"Training/ELA/NonNullOccurrences/Mean": averaged_value}, commit=False)
            wandb.log({f"Training/ELA/NonNullOccurrences/FreqMean": meanFreq}, commit=False)
            wandb.log({f"Training/ELA/NonNullOccurrences/FreqMedian": medianFreq}, commit=False)
            wandb.log({f"Training/ELA/NonNullOccurrences/FreqMax": maxFreq}, commit=False)
            wandb.log({f"Training/ELA/NonNullOccurrences/FreqQ3": q3Freq}, commit=False)
            wandb.log({f"Training/ELA/NonNullOccurrences/Std": std_value}, commit=False)
            wandb.log({f"Training/ELA/NonNullOccurrences/Median": median_value}, commit=False)
            wandb.log({f"Training/ELA/NonNullOccurrences/Max": max_value}, commit=False)
            wandb.log({f"Training/ELA/NonNullOccurrences/Min": min_value}, commit=False)
            wandb.log({f"Training/ELA/NonNullOccurrences/Q1": q1_value}, commit=False)
            wandb.log({f"Training/ELA/NonNullOccurrences/Q3": q3_value}, commit=False)
            wandb.log({f"Training/ELA/NonNullOccurrences/IQR": iqr}, commit=False)
            wandb.log({f"Training/ELA/NonUniqueDataRatio":float(self.non_unique_data)/(self.nbr_data+1)}, commit=False)
            wandb.log({f"Training/ELA/NonUniqueDataNbr": self.non_unique_data}, commit=False)
        
            if self.kwargs['ELA_rg_filter_out_non_unique'] \
            and not unique:  
                return

        self.rg_storages[actor_index].add(exp_dict, test_set=test_set)
        
        if "symbolic_image" in exp_dict['info']:
            # Regularising:
            if self.rg_storages[actor_index].latest_addition_to_test_set \
            and lastTestSymbolicRepr is not None:
                self.uniqueObs2Hist[ht(lastTestSymbolicRepr)] -= 1
            elif lastSymbolicRepr is not None:
                self.uniqueObs2Hist[ht(lastSymbolicRepr)] -= 1
        
        return
             
    def compute_captions(
        self,
        exp:List[Dict[str,object]], 
        feedbacks:Dict[str,float]={"failure":-1, "success":0},
        reward_shape:List[int]=[1,1],
        **kwargs:Dict[str,object],
    ):
        '''
        '''
        episode_length = len(exp)
        state = torch.stack(
            [e['succ_s'] for e in exp],
            dim=0,
        )
        
        rnn_states = _concatenate_list_hdict(
            lhds=[e['next_rnn_states'] for e in exp], 
            concat_fn=archi_concat_fn,
            preprocess_fn=(lambda x:x),
        )
        with torch.no_grad():
            training = self.predictor.training
            self.predictor.train(False)
            prediction = self.predictor(
                x=state, 
                rnn_states=rnn_states,
            )
            captions = prediction['output'][0].cpu()
            self.predictor.train(training)
        
        ## Logging perplexity:
        wandb_dict = {}
        metrics = {
            'caption_perplexity': prediction['next_rnn_states']['CaptionGenerator']['input0_prediction_perplexities'][0].cpu().numpy(),
            'caption_likelihood': prediction['next_rnn_states']['CaptionGenerator']['input0_prediction_likelihoods'][0].cpu().numpy(),
        }
        for k in metrics.keys():
            hist = metrics[k]
            median = np.median(hist)
            mean = np.mean(hist)
            std = np.std(hist)
            minv = np.min(hist)
            maxv = np.max(hist)
            wandb_dict[f"PerEpisode/{k}/Max"] = maxv
            wandb_dict[f"PerEpisode/{k}/Min"] = minv
            wandb_dict[f"PerEpisode/{k}/Median"] = median
            wandb_dict[f"PerEpisode/{k}/Mean"] = mean
            wandb_dict[f"PerEpisode/{k}/Std"] = std
        wandb.log(
            wandb_dict,
            commit=False,
        )
        
        visited_captions = []
        reward_mask = torch.zeros(episode_length)
        for idx, caption in enumerate(captions):
            if caption.tolist() not in visited_captions:
                visited_captions.append(caption.tolist())
                reward_mask[idx] = 1
        reward_mask = reward_mask.bool()
        feedbacks_type =  self.kwargs.get('ELA_feedbacks_type', 'normal')
        if 'hurry' in feedbacks_type:
            assert '-' in feedbacks_type
            max_episode_length = int(feedbacks_type.split('-')[-1])
            feedbacks_type = 'hurry'
        if feedbacks_type == 'normal':
            reward = reward_mask.unsqueeze(-1)*feedbacks["success"]*torch.ones(reward_shape)
        elif feedbacks_type == 'hurry':
            assert feedbacks['success'] > 0.0
            reward = (1.0-torch.arange(episode_length).float()/max_episode_length).clamp(min=0.1).unsqueeze(-1)
            reward *= reward_mask.unsqueeze(-1)*feedbacks["success"]*torch.ones(reward_shape)
        else:
            raise NotImplementedError
        reward += (~reward_mask.unsqueeze(-1))*feedbacks["failure"]*torch.ones(reward_shape)
        return reward, captions
    
    def record_metrics(self, exp_dict, actor_index=0):
        if not hasattr(self, 'per_actor_metrics'):
            self.per_actor_metrics = {} 

        last_info = exp_dict['succ_info']
        metrics = last_info.get('metrics', None)
        if metrics is None:
            return
        if actor_index not in self.per_actor_metrics:
            self.per_actor_metrics[actor_index] = []
        self.per_actor_metrics[actor_index].append(metrics)
        
        # Logging new values over actor list :
        wandb_dict = {}
        '''
        for k in metrics.keys():
            hist = [self.per_actor_metrics[actor_index][i][k] 
                    for i in range(len(self.per_actor_metrics[actor_index]))]
            #wandb_hist = wandb.Histogram(hist)
            mean = np.mean(hist)
            std = np.std(hist)
            minv = min(hist)
            maxv = max(hist)
            wandb_dict[f"PerActor/Metrics/{actor_index}_{k}_Max"] = max
            wandb_dict[f"PerActor/Metrics/{actor_index}_{k}_Min"] = min
            wandb_dict[f"PerActor/Metrics/{actor_index}_{k}_Mean"] = mean
            wandb_dict[f"PerActor/Metrics/{actor_index}_{k}_Std"] = std
        wandb.log(
            wandb_dict,
            commit=False,
        )
        '''
        if len(self.per_actor_metrics[actor_index]) >= 32:
            self.per_actor_metrics[actor_index].pop(0)
        
        # Logging new values, independantly of actor:
        if not hasattr(self, 'metrics_counter'):
            self.metrics_counter = -1
        self.metrics_counter += 1
        wandb_dict = {}
        wandb_dict['PerEpisode/Metrics/MetricsIndex'] = self.metrics_counter
        for k,v in metrics.items():
            wandb_dict[f"PerEpisode/Metrics/{k}"] = v
        
        wandb.log(
            wandb_dict,
            commit=False,#True,
        )
        return 
    
    def store(self, exp_dict, actor_index=0):
        #################
        #################
        # Vocabulary logging:
        if not hasattr(self, "w2idx"):
            self.w2idx = self.predictor.model.modules['CaptionGenerator'].w2idx
            vocab_data = {"token_idx": list(self.w2idx.values()), "token": list(self.w2idx.keys())}
            vocab_df = pd.DataFrame(vocab_data)
            wandb.log({"VocabularyTable":wandb.Table(data=vocab_df),}, commit=True)
         
        self.episode_buffer[actor_index].append(exp_dict)
        self.nbr_buffered_predictor_experience += 1

        successful_traj = False

        if not(exp_dict['non_terminal']):
            self.record_metrics(exp_dict, actor_index=actor_index)

            self.episode_count += 1
            episode_length = len(self.episode_buffer[actor_index])
            self.reward_shape = exp_dict['r'].shape

            # Assumes non-successful rewards are non-positive:
            successful_traj = all(self.episode_buffer[actor_index][-1]['r']>self.kwargs['success_threshold'])
            if successful_traj: self.nbr_successfull_traj += 1

            episode_rewards = []
            per_episode_d2store = {}
            previous_d2stores = [] 

            self.nbr_relabelled_traj += 1
            if self.kwargs['ELA_use_ELA']:
                batched_exp = self.episode_buffer[actor_index]
                batched_new_r, batched_captions = self.compute_captions(
                    exp=batched_exp, 
                    feedbacks=self.feedbacks,
                    reward_shape=self.reward_shape,
                )
                positive_new_r_mask = (batched_new_r.detach() == self.feedbacks['success']).cpu().reshape(-1)
            else:
                batched_new_r = None
                batched_r = torch.cat([
                    self.episode_buffer[actor_index][idx]['r']
                    for idx in range(episode_length)],
                    dim=0,
                )
                feedbacks_type =  self.kwargs.get('ELA_feedbacks_type', 'normal')
                if 'hurry' in feedbacks_type:
                    assert '-' in feedbacks_type
                    max_episode_length = int(feedbacks_type.split('-')[-1])
                    feedbacks_type = 'hurry'
                
                if feedbacks_type == 'normal':
                    pass #reward = reward_mask.unsqueeze(-1)*feedbacks["success"]*torch.ones(reward_shape)
                elif feedbacks_type == 'hurry':
                    reward_scaler = (1.0-torch.arange(episode_length).float()/max_episode_length).clamp(min=0.1).unsqueeze(-1)
                    batched_r *= reward_scaler
                else:
                    raise NotImplementedError
                positive_new_r_mask = (batched_r > 0.0).reshape(-1)
                
            positive_new_r_step_positions = torch.arange(episode_length).masked_select(positive_new_r_mask)
            positive_new_r_step_histogram = wandb.Histogram(positive_new_r_step_positions)
            
            hist_index = self.nbr_relabelled_traj
            wandb.log({
                "PerEpisode/ELA_Predicate/StepHistogram": positive_new_r_step_histogram,
                "PerEpisode/ELA_Predicate/RelabelledEpisodeGoalSimilarityRatioOverEpisode": positive_new_r_mask.float().sum()/episode_length,
                "PerEpisode/ELA_Predicate/RelabelledEpisodeGoalSimilarityCount": positive_new_r_mask.float().sum(),
                "PerEpisode/ELA_Predicate/RelabelledEpisodeLength": episode_length,
                "PerEpisode/ELA_Predicate/StepHistogramIndex": hist_index,
                }, 
                commit=False,
            )
            
            new_rs = []
            for idx in range(episode_length):
                s = self.episode_buffer[actor_index][idx]['s']
                a = self.episode_buffer[actor_index][idx]['a']
                r = self.episode_buffer[actor_index][idx]['r']
                
                new_r = self.extrinsic_weight*r
                if self.kwargs['ELA_use_ELA']:
                    new_r += self.intrinsic_weight*batched_new_r[idx:idx+1]
                else:
                    assert self.extrinsic_weight > 0
                    if feedbacks_type == 'normal':
                        pass
                    elif feedbacks_type == 'hurry':
                        new_r *= reward_scaler[idx].item()
                    else:
                        raise NotImplementedError
                new_rs.append(new_r)

                succ_s = self.episode_buffer[actor_index][idx]['succ_s']
                non_terminal = self.episode_buffer[actor_index][idx]['non_terminal']

                info = self.episode_buffer[actor_index][idx]['info']
                succ_info = self.episode_buffer[actor_index][idx]['succ_info']
                rnn_states = self.episode_buffer[actor_index][idx]['rnn_states']
                next_rnn_states = self.episode_buffer[actor_index][idx]['next_rnn_states']
                
                episode_rewards.append(r)

                d2store_ela = {
                    's':s, 
                    'a':a, 
                    'r':new_r, 
                    'succ_s':succ_s, 
                    'non_terminal':non_terminal, 
                    'rnn_states':copy_hdict(rnn_states),
                    'next_rnn_states':copy_hdict(next_rnn_states),
                    'info': info,
                    'succ_info': succ_info,
                }
                for hook_fn in self.hook_fns:
                    hook_fn(
                        exp_dict=d2store_ela,
                        actor_index=actor_index,
                        negative=False,
                        self=self,
                    )
                # Adding all the other elements back into the dict
                # e.g. v values , and entropy ...
                for key, value in self.episode_buffer[actor_index][idx].items():
                    if key not in d2store_ela:
                        d2store_ela[key] = value
                self.algorithm.store(d2store_ela, actor_index=actor_index)
                
                if idx==(episode_length-1):
                    wandb.log({'PerEpisode/ExtrinsicWeight': self.extrinsic_weight}, commit=False)
                    wandb.log({'PerEpisode/IntrinsicWeight': self.intrinsic_weight}, commit=False)
                    wandb.log({'PerEpisode/EpisodeLength': episode_length}, commit=False)
                    
                    wandb.log({'PerEpisode/ELA_Return': sum(new_rs).item(),}, commit=False) 
                    wandb.log({
                        'PerEpisode/ELA_Success': float(new_r.item()>0.5), #1+her_r.mean().item(),
                    }, commit=False) 
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

                    if self.algorithm.unwrapped.summary_writer is not None:
                        self.algorithm.unwrapped.summary_writer.add_scalar('PerEpisode/Success', (self.rewards['success']==her_r).float().mean().item(), self.episode_count)
                        self.algorithm.unwrapped.summary_writer.add_histogram('PerEpisode/Rewards', episode_rewards, self.episode_count)
            self.episode_buffer[actor_index] = []
        self.update_predictor(successful_traj=successful_traj)
	   
    def init_referential_game(self):
        ReferentialGym.datasets.dataset.DSS_version = self.kwargs["ELA_rg_distractor_sampling_scheme_version"]
        ReferentialGym.datasets.dataset.OC_version = self.kwargs["ELA_rg_object_centric_version"]
        print(f"OC_version = {ReferentialGym.datasets.dataset.OC_version}.")
        ReferentialGym.datasets.dataset.DC_version = self.kwargs["ELA_rg_descriptive_version"]
        #if args.descriptive_version == 2:
        #    args.batch_size = args.batch_size // 2
        print(f"DC_version = {ReferentialGym.datasets.dataset.DC_version} and BS={self.kwargs['ELA_rg_batch_size']}.")
        
        try:
            obs_instance = getattr(self.rg_storages[0], self.kwargs['ELA_exp_key'])[0][0]
            obs_shape = obs_instance.shape
            # Format is [ batch_size =1  x depth x w x h]
            stimulus_depth_dim = obs_shape[1]
        except Exception as e:
            print(e)
            obs_instance = None
            obs_shape = self.kwargs["preprocessed_observation_shape"]
            # Format is [ depth x w x h]
            stimulus_depth_dim = obs_shape[0]

        stimulus_resize_dim = obs_shape[-1] #args.resizeDim #64 #28
        normalize_rgb_values = False 
        transformations = []
        rgb_scaler = 1.0 #255.0
        '''
        from ReferentialGym.datasets.utils import ResizeNormalize
        transform = ResizeNormalize(
            size=stimulus_resize_dim, 
            normalize_rgb_values=normalize_rgb_values,
            rgb_scaler=rgb_scaler,
            use_cuda=False, #subprocess issue...s
        )
        transformations.append(transform)
        '''
        if self.kwargs["ELA_rg_with_color_jitter_augmentation"]:
            transformations = [T.RandomApply([
                SplitImg(
                    T.ColorJitter(
                        brightness=0.8,
                        contrast=0.8,
                        saturation=0.8,
                        hue=0.1,
                    ),
                    input_channel_dim=0,
                    transform_channel_dim=0,
                    output_channel_dim=0,
                )], 
                p=self.kwargs['ELA_rg_color_jitter_prob'])]+transformations
        
        if self.kwargs["ELA_rg_with_gaussian_blur_augmentation"]:
            transformations = [T.RandomApply([
                SplitImg(
                    GaussianBlur(
                        sigma=[0.1,0.5],
                        #sigma=(0.1, 0.5),
                    ),
                    input_channel_dim=0,
                    transform_channel_dim=-1,
                    output_channel_dim=0,
                )], 
                p=self.kwargs['ELA_rg_gaussian_blur_prob'])]+transformations
        
        from ReferentialGym.datasets.utils import AddEgocentricInvariance
        ego_inv_transform = AddEgocentricInvariance()
        
        transform_degrees = self.kwargs["ELA_rg_egocentric_tr_degrees"]
        transform_translate = float(self.kwargs["ELA_rg_egocentric_tr_xy"])/stimulus_resize_dim
        transform_translate = (transform_translate, transform_translate)
        
        if self.kwargs["ELA_rg_egocentric"]:
            split_img_ego_tr = SplitImg(
                ego_inv_transform,
                input_channel_dim=0,
                transform_channel_dim=-1,
                output_channel_dim=0,
            )
            '''
            rand_split_img_ego_tr = RandomApply( #T.RandomApply(
                [split_img_ego_tr],
                p=0.5,
            )
            '''
            affine_tr = T.RandomAffine(
                degrees=transform_degrees, 
                translate=transform_translate, 
                scale=None, 
                shear=None, 
                interpolation=T.InterpolationMode.BILINEAR, 
                fill=0,
            )
            split_img_affine_tr = SplitImg(
                affine_tr,
                input_channel_dim=0,
                transform_channel_dim=0,
                output_channel_dim=0,
            )
            '''
            rand_split_img_affine_tr = T.RandomApply(
                [split_img_affine_tr],
                p=0.5,
            ),
            '''
            #rand_split_img_ego_affine_tr = RandomApply(
            rand_split_img_ego_affine_tr = T.RandomApply(
                [split_img_ego_tr, split_img_affine_tr],
                p=self.kwargs['ELA_rg_egocentric_prob'],
            )
            transformations = [
                #rand_split_img_ego_tr,
                #rand_split_img_affine_tr,
                rand_split_img_ego_affine_tr,
                *transformations,
            ]
        
        self.rg_transformation = T.Compose(transformations)
        
        default_descriptive_ratio = 1-(1/(self.kwargs["ELA_rg_nbr_train_distractors"]+2))
        # Default: 1-(1/(nbr_distractors+2)), 
        # otherwise the agent find the local minimum
        # where it only predicts "no-target"...
        if self.kwargs["ELA_rg_descriptive_ratio"] <=0.001:
            descriptive_ratio = default_descriptive_ratio
        else:
            descriptive_ratio = self.kwargs["ELA_rg_descriptive_ratio"]

        
        rg_config = {
            "observability":            self.kwargs["ELA_rg_observability"],
            "max_sentence_length":      self.kwargs["ELA_rg_max_sentence_length"],
            "nbr_communication_round":  1,
            "nbr_distractors":          {"train":self.kwargs["ELA_rg_nbr_train_distractors"], "test":self.kwargs["ELA_rg_nbr_test_distractors"]},
            "distractor_sampling":      self.kwargs["ELA_rg_distractor_sampling"],
            # Default: use "similarity-50"
            # otherwise the emerging language 
            # will have very high ambiguity...
            # Speakers find the strategy of uttering
            # a word that is relevant to the class/label
            # of the target, seemingly.  
            
            "descriptive":              self.kwargs["ELA_rg_descriptive"],
            "descriptive_target_ratio": descriptive_ratio,
            
            "object_centric":           self.kwargs["ELA_rg_object_centric"],
            "object_centric_type":      self.kwargs["ELA_rg_object_centric_type"],
            "nbr_stimulus":             1,
            
            "graphtype":                self.kwargs["ELA_rg_graphtype"],
            "tau0":                     0.2,
            "gumbel_softmax_eps":       1e-6,
            "vocab_size":               self.kwargs["ELA_rg_vocab_size"],
            "force_eos":                self.kwargs["ELA_rg_force_eos"],
            "symbol_embedding_size":    self.kwargs["ELA_rg_symbol_embedding_size"], #64
            
            "agent_architecture":       self.kwargs["ELA_rg_arch"], #'CoordResNet18AvgPooled-2', #'BetaVAE', #'ParallelMONet', #'BetaVAE', #'CNN[-MHDPA]'/'[pretrained-]ResNet18[-MHDPA]-2'
            "shared_architecture":      self.kwargs["ELA_rg_shared_architecture"],
            "normalize_features":       self.kwargs["ELA_rg_normalize_features"],
            "agent_learning":           "learning",  #"transfer_learning" : CNN"s outputs are detached from the graph...
            "agent_loss_type":          self.kwargs["ELA_rg_agent_loss_type"], #"NLL"
            
            #"cultural_pressure_it_period": self.kwargs["ELA_rg_cultural_pressure_it_period"],
            #"cultural_speaker_substrate_size":  self.kwargs["ELA_rg_cultural_speaker_substrate_size"],
            #"cultural_listener_substrate_size":  self.kwargs["ELA_rg_cultural_listener_substrate_size"],
            #"cultural_reset_strategy":  self.kwargs["ELA_rg_cultural_reset_strategy"], #"oldestL", # "uniformSL" #"meta-oldestL-SGD"
            #"cultural_pressure_parameter_filtering_fn":  cultural_pressure_param_filtering_fn,
            #"cultural_pressure_meta_learning_rate":  self.kwargs["ELA_rg_cultural_pressure_meta_learning_rate"],
            
            # Cultural Bottleneck:
            #"iterated_learning_scheme": self.kwargs["ELA_rg_iterated_learning_scheme"],
            #"iterated_learning_period": self.kwargs["ELA_rg_iterated_learning_period"],
            #"iterated_learning_rehearse_MDL": self.kwargs["ELA_rg_iterated_learning_rehearse_MDL"],
            #"iterated_learning_rehearse_MDL_factor": self.kwargs["ELA_rg_iterated_learning_rehearse_MDL_factor"],
             
            # Obverter Hyperparameters:
            "obverter_stop_threshold":  self.kwargs["ELA_rg_obverter_threshold_to_stop_message_generation"],  #0.0 if not in use.
            "obverter_nbr_games_per_round": self.kwargs["ELA_rg_obverter_nbr_games_per_round"],
            
            "obverter_least_effort_loss": False,
            "obverter_least_effort_loss_weights": [1.0 for x in range(0, 10)],
            
            "batch_size":               self.kwargs["ELA_rg_batch_size"],
            "dataloader_num_worker":    self.kwargs["ELA_rg_dataloader_num_worker"],
            "stimulus_depth_dim":       stimulus_depth_dim, #1 if "dSprites" in args.dataset else 3,
            "stimulus_resize_dim":      stimulus_resize_dim, 
            
            "learning_rate":            self.kwargs["ELA_rg_learning_rate"], #1e-3,
            "weight_decay":             self.kwargs["ELA_rg_weight_decay"],
            "adam_eps":                 1e-16,
            "dropout_prob":             self.kwargs["ELA_rg_dropout_prob"],
            "embedding_dropout_prob":   self.kwargs["ELA_rg_emb_dropout_prob"],
            
            "with_gradient_clip":       False,
            "gradient_clip":            1e0,
            
            "use_homoscedastic_multitasks_loss": self.kwargs["ELA_rg_homoscedastic_multitasks_loss"],
            
            "use_feat_converter":       self.kwargs["ELA_rg_use_feat_converter"],
            
            "use_curriculum_nbr_distractors": self.kwargs["ELA_rg_use_curriculum_nbr_distractors"],
            "init_curriculum_nbr_distractors": self.kwargs["ELA_rg_init_curriculum_nbr_distractors"],
            "curriculum_distractors_window_size": 25, #100,
            
            "unsupervised_segmentation_factor": None, #1e5
            "nbr_experience_repetition":  self.kwargs["ELA_rg_nbr_experience_repetition"],
            
            "with_utterance_penalization":  False,
            "with_utterance_promotion":     False,
            "utterance_oov_prob":  0.5,  # Expected penalty of observing out-of-vocabulary words. 
            # The greater this value, the greater the loss/cost.
            "utterance_factor":    1e-2,
            
            "with_speaker_entropy_regularization":  False,
            "with_listener_entropy_regularization":  False,
            "entropy_regularization_factor":    -1e-2,
            
            "with_logits_mdl_principle":       self.kwargs['ELA_rg_with_logits_mdl_principle'],
            "logits_mdl_principle_normalization":     self.kwargs['ELA_rg_logits_mdl_principle_normalization'],
            "logits_mdl_principle_use_inst_accuracy":     self.kwargs['ELA_rg_logits_mdl_principle_use_inst_accuracy'],
            "logits_mdl_principle_factor":     self.kwargs['ELA_rg_logits_mdl_principle_factor'],
            "logits_mdl_principle_accuracy_threshold":     self.kwargs['ELA_rg_logits_mdl_principle_accuracy_threshold'],
            
            "with_mdl_principle":       False,
            "mdl_principle_factor":     5e-2,
            
            "with_weight_maxl1_loss":   False,
            
            "use_cuda":                 self.kwargs["ELA_rg_use_cuda"],
            
            "train_transform":            self.rg_transformation,
            "test_transform":             self.rg_transformation,
        }
        
        ## Agent Configuration:
        agent_config = copy.deepcopy(rg_config)
        agent_config["nbr_distractors"] = rg_config["nbr_distractors"]["train"] if rg_config["observability"] == "full" else 0

        # Obverter:
        if 'obverter' in self.kwargs["ELA_rg_graphtype"]:
            agent_config["use_obverter_threshold_to_stop_message_generation"] = self.kwargs["ELA_rg_obverter_threshold_to_stop_message_generation"]

        # Recurrent Convolutional Architecture:
        agent_config["architecture"] = rg_config["agent_architecture"]
         
        if "3xCNN" in agent_config["architecture"]:
            if "BN" in self.kwargs["ELA_rg_arch"]:
                agent_config["cnn_encoder_channels"] = ["BN32","BN64","BN128"]
            else:
                agent_config["cnn_encoder_channels"] = [32,64,128]
         
            if "3x3" in agent_config["architecture"]:
                agent_config["cnn_encoder_kernels"] = [3,3,3]
                agent_config["cnn_encoder_strides"] = [2,2,2]
            elif "7x4x3" in agent_config["architecture"]:
                agent_config["cnn_encoder_kernels"] = [7,4,3]
                agent_config["cnn_encoder_strides"] = [4,2,1]
            else:
                agent_config["cnn_encoder_kernels"] = [4,4,4]
                agent_config["cnn_encoder_strides"] = [2,2,2]
            agent_config["cnn_encoder_paddings"] = [1,1,1]
            agent_config["cnn_encoder_fc_hidden_units"] = []#[128,] 
            # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
            
            # For a fair comparison between CNN an VAEs:
            #agent_config["cnn_encoder_feature_dim"] = args.vae_nbr_latent_dim
            agent_config["cnn_encoder_feature_dim"] = self.kwargs["ELA_rg_agent_nbr_latent_dim"]
            #agent_config["cnn_encoder_feature_dim"] = self.kwargs["ELA_rg_symbol_processing_nbr_hidden_units"]
            # N.B.: if cnn_encoder_fc_hidden_units is [],
            # then this last parameter does not matter.
            # The cnn encoder is not topped by a FC network.
            
            agent_config["cnn_encoder_mini_batch_size"] = self.kwargs["ELA_rg_mini_batch_size"]
            agent_config["feat_converter_output_size"] = 256
            
            agent_config["temporal_encoder_nbr_hidden_units"] = 0
            agent_config["temporal_encoder_nbr_rnn_layers"] = 0
            agent_config["temporal_encoder_mini_batch_size"] = self.kwargs["ELA_rg_mini_batch_size"]
            agent_config["symbol_processing_nbr_hidden_units"] = self.kwargs["ELA_rg_symbol_processing_nbr_hidden_units"]
            agent_config["symbol_processing_nbr_rnn_layers"] = 1
        else:
            raise NotImplementedError

        batch_size = 4
        nbr_distractors = 1 if 'partial' in rg_config['observability'] else agent_config['nbr_distractors']['train']
        nbr_stimulus = agent_config['nbr_stimulus']
        
        obs_shape = [
            nbr_distractors+1,
            nbr_stimulus, 
            rg_config['stimulus_depth_dim'],
            rg_config['stimulus_resize_dim'],
            rg_config['stimulus_resize_dim']
        ]
        
        vocab_size = rg_config['vocab_size']
        max_sentence_length = rg_config['max_sentence_length']
        
        speaker = self.predictor
        speaker.speaker_init(
            kwargs=agent_config, 
            obs_shape=obs_shape, 
            vocab_size=vocab_size, 
            max_sentence_length=max_sentence_length,
            agent_id='s0',
            logger=None
        )
        print("Speaker:", speaker)
        self.speaker = speaker

        listener_config = copy.deepcopy(agent_config)
        if self.kwargs["ELA_rg_shared_architecture"]:
            listener_config['cnn_encoder'] = speaker.cnn_encoder 
        listener_config['nbr_distractors'] = rg_config['nbr_distractors']['train']
        
        if 'obverter' in self.kwargs["ELA_rg_graphtype"]:
            listener = copy.deepcopy(self.predictor)
            listener.listener_init(
                kwargs=listener_config,
                obs_shape=obs_shape,
                vocab_size=vocab_size,
                max_sentence_length=max_sentence_length,
                agent_id='l0',
                logger=None,
            )
        else:
            listener = LSTMCNNListener(
                kwargs=listener_config, 
                obs_shape=obs_shape, 
                vocab_size=vocab_size, 
                max_sentence_length=max_sentence_length,
                agent_id='l0',
                logger=None
            )
        print("Listener:", listener)
        self.listener = listener

        ## Train set:
        self.rg_split_strategy = self.kwargs["ELA_split_strategy"]
        self.rg_train_dataset_length = self.kwargs["ELA_train_dataset_length"]
        self.rg_test_dataset_length = self.kwargs["ELA_test_dataset_length"]
        self.rg_exp_key = self.kwargs["ELA_exp_key"]
        
        ## Modules:
        modules = {}
        modules[speaker.id] = speaker 
        modules[listener.id] = listener 
        
        from ReferentialGym import modules as rg_modules
        
        if self.kwargs["ELA_rg_use_obverter_sampling"]:
            obverter_sampling_id = "obverter_sampling_0"
            obverter_sampling_config = {
                "batch_size": rg_config["batch_size"],
                "round_alternation_only": self.kwargs["ELA_rg_obverter_sampling_round_alternation_only"],
                "obverter_nbr_games_per_round": self.kwargs["ELA_rg_obverter_nbr_games_per_round"],
                "repeat_experiences": self.kwargs["ELA_rg_obverter_sampling_repeat_experiences"],
            }
            
            modules[obverter_sampling_id] = rg_modules.ObverterDatasamplingModule(
                id=obverter_sampling_id,
                config=obverter_sampling_config,
            )
  
        # Population:
        population_handler_id = "population_handler_0"
        population_handler_config = copy.deepcopy(rg_config)
        population_handler_config["verbose"] = self.kwargs["ELA_rg_verbose"]
        population_handler_stream_ids = {
            "current_speaker_streams_dict":"modules:current_speaker",
            "current_listener_streams_dict":"modules:current_listener",
            "epoch":"signals:epoch",
            "mode":"signals:mode",
            "global_it_datasample":"signals:global_it_datasample",
        }
        
        # Current Speaker:
        current_speaker_id = "current_speaker"
        
        # Current Listener:
        current_listener_id = "current_listener"
        
        modules[population_handler_id] = rg_modules.build_PopulationHandlerModule(
            id=population_handler_id,
            prototype_speaker=speaker,
            prototype_listener=listener,
            config=population_handler_config,
            input_stream_ids=population_handler_stream_ids,
        )
            
        modules[current_speaker_id] = rg_modules.CurrentAgentModule(id=current_speaker_id,role="speaker")
        modules[current_listener_id] = rg_modules.CurrentAgentModule(id=current_listener_id,role="listener")
        
        if self.kwargs.get("ELA_rg_use_semantic_cooccurrence_grounding", False):
            sem_cooc_grounding_id = "sem_cooccurrence_grounding_0"
            sem_cooc_grounding_config = {
                "lambda_factor": self.kwargs.get("ELA_rg_semantic_cooccurrence_grounding_lambda", 1.0),
                "noise_magnitude": self.kwargs.get("ELA_rg_semantic_cooccurrence_grounding_noise_magnitude", 0.0),
            }
            modules[sem_cooc_grounding_id] = rg_modules.build_CoOccurrenceSemanticGroundingLossModule(
                id=sem_cooc_grounding_id,
                config=sem_cooc_grounding_config,
            )

        if self.kwargs.get("ELA_rg_with_semantic_grounding_metric", False):
            sem_grounding_id = "sem_grounding_metric_0"
            sem_grounding_config = {
                'idx2w':self.idx2w,
            }
            modules[sem_grounding_id] = rg_modules.build_SemanticGroundingMetricModule(
                id=sem_grounding_id,
                config=sem_grounding_config,
            )

        ## Pipelines:
        pipelines = {}
        
        # 0) Now that all the modules are known, let us build the optimization module:
        optim_id = "global_optim"
        optim_config = {
            "modules":modules,
            "learning_rate":self.kwargs["ELA_rg_learning_rate"],
            "weight_decay":self.kwargs["ELA_rg_weight_decay"],
            "l1_reg_lambda":self.kwargs["ELA_rg_l1_weight_decay"],
            "l2_reg_lambda":self.kwargs["ELA_rg_l2_weight_decay"],
            "optimizer_type":self.kwargs["ELA_rg_optimizer_type"],
            "with_gradient_clip":rg_config["with_gradient_clip"],
            "adam_eps":rg_config["adam_eps"],
            "optimize":self.kwargs.get("ELA_with_rg_optimize", True),
        }
        
        optim_module = rg_modules.build_OptimizationModule(
            id=optim_id,
            config=optim_config,
        )
        modules[optim_id] = optim_module
        
        if self.kwargs["ELA_rg_homoscedastic_multitasks_loss"]:
            homo_id = "homo0"
            homo_config = {"use_cuda":self.kwargs["ELA_rg_use_cuda"]}
            modules[homo_id] = rg_modules.build_HomoscedasticMultiTasksLossModule(
                id=homo_id,
                config=homo_config,
            )
        
        grad_recorder_id = "grad_recorder"
        grad_recorder_module = rg_modules.build_GradRecorderModule(id=grad_recorder_id)
        modules[grad_recorder_id] = grad_recorder_module
        
        speaker_topo_sim_metric_id = f"{speaker.id}_topo_sim2_metric"
        speaker_topo_sim_metric_input_stream_ids = {
            #"model":"modules:current_speaker:ref:ref_agent",
            "model":f"modules:{speaker.id}:ref:_utter",
            "features":"modules:current_speaker:ref:ref_agent:features",
            "representations":"modules:current_speaker:sentences_widx",
            "experiences":"current_dataloader:sample:speaker_experiences", 
            "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
            "latent_representations_values":"current_dataloader:sample:speaker_exp_latents_values", 
            "latent_representations_one_hot_encoded":"current_dataloader:sample:speaker_exp_latents_one_hot_encoded", 
            "indices":"current_dataloader:sample:speaker_indices", 
        }
        
        def agent_preprocess_fn(x):
            if self.kwargs["ELA_rg_use_cuda"]:
                x = x.cuda()
            # adding distractor x stimuli-dim dims:
            x = x.unsqueeze(1).unsqueeze(1)
            return x 
        
        def agent_postprocess_fn(x):
            x = x[1].cpu().detach()
            x = x.reshape((x.shape[0],-1)).numpy()
            return x 
        
        def agent_features_postprocess_fn(x):
            x = x[-1].cpu().detach()
            x = x.reshape((x.shape[0],-1)).numpy()
            return x 
        
        speaker_topo_sim_metric_module = rg_modules.build_TopographicSimilarityMetricModule2(
            id=speaker_topo_sim_metric_id,
            config = {
                "metric_fast": self.kwargs["ELA_rg_metric_fast"],
                "pvalue_significance_threshold": 0.05,
                "parallel_TS_computation_max_workers":self.kwargs["ELA_rg_parallel_TS_worker"],
                "filtering_fn":(lambda kwargs: speaker.role=="speaker"),
                #"postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
                # cf outputs of _utter:
                "postprocess_fn": agent_postprocess_fn, #(lambda x: x[1].cpu().detach().numpy()),
                # not necessary if providing a preprocess_fn, 
                # that computes the features/_sense output, but here it is in order to deal with shapes:
                "features_postprocess_fn": agent_features_postprocess_fn, #(lambda x: x[-1].cpu().detach().numpy()),
                #"preprocess_fn": (lambda x: x.cuda() if self.kwargs["ELA_rg_use_cuda"] else x),
                # cf _sense:
                "preprocess_fn": (lambda x: speaker._sense(agent_preprocess_fn(x))),
                #"epoch_period":args.epoch-1, 
                "epoch_period": self.kwargs["ELA_rg_metric_epoch_period"],
                "batch_size":self.kwargs["ELA_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ELA_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ELA_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ELA_rg_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ELA_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ELA_rg_metric_active_factors_only"],
            },
            input_stream_ids=speaker_topo_sim_metric_input_stream_ids,
        )
        modules[speaker_topo_sim_metric_id] = speaker_topo_sim_metric_module
        
        # Modularity: Speaker
        speaker_modularity_disentanglement_metric_id = f"{speaker.id}_modularity_disentanglement_metric"
        speaker_modularity_disentanglement_metric_input_stream_ids = {
            "model":f"modules:{speaker.id}:ref:cnn_encoder",
            "representations":f"modules:{current_speaker_id}:ref:ref_agent:features",
            "experiences":f"modules:{current_speaker_id}:ref:ref_agent:experiences", 
            "latent_representations":f"modules:{current_speaker_id}:ref:ref_agent:exp_latents", 
            "indices":f"modules:{current_speaker_id}:ref:ref_agent:indices", 
        }
        speaker_modularity_disentanglement_metric_module = rg_modules.build_ModularityDisentanglementMetricModule(
            id=speaker_modularity_disentanglement_metric_id,
            input_stream_ids=speaker_modularity_disentanglement_metric_input_stream_ids,
            config = {
                "filtering_fn":(lambda kwargs: speaker.role=="speaker"),
                #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
                # dealing with extracting z (mu in pos 1):
                "postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in agent_config["architecture"] else x.cpu().detach().numpy()),
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ELA_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ELA_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ELA_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ELA_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ELA_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ELA_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ELA_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ELA_rg_metric_active_factors_only"],
            },
        )
        modules[speaker_modularity_disentanglement_metric_id] = speaker_modularity_disentanglement_metric_module
        
        # Modularity: Listener
        listener_modularity_disentanglement_metric_id = f"{listener.id}_modularity_disentanglement_metric"
        listener_modularity_disentanglement_metric_input_stream_ids = {
            "model":f"modules:{listener.id}:ref:cnn_encoder",
            "representations":f"modules:{current_listener_id}:ref:ref_agent:features",
            "experiences":f"modules:{current_listener_id}:ref:ref_agent:experiences", 
            "latent_representations":f"modules:{current_listener_id}:ref:ref_agent:exp_latents", 
            "indices":f"modules:{current_listener_id}:ref:ref_agent:indices", 
        }
        listener_modularity_disentanglement_metric_module = rg_modules.build_ModularityDisentanglementMetricModule(
            id=listener_modularity_disentanglement_metric_id,
            input_stream_ids=listener_modularity_disentanglement_metric_input_stream_ids,
            config = {
                "filtering_fn": (lambda kwargs: listener.role=="speaker"),
                #"filtering_fn": (lambda kwargs: True),
                #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
                "postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in agent_config["architecture"] else x.cpu().detach().numpy()),
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ELA_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ELA_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ELA_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ELA_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ELA_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ELA_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ELA_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ELA_rg_metric_active_factors_only"],
            },
        )
        modules[listener_modularity_disentanglement_metric_id] = listener_modularity_disentanglement_metric_module
        
        inst_coord_metric_id = f"inst_coord_metric"
        inst_coord_input_stream_ids = {
            "logger":"modules:logger:ref",
            "logs_dict":"logs_dict",
            "epoch":"signals:epoch",
            "mode":"signals:mode",
        
            "end_of_dataset":"signals:end_of_dataset",  
            # boolean: whether the current batch/datasample is the last of the current dataset/mode.
            "end_of_repetition_sequence":"signals:end_of_repetition_sequence",
            # boolean: whether the current sample(observation from the agent of the current batch/datasample) 
            # is the last of the current sequence of repetition.
            "end_of_communication":"signals:end_of_communication",
            # boolean: whether the current communication round is the last of 
            # the current dialog.
            "dataset":"current_dataset:ref",
            
            "vocab_size":"config:vocab_size",
            "max_sentence_length":"config:max_sentence_length",
            "sentences_widx":"modules:current_speaker:sentences_widx", 
            "decision_probs":"modules:current_listener:decision_probs",
            "listener_indices":"current_dataloader:sample:listener_indices",
        }
        inst_coord_metric_module = rg_modules.build_InstantaneousCoordinationMetricModule(
            id=inst_coord_metric_id,
            config = {
                "filtering_fn":(lambda kwargs: True),
                "epoch_period":1,
            },
            input_stream_ids=inst_coord_input_stream_ids,
        )
        modules[inst_coord_metric_id] = inst_coord_metric_module
        
        # FactorVAE Disentanglement Metric :
        speaker_factor_vae_disentanglement_metric_input_stream_ids = {
            "model":f"modules:{speaker.id}:ref:cnn_encoder",
            "representations":f"modules:{current_speaker_id}:ref:ref_agent:features",
            "experiences":f"modules:{current_speaker_id}:ref:ref_agent:experiences", 
            "latent_representations":f"modules:{current_speaker_id}:ref:ref_agent:exp_latents", 
            "latent_values_representations":f"modules:{current_speaker_id}:ref:ref_agent:exp_latents_values",
            "indices":f"modules:{current_speaker_id}:ref:ref_agent:indices", 
        }
        speaker_factor_vae_disentanglement_metric_id = f"{speaker.id}_factor_vae_disentanglement_metric"
        speaker_factor_vae_disentanglement_metric_module = rg_modules.build_FactorVAEDisentanglementMetricModule(
            id=speaker_factor_vae_disentanglement_metric_id,
            input_stream_ids=speaker_factor_vae_disentanglement_metric_input_stream_ids,
            config = {
                "filtering_fn": (lambda kwargs: speaker.role=="speaker"),
                #"filtering_fn": (lambda kwargs: True),
                #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
                "postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in agent_config["architecture"] else x.cpu().detach().numpy()),
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ELA_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ELA_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ELA_rg_metric_batch_size"],#5,
                "nbr_train_points": self.kwargs["ELA_rg_nbr_train_points"],#3000,
                "nbr_eval_points": self.kwargs["ELA_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ELA_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ELA_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ELA_rg_metric_active_factors_only"],
            }
        )
        modules[speaker_factor_vae_disentanglement_metric_id] = speaker_factor_vae_disentanglement_metric_module
        
        listener_factor_vae_disentanglement_metric_input_stream_ids = {
            "model":f"modules:{listener.id}:ref:cnn_encoder",
            "representations":f"modules:{current_listener_id}:ref:ref_agent:features",
            "experiences":f"modules:{current_listener_id}:ref:ref_agent:experiences", 
            "latent_representations":f"modules:{current_listener_id}:ref:ref_agent:exp_latents", 
            "latent_values_representations":f"modules:{current_listener_id}:ref:ref_agent:exp_latents_values",
            "indices":f"modules:{current_listener_id}:ref:ref_agent:indices", 
        }
        listener_factor_vae_disentanglement_metric_id = f"{listener.id}_factor_vae_disentanglement_metric"
        listener_factor_vae_disentanglement_metric_module = rg_modules.build_FactorVAEDisentanglementMetricModule(
            id=listener_factor_vae_disentanglement_metric_id,
            input_stream_ids=listener_factor_vae_disentanglement_metric_input_stream_ids,
            config = {
                "filtering_fn": (lambda kwargs: listener.role=="speaker"),
                #"filtering_fn": (lambda kwargs: True),
                #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
                "postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in agent_config["architecture"] else x.cpu().detach().numpy()),
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ELA_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ELA_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ELA_rg_metric_batch_size"],#5,
                "nbr_train_points": self.kwargs["ELA_rg_nbr_train_points"],#3000,
                "nbr_eval_points": self.kwargs["ELA_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ELA_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ELA_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ELA_rg_metric_active_factors_only"],
            }
        )
        modules[listener_factor_vae_disentanglement_metric_id] = listener_factor_vae_disentanglement_metric_module
        
        # Mutual Information Gap:
        speaker_mig_disentanglement_metric_input_stream_ids = {
            "model":f"modules:{speaker.id}:ref:cnn_encoder",
            "representations":f"modules:{current_speaker_id}:ref:ref_agent:features",
            "experiences":f"modules:{current_speaker_id}:ref:ref_agent:experiences", 
            "latent_representations":f"modules:{current_speaker_id}:ref:ref_agent:exp_latents", 
            "indices":f"modules:{current_speaker_id}:ref:ref_agent:indices", 
        }
        speaker_mig_disentanglement_metric_id = f"{speaker.id}_mig_disentanglement_metric"
        speaker_mig_disentanglement_metric_module = rg_modules.build_MutualInformationGapDisentanglementMetricModule(
            id=speaker_mig_disentanglement_metric_id,
            input_stream_ids=speaker_mig_disentanglement_metric_input_stream_ids,
            config = {
                "filtering_fn": (lambda kwargs: speaker.role=="speaker"),
                #"filtering_fn": (lambda kwargs: True),
                #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
                "postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in agent_config["architecture"] else x.cpu().detach().numpy()),
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ELA_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ELA_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ELA_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ELA_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ELA_rg_nbr_eval_points"],#2000,
                "resample":self.kwargs["ELA_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ELA_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ELA_rg_metric_active_factors_only"],
            }
        )
        modules[speaker_mig_disentanglement_metric_id] = speaker_mig_disentanglement_metric_module
    
        listener_mig_disentanglement_metric_input_stream_ids = {
            "model":f"modules:{listener.id}:ref:cnn_encoder",
            "representations":f"modules:{current_listener_id}:ref:ref_agent:features",
            "experiences":f"modules:{current_listener_id}:ref:ref_agent:experiences", 
            "latent_representations":f"modules:{current_listener_id}:ref:ref_agent:exp_latents", 
            "indices":f"modules:{current_listener_id}:ref:ref_agent:indices", 
        }
        listener_mig_disentanglement_metric_id = f"{listener.id}_mig_disentanglement_metric"
        listener_mig_disentanglement_metric_module = rg_modules.build_MutualInformationGapDisentanglementMetricModule(
            id=listener_mig_disentanglement_metric_id,
            input_stream_ids=listener_mig_disentanglement_metric_input_stream_ids,
            config = {
                "filtering_fn": (lambda kwargs: listener.role=="speaker"),
                #"filtering_fn": (lambda kwargs: True),
                #"postprocess_fn": (lambda x: x.cpu().detach().numpy()),
                "postprocess_fn": (lambda x: x[2].cpu().detach().numpy() if "BetaVAE" in agent_config["architecture"] else x.cpu().detach().numpy()),
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ELA_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ELA_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ELA_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ELA_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ELA_rg_nbr_eval_points"],#2000,
                "resample":self.kwargs["ELA_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ELA_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ELA_rg_metric_active_factors_only"],
            }
        )
        modules[listener_mig_disentanglement_metric_id] = listener_mig_disentanglement_metric_module

        compactness_ambiguity_metric_id = "compactness_ambiguity_metric"
        compactness_ambiguity_metric_input_stream_ids = {
            #"model":"modules:current_speaker:ref:ref_agent",
            "model":"modules:current_speaker:ref:ref_agent:_utter",
            "representations":"modules:current_speaker:sentences_widx",
            "experiences":"current_dataloader:sample:speaker_experiences", 
            "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
            #"latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
            "indices":"current_dataloader:sample:speaker_indices", 
        }
        if self.kwargs.get("ELA_rg_sanity_check_compactness_ambiguity_metric", False):
            compactness_ambiguity_metric_input_stream_ids["representations"] = \
                "current_dataloader:sample:speaker_grounding_signal"
            compactness_ambiguity_metric_input_stream_ids["top_view"] = "current_dataloader:sample:speaker_top_view" 
            compactness_ambiguity_metric_input_stream_ids["agent_pos_in_top_view"] = "current_dataloader:sample:speaker_agent_pos_in_top_view" 
        if "natural" in self.kwargs.get("ELA_rg_compactness_ambiguity_metric_language_specs", "emergent"):
            compactness_ambiguity_metric_input_stream_ids["natural_representations"] = "current_dataloader:sample:speaker_natural_language_sentences_widx"        
        
        compactness_ambiguity_metric_module = rg_modules.build_CompactnessAmbiguityMetricModule(
            id=compactness_ambiguity_metric_id,
            input_stream_ids=compactness_ambiguity_metric_input_stream_ids,
            config = {
                "with_ordering": self.kwargs["ELA_rg_compactness_ambiguity_metric_with_ordering"],
                'sanity_check_shuffling': False,
                "show_stimuli": False, #True,
                "postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ELA_rg_use_cuda"] else x),
                "epoch_period":1,#self.kwargs["ELA_rg_metric_epoch_period"],
                "batch_size":self.kwargs["ELA_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ELA_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ELA_rg_nbr_eval_points"],#2000,
                "resample": False, #self.kwargs["ELA_rg_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ELA_rg_seed"],
                "nbr_shuffled_entities":4,
                "verbose":False,
                "idx2w": self.idx2w,
                "language_specs_to_compute":self.kwargs["ELA_rg_compactness_ambiguity_metric_language_specs"].split('+'),
                "kwargs": self.kwargs,
            }
        )
        modules[compactness_ambiguity_metric_id] = compactness_ambiguity_metric_module

        posbosdis_disentanglement_metric_id = "posbosdis_disentanglement_metric"
        posbosdis_disentanglement_metric_input_stream_ids = {
            #"model":"modules:current_speaker:ref:ref_agent",
            "model":"modules:current_speaker:ref:ref_agent:_utter",
            "representations":"modules:current_speaker:sentences_widx",
            "experiences":"current_dataloader:sample:speaker_experiences", 
            "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
            #"latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
            "indices":"current_dataloader:sample:speaker_indices", 
        }

        posbosdis_disentanglement_metric_module = rg_modules.build_PositionalBagOfSymbolsDisentanglementMetricModule(
            id=posbosdis_disentanglement_metric_id,
            input_stream_ids=posbosdis_disentanglement_metric_input_stream_ids,
            config = {
                "postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ELA_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ELA_rg_metric_epoch_period"],
                "batch_size":self.kwargs["ELA_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ELA_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ELA_rg_nbr_eval_points"],#2000,
                "resample":self.kwargs["ELA_rg_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ELA_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ELA_rg_metric_active_factors_only"],
            }
        )
        modules[posbosdis_disentanglement_metric_id] = posbosdis_disentanglement_metric_module

        logger_id = "per_epoch_logger"
        logger_module = rg_modules.build_PerEpochLoggerModule(id=logger_id)
        modules[logger_id] = logger_module
      
        pipelines["referential_game"] = [population_handler_id]
        if self.kwargs["ELA_rg_use_obverter_sampling"]:
            pipelines["referential_game"].append(obverter_sampling_id)
        if "synthetic" in self.kwargs["ELA_rg_graphtype"]:
            pipelines["referential_game"] += [
                current_speaker_id,
                intervention_id,
                current_listener_id
            ]
        else:
            pipelines["referential_game"] += [
                current_speaker_id,
                current_listener_id
            ]
        
        pipelines[optim_id] = []
        if self.kwargs.get("ELA_rg_use_semantic_cooccurrence_grounding", False):
            pipelines[optim_id].append(sem_cooc_grounding_id)
        if self.kwargs.get("ELA_rg_with_semantic_grounding_metric", False):
            pipelines[optim_id].append(sem_grounding_id)
        if self.kwargs["ELA_rg_homoscedastic_multitasks_loss"]:
            pipelines[optim_id].append(homo_id)
        pipelines[optim_id].append(optim_id)
        """
        # Add gradient recorder module for debugging purposes:
        pipelines[optim_id].append(grad_recorder_id)
        """
        if not(self.kwargs["ELA_rg_shared_architecture"]):
            pipelines[optim_id].append(listener_factor_vae_disentanglement_metric_id)
            pipelines[optim_id].append(listener_modularity_disentanglement_metric_id)
            pipelines[optim_id].append(listener_mig_disentanglement_metric_id)
        #pipelines[optim_id].append(speaker_factor_vae_disentanglement_metric_id)
        #pipelines[optim_id].append(speaker_modularity_disentanglement_metric_id)
        #pipelines[optim_id].append(speaker_mig_disentanglement_metric_id)
    
        #pipelines[optim_id].append(topo_sim_metric_id)
        pipelines[optim_id].append(speaker_topo_sim_metric_id)
        #pipelines[optim_id].append(posbosdis_disentanglement_metric_id)
        pipelines[optim_id].append(compactness_ambiguity_metric_id)
        #pipelines[optim_id].append(speaker_posbosdis_metric_id)
        '''
        if "obverter" in self.kwargs["ELA_rg_graphtype"]:
            pipelines[optim_id].append(listener_topo_sim_metric_id)
            pipelines[optim_id].append(listener_posbosdis_metric_id)
        '''
        pipelines[optim_id].append(inst_coord_metric_id)
        
        pipelines[optim_id].append(logger_id)
        
        rg_config["modules"] = modules
        rg_config["pipelines"] = pipelines

        self.rg_config = rg_config
        self.population_handler_id = population_handler_id
           
    def launch_referential_game(self, nbr_epoch=1):
        torch.set_grad_enabled(True)
        self.predictor.train(True)
        self.referential_game.train(
            #nbr_epoch=self.rg_iteration+nbr_epoch,
            nbr_epoch=nbr_epoch,
            logger=self.logger,
            verbose_period=1,
        )
        self.predictor.train(False)
        torch.set_grad_enabled(False)
        
        self.rg_iteration+=1 #nbr_epoch
        #self.referential_game.save(os.path.join(self.save_path, f"{self.rg_iteration}.rg"))
        self.logger.flush()
 
    def update_datasets(self):
        kwargs = {'same_episode_target': self.kwargs.get('ELA_rg_same_episode_target', False)}
        if 'episodic-dissimilarity' in self.rg_config['distractor_sampling']:
            assert self.kwargs['ELA_rg_same_episode_target']
        #TODO: investigate what type of target we want with similarity:
        elif 'similarity-' in self.rg_config['distractor_sampling']:
            assert self.kwargs['ELA_rg_same_episode_target']
        
        extra_keys_dict = {
            "grounding_signal":self.kwargs.get("ELA_grounding_signal_key", None),
        }
        if self.kwargs.get("ELA_rg_sanity_check_compactness_ambiguity_metric", False):
            extra_keys_dict.update({
                "top_view":"info:top_view",
                "agent_pos_in_top_view":"info:agent_pos_in_top_view",
            })
        if self.kwargs.get("ELA_rg_with_semantic_grounding_metric", False):
            extra_keys_dict.update({
                "semantic_signal":"info:symbolic_image",
            })
        if "natural" in self.kwargs.get("ELA_rg_compactness_ambiguity_metric_language_specs","emergent"):
            extra_keys_dict.update({
                "natural_language_sentences_widx":"info:achieved_goal",
            })

        self.rg_train_dataset = DemonstrationDataset(
            replay_storage=self.rg_storages[0],
            train=True,
            transform=self.rg_transformation,
            split_strategy=self.rg_split_strategy,
            dataset_length=self.rg_train_dataset_length,
            exp_key=self.rg_exp_key,
            extra_keys_dict=extra_keys_dict,
            kwargs=kwargs,
        )
        
        self.rg_test_dataset = DemonstrationDataset(
            replay_storage=self.rg_storages[0],
            #replay_storage=self.rg_storages[0].test_storage,
            train=False,
            transform=self.rg_transformation,
            split_strategy=self.rg_split_strategy,
            dataset_length=self.rg_test_dataset_length,
            exp_key=self.rg_exp_key,
            extra_keys_dict=extra_keys_dict,
            kwargs=kwargs,
        )
        
        need_dict_wrapping = {}
        dataset_args = {"modes":["train", "test"]}
        dataset_args["train"] = {
            "dataset_class":            "DualLabeledDataset",
            "modes": {
                "train": self.rg_train_dataset,
                "test": self.rg_test_dataset,
            },
            "need_dict_wrapping":       need_dict_wrapping,
            "nbr_stimulus":             self.rg_config["nbr_stimulus"],
            "distractor_sampling":      self.rg_config["distractor_sampling"],
            "nbr_distractors":          self.rg_config["nbr_distractors"],
            "observability":            self.rg_config["observability"],
            "object_centric":           self.rg_config["object_centric"],
            "object_centric_type":      self.rg_config["object_centric_type"],
            "descriptive":              self.rg_config["descriptive"],
            "descriptive_target_ratio": self.rg_config["descriptive_target_ratio"],
            'with_replacement':         self.kwargs['ELA_rg_distractor_sampling_with_replacement'],
        }
        dataset_args["test"] = {
            "dataset_class":            "DualLabeledDataset",
            "modes": {
                "train": self.rg_train_dataset,
                "test": self.rg_test_dataset,
            },
            "need_dict_wrapping":       need_dict_wrapping,
            "nbr_stimulus":             self.rg_config["nbr_stimulus"],
            "distractor_sampling":      self.rg_config["distractor_sampling"],
            "nbr_distractors":          self.rg_config["nbr_distractors"],
            "observability":            self.rg_config["observability"],
            "object_centric":           self.rg_config["object_centric"],
            "object_centric_type":      self.rg_config["object_centric_type"],
            "descriptive":              self.rg_config["descriptive"],
            "descriptive_target_ratio": self.rg_config["descriptive_target_ratio"],
            'with_replacement':         self.kwargs['ELA_rg_distractor_sampling_with_replacement'],
        }

        self.dataset_args = dataset_args

    def update_predictor(self, successful_traj=False):
        '''
        Every training_period:
            - reset the previous period count check to the current count
            - so that it is possible to update the training_period adaptively if needs be.
        '''
        # RG Update:
        period_check = self.kwargs['ELA_rg_training_period']
        period_count_check = self.nbr_buffered_predictor_experience-getattr(self, 'previous_ELA_period_count_check',0)
        can_rg_train = False
        if self.kwargs["ELA_with_rg_training"] \
        and len(self.rg_storages[0])>=self.kwargs['ELA_replay_capacity']:
            can_rg_train = True
        quotient = period_count_check // period_check
        #previous_quotient = getattr(self, 'previous_ELA_quotient', -1)
        if can_rg_train \
        and quotient != 0: #previous_quotient:
        #and (period_count_check % period_check == 0):
            #self.previous_ELA_quotient = quotient
            self.previous_ELA_period_count_check = self.nbr_buffered_predictor_experience
            if self.kwargs['ELA_use_ELA']:
                self._rg_training()
        
        wandb.log({'Training/ELA/storage_length': len(self.rg_storages[0])}, commit=False)

    def _rg_training(self):
        full_update = True
        for it in range(self.kwargs['ELA_rg_nbr_epoch_per_update']):
            self.test_acc = self.finetune_predictor(update=(it==0))
            if self.test_acc >= self.kwargs['ELA_rg_accuracy_threshold']:
                full_update = False
                break
        # Update training period:
        if self.kwargs['ELA_rg_training_adaptive_period']:
            if full_update: self.kwargs['ELA_rg_training_period'] /= 2
            else:   self.kwargs['ELA_rg_training_period'] *= 2
        wandb.log({f"Training/ELA/RGTrainingPeriod":self.kwargs['ELA_rg_training_period']}, commit=False)
        wandb.log({f"Training/ELA/TestAccuracy":self.test_acc}, commit=False)
        wandb.log({f"Training/ELA/FullUpdate":int(full_update)}, commit=False)

    def finetune_predictor(self, update=False):
        if self.rg_iteration == 0:
            ###
            save_path = os.path.join(wandb.run.dir, f"referential_game")
            print(f"ELA: Referential Game NEW PATH: {save_path}")
            self.save_path = save_path 
            ###
            
            ###
            from ReferentialGym.utils import statsLogger
            logger = statsLogger(path=save_path,dumpPeriod=100)
            self.logger = logger
            self.listener.logger = logger
            self.speaker.logger = logger
            self.rg_config['modules'][self.population_handler_id].logger = logger
            self.rg_config['modules'][self.population_handler_id].config['save_path'] = self.save_path
            ###
            
        if update:
            self.update_datasets()
            if self.rg_iteration==0:
                self.referential_game = ReferentialGym.make(
                    config=self.rg_config, 
                    dataset_args=self.dataset_args,
                    save_path=self.save_path,
                )
            else:
                self.referential_game.update_datasets(dataset_args=self.dataset_args)
         
        start = time.time()
        #self.launch_referential_game(nbr_epoch=self.kwargs["ELA_rg_nbr_epoch_per_update"])
        self.launch_referential_game(nbr_epoch=1)
        end = time.time()
        
        wandb.log({'PerELAUpdate/TimeComplexity/ReferentialGame':  end-start}, commit=False) # self.param_update_counter)
        
        logs_dict = self.referential_game.modules['per_epoch_logger'].latest_logs
        test_acc = logs_dict["PerEpoch/test/repetition0/comm_round0/referential_game_accuracy/Mean"]
        train_acc = logs_dict["PerEpoch/train/repetition0/comm_round0/referential_game_accuracy/Mean"]
        return test_acc 

    def clone(
        self, 
        with_replay_buffer: bool=False, 
        clone_proxies: bool=False, 
        minimal: bool=False
    ):        
        cloned_algo = ELAAlgorithmWrapper(
            algorithm=self.algorithm.clone(
                with_replay_buffer=with_replay_buffer,
                clone_proxies=clone_proxies,
                minimal=minimal
            ), 
            predictor=self.predictor, 
        ) 
        return cloned_algo

