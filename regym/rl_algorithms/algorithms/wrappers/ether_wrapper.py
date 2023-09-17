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


from regym.rl_algorithms.algorithms.wrappers.ther_wrapper2 import THERAlgorithmWrapper2
from regym.rl_algorithms.algorithms.wrappers.ther_wrapper2 import batched_predictor_based_goal_predicated_reward_fn2

from regym.rl_algorithms.replay_buffers import PrioritizedReplayStorage, SplitReplayStorage, SplitPrioritizedReplayStorage
from regym.rl_algorithms.utils import archi_concat_fn, _extract_rnn_states_from_batch_indices, _concatenate_hdict, _concatenate_list_hdict, copy_hdict

import ReferentialGym
from ReferentialGym.datasets import DemonstrationDataset
from ReferentialGym.agents import DiscriminativeListener, LSTMCNNListener

###########################################################
###########################################################
###########################################################

# Adapted from: 
# https://github.com/facebookresearch/EGG/blob/3418b21f579743e7951f05e26562af235dcefa46/egg/zoo/emcom_as_ssl/data.py#L55
from PIL import ImageFilter
import random
class GaussianBlur:
    """
    Gaussian blur augmentation as in SimCLR https://arxiv.org/abs/2002.05709
    """
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
        self.kernel_size = (5,5)

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        if isinstance(x, torch.Tensor):
            y = T.functional.gaussian_blur(img=x, kernel_size=self.kernel_size, sigma=(sigma, sigma))     
        else:
            #raise NotImplementedError("This fn is expecting PIL images, not torch.Tensor")
            y = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return y

###########################################################
###########################################################
###########################################################

class RandomApply:
    def __init__(
        self,
        transforms,
        p=0.5,
    ):
        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        rand = torch.rand(1).item()
        wandb.log({'RandomApply/RandomValue':rand}, commit=True)
        y = x
        if self.p > rand:
            for transform in self.transforms:
                y = transform(y)
        return y

class SplitImg:
    def __init__(
        self, 
        transform,
        input_channel_dim=-1,
        transform_channel_dim=-1,
        output_channel_dim=None,
    ):
        self.transform = transform
        self.input_channel_dim = input_channel_dim
        self.transform_channel_dim = transform_channel_dim
        if output_channel_dim is None:
            output_channel_dim = input_channel_dim
        self.output_channel_dim = output_channel_dim
        
    def __call__(self, x):
        assert len(x.shape)==3
        if self.input_channel_dim!=self.transform_channel_dim:
            x = x.transpose(self.transform_channel_dim,self.input_channel_dim)
        xis = []
        for xi in x.split(split_size=3,dim=self.transform_channel_dim):
            tcdim = self.transform_channel_dim
            out = self.transform(xi)
            if not isinstance(out, torch.Tensor):
                out = T.ToTensor()(out)
                tcdim = 0
            if tcdim!=self.output_channel_dim:
                out = out.transpose(self.transform_channel_dim,self.output_channel_dim)
            xis.append(out)
        xis = torch.cat(xis, dim=self.output_channel_dim)
        return xis


class ListenerWrapper(nn.Module):
    def __init__(
        self,
        listener_agent:DiscriminativeListener,
        predicate_threshold:float=0.5,
    ):
        super(ListenerWrapper, self).__init__()
        self.listener_agent = listener_agent
        self.predicate_threshold = getattr(listener_agent, "predicate_threshold", predicate_threshold)
    
    def get_final_decision(
        self,
        sentences_token_idx,
        decision_probs,
    ) -> torch.Tensor :
        batch_size = sentences_token_idx.shape[0]
        #(batch_size, max_sentence_length)
        eos_mask = (sentences_token_idx==self.listener_agent.vocab_stop_idx)
        padding_with_eos = (eos_mask.cumsum(-1).sum()>batch_size)
        # Include first EoS Symbol:
        if padding_with_eos:
            token_mask = ((eos_mask.cumsum(-1)>1)<=0)
            lengths = token_mask.sum(-1)
            #(batch_size, )
        else:
            token_mask = ((eos_mask.cumsum(-1)>0)<=0)
            lengths = token_mask.sum(-1)
        
        if not(padding_with_eos):
            # If excluding first EoS:
            lengths = lengths.add(1)
        sentences_lengths = lengths.clamp(max=self.listener_agent.max_sentence_length)
        #(batch_size, )
    
        sentences_lengths = sentences_lengths.reshape(-1,1,1).expand(
            decision_probs.shape[0],
            1,
            decision_probs.shape[2]
        )
    
        final_decision_probs = decision_probs.gather(
            dim=1, 
            index=(sentences_lengths-1),
        ).squeeze(1)
        
        return final_decision_probs

    def forward(
        self,
        x:torch.Tensor,
        rnn_states:Dict[str,object],
        goal:torch.Tensor,
    ) -> torch.Tensor :
        
        self.listener_agent._tidyup()
        
        batch_size = x.shape[0]
        obs_shape = x.shape[1:]
        experiences = x.reshape((batch_size, 1, *obs_shape))
        
        max_sentence_length = self.listener_agent.max_sentence_length
        vocab_size = self.listener_agent.vocab_size

        if goal.shape[0] == 1:
            goal = goal.repeat(batch_size, *[1 for _ in goal.shape[1:]])
        sentences_widx = goal.long().reshape((batch_size, max_sentence_length))
        sentences_one_hots = nn.functional.one_hot(
            sentences_widx, 
            num_classes=vocab_size,
        ).float()
        
        if self.listener_agent.kwargs['use_cuda']:
            sentences_widx = sentences_widx.cuda()
            sentences_one_hots = sentences_one_hots.cuda()
            experiences = experiences.cuda()

        output_dict = self.listener_agent.forward(
            sentences=sentences_one_hots, 
            experiences=experiences, 
        )
        
        decision_probs = output_dict['decision']
        if self.listener_agent.kwargs['descriptive']:
            decision_probs = decision_probs.softmax(dim=-1)
        # (batch_size x max_sentence_length x nbr_distractors+2)
        elif self.listener_agent.kwargs['normalize_features']:
            # if not descriptive then we need to assert that we have normalized the features
            # and therefore the probs are between -1 and 1 and we can format them between 0 and 1:
            decision_probs = (decision_probs+1)/2
        # (batch_size x max_sentence_length x nbr_distractors+1)
        else:
            # since we do not know what are the possible values, we propose to apply tanh to squash the values:
            # it is applied inside the listener : decision_probs = torch.Tanh(decision_probs)
            decision_probs = (decision_probs+1)/2
        # (batch_size x max_sentence_length x nbr_distractors+1)
        
        final_decision_probs = self.get_final_decision(
            sentences_token_idx=sentences_widx,
            decision_probs=decision_probs,
        )
       
        decision = final_decision_probs[:, 0]
        
        #################################
        try:
            wandb.log({f"ListenerWrapper/PredicateThresholdDecisionProbs": self.predicate_threshold}, commit=False)
            wandb.log({f"ListenerWrapper/MeanDecisionProbsPerBatch":decision.sum().item()/batch_size}, commit=False)
            wandb.log({f"ListenerWrapper/StdDecisionProbsPerBatch":decision.std().item()}, commit=False)
            wandb.log({f"ListenerWrapper/MinDecisionProbsPerBatch":decision.min().item()}, commit=False)
            wandb.log({f"ListenerWrapper/MaxDecisionProbsPerBatch":decision.max().item()}, commit=False)
        except Exception as e:
            print(f"WARNING: ListenerWrapper :: W&B Logging: {e}")
        #################################
        
        self.listener_agent._tidyup()

        return decision


def batched_listener_based_goal_predicated_reward_fn(
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
    Relabelling an unsuccessful trajectory, so the desired_exp's goal is not achieved.
    We want to know whether the goal that is achieved on the :param target_exp:'s succ_s
    is achieved on the :param achieved_exp:'s succ_s.
    
    :param kwargs: must contain an entry 'listener' whose value is the listener agent to
    query to evaluate the alignement between the previously-mentioned goal and achieved succ_s.
    
    Returns :param feedbacks['failure']: for failure and :param feedbacks['success']: for success,
    unless :param kwargs['use_continuous_feedback']: is provided and True, then an interpolation
    between the previously mentioned values is performed.
    '''
    listener = kwargs['listener']

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
        listener_training = listener.training
        listener.train(False)

        target_pred_goal = predictor(
            x=target_state, 
            rnn_states=target_rnn_states,
        )
        target_descriptive_probs = listener(
            x=target_state, 
            rnn_states=target_rnn_states, 
            goal=target_pred_goal,
        ).cpu()
        
        #achieved_pred_goal = predictor(x=state, rnn_states=rnn_states).cpu()
        descriptive_probs = listener(x=state, rnn_states=rnn_states, goal=target_pred_goal).cpu()
        
        predictor.train(training)
        listener.train(listener_training)
    
    target_pred_goal = target_pred_goal.cpu()
    listener.predicate_threshold = target_descriptive_probs.item()-1.0e-4
    wandb.log({f"ListenerWrapper/TargerPredicateDecisionProbs": target_descriptive_probs.item()}, commit=False)
    
    if kwargs.get("use_continuous_feedback", False):
        reward_range = feedbacks['success']-feedbacks['failure']
        # (batch_size, )
        reward = descriptive_probs.unsqueeze(-1)*reward_range + feedbacks['failure']
        # (batch_size, 1)
    else:
        reward_mask = descriptive_probs > listener.predicate_threshold
        reward = reward_mask.unsqueeze(-1)*feedbacks["success"]*torch.ones(reward_shape)
        reward += (~reward_mask.unsqueeze(-1))*feedbacks["failure"]*torch.ones(reward_shape)
    
    return reward, target_pred_goal, target_latent_goal


class ETHERAlgorithmWrapper(THERAlgorithmWrapper2):
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
        feedbacks={"failure":-1, "success":0},
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
        
        super(ETHERAlgorithmWrapper, self).__init__(
            algorithm=algorithm,
            extra_inputs_infos=extra_inputs_infos,
            predictor=predictor, 
            predictor_loss_fn=predictor_loss_fn, 
            strategy=strategy, 
            goal_predicated_reward_fn=goal_predicated_reward_fn,
            _extract_goal_from_info_fn=_extract_goal_from_info_fn,
            achieved_goal_key_from_info=achieved_goal_key_from_info,
            target_goal_key_from_info=target_goal_key_from_info,
            achieved_latent_goal_key_from_info=achieved_latent_goal_key_from_info,
            target_latent_goal_key_from_info=target_latent_goal_key_from_info,
            filtering_fn=filtering_fn,
            feedbacks=feedbacks,
            relabel_terminal=relabel_terminal,
            filter_predicate_fn=filter_predicate_fn,
            filter_out_timed_out_episode=filter_out_timed_out_episode,
            timing_out_episode_length_threshold=timing_out_episode_length_threshold,
            episode_length_reward_shaping=episode_length_reward_shaping,
            train_contrastively=train_contrastively,
            contrastive_training_nbr_neg_examples=contrastive_training_nbr_neg_examples,
        )
        
        self.rg_storages = None
        self._reset_rg_storages()
        self.hook_fns.append(
            ETHERAlgorithmWrapper.referential_game_store,
        )
        self.non_unique_data = 0
        self.nbr_data = 0 

        self.ether_test_acc = 0.0
        
        self.rg_iteration = 0
        self.vocabulary = self.predictor.model.modules['InstructionGenerator'].vocabulary
        self.idx2w = self.predictor.model.modules['InstructionGenerator'].idx2w
        
        self.init_referential_game()
        self.goal_predicated_reward_fn_kwargs = {
            'listener': ListenerWrapper(self.listener),
            'use_continuous_feedback': self.kwargs.get('ETHER_use_continuous_feedback', False),
        }
        
        if self.kwargs.get("ETHER_listener_based_predicated_reward_fn", False):
            self.goal_predicated_reward_fn = partial(
                    #batched_predictor_based_goal_predicated_reward_fn2, 
                    batched_listener_based_goal_predicated_reward_fn,
                    predictor=self.predictor,
                )


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
        if 'ETHER_rg_PER_beta_increase_interval' in self.kwargs and self.kwargs['ETHER_rg_PER_beta_increase_interval']!='None':
            beta_increase_interval = float(self.kwargs['ETHER_rg_PER_beta_increase_interval'])  

        for i in range(nbr_storages):
            if self.kwargs.get('ETHER_use_PER', False):
                raise NotImplementedError
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
                    )
                )
            else:
                self.rg_storages.append(
                    SplitReplayStorage(
                        capacity=int(self.kwargs['ETHER_replay_capacity']),
                        keys=keys,
                        circular_keys=circular_keys,
                        circular_offsets=circular_offsets,
                        test_train_split_interval=self.kwargs['ETHER_test_train_split_interval'],
                        test_capacity=int(self.kwargs['ETHER_test_replay_capacity']),
                        lock_test_storage=self.kwargs['ETHER_lock_test_storage'],
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
        if self.kwargs['ETHER_use_PER']:
            init_sampling_priority = None 
            self.rg_storages[actor_index].add(exp_dict, priority=init_sampling_priority, test_set=test_set)
        else:
        '''
        # CHECK for uniqueness:
        self.nbr_data += 1 
        wandb.log({f"Training/ETHER/NbrData":self.nbr_data}, commit=False)
        if "symbolic_image" in exp_dict['info']:
            unique = True
            for idx in range(len(self.rg_storages[actor_index])):
                if all((self.rg_storages[actor_index].info[0][idx]['symbolic_image'] == exp_dict['info']['symbolic_image']).reshape(-1)):
                    self.non_unique_data += 1
                    #self.nbr_data = self.rg_storages[actor_index].get_size()+self.rg_storages[actor_index].get_size(test=True)
                    unique = False
                    break
        
            wandb.log({f"Training/ETHER/NonUniqueDataRatio":float(self.non_unique_data)/(self.nbr_data+1)}, commit=False)
            wandb.log({f"Training/ETHER/NonUniqueDataNbr": self.non_unique_data}, commit=False)
        
            if self.kwargs['ETHER_rg_filter_out_non_unique'] \
            and not unique:  
                return

        self.rg_storages[actor_index].add(exp_dict, test_set=test_set)

    def init_referential_game(self):
        ReferentialGym.datasets.dataset.DSS_version = self.kwargs["ETHER_rg_distractor_sampling_scheme_version"]
        print(f"DSS_version = {ReferentialGym.datasets.dataset.DSS_version}.")
        ReferentialGym.datasets.dataset.OC_version = self.kwargs["ETHER_rg_object_centric_version"]
        print(f"OC_version = {ReferentialGym.datasets.dataset.OC_version}.")
        ReferentialGym.datasets.dataset.DC_version = self.kwargs["ETHER_rg_descriptive_version"]
        #if args.descriptive_version == 2:
        #    args.batch_size = args.batch_size // 2
        print(f"DC_version = {ReferentialGym.datasets.dataset.DC_version} and BS={self.kwargs['ETHER_rg_batch_size']}.")
        
        try:
            obs_instance = getattr(self.rg_storages[0], self.kwargs['ETHER_exp_key'])[0][0]
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
        if self.kwargs["ETHER_rg_with_color_jitter_augmentation"]:
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
                p=self.kwargs['ETHER_rg_color_jitter_prob'])]+transformations
        
        if self.kwargs["ETHER_rg_with_gaussian_blur_augmentation"]:
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
                p=self.kwargs['ETHER_rg_gaussian_blur_prob'])]+transformations
        
        from ReferentialGym.datasets.utils import AddEgocentricInvariance
        ego_inv_transform = AddEgocentricInvariance()
        
        transform_degrees = self.kwargs["ETHER_rg_egocentric_tr_degrees"]
        transform_translate = float(self.kwargs["ETHER_rg_egocentric_tr_xy"])/stimulus_resize_dim
        transform_translate = (transform_translate, transform_translate)
        
        if self.kwargs["ETHER_rg_egocentric"]:
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
                p=self.kwargs['ETHER_rg_egocentric_prob'],
            )
            transformations = [
                #rand_split_img_ego_tr,
                #rand_split_img_affine_tr,
                rand_split_img_ego_affine_tr,
                *transformations,
            ]
        
        self.rg_transformation = T.Compose(transformations)
        
        default_descriptive_ratio = 1-(1/(self.kwargs["ETHER_rg_nbr_train_distractors"]+2))
        # Default: 1-(1/(nbr_distractors+2)), 
        # otherwise the agent find the local minimum
        # where it only predicts "no-target"...
        if self.kwargs["ETHER_rg_descriptive_ratio"] <=0.001:
            descriptive_ratio = default_descriptive_ratio
        else:
            descriptive_ratio = self.kwargs["ETHER_rg_descriptive_ratio"]

        
        rg_config = {
            "observability":            self.kwargs["ETHER_rg_observability"],
            "max_sentence_length":      self.kwargs["ETHER_rg_max_sentence_length"],
            "nbr_communication_round":  1,
            "nbr_distractors":          {"train":self.kwargs["ETHER_rg_nbr_train_distractors"], "test":self.kwargs["ETHER_rg_nbr_test_distractors"]},
            "distractor_sampling":      self.kwargs["ETHER_rg_distractor_sampling"],
            # Default: use "similarity-0.5"
            # otherwise the emerging language 
            # will have very high ambiguity...
            # Speakers find the strategy of uttering
            # a word that is relevant to the class/label
            # of the target, seemingly.  
            
            "descriptive":              self.kwargs["ETHER_rg_descriptive"],
            "descriptive_target_ratio": descriptive_ratio,
            
            "object_centric":           self.kwargs["ETHER_rg_object_centric"],
            "nbr_stimulus":             1,
            
            "graphtype":                self.kwargs["ETHER_rg_graphtype"],
            "tau0":                     0.2,
            "gumbel_softmax_eps":       1e-6,
            "vocab_size":               self.kwargs["ETHER_rg_vocab_size"],
            "force_eos":                self.kwargs["ETHER_rg_force_eos"],
            "symbol_embedding_size":    self.kwargs["ETHER_rg_symbol_embedding_size"], #64
            
            "agent_architecture":       self.kwargs["ETHER_rg_arch"], #'CoordResNet18AvgPooled-2', #'BetaVAE', #'ParallelMONet', #'BetaVAE', #'CNN[-MHDPA]'/'[pretrained-]ResNet18[-MHDPA]-2'
            "shared_architecture":      self.kwargs["ETHER_rg_shared_architecture"],
            "normalize_features":       self.kwargs["ETHER_rg_normalize_features"],
            "agent_learning":           "learning",  #"transfer_learning" : CNN"s outputs are detached from the graph...
            "agent_loss_type":          self.kwargs["ETHER_rg_agent_loss_type"], #"NLL"
            
            #"cultural_pressure_it_period": self.kwargs["ETHER_rg_cultural_pressure_it_period"],
            #"cultural_speaker_substrate_size":  self.kwargs["ETHER_rg_cultural_speaker_substrate_size"],
            #"cultural_listener_substrate_size":  self.kwargs["ETHER_rg_cultural_listener_substrate_size"],
            #"cultural_reset_strategy":  self.kwargs["ETHER_rg_cultural_reset_strategy"], #"oldestL", # "uniformSL" #"meta-oldestL-SGD"
            #"cultural_pressure_parameter_filtering_fn":  cultural_pressure_param_filtering_fn,
            #"cultural_pressure_meta_learning_rate":  self.kwargs["ETHER_rg_cultural_pressure_meta_learning_rate"],
            
            # Cultural Bottleneck:
            #"iterated_learning_scheme": self.kwargs["ETHER_rg_iterated_learning_scheme"],
            #"iterated_learning_period": self.kwargs["ETHER_rg_iterated_learning_period"],
            #"iterated_learning_rehearse_MDL": self.kwargs["ETHER_rg_iterated_learning_rehearse_MDL"],
            #"iterated_learning_rehearse_MDL_factor": self.kwargs["ETHER_rg_iterated_learning_rehearse_MDL_factor"],
             
            # Obverter Hyperparameters:
            "obverter_stop_threshold":  self.kwargs["ETHER_rg_obverter_threshold_to_stop_message_generation"],  #0.0 if not in use.
            "obverter_nbr_games_per_round": self.kwargs["ETHER_rg_obverter_nbr_games_per_round"],
            
            "obverter_least_effort_loss": False,
            "obverter_least_effort_loss_weights": [1.0 for x in range(0, 10)],
            
            "batch_size":               self.kwargs["ETHER_rg_batch_size"],
            "dataloader_num_worker":    self.kwargs["ETHER_rg_dataloader_num_worker"],
            "stimulus_depth_dim":       stimulus_depth_dim, #1 if "dSprites" in args.dataset else 3,
            "stimulus_resize_dim":      stimulus_resize_dim, 
            
            "learning_rate":            self.kwargs["ETHER_rg_learning_rate"], #1e-3,
            "weight_decay":             self.kwargs["ETHER_rg_weight_decay"],
            "adam_eps":                 1e-16,
            "dropout_prob":             self.kwargs["ETHER_rg_dropout_prob"],
            "embedding_dropout_prob":   self.kwargs["ETHER_rg_emb_dropout_prob"],
            
            "with_gradient_clip":       False,
            "gradient_clip":            1e0,
            
            "use_homoscedastic_multitasks_loss": self.kwargs["ETHER_rg_homoscedastic_multitasks_loss"],
            
            "use_feat_converter":       self.kwargs["ETHER_rg_use_feat_converter"],
            
            "use_curriculum_nbr_distractors": self.kwargs["ETHER_rg_use_curriculum_nbr_distractors"],
            "init_curriculum_nbr_distractors": self.kwargs["ETHER_rg_init_curriculum_nbr_distractors"],
            "curriculum_distractors_window_size": 25, #100,
            
            "unsupervised_segmentation_factor": None, #1e5
            "nbr_experience_repetition":  self.kwargs["ETHER_rg_nbr_experience_repetition"],
            
            "with_utterance_penalization":  False,
            "with_utterance_promotion":     False,
            "utterance_oov_prob":  0.5,  # Expected penalty of observing out-of-vocabulary words. 
            # The greater this value, the greater the loss/cost.
            "utterance_factor":    1e-2,
            
            "with_speaker_entropy_regularization":  False,
            "with_listener_entropy_regularization":  False,
            "entropy_regularization_factor":    -1e-2,
            
            "with_logits_mdl_principle":       self.kwargs['ETHER_rg_with_logits_mdl_principle'],
            "logits_mdl_principle_factor":     self.kwargs['ETHER_rg_logits_mdl_principle_factor'],
            "logits_mdl_principle_accuracy_threshold":     self.kwargs['ETHER_rg_logits_mdl_principle_accuracy_threshold'],
            
            "with_mdl_principle":       False,
            "mdl_principle_factor":     5e-2,
            
            "with_weight_maxl1_loss":   False,
            
            "use_cuda":                 self.kwargs["ETHER_rg_use_cuda"],
            
            "train_transform":            self.rg_transformation,
            "test_transform":             self.rg_transformation,
        }
        
        ## Agent Configuration:
        agent_config = copy.deepcopy(rg_config)
        agent_config["nbr_distractors"] = rg_config["nbr_distractors"]["train"] if rg_config["observability"] == "full" else 0

        # Obverter:
        if 'obverter' in self.kwargs["ETHER_rg_graphtype"]:
            agent_config["use_obverter_threshold_to_stop_message_generation"] = self.kwargs["ETHER_rg_obverter_threshold_to_stop_message_generation"]

        # Recurrent Convolutional Architecture:
        agent_config["architecture"] = rg_config["agent_architecture"]
         
        if "3xCNN" in agent_config["architecture"]:
            if "BN" in self.kwargs["ETHER_rg_arch"]:
                agent_config["cnn_encoder_channels"] = ["BN32","BN64","BN128"]
            else:
                agent_config["cnn_encoder_channels"] = [32,64,128]
         
            if "3x3" in agent_config["architecture"]:
                agent_config["cnn_encoder_kernels"] = [3,3,3]
            elif "7x4x3" in agent_config["architecture"]:
                agent_config["cnn_encoder_kernels"] = [7,4,3]
            else:
                agent_config["cnn_encoder_kernels"] = [4,4,4]
            agent_config["cnn_encoder_strides"] = [2,2,2]
            agent_config["cnn_encoder_paddings"] = [1,1,1]
            agent_config["cnn_encoder_fc_hidden_units"] = []#[128,] 
            # the last FC layer is provided by the cnn_encoder_feature_dim parameter below...
            
            # For a fair comparison between CNN an VAEs:
            #agent_config["cnn_encoder_feature_dim"] = args.vae_nbr_latent_dim
            agent_config["cnn_encoder_feature_dim"] = self.kwargs["ETHER_rg_agent_nbr_latent_dim"]
            #agent_config["cnn_encoder_feature_dim"] = self.kwargs["ETHER_rg_symbol_processing_nbr_hidden_units"]
            # N.B.: if cnn_encoder_fc_hidden_units is [],
            # then this last parameter does not matter.
            # The cnn encoder is not topped by a FC network.
            
            agent_config["cnn_encoder_mini_batch_size"] = self.kwargs["ETHER_rg_mini_batch_size"]
            agent_config["feat_converter_output_size"] = 256
            
            agent_config["temporal_encoder_nbr_hidden_units"] = 0
            agent_config["temporal_encoder_nbr_rnn_layers"] = 0
            agent_config["temporal_encoder_mini_batch_size"] = self.kwargs["ETHER_rg_mini_batch_size"]
            agent_config["symbol_processing_nbr_hidden_units"] = self.kwargs["ETHER_rg_symbol_processing_nbr_hidden_units"]
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
        speaker.set_vocabulary(self.vocabulary)
        print("Speaker:", speaker)
        self.speaker = speaker

        listener_config = copy.deepcopy(agent_config)
        if self.kwargs["ETHER_rg_shared_architecture"]:
            listener_config['cnn_encoder'] = speaker.cnn_encoder 
        listener_config['nbr_distractors'] = rg_config['nbr_distractors']['train']
        
        if 'obverter' in self.kwargs["ETHER_rg_graphtype"]:
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
        listener.set_vocabulary(self.vocabulary)
        print("Listener:", listener)
        self.listener = listener

        ## Train set:
        self.rg_split_strategy = self.kwargs["ETHER_split_strategy"]
        self.rg_train_dataset_length = self.kwargs["ETHER_train_dataset_length"]
        self.rg_test_dataset_length = self.kwargs["ETHER_test_dataset_length"]
        self.rg_exp_key = self.kwargs["ETHER_exp_key"]
        
        ## Modules:
        modules = {}
        modules[speaker.id] = speaker 
        modules[listener.id] = listener 
        
        from ReferentialGym import modules as rg_modules
        
        if self.kwargs["ETHER_rg_use_obverter_sampling"]:
            obverter_sampling_id = "obverter_sampling_0"
            obverter_sampling_config = {
                "batch_size": rg_config["batch_size"],
                "round_alternation_only": self.kwargs["ETHER_rg_obverter_sampling_round_alternation_only"],
                "obverter_nbr_games_per_round": self.kwargs["ETHER_rg_obverter_nbr_games_per_round"],
                "repeat_experiences": self.kwargs["ETHER_rg_obverter_sampling_repeat_experiences"],
            }
            
            modules[obverter_sampling_id] = rg_modules.ObverterDatasamplingModule(
                id=obverter_sampling_id,
                config=obverter_sampling_config,
            )
  
        # Population:
        population_handler_id = "population_handler_0"
        population_handler_config = copy.deepcopy(rg_config)
        population_handler_config["verbose"] = self.kwargs["ETHER_rg_verbose"]
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
        
        ortho_id = "ortho_0"
        ortho_config = {}
        ortho_input_stream_ids = {
            "logger":"modules:logger:ref",
            "logs_dict":"logs_dict",
            "epoch":"signals:epoch",
            "it_rep":"signals:it_sample",
            "it_comm_round":"signals:it_step",
            "mode":"signals:mode",

            "agent":"modules:current_speaker:ref:ref_agent",
            "representations":"modules:current_speaker:ref:ref_agent:model:modules:InstructionGenerator:semantic_embedding:weight",
        }
        
        if self.kwargs.get("ETHER_rg_with_ortho_metric", False):
            modules[ortho_id] = rg_modules.build_OrthogonalityMetricModule(
                id=ortho_id,
                config=ortho_config,
                input_stream_ids=ortho_input_stream_ids,
            )

        if self.kwargs.get("ETHER_rg_use_semantic_cooccurrence_grounding", False):
            sem_cooc_grounding_id = "sem_cooccurrence_grounding_0"
            sem_cooc_grounding_config = {
                "lambda_factor": self.kwargs.get("ETHER_rg_semantic_cooccurrence_grounding_lambda", 1.0),
                "sentence_level_lambda_factor": self.kwargs.get("ETHER_rg_semantic_cooccurrence_grounding_sentence_level_lambda", 1.0),
                "noise_magnitude": self.kwargs.get("ETHER_rg_semantic_cooccurrence_grounding_noise_magnitude", 0.0),
                "semantic_level_grounding": self.kwargs.get("ETHER_rg_semantic_cooccurrence_grounding_semantic_level", False),
                "semantic_level_ungrounding": self.kwargs.get("ETHER_rg_semantic_cooccurrence_grounding_semantic_level_ungrounding", False),
                "sentence_level_grounding": self.kwargs.get("ETHER_rg_semantic_cooccurrence_grounding_sentence_level", False),
                "sentence_level_ungrounding": self.kwargs.get("ETHER_rg_semantic_cooccurrence_grounding_sentence_level_ungrounding", False),
            }
            modules[sem_cooc_grounding_id] = rg_modules.build_CoOccurrenceSemanticGroundingLossModule(
                id=sem_cooc_grounding_id,
                config=sem_cooc_grounding_config,
            )

        if self.kwargs.get("ETHER_rg_with_semantic_grounding_metric", False):
            sem_grounding_id = "sem_grounding_metric_0"
            sem_grounding_config = {
                'idx2w':self.idx2w,
                'semantic_percentiles': [50,75,90,95],
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
            "learning_rate":self.kwargs["ETHER_rg_learning_rate"],
            "weight_decay":self.kwargs["ETHER_rg_weight_decay"],
            "l1_reg_lambda":self.kwargs["ETHER_rg_l1_weight_decay"],
            "l2_reg_lambda":self.kwargs["ETHER_rg_l2_weight_decay"],
            "optimizer_type":self.kwargs["ETHER_rg_optimizer_type"],
            "with_gradient_clip":rg_config["with_gradient_clip"],
            "adam_eps":rg_config["adam_eps"],
        }
        
        optim_module = rg_modules.build_OptimizationModule(
            id=optim_id,
            config=optim_config,
        )
        modules[optim_id] = optim_module
        
        if self.kwargs["ETHER_rg_homoscedastic_multitasks_loss"]:
            homo_id = "homo0"
            homo_config = {"use_cuda":self.kwargs["ETHER_rg_use_cuda"]}
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
            if self.kwargs["ETHER_rg_use_cuda"]:
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
                "metric_fast": self.kwargs["ETHER_rg_metric_fast"],
                "pvalue_significance_threshold": 0.05,
                "parallel_TS_computation_max_workers":self.kwargs["ETHER_rg_parallel_TS_worker"],
                "filtering_fn":(lambda kwargs: speaker.role=="speaker"),
                #"postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
                # cf outputs of _utter:
                "postprocess_fn": agent_postprocess_fn, #(lambda x: x[1].cpu().detach().numpy()),
                # not necessary if providing a preprocess_fn, 
                # that computes the features/_sense output, but here it is in order to deal with shapes:
                "features_postprocess_fn": agent_features_postprocess_fn, #(lambda x: x[-1].cpu().detach().numpy()),
                #"preprocess_fn": (lambda x: x.cuda() if self.kwargs["ETHER_rg_use_cuda"] else x),
                # cf _sense:
                "preprocess_fn": (lambda x: speaker._sense(agent_preprocess_fn(x))),
                #"epoch_period":args.epoch-1, 
                "epoch_period": self.kwargs["ETHER_rg_metric_epoch_period"],
                "batch_size":self.kwargs["ETHER_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ETHER_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ETHER_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ETHER_rg_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ETHER_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ETHER_rg_metric_active_factors_only"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ETHER_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ETHER_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ETHER_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ETHER_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ETHER_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ETHER_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ETHER_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ETHER_rg_metric_active_factors_only"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ETHER_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ETHER_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ETHER_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ETHER_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ETHER_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ETHER_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ETHER_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ETHER_rg_metric_active_factors_only"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ETHER_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ETHER_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ETHER_rg_metric_batch_size"],#5,
                "nbr_train_points": self.kwargs["ETHER_rg_nbr_train_points"],#3000,
                "nbr_eval_points": self.kwargs["ETHER_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ETHER_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ETHER_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ETHER_rg_metric_active_factors_only"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ETHER_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ETHER_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ETHER_rg_metric_batch_size"],#5,
                "nbr_train_points": self.kwargs["ETHER_rg_nbr_train_points"],#3000,
                "nbr_eval_points": self.kwargs["ETHER_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ETHER_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ETHER_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ETHER_rg_metric_active_factors_only"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ETHER_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ETHER_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ETHER_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ETHER_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ETHER_rg_nbr_eval_points"],#2000,
                "resample":self.kwargs["ETHER_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ETHER_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ETHER_rg_metric_active_factors_only"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ETHER_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ETHER_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ETHER_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ETHER_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ETHER_rg_nbr_eval_points"],#2000,
                "resample":self.kwargs["ETHER_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ETHER_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ETHER_rg_metric_active_factors_only"],
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
        if self.kwargs.get("ETHER_rg_sanity_check_compactness_ambiguity_metric", False):
            compactness_ambiguity_metric_input_stream_ids["representations"] = \
                "current_dataloader:sample:speaker_grounding_signal"
            compactness_ambiguity_metric_input_stream_ids["top_view"] = "current_dataloader:sample:speaker_top_view" 
            compactness_ambiguity_metric_input_stream_ids["agent_pos_in_top_view"] = "current_dataloader:sample:speaker_agent_pos_in_top_view" 
            
        compactness_ambiguity_metric_module = rg_modules.build_CompactnessAmbiguityMetricModule(
            id=compactness_ambiguity_metric_id,
            input_stream_ids=compactness_ambiguity_metric_input_stream_ids,
            config = {
                'sanity_check_shuffling': False,
                "show_stimuli": False, #True,
                "postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ETHER_rg_use_cuda"] else x),
                "epoch_period":1,#self.kwargs["ETHER_rg_metric_epoch_period"],
                "batch_size":self.kwargs["ETHER_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ETHER_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ETHER_rg_nbr_eval_points"],#2000,
                "resample": False, #self.kwargs["ETHER_rg_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ETHER_rg_seed"],
                "verbose":False,
                "idx2w": self.idx2w,
                "kwargs": self.kwargs,
            }
        )
        modules[compactness_ambiguity_metric_id] = compactness_ambiguity_metric_module

        speaker_posbosdis_metric_id = "speaker_posbosdis_metric"
        speaker_posbosdis_metric_input_stream_ids = {
            #"model":"modules:current_speaker:ref:ref_agent",
            "model":"modules:current_speaker:ref:ref_agent:_utter",
            "representations":"modules:current_speaker:sentences_widx",
            "experiences":"current_dataloader:sample:speaker_experiences", 
            "latent_representations":"current_dataloader:sample:speaker_exp_latents", 
            #"latent_values_representations":"current_dataloader:sample:speaker_exp_latents_values",
            "indices":"current_dataloader:sample:speaker_indices", 
        }

        speaker_posbosdis_metric_module = rg_modules.build_PositionalBagOfSymbolsDisentanglementMetricModule(
            id=speaker_posbosdis_metric_id,
            input_stream_ids=speaker_posbosdis_metric_input_stream_ids,
            config = {
                "postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ETHER_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ETHER_rg_metric_epoch_period"],
                "batch_size":self.kwargs["ETHER_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ETHER_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ETHER_rg_nbr_eval_points"],#2000,
                "resample":self.kwargs["ETHER_rg_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ETHER_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ETHER_rg_metric_active_factors_only"],
            }
        )
        modules[speaker_posbosdis_metric_id] = speaker_posbosdis_metric_module

        logger_id = "per_epoch_logger"
        logger_module = rg_modules.build_PerEpochLoggerModule(id=logger_id)
        modules[logger_id] = logger_module
      
        pipelines["referential_game"] = [population_handler_id]
        if self.kwargs["ETHER_rg_use_obverter_sampling"]:
            pipelines["referential_game"].append(obverter_sampling_id)
        if "synthetic" in self.kwargs["ETHER_rg_graphtype"]:
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
        if self.kwargs.get("ETHER_rg_with_ortho_metric", False):
            pipelines[optim_id].append(ortho_id)
        if self.kwargs.get("ETHER_rg_use_semantic_cooccurrence_grounding", False):
            pipelines[optim_id].append(sem_cooc_grounding_id)
        if self.kwargs.get("ETHER_rg_with_semantic_grounding_metric", False):
            pipelines[optim_id].append(sem_grounding_id)
        if self.kwargs["ETHER_rg_homoscedastic_multitasks_loss"]:
            pipelines[optim_id].append(homo_id)
        pipelines[optim_id].append(optim_id)
        """
        # Add gradient recorder module for debugging purposes:
        pipelines[optim_id].append(grad_recorder_id)
        """
        if self.kwargs["ETHER_rg_dis_metric_epoch_period"] != 0:
            if not(self.kwargs["ETHER_rg_shared_architecture"]):
                pipelines[optim_id].append(listener_factor_vae_disentanglement_metric_id)
                pipelines[optim_id].append(listener_modularity_disentanglement_metric_id)
                pipelines[optim_id].append(listener_mig_disentanglement_metric_id)
            pipelines[optim_id].append(speaker_factor_vae_disentanglement_metric_id)
            pipelines[optim_id].append(speaker_modularity_disentanglement_metric_id)
            pipelines[optim_id].append(speaker_mig_disentanglement_metric_id)
    
        pipelines[optim_id].append(speaker_topo_sim_metric_id)
        pipelines[optim_id].append(compactness_ambiguity_metric_id)
        pipelines[optim_id].append(speaker_posbosdis_metric_id)
        if "obverter" in self.kwargs["ETHER_rg_graphtype"]:
            pipelines[optim_id].append(listener_topo_sim_metric_id)
            pipelines[optim_id].append(listener_posbosdis_metric_id)
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
        assert len(self.predictor_storages)==1
        kwargs = {'same_episode_target': False}
        if 'similarity' in self.rg_config['distractor_sampling']:
            kwargs['same_episode_target'] = True 

        extra_keys_dict = {
            #"rnn_states":"rnn_states",
            "grounding_signal":self.kwargs.get("ETHER_grounding_signal_key", None),
        }
        if self.kwargs.get("ETHER_with_Oracle", False):
            extra_keys_dict["rnn_states"] = "info:achieved_goal"

        if self.kwargs.get("ETHER_rg_sanity_check_compactness_ambiguity_metric", False):
            extra_keys_dict.update({
                "top_view":"info:top_view",
                "agent_pos_in_top_view":"info:agent_pos_in_top_view",
            })
        if self.kwargs.get("ETHER_rg_with_semantic_grounding_metric", False):
            extra_keys_dict.update({
                "semantic_signal":"info:symbolic_image",
            })
         
        self.rg_train_dataset = DemonstrationDataset(
            replay_storage=self.rg_storages[0],
            train=True,
            transform=self.rg_transformation,
            split_strategy=self.rg_split_strategy,
            dataset_length=self.rg_train_dataset_length,
            exp_key=self.rg_exp_key,
            extra_keys_dict=extra_keys_dict,
            latents_build_fn=self.kwargs['ETHER_rg_latents_build_fn'],
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
            latents_build_fn=self.kwargs['ETHER_rg_latents_build_fn'],
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
            "descriptive":              self.rg_config["descriptive"],
            "descriptive_target_ratio": self.rg_config["descriptive_target_ratio"],
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
            "descriptive":              self.rg_config["descriptive"],
            "descriptive_target_ratio": self.rg_config["descriptive_target_ratio"],
        }

        self.dataset_args = dataset_args

    def update_predictor(self, successful_traj=False):
        period_check = self.kwargs['THER_replay_period']
        period_count_check = self.nbr_buffered_predictor_experience
        
        # Update predictor:
        can_update_predictor = False
        if self.kwargs.get('THER_use_THER_predictor_supervised_training', False):
            assert self.kwargs['THER_use_THER_predictor_supervised_training_data_collection']
            if self.nbr_handled_predictor_experience >= self.kwargs['THER_min_capacity']:
                can_update_predictor = True
        if can_update_predictor \
        and ((period_count_check % period_check == 0) or (self.kwargs['THER_train_on_success'] and successful_traj)):
            self._update_predictor()
        
        # RG Update:
        period_check = self.kwargs['ETHER_rg_training_period']
        period_count_check = self.nbr_buffered_predictor_experience
        can_rg_train = False
        if len(self.rg_storages[0])>=self.kwargs['ETHER_replay_capacity']:
            can_rg_train = True
        quotient = period_count_check // period_check
        previous_quotient = getattr(self, 'previous_ETHER_quotient', 0)
        if can_rg_train \
        and quotient != previous_quotient:
        #and (period_count_check % period_check == 0):
            self.previous_ETHER_quotient = quotient
            if self.kwargs['ETHER_use_ETHER']:
                self._rg_training()
        
        wandb.log({'Training/ETHER/storage_length': len(self.rg_storages[0])}, commit=False)

    def _update_predictor(self):	
        full_update = True
        for it in range(self.kwargs['THER_nbr_training_iteration_per_update']):
            self.test_acc = self.train_predictor()
            if self.test_acc >= self.kwargs['THER_predictor_accuracy_threshold']:
                full_update = False
                break
        wandb.log({f"Training/THER_Predictor/TestAccuracy":self.test_acc}, commit=False)
        wandb.log({f"Training/THER_Predictor/FullUpdate":int(full_update)}, commit=False)
    
    def _rg_training(self):
        full_update = True
        for it in range(self.kwargs['ETHER_rg_nbr_epoch_per_update']):
            #self.test_acc = self.train_predictor()
            if self.kwargs['ETHER_use_supervised_training']:
                assert self.kwargs['THER_use_THER_predictor_supervised_training_data_collection']
                self._update_predictor()
            self.ether_test_acc = self.finetune_predictor(update=(it==0))
            if self.ether_test_acc >= self.kwargs['ETHER_rg_accuracy_threshold']:
                full_update = False
                break
        wandb.log({f"Training/ETHER/TestAccuracy":self.ether_test_acc}, commit=False)
        wandb.log({f"Training/ETHER/FullUpdate":int(full_update)}, commit=False)

    def finetune_predictor(self, update=False):
        if self.rg_iteration == 0:
            ###
            save_path = os.path.join(wandb.run.dir, f"referential_game")
            print(f"ETHER: Referential Game NEW PATH: {save_path}")
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
            '''
            wandb.watch(
                self.speaker, 
                log='gradients',
                log_freq=32,
                log_graph=False,
            )
            '''
            wandb.watch(
                self.listener, 
                log='gradients',
                log_freq=32,
                log_graph=False,
            )
            
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
        #self.launch_referential_game(nbr_epoch=self.kwargs["ETHER_rg_nbr_epoch_per_update"])
        self.launch_referential_game(nbr_epoch=1)
        end = time.time()
        
        wandb.log({'PerETHERUpdate/TimeComplexity/ReferentialGame':  end-start}, commit=False) # self.param_update_counter)
        
        '''
        start = time.time()
        end = time.time()
        
        wandb.log({'PerTHERPredictorUpdate/TimeComplexity/OptimizeModelFn':  end-start}, commit=False) # self.param_update_counter)
        
        successful_update = int(updated_acc >= best_acc)
        wandb.log({f"Training/THER_Predictor/SuccessfulUpdate":successful_update}, commit=False)
        if not successful_update:
            self.predictor.load_state_dict(self.best_predictor.state_dict())
            self.predictor_optimizer.load_state_dict(self.best_predictor_optimizer_sd)
            acc = best_acc
        else:
            self.best_predictor.load_state_dict(self.predictor.state_dict())
            self.best_predictor_optimizer_sd = self.predictor_optimizer.state_dict()
            acc = updated_acc 

        wandb.log({'PerTHERPredictorUpdate/TestSentenceAccuracy': acc, "ther_predictor_update_count":self.param_predictor_update_counter}, commit=True)
        '''
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
        cloned_algo = ETHERAlgorithmWrapper(
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

