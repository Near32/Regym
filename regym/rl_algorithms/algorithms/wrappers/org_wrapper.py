from typing import Dict, Optional, List, Union

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

from regym.rl_algorithms.algorithms.algorithm import Algorithm 
from regym.rl_algorithms.algorithms.wrappers.algorithm_wrapper import AlgorithmWrapper

from regym.rl_algorithms.utils import (
    archi_concat_fn, 
    _extract_rnn_states_from_batch_indices, 
    _extract_rnn_states_from_seq_indices, 
    apply_on_hdict,
    _concatenate_hdict, 
    _concatenate_list_hdict, 
    copy_hdict,
)

import ReferentialGym
from ReferentialGym.agents import (
    DiscriminativeListener, 
    LSTMObsListener, 
    LSTMCNNListener,
    LSTMObsSpeaker,
)



def S2B_postprocess_fn(
    output:Dict[str,object],
    predictor:nn.Module,
    algorithm:Union[Algorithm,AlgorithmWrapper],
    **kwargs,
    ):
    
    current_actor = algorithm.current_actor
    current_env = algorithm.venv.unwrapped.env_processes[current_actor].env.env.env
    
    action = output['a']
    decoded_action = current_env._decode_action(action.squeeze(-1).detach().cpu().numpy())
    sentences_widx = np.concatenate(
        [d['communication_channel'] for d in decoded_action],
        axis=0,
    )
    
    batch_size = action.shape[0]
    max_sentence_length = predictor.max_sentence_length
    vocab_size = predictor.vocab_size
     
    unreg_sentences_widx = torch.from_numpy(sentences_widx).float().to(action.device)
    # (batch_size, nbr_latents)
    sentences_widx = predictor.vocab_stop_idx*torch.ones((batch_size, max_sentence_length)).to(action.device)
    sentences_widx[:,:unreg_sentences_widx.shape[1]] = unreg_sentences_widx
    
    if predictor.generator_name not in output['next_rnn_states']:
        output['next_rnn_states'][predictor.generator_name] = {}
     
    output['next_rnn_states'][predictor.generator_name]['processed_input0'] = [sentences_widx]
    
    predicted_logits = torch.zeros(
        batch_size, max_sentence_length, vocab_size,
    ).to(sentences_widx.device)
    # batch_size x max_sentence_length x vocab_size 
    predicted_logits = predicted_logits.scatter_(
        dim=-1,
        index=sentences_widx.unsqueeze(-1,
            ).repeat(1,1,vocab_size).long(),
        src=torch.ones_like(predicted_logits),
    )
    hidden_dim = 512 #predictor.kwargs['processing_hidden_units']
    hidden_states = torch.zeros(batch_size, max_sentence_length, hidden_dim).to(sentences_widx.device)
    # batch_size x max_sentence_length x hidden_state_dim=1=dummy

    output["next_rnn_states"][predictor.generator_name]["input0_prediction_logits"] = [predicted_logits]
    output["next_rnn_states"][predictor.generator_name]["input0_hidden_states"] = [hidden_states]
    
    return output


###########################################################
###########################################################
###########################################################


class OnlineReferentialGameAlgorithmWrapper(AlgorithmWrapper):
    def __init__(
        self, 
        algorithm, 
        predictor, 
        ):
        """
        """
        
        super(OnlineReferentialGameAlgorithmWrapper, self).__init__(
            algorithm=algorithm,
        )
        
        self.predictor = predictor 
        self.episode_count = 0
        
        self.rg_iteration = 0
        
        self.init_referential_game()
    
    def async_actor(self):
        return self.algorithm.async_actor()
        
    def set_venv(self, venv):
        self.venv = venv
        return 

    def store(self, exp_dict, actor_index=0):
        self.algorithm.store(exp_dict, actor_index=actor_index)
        if not(exp_dict['non_terminal']):
            self.episode_count += 1
            
            self.current_actor = actor_index 
            self.update_agents(exp_dict)
            self.run()
            self.regularise_agents()
        else:
            pass
    
    def update_agents(self, exp_dict):
        # Extract rnn_states from current_actor
        self.new_reset_states = copy_hdict(exp_dict['next_rnn_states'])
        # Squeezing out:
        if self.kwargs['vdn']:
            self.new_reset_states = _extract_rnn_states_from_seq_indices(
                self.new_reset_states,
                seq_indices=[0], #Assuming we are recovering the speaker agent here.
                filter_fn= lambda x: len(x.shape)>=3,
            )
            def reg(x):
                if len(x.shape)==3:
                    outx = x.squeeze(-2)
                elif len(x.shape)==1:
                    outx = x.unsqueeze(-1)
                else:
                    outx = x
                return outx
            self.new_reset_states = apply_on_hdict(
                hdict=self.new_reset_states,
                fn=reg,
            )
        
        # Set rnn_states as default reset_state for copy of model in agents
        if hasattr(self.speaker, 'get_reset_states'):
            self.old_reset_states = self.speaker.get_reset_states()
            self.speaker.set_reset_states(self.new_reset_states)

        ## Listener :
        if self.kwargs['ORG_rg_reset_listener_each_training']:
            self.listener.reset_weights(whole=True)
        return 
    
    def regularise_agents(self):
        if hasattr(self.speaker, 'set_reset_states'):
            self.speaker.set_reset_states(
                self.old_reset_states,
            )
        return 
            
    def init_referential_game(self):
        ReferentialGym.datasets.dataset.DSS_version = self.kwargs["ORG_rg_distractor_sampling_scheme_version"]
        print(f"DSS_version = {ReferentialGym.datasets.dataset.DSS_version}.")
        ReferentialGym.datasets.dataset.OC_version = self.kwargs["ORG_rg_object_centric_version"]
        print(f"OC_version = {ReferentialGym.datasets.dataset.OC_version}.")
        ReferentialGym.datasets.dataset.DC_version = self.kwargs["ORG_rg_descriptive_version"]
        #if args.descriptive_version == 2:
        #    args.batch_size = args.batch_size // 2
        print(f"DC_version = {ReferentialGym.datasets.dataset.DC_version} and BS={self.kwargs['ORG_rg_batch_size']}.")
        
        obs_instance = None
        obs_shape = self.kwargs["preprocessed_observation_shape"]
        # Format is [ depth [x w x h]]
        #stimulus_depth_dim = obs_shape[0]

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
        if self.kwargs["ORG_rg_with_color_jitter_augmentation"]:
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
                p=self.kwargs['ORG_rg_color_jitter_prob'])]+transformations
        
        if self.kwargs["ORG_rg_with_gaussian_blur_augmentation"]:
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
                p=self.kwargs['ORG_rg_gaussian_blur_prob'])]+transformations
        
        from ReferentialGym.datasets.utils import AddEgocentricInvariance
        ego_inv_transform = AddEgocentricInvariance()
        
        transform_degrees = self.kwargs["ORG_rg_egocentric_tr_degrees"]
        transform_translate = float(self.kwargs["ORG_rg_egocentric_tr_xy"])/stimulus_resize_dim
        transform_translate = (transform_translate, transform_translate)
        
        if self.kwargs["ORG_rg_egocentric"]:
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
                p=self.kwargs['ORG_rg_egocentric_prob'],
            )
            transformations = [
                #rand_split_img_ego_tr,
                #rand_split_img_affine_tr,
                rand_split_img_ego_affine_tr,
                *transformations,
            ]
        
        self.rg_transformation = T.Compose(transformations)
        
        default_descriptive_ratio = 1-(1/(self.kwargs["ORG_rg_nbr_train_distractors"]+2))
        # Default: 1-(1/(nbr_distractors+2)), 
        # otherwise the agent find the local minimum
        # where it only predicts "no-target"...
        if self.kwargs["ORG_rg_descriptive_ratio"] <=0.001:
            descriptive_ratio = default_descriptive_ratio
        else:
            descriptive_ratio = self.kwargs["ORG_rg_descriptive_ratio"]

        
        rg_config = {
            "observability":            self.kwargs["ORG_rg_observability"],
            "max_sentence_length":      self.kwargs["ORG_rg_max_sentence_length"],
            "nbr_communication_round":  1,
            "nbr_distractors":          {"train":self.kwargs["ORG_rg_nbr_train_distractors"], "test":self.kwargs["ORG_rg_nbr_test_distractors"]},
            "distractor_sampling":      self.kwargs["ORG_rg_distractor_sampling"],
            # Default: use "similarity-0.5"
            # otherwise the emerging language 
            # will have very high ambiguity...
            # Speakers find the strategy of uttering
            # a word that is relevant to the class/label
            # of the target, seemingly.  
            
            "descriptive":              self.kwargs["ORG_rg_descriptive"],
            "descriptive_target_ratio": descriptive_ratio,
            
            "object_centric":           self.kwargs["ORG_rg_object_centric"],
            "object_centric_type":      self.kwargs["ORG_rg_object_centric_type"],
            "nbr_stimulus":             1,
            
            "graphtype":                self.kwargs["ORG_rg_graphtype"],
            "tau0":                     self.kwargs["ORG_rg_tau0"],
            "gumbel_softmax_eps":       1e-6,
            "vocab_size":               self.kwargs["ORG_rg_vocab_size"],
            "force_eos":                self.kwargs["ORG_rg_force_eos"],
            "symbol_embedding_size":    self.kwargs["ORG_rg_symbol_embedding_size"], #64
            
            "agent_architecture":       self.kwargs["ORG_rg_arch"], #'CoordResNet18AvgPooled-2', #'BetaVAE', #'ParallelMONet', #'BetaVAE', #'CNN[-MHDPA]'/'[pretrained-]ResNet18[-MHDPA]-2'
            "shared_architecture":      self.kwargs["ORG_rg_shared_architecture"],
            "normalize_features":       self.kwargs["ORG_rg_normalize_features"],
            "agent_learning":           "learning",  #"transfer_learning" : CNN"s outputs are detached from the graph...
            "agent_loss_type":          self.kwargs["ORG_rg_agent_loss_type"], #"NLL"
            
            #"cultural_pressure_it_period": self.kwargs["ORG_rg_cultural_pressure_it_period"],
            #"cultural_speaker_substrate_size":  self.kwargs["ORG_rg_cultural_speaker_substrate_size"],
            #"cultural_listener_substrate_size":  self.kwargs["ORG_rg_cultural_listener_substrate_size"],
            #"cultural_reset_strategy":  self.kwargs["ORG_rg_cultural_reset_strategy"], #"oldestL", # "uniformSL" #"meta-oldestL-SGD"
            #"cultural_pressure_parameter_filtering_fn":  cultural_pressure_param_filtering_fn,
            #"cultural_pressure_meta_learning_rate":  self.kwargs["ORG_rg_cultural_pressure_meta_learning_rate"],
            
            # Cultural Bottleneck:
            #"iterated_learning_scheme": self.kwargs["ORG_rg_iterated_learning_scheme"],
            #"iterated_learning_period": self.kwargs["ORG_rg_iterated_learning_period"],
            #"iterated_learning_rehearse_MDL": self.kwargs["ORG_rg_iterated_learning_rehearse_MDL"],
            #"iterated_learning_rehearse_MDL_factor": self.kwargs["ORG_rg_iterated_learning_rehearse_MDL_factor"],
             
            # Obverter Hyperparameters:
            "obverter_stop_threshold":  self.kwargs["ORG_rg_obverter_threshold_to_stop_message_generation"],  #0.0 if not in use.
            "obverter_nbr_games_per_round": self.kwargs["ORG_rg_obverter_nbr_games_per_round"],
            
            "obverter_least_effort_loss": False,
            "obverter_least_effort_loss_weights": [1.0 for x in range(0, 10)],
            
            "batch_size":               self.kwargs["ORG_rg_batch_size"],
            "dataloader_num_worker":    self.kwargs["ORG_rg_dataloader_num_worker"],
            #"stimulus_depth_dim":       stimulus_depth_dim, #1 if "dSprites" in args.dataset else 3,
            "stimulus_resize_dim":      stimulus_resize_dim, 
            
            "learning_rate":            self.kwargs["ORG_rg_learning_rate"], #1e-3,
            "weight_decay":             self.kwargs["ORG_rg_weight_decay"],
            "adam_eps":                 1e-16,
            "dropout_prob":             self.kwargs["ORG_rg_dropout_prob"],
            "embedding_dropout_prob":   self.kwargs["ORG_rg_emb_dropout_prob"],
            
            "with_gradient_clip":       False,
            "gradient_clip":            1e0,
            
            "use_homoscedastic_multitasks_loss": self.kwargs["ORG_rg_homoscedastic_multitasks_loss"],
            
            "use_feat_converter":       self.kwargs["ORG_rg_use_feat_converter"],
            
            "use_curriculum_nbr_distractors": self.kwargs["ORG_rg_use_curriculum_nbr_distractors"],
            "init_curriculum_nbr_distractors": self.kwargs["ORG_rg_init_curriculum_nbr_distractors"],
            "curriculum_distractors_window_size": 25, #100,
            
            "unsupervised_segmentation_factor": None, #1e5
            "nbr_experience_repetition":  self.kwargs["ORG_rg_nbr_experience_repetition"],
            
            "with_utterance_penalization":  False,
            "with_utterance_promotion":     False,
            "utterance_oov_prob":  0.5,  # Expected penalty of observing out-of-vocabulary words. 
            # The greater this value, the greater the loss/cost.
            "utterance_factor":    1e-2,
            
            "with_speaker_entropy_regularization":  False,
            "with_listener_entropy_regularization":  False,
            "entropy_regularization_factor":    -1e-2,
            
            "with_logits_mdl_principle":       self.kwargs['ORG_rg_with_logits_mdl_principle'],
            "logits_mdl_principle_factor":     self.kwargs['ORG_rg_logits_mdl_principle_factor'],
            "logits_mdl_principle_accuracy_threshold":     self.kwargs['ORG_rg_logits_mdl_principle_accuracy_threshold'],
            
            "with_mdl_principle":       False,
            "mdl_principle_factor":     5e-2,
            
            "with_weight_maxl1_loss":   False,
            
            "use_cuda":                 self.kwargs["ORG_rg_use_cuda"],
            
            "train_transform":            self.rg_transformation,
            "test_transform":             self.rg_transformation,
        }
        
        ## Agent Configuration:
        agent_config = copy.deepcopy(rg_config)
        agent_config["nbr_distractors"] = rg_config["nbr_distractors"]["train"] if rg_config["observability"] == "full" else 0

        # Obverter:
        if 'obverter' in self.kwargs["ORG_rg_graphtype"]:
            agent_config["use_obverter_threshold_to_stop_message_generation"] = self.kwargs["ORG_rg_obverter_threshold_to_stop_message_generation"]

        # Recurrent Convolutional Architecture:
        agent_config["architecture"] = rg_config["agent_architecture"]
         
        if "3xCNN" in agent_config["architecture"]:
            if "BN" in self.kwargs["ORG_rg_arch"]:
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
            agent_config["cnn_encoder_feature_dim"] = self.kwargs["ORG_rg_agent_nbr_latent_dim"]
            #agent_config["cnn_encoder_feature_dim"] = self.kwargs["ORG_rg_symbol_processing_nbr_hidden_units"]
            # N.B.: if cnn_encoder_fc_hidden_units is [],
            # then this last parameter does not matter.
            # The cnn encoder is not topped by a FC network.
            
            agent_config["cnn_encoder_mini_batch_size"] = self.kwargs["ORG_rg_mini_batch_size"]
            agent_config["feat_converter_output_size"] = 256
            
            agent_config["temporal_encoder_nbr_hidden_units"] = 0
            agent_config["temporal_encoder_nbr_rnn_layers"] = 0
            agent_config["temporal_encoder_mini_batch_size"] = self.kwargs["ORG_rg_mini_batch_size"]
            agent_config["symbol_processing_nbr_hidden_units"] = self.kwargs["ORG_rg_symbol_processing_nbr_hidden_units"]
            agent_config["symbol_processing_nbr_rnn_layers"] = 1
        elif 'MLP' in agent_config["architecture"]:
            """ 
            if "BN" in self.kwargs["ORG_rg_arch"]:
                agent_config["fc_hidden_units"] = ["BN32", "BN64", "BN128"]#[128,] 
                agent_config["fc_hidden_units"] = ["BN32", "BN64", "BN128"]#[128,] 
            else:
                agent_config["fc_hidden_units"] = [32, 64, 128]#[128,] 
            agent_config["mini_batch_size"] = self.kwargs["ORG_rg_mini_batch_size"]
            agent_config["feat_converter_output_size"] = 128 
            """ 
            agent_config["temporal_encoder_nbr_hidden_units"] = 0
            agent_config["temporal_encoder_nbr_rnn_layers"] = 0
            agent_config["temporal_encoder_mini_batch_size"] = self.kwargs["ORG_rg_mini_batch_size"]
            
            agent_config['use_feat_converter'] = False 
            agent_config["mini_batch_size"] = self.kwargs["ORG_rg_mini_batch_size"]
            agent_config["fc_hidden_units"] = [64,64 ]#[128,] 
            agent_config["feat_converter_output_size"] = 64 
            agent_config["symbol_processing_nbr_hidden_units"] = 64 #self.kwargs["ORG_rg_symbol_processing_nbr_hidden_units"]
            agent_config["symbol_processing_nbr_rnn_layers"] = 1
        else:
            raise NotImplementedError

        batch_size = 4
        nbr_distractors = 1 if 'partial' in rg_config['observability'] else agent_config['nbr_distractors']['train']
        nbr_stimulus = agent_config['nbr_stimulus']
        
        agent_obs_shape = [
            nbr_distractors+1,
            nbr_stimulus, 
            *obs_shape, #*kwargs['preprocessed_observation_shape'],
            #rg_config['stimulus_depth_dim'],
            #rg_config['stimulus_resize_dim'],
            #rg_config['stimulus_resize_dim']
        ]
        
        vocab_size = rg_config['vocab_size']
        max_sentence_length = rg_config['max_sentence_length']
        
        nbr_obs_dim = 15 #30
        
        if self.kwargs['ORG_use_predictor_as_speaker']:
            speaker = self.predictor
            speaker.speaker_init(
                kwargs=agent_config, 
                obs_shape=agent_obs_shape, 
                vocab_size=vocab_size, 
                max_sentence_length=max_sentence_length,
                agent_id='s0',
                logger=None
            )
            # It is not necessarry to change the input to be latents,
            # because the rnn_states is being replaced with containing the latents
            # by the archi predictor speaker.
            print("Speaker:", speaker)
        else:
            if self.kwargs["ORG_with_Oracle_listener"]:
                agent_obs_shape[-1] = nbr_obs_dim #15 #9 #15 
            speaker = LSTMObsSpeaker(
                kwargs=agent_config, 
                obs_shape=agent_obs_shape, 
                vocab_size=vocab_size, 
                max_sentence_length=max_sentence_length,
                agent_id='s0',
                logger=None
            )
            print("Speaker:", speaker)
            import ipdb; ipdb.set_trace()
            if self.kwargs["ORG_with_Oracle_listener"]:
                speaker.input_stream_ids["speaker"]["experiences"] = "current_dataloader:sample:speaker_exp_latents_one_hot_encoded.float" 
         
        if hasattr(self, 'vocabulary'):
            speaker.set_vocabulary(self.vocabulary)
        elif hasattr(speaker, 'vocabulary'):
            self.vocabulary = speaker.vocabulary
            self.idx2w = speaker.idx2w
        else:
            raise NotImplementedError
        self.speaker = speaker

        listener_config = copy.deepcopy(agent_config)
        #TODO : 
        if self.kwargs["ORG_with_Oracle_listener"]:
            agent_obs_shape[-1] = nbr_obs_dim #15 #9 #15 
        if self.kwargs["ORG_rg_shared_architecture"]:
            if len(obs_shape)==1:
                listener_config['obs_encoder'] = speaker.obs_encoder
            else:
                listener_config['cnn_encoder'] = speaker.cnn_encoder 
        listener_config['nbr_distractors'] = rg_config['nbr_distractors']['train']
        
        if self.kwargs['ORG_use_predictor_as_listener'] \
        or 'obverter' in self.kwargs["ORG_rg_graphtype"]:
            listener = copy.deepcopy(self.predictor)
            listener.listener_init(
                kwargs=listener_config,
                obs_shape=agent_obs_shape,
                vocab_size=vocab_size,
                max_sentence_length=max_sentence_length,
                agent_id='l0',
                logger=None,
            )
        elif len(obs_shape)==1:
            listener = LSTMObsListener(
                kwargs=listener_config, 
                obs_shape=agent_obs_shape, 
                vocab_size=vocab_size, 
                max_sentence_length=max_sentence_length,
                agent_id='l0',
                logger=None
            )
        elif len(obs_shape)==3:
            listener = LSTMCNNListener(
                kwargs=listener_config, 
                obs_shape=agent_obs_shape, 
                vocab_size=vocab_size, 
                max_sentence_length=max_sentence_length,
                agent_id='l0',
                logger=None
            )
        listener.set_vocabulary(self.vocabulary)
        print("Listener:", listener)
        import ipdb; ipdb.set_trace()
        if self.kwargs["ORG_with_Oracle_listener"]:
            #listener.input_stream_ids["listener"]["experiences"] = "current_dataloader:sample:listener_exp_latents" 
            listener.input_stream_ids["listener"]["experiences"] = "current_dataloader:sample:listener_exp_latents_one_hot_encoded" 
        self.listener = listener

        ## Train set:
        self.rg_split_strategy = self.kwargs["ORG_split_strategy"]
        self.rg_exp_key = self.kwargs["ORG_exp_key"]
        
        ## Modules:
        modules = {}
        modules[speaker.id] = speaker 
        modules[listener.id] = listener 
        
        from ReferentialGym import modules as rg_modules
        
        if self.kwargs["ORG_rg_use_aita_sampling"]:
            aita_sampling_id = "aita_sampling_0"
            aita_sampling_config = {
                "update_epoch_period": self.kwargs['ORG_rg_aita_update_epoch_period'],
                "max_workers": 8,
                "comprange": self.kwargs['ORG_rg_aita_levenshtein_comprange'],
            }
            
            modules[aita_sampling_id] = rg_modules.AITAModule(
                id=aita_sampling_id,
                config=aita_sampling_config,
            )
  
        if self.kwargs["ORG_rg_use_obverter_sampling"]:
            obverter_sampling_id = "obverter_sampling_0"
            obverter_sampling_config = {
                "batch_size": rg_config["batch_size"],
                "round_alternation_only": self.kwargs["ORG_rg_obverter_sampling_round_alternation_only"],
                "obverter_nbr_games_per_round": self.kwargs["ORG_rg_obverter_nbr_games_per_round"],
                "repeat_experiences": self.kwargs["ORG_rg_obverter_sampling_repeat_experiences"],
            }
            
            modules[obverter_sampling_id] = rg_modules.ObverterDatasamplingModule(
                id=obverter_sampling_id,
                config=obverter_sampling_config,
            )
  
        # Population:
        population_handler_id = "population_handler_0"
        population_handler_config = copy.deepcopy(rg_config)
        population_handler_config["verbose"] = self.kwargs["ORG_rg_verbose"]
        population_handler_config["agent_saving"] = True #False
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
        
        if self.kwargs.get("ORG_rg_with_ortho_metric", False):
            modules[ortho_id] = rg_modules.build_OrthogonalityMetricModule(
                id=ortho_id,
                config=ortho_config,
                input_stream_ids=ortho_input_stream_ids,
            )

        if self.kwargs.get("ORG_rg_use_semantic_cooccurrence_grounding", False):
            sem_cooc_grounding_id = "sem_cooccurrence_grounding_0"
            sem_cooc_grounding_config = {
                "lambda_factor": self.kwargs.get("ORG_rg_semantic_cooccurrence_grounding_lambda", 1.0),
                "sentence_level_lambda_factor": self.kwargs.get("ORG_rg_semantic_cooccurrence_grounding_sentence_level_lambda", 1.0),
                "noise_magnitude": self.kwargs.get("ORG_rg_semantic_cooccurrence_grounding_noise_magnitude", 0.0),
                "semantic_level_grounding": self.kwargs.get("ORG_rg_semantic_cooccurrence_grounding_semantic_level", False),
                "semantic_level_ungrounding": self.kwargs.get("ORG_rg_semantic_cooccurrence_grounding_semantic_level_ungrounding", False),
                "sentence_level_grounding": self.kwargs.get("ORG_rg_semantic_cooccurrence_grounding_sentence_level", False),
                "sentence_level_ungrounding": self.kwargs.get("ORG_rg_semantic_cooccurrence_grounding_sentence_level_ungrounding", False),
            }
            modules[sem_cooc_grounding_id] = rg_modules.build_CoOccurrenceSemanticGroundingLossModule(
                id=sem_cooc_grounding_id,
                config=sem_cooc_grounding_config,
            )

        if self.kwargs.get("ORG_rg_with_semantic_grounding_metric", False):
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
            "learning_rate":self.kwargs["ORG_rg_learning_rate"],
            "weight_decay":self.kwargs["ORG_rg_weight_decay"],
            "l1_reg_lambda":self.kwargs["ORG_rg_l1_weight_decay"],
            "l2_reg_lambda":self.kwargs["ORG_rg_l2_weight_decay"],
            "optimizer_type":self.kwargs["ORG_rg_optimizer_type"],
            "with_gradient_clip":rg_config["with_gradient_clip"],
            "adam_eps":rg_config["adam_eps"],
        }
        
        optim_module = rg_modules.build_OptimizationModule(
            id=optim_id,
            config=optim_config,
        )
        modules[optim_id] = optim_module
        
        if self.kwargs["ORG_rg_homoscedastic_multitasks_loss"]:
            homo_id = "homo0"
            homo_config = {"use_cuda":self.kwargs["ORG_rg_use_cuda"]}
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
            if self.kwargs["ORG_rg_use_cuda"]:
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
                "metric_fast": self.kwargs["ORG_rg_metric_fast"],
                "pvalue_significance_threshold": 0.05,
                "parallel_TS_computation_max_workers":self.kwargs["ORG_rg_parallel_TS_worker"],
                "filtering_fn":(lambda kwargs: speaker.role=="speaker"),
                #"postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
                # cf outputs of _utter:
                "postprocess_fn": agent_postprocess_fn, #(lambda x: x[1].cpu().detach().numpy()),
                # not necessary if providing a preprocess_fn, 
                # that computes the features/_sense output, but here it is in order to deal with shapes:
                "features_postprocess_fn": agent_features_postprocess_fn, #(lambda x: x[-1].cpu().detach().numpy()),
                #"preprocess_fn": (lambda x: x.cuda() if self.kwargs["ORG_rg_use_cuda"] else x),
                # cf _sense:
                "preprocess_fn": (lambda x: speaker._sense(agent_preprocess_fn(x))),
                #"epoch_period":args.epoch-1, 
                "epoch_period": self.kwargs["ORG_rg_metric_epoch_period"],
                "batch_size":self.kwargs["ORG_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ORG_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ORG_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ORG_rg_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ORG_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ORG_rg_metric_active_factors_only"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ORG_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ORG_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ORG_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ORG_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ORG_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ORG_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ORG_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ORG_rg_metric_active_factors_only"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ORG_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ORG_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ORG_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ORG_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ORG_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ORG_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ORG_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ORG_rg_metric_active_factors_only"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ORG_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ORG_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ORG_rg_metric_batch_size"],#5,
                "nbr_train_points": self.kwargs["ORG_rg_nbr_train_points"],#3000,
                "nbr_eval_points": self.kwargs["ORG_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ORG_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ORG_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ORG_rg_metric_active_factors_only"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ORG_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ORG_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ORG_rg_metric_batch_size"],#5,
                "nbr_train_points": self.kwargs["ORG_rg_nbr_train_points"],#3000,
                "nbr_eval_points": self.kwargs["ORG_rg_nbr_eval_points"],#2000,
                "resample": self.kwargs["ORG_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ORG_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ORG_rg_metric_active_factors_only"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ORG_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ORG_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ORG_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ORG_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ORG_rg_nbr_eval_points"],#2000,
                "resample":self.kwargs["ORG_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ORG_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ORG_rg_metric_active_factors_only"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ORG_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ORG_rg_dis_metric_epoch_period"],
                "batch_size":self.kwargs["ORG_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ORG_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ORG_rg_nbr_eval_points"],#2000,
                "resample":self.kwargs["ORG_rg_dis_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ORG_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ORG_rg_metric_active_factors_only"],
            }
        )
        modules[listener_mig_disentanglement_metric_id] = listener_mig_disentanglement_metric_module

        if self.kwargs['ORG_with_compactness_ambiguity_metric']:
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
            if self.kwargs.get("ORG_rg_sanity_check_compactness_ambiguity_metric", False):
                compactness_ambiguity_metric_input_stream_ids["representations"] = \
                    "current_dataloader:sample:speaker_grounding_signal"
                compactness_ambiguity_metric_input_stream_ids["top_view"] = "current_dataloader:sample:speaker_top_view" 
                compactness_ambiguity_metric_input_stream_ids["agent_pos_in_top_view"] = "current_dataloader:sample:speaker_agent_pos_in_top_view" 
                
            compactness_ambiguity_metric_module = rg_modules.build_CompactnessAmbiguityMetricModule(
                id=compactness_ambiguity_metric_id,
                input_stream_ids=compactness_ambiguity_metric_input_stream_ids,
                config = {
                    "sanity_check_shuffling": self.kwargs.get("ORG_rg_shuffling_sanity_check_compactness_ambiguity_metric", False),
                    'sanity_check_shuffling': False,
                    "show_stimuli": False, #True,
                    "postprocess_fn": (lambda x: x["sentences_widx"].cpu().detach().numpy()),
                    "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ORG_rg_use_cuda"] else x),
                    "epoch_period":1,#self.kwargs["ORG_rg_metric_epoch_period"],
                    "batch_size":self.kwargs["ORG_rg_metric_batch_size"],#5,
                    "nbr_train_points":self.kwargs["ORG_rg_nbr_train_points"],#3000,
                    "nbr_eval_points":self.kwargs["ORG_rg_nbr_eval_points"],#2000,
                    "resample": False, #self.kwargs["ORG_rg_metric_resampling"],
                    "threshold":5e-2,#0.0,#1.0,
                    "random_state_seed":self.kwargs["ORG_rg_seed"],
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
                "preprocess_fn": (lambda x: x.cuda() if self.kwargs["ORG_rg_use_cuda"] else x),
                "epoch_period":self.kwargs["ORG_rg_metric_epoch_period"],
                "batch_size":self.kwargs["ORG_rg_metric_batch_size"],#5,
                "nbr_train_points":self.kwargs["ORG_rg_nbr_train_points"],#3000,
                "nbr_eval_points":self.kwargs["ORG_rg_nbr_eval_points"],#2000,
                "resample":self.kwargs["ORG_rg_metric_resampling"],
                "threshold":5e-2,#0.0,#1.0,
                "random_state_seed":self.kwargs["ORG_rg_seed"],
                "verbose":False,
                "active_factors_only":self.kwargs["ORG_rg_metric_active_factors_only"],
            }
        )
        modules[speaker_posbosdis_metric_id] = speaker_posbosdis_metric_module

        logger_id = "per_epoch_logger"
        logger_module = rg_modules.build_PerEpochLoggerModule(id=logger_id)
        modules[logger_id] = logger_module
      
        pipelines["referential_game"] = [population_handler_id]
        if self.kwargs["ORG_rg_use_obverter_sampling"]:
            pipelines["referential_game"].append(obverter_sampling_id)
        if "synthetic" in self.kwargs["ORG_rg_graphtype"]:
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
        if self.kwargs.get("ORG_rg_with_ortho_metric", False):
            pipelines[optim_id].append(ortho_id)
        if self.kwargs.get("ORG_rg_use_semantic_cooccurrence_grounding", False):
            pipelines[optim_id].append(sem_cooc_grounding_id)
        if self.kwargs.get("ORG_rg_with_semantic_grounding_metric", False):
            pipelines[optim_id].append(sem_grounding_id)
        if self.kwargs["ORG_rg_homoscedastic_multitasks_loss"]:
            pipelines[optim_id].append(homo_id)
        pipelines[optim_id].append(optim_id)
        """
        # Add gradient recorder module for debugging purposes:
        pipelines[optim_id].append(grad_recorder_id)
        """
        if self.kwargs["ORG_rg_dis_metric_epoch_period"] != 0:
            if not(self.kwargs["ORG_rg_shared_architecture"]):
                pipelines[optim_id].append(listener_factor_vae_disentanglement_metric_id)
                pipelines[optim_id].append(listener_modularity_disentanglement_metric_id)
                pipelines[optim_id].append(listener_mig_disentanglement_metric_id)
            pipelines[optim_id].append(speaker_factor_vae_disentanglement_metric_id)
            pipelines[optim_id].append(speaker_modularity_disentanglement_metric_id)
            pipelines[optim_id].append(speaker_mig_disentanglement_metric_id)
    
        pipelines[optim_id].append(speaker_topo_sim_metric_id)
        if self.kwargs['ORG_with_compactness_ambiguity_metric']:
            pipelines[optim_id].append(compactness_ambiguity_metric_id)
        pipelines[optim_id].append(speaker_posbosdis_metric_id)
        if "obverter" in self.kwargs["ORG_rg_graphtype"]:
            pipelines[optim_id].append(listener_topo_sim_metric_id)
            pipelines[optim_id].append(listener_posbosdis_metric_id)
        pipelines[optim_id].append(inst_coord_metric_id)
        if self.kwargs["ORG_rg_use_aita_sampling"]:
            pipelines[optim_id].append(aita_sampling_id)
         
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
        kwargs = {'same_episode_target': False}
        if 'similarity' in self.rg_config['distractor_sampling']:
            kwargs['same_episode_target'] = True 
        
        self.rg_train_dataset = self.venv.unwrapped.env_processes[self.current_actor].unwrapped.datasets['test'].previous_datasets['train']
        self.rg_test_dataset = self.venv.unwrapped.env_processes[self.current_actor].unwrapped.datasets['test'].previous_datasets['test']
        
        #####
        self.rg_train_dataset.sampling_strategy = None
        self.rg_train_dataset.reset_sampling()
        self.rg_test_dataset.__init__( 
            train=False, 
            transform=self.rg_train_dataset.transform, 
            sampling_strategy=None,
            split_strategy=self.rg_train_dataset.split_strategy, 
            nbr_latents=self.rg_train_dataset.nbr_latents, 
            min_nbr_values_per_latent=self.rg_train_dataset.min_nbr_values_per_latent, 
            max_nbr_values_per_latent=self.rg_train_dataset.max_nbr_values_per_latent, 
            nbr_object_centric_samples=self.rg_train_dataset.nbr_object_centric_samples,
            prototype=self.rg_train_dataset,
        )
        ##### 
        
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
            'with_replacement':         self.kwargs['ORG_rg_distractor_sampling_with_replacement'],
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
            'with_replacement':         self.kwargs['ORG_rg_distractor_sampling_with_replacement'],
        }

        self.dataset_args = dataset_args

    def run(self):
        # RG Update:
        period_check = self.kwargs['ORG_rg_training_period']
        period_count_check = self.episode_count
        quotient = period_count_check // period_check
        previous_quotient = getattr(self, 'previous_ORG_quotient', 0)
        if quotient != previous_quotient:
            self.previous_ORG_quotient = quotient
            self._rg_training()
        return 
    
    def _rg_training(self):
        full_update = True
        for it in range(self.kwargs['ORG_rg_nbr_epoch_per_update']):
            self.test_acc = self.run_rg(update=(it==0))
            if self.test_acc >= self.kwargs['ORG_rg_accuracy_threshold']:
                full_update = False
                break
        wandb.log({f"Training/ORG/TestAccuracy":self.test_acc}, commit=False)
        wandb.log({f"Training/ORG/FullUpdate":int(full_update)}, commit=False)

    def run_rg(self, update=False):
        if self.rg_iteration == 0:
            ###
            save_path = os.path.join(wandb.run.dir, f"referential_game")
            print(f"ORG: Referential Game NEW PATH: {save_path}")
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
                log_freq=2,
                log_graph=False,
            )
            wandb.watch(
                self.listener, 
                log='gradients',
                log_freq=32,
                log_graph=False,
            )
            '''
            
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
        self.launch_referential_game(nbr_epoch=1)
        end = time.time()
        
        wandb.log({'PerORGUpdate/TimeComplexity/ReferentialGame':  end-start}, commit=False) # self.param_update_counter)
        
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
        cloned_algo = OnlineReferentialGameAlgorithmWrapper(
            algorithm=self.algorithm.clone(
                with_replay_buffer=with_replay_buffer,
                clone_proxies=clone_proxies,
                minimal=minimal
            ), 
            predictor=self.predictor.clone(
                clone_proxies=clone_proxies,
                minimal=minimal,
            ), 
        )
        return cloned_algo

