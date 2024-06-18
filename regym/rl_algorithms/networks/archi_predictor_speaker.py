from typing import Dict

from functools import partial
import copy
import numpy as np
import torch
import torch.nn as nn

from .archi_predictor import ArchiPredictor

from ReferentialGym.agents import Speaker
from ReferentialGym.networks import layer_init, BetaVAE
from ReferentialGym.utils import gumbel_softmax
from regym.rl_algorithms.utils import concat_fn, _concatenate_list_hdict

class ArchiPredictorSpeaker(ArchiPredictor, Speaker):
    def __init__(
        self,
        model,
        pipeline_name="instruction_generator",
        generator_name="InstructionGenerator",
        trainable=True,
        postprocess_fn=None,
        **kwargs,
    ):
        ArchiPredictor.__init__(
            self,
            model=model,
            pipeline_name=pipeline_name,
            generator_name=generator_name,
            **kwargs,
        )
        self.logger = None
        self.trainable = trainable
        self.postprocess_fn = postprocess_fn
         
        '''
        Speaker.__init__(
            self,
            obs_shape=kwargs.get("obs_shape", [16,56,56]),
            vocab_size=kwargs.get("vocab_size", 100),
            max_sentence_length=kwargs.get("max_sentence_length", 10),
            agent_id="s0",
            logger=None,
            kwargs=kwargs,
        )
        '''
    
    def set_postprocess_fn(self, postprocess_fn):
        self.postprocess_fn = postprocess_fn
            
    def save(self, path, filename=None):
        postprocess_fn = self.postprocess_fn
        self.postprocess_fn = None 
        
        logger = self.logger
        self.logger = None
        
        if filename is None:
            filepath = path+self.id+".agent"
        else:
            filepath = os.path.join(path, filename)
        torch.save(self, filepath)
        
        self.logger = logger 
        self.postprocess_fn = postprocess_fn
        return 

    def clone(self, clone_proxies=False, minimal=False):
        self.reset()
        postprocess_fn = self.postprocess_fn
        self.postprocess_fn = None
        cloned = Speaker.clone(self)
        if clone_proxies:
            cloned.postprocess_fn = postprocess_fn
        self.postprocess_fn = postprocess_fn
        return cloned

    def speaker_init(
        self,
        obs_shape,
        vocab_size,
        max_sentence_length,
        agent_id,
        kwargs,
        logger=None,
    ):
        model = self.model
        pkwargs = self.archi_kwargs

        Speaker.__init__(
            self,
            obs_shape=obs_shape,
            vocab_size=vocab_size,
            max_sentence_length=max_sentence_length,
            agent_id=agent_id,
            logger=logger,
            kwargs=kwargs,
        )

        ArchiPredictor.__init__(
            self,
            model=model,
            pipeline_name=self.pipeline_name,
            generator_name=self.generator_name,
            **pkwargs,
        )

        feature_module = self.model.pipelines[self.pipeline_name][0]
        self.cnn_encoder = self.model.modules[feature_module]

        self.tau_fc = nn.Sequential(
            nn.Linear(self.archi_kwargs['hyperparameters']['hidden_dim'], 1,bias=False),
            nn.Softplus(),
        )
        
        self.reset_weights(reset_language_model=True)

    def _tidyup(self):
        """
        Called at the agent level at the end of the `compute` function.
        """
        self.embedding_tf_final_outputs = None

        if isinstance(self.cnn_encoder, BetaVAE):
            self.VAE_losses = list()
            self.compactness_losses.clear()
            self.buffer_cnn_output_dict = dict()

    def reset(self):
        self.features = None
        if hasattr(self, 'tau'):
            self.tau = None
        self._reset_rnn_states()

    def reset_weights(self, reset_language_model=False): 
        # TODO: implement language model reset if
        # wanting to use iterated learning or cultural pressures...
        self.features = None
        if hasattr(self, 'tau_fc'):
            self.tau_fc.apply(layer_init)
        if hasattr(self, 'tau'):
            self.tau = None
        self._reset_rnn_states()
  
    def parameters(self):
        params = []
        
        if not self.trainable:
            return params
        
        if hasattr(self, 'tau_fc'):
            #print(f"WARNING: Speaker INIT: Tau_FC parameters included for optimization")
            params += self.tau_fc.parameters()
        
        for km, module in self.model.modules.items():
            if km in self.model.pipelines[self.pipeline_name]: #["instruction_generator"]:
                params += module.parameters()
        
        return params

    def _compute_tau(self, tau0, h):
        batch_size = h.shape[0]
        if self.trainable:
            tau = 1.0 / (self.tau_fc(h).squeeze() + tau0)
        else:
            tau = tau0*torch.ones(batch_size, 1).to(h.device)
        return tau
    
    def _utter(self, features, sentences=None, rnn_states=None):
        """
        Reasons about the features and the listened sentences, if multi_round, to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: None, or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - word indices: Tensor of shape `(batch_size, max_sentence_length, 1)` of type `long` containing the indices of the words that make up the sentences.
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        """
        batch_size = features.shape[0]
        extra_rnn_states = self.model.get_reset_states({
            'repeat':batch_size,
            'cuda':self.kwargs['use_cuda'],
            }
        )
        if rnn_states is None:
            rnn_states = extra_rnn_states
        else:
            for k,v in extra_rnn_states.items():
                if k in rnn_states: continue
                rnn_states[k] = v
				
        input_dict = {
            'obs':features,
            'rnn_states': rnn_states,
        }
        gt_sentences = None 
        return_feature_only = None 
            
        output = self.model.forward(
            **input_dict,
            pipelines={
                #"instruction_generator":
                self.pipeline_name:self.archi_kwargs["pipelines"][self.pipeline_name], #"instruction_generator"]
            },
            return_feature_only=return_feature_only,
        )
        
        if self.postprocess_fn is not None:
            output = self.postprocess_fn(
                output=output,
                predictor=self,
            )
        
        sentences_widx = output["next_rnn_states"][self.generator_name]["processed_input0"][0].unsqueeze(-1)
        sentences_logits = output["next_rnn_states"][self.generator_name]["input0_prediction_logits"][0]
        sentences_hidden_states = output["next_rnn_states"][self.generator_name]["input0_hidden_states"][0]
        sentences_one_hots = nn.functional.one_hot(
                sentences_widx.long(), 
                num_classes=self.vocab_size,
        ).float()
        # (batch_size, max_sentence_length, vocab_size)
        feature_module = self.model.pipelines[self.pipeline_name][0]
        if 'processed_input' in output['next_rnn_states'][feature_module].keys():
            self.features = output['next_rnn_states'][feature_module]['processed_input'][0]
        else:
            self.features = sentences_hidden_states
         
        temporal_features = None

        return sentences_hidden_states, sentences_widx, sentences_logits, sentences_one_hots, temporal_features
   
    def forward(self, **kwargs):
        if 'experiences' in kwargs:
            return self.speaker_forward(**kwargs)
        
        return self.predictor_forward(**kwargs)

    def predictor_forward(
        self,
        x,
        gt_sentences=None,
        rnn_states=None,
    ):
        batch_size = x.shape[0]
        if rnn_states is None:
            rnn_states = self.model.get_reset_states({
                'repeat':batch_size,
                'cuda':self.kwargs['use_cuda'],
                }
            )
        else:
            for k,v in self.model.get_reset_states().items():
                if k in rnn_states: continue
                rnn_states[k] = v
				
        input_dict = {
            'obs':x,
            'rnn_states': rnn_states,
        }
         
        if gt_sentences is None:
            return_feature_only=self.archi_kwargs["features_id"][self.pipeline_name]
        else:
            return_feature_only = None 
            input_dict['rnn_states']['gt_sentences'] = gt_sentences
            
        if return_feature_only is None:
            output = self.model.forward(
                **input_dict,
                pipelines={
                    self.pipeline_name:self.archi_kwargs["pipelines"][self.pipeline_name]
                },
                return_feature_only=return_feature_only,
            )
        else:
            prediction = self.model.forward(
                **input_dict,
                pipelines={
                    self.pipeline_name:self.archi_kwargs["pipelines"][self.pipeline_name]
                },
                return_feature_only=None, #return_feature_only,
            )
            output = self.model.stream_handler[return_feature_only]
            prediction['output'] = output
            return prediction
             
        return output
    
    def speaker_forward(
        self, 
        experiences, 
        sentences=None, 
        multi_round=False, 
        graphtype="straight_through_gumbel_softmax", 
        tau0=0.2,
        sample=None,
    ):
        """
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                            `experiences[:,0]` is assumed as the target experience, while the others are distractors, if any. 
        :param multi_round: Boolean defining whether to utter a sentence back or not.
        :param graphtype: String defining the type of symbols used in the output sentence:
                    - `'categorical'`: one-hot-encoded symbols.
                    - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                    - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                    - `'obverter'`: obverter training scheme...
        :param tau0: Float, temperature with which to apply gumbel-softmax estimator.
        """
        batch_size = experiences.size(0)
        #features = self._sense(experiences=experiences, sentences=sentences)
        if len(experiences.shape)==5:
            features = experiences.reshape(-1, experiences.shape[-1])
        else:
            features = experiences.view(-1, *(experiences.size()[2:]))
        rnn_states = None
        if sample is not None \
        and hasattr(sample, 'speaker_rnn_states'):
            rnn_states = sample.speaker_rnn_states
            '''
            rnn_states = _concatenate_list_hdict(
                lhds=[rnn_states[idx][0][0] for idx in range(batch_size)],
                concat_fn=partial(torch.cat, dim=0),
                #preprocess_fn=(lambda x:
                #    torch.from_numpy(x).unsqueeze(0) if isinstance(x, np.ndarray) else torch.ones(1, 1)*x                   ),
            )
            '''
            rnn_states = {'InstructionGenerator':{'achieved_goal':rnn_states}}
        utter_outputs = self._utter(features=features, sentences=None, rnn_states=rnn_states)
        if len(utter_outputs) == 5:
            next_sentences_hidden_states, next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
        else:
            next_sentences_hidden_states = None
            next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
        
        if self.training and self.trainable:
            if "gumbel_softmax" in graphtype:    
                if next_sentences_hidden_states is None: 
                    self.tau = self._compute_tau(tau0=tau0)
                    #tau = self.tau.view((-1,1,1)).repeat(1, self.max_sentence_length, self.vocab_size)
                    tau = self.tau.view((-1))
                    # (batch_size)
                else:
                    self.tau = []
                    for hs in next_sentences_hidden_states:
                        self.tau.append( self._compute_tau(tau0=tau0, h=hs).view((-1)))
                    # list of size batch_size containing Tensors of shape (sentence_length)
                    tau = self.tau 
                    
                straight_through = (graphtype == "straight_through_gumbel_softmax")
                
                next_sentences_stgs = []
                for bidx in range(len(next_sentences_logits)):
                    nsl_in = next_sentences_logits[bidx]
                    # (sentence_length<=max_sentence_length, vocab_size)
                    tau_in = tau[bidx].view((-1,1))
                    # (1, 1) or (sentence_length, 1)
                    stgs = gumbel_softmax(logits=nsl_in, tau=tau_in, hard=straight_through, dim=-1, eps=self.kwargs["gumbel_softmax_eps"])
                    
                    next_sentences_stgs.append(stgs)
                    #next_sentences_stgs.append( nn.functional.gumbel_softmax(logits=nsl_in, tau=tau_in, hard=straight_through, dim=-1))
                next_sentences = next_sentences_stgs
                if isinstance(next_sentences, list): 
                    next_sentences = nn.utils.rnn.pad_sequence(next_sentences, batch_first=True, padding_value=0.0).float()
                    # (batch_size, max_sentence_length<=max_sentence_length, vocab_size)

        output_dict = {"sentences_widx":next_sentences_widx, 
                       "sentences_logits":next_sentences_logits, 
                       "sentences_one_hot":next_sentences,
                       #"features":features,
                       "temporal_features":temporal_features}
        
        if not multi_round:
            self._reset_rnn_states()

        return output_dict


