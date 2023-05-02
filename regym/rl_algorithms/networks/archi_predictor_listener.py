from typing import Dict

import copy
import torch
import torch.nn as nn

from .archi_predictor import ArchiPredictor

from ReferentialGym.agents import DiscriminativeListener
from ReferentialGym.networks import layer_init
from ReferentialGym.utils import gumbel_softmax


class ArchiPredictorListener(ArchiPredictor, DiscriminativeListener):
    def __init__(
        self,
        model,
        **kwargs,
    ):
        ArchiPredictor.__init__(
            self,
            model=model,
            **kwargs,
        )
        self.logger = None
        '''
        DiscriminativeListener.__init__(
            self,
            obs_shape=kwargs.get("obs_shape", [16,56,56]),
            vocab_size=kwargs.get("vocab_size", 100),
            max_sentence_length=kwargs.get("max_sentence_length", 10),
            agent_id="s0",
            logger=None,
            kwargs=kwargs,
        )
        '''
    
    def clone(self):
        self.reset()
        return DiscriminativeListener.clone(self)

    def listener_init(
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

        DiscriminativeListener.__init__(
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
            **pkwargs,
        )
        
        self.cnn_encoder = self.model.modules['SharedObsEncoder']

        self.tau_fc = nn.Sequential(
            nn.Linear(self.archi_kwargs['hyperparameters']['hidden_dim'], 1,bias=False),
            nn.Softplus(),
        )
        
        self.reset()

    def reset(self, reset_language_model=False):
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
        for km, module in self.model.modules.items():
            if km in self.model.pipelines["instruction_generator"]:
                params += module.parameters()
        if hasattr(self, 'tau_fc'):
            #print(f"WARNING: Speaker INIT: Tau_FC parameters included for optimization")
            params += self.tau_fc.parameters()
        return params

    def _compute_tau(self, tau0, h):
        tau = 1.0 / (self.tau_fc(h).squeeze() + tau0)
        return tau
    
    def _reason(self, sentences, features):
        """
        Reasons about the features and sentences to yield the target-prediction logits.
        
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        
        :returns:
            - decision_logits: Tensor of shape `(batch_size, self.obs_shape[1])` containing the target-prediction logits.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        """
        rnn_states = self.model.get_reset_states()
        input_dict = {
            'obs':features,
            'rnn_states': {
                **rnn_states,
                'sentences': sentences,
            },
        }
        gt_sentences = None 
        return_feature_only = None 
            
        output = self.model.forward(
            **input_dict,
            pipelines={
                "decision_generator":self.archi_kwargs["pipelines"]["decision_generator"]
            },
            return_feature_only=return_feature_only,
        )
        
        self.features = output['next_rnn_states']['SharedObsEncoder']['processed_input'][0]

        decision_logits = output["next_rnn_states"]["DecisionGenerator"]["decision"][0]
        import ipdb; ipdb.set_trace()
        # TODO : find out dimensions :
        # (batch_size, max sentence_lengths, (nbr_distractors+1)=1 most likely, 2 )
        
        temporal_features = None

        return decision_logits, temporal_features
   
     
    def _utter(self, features, sentences=None):
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
        rnn_states = self.model.get_reset_states()
        input_dict = {
            'obs':features,
            'rnn_states': {
                **rnn_states,
                'sentences': None,
            },
        }
        gt_sentences = None 
        return_feature_only = None 
            
        output = self.model.forward(
            **input_dict,
            pipelines={
                "instruction_generator":self.archi_kwargs["pipelines"]["instruction_generator"]
            },
            return_feature_only=return_feature_only,
        )
        
        self.features = output['next_rnn_states']['SharedObsEncoder']['processed_input'][0]

        sentences_widx = output["next_rnn_states"]["InstructionGenerator"]["processed_input0"][0].unsqueeze(-1)
        sentences_logits = output["next_rnn_states"]["InstructionGenerator"]["input0_prediction_logits"][0]
        sentences_hidden_states = output["next_rnn_states"]["InstructionGenerator"]["input0_hidden_states"][0]
        sentences_one_hots = nn.functional.one_hot(
                sentences_widx.long(), 
                num_classes=self.vocab_size,
        ).float()
        # (batch_size, max_sentence_length, vocab_size)
        
        temporal_features = None

        return sentences_hidden_states, sentences_widx, sentences_logits, sentences_one_hots, temporal_features
   
    def forward(self, **kwargs):
        if 'experiences' in kwargs:
            return self.listener_forward(**kwargs)
        
        # TODO : add condition for predicate forward by checking the ins and out of batched )listener-based) predicate fn :
        if False:
            return self.predicate_forward(**kwargs)

        return self.predictor_forward(**kwargs)

    def predictor_forward(
        self,
        x,
        gt_sentences=None,
        rnn_states=None,
    ):
        if rnn_states is None:
            rnn_states = self.model.get_reset_states()

        input_dict = {
            'obs':x,
            'rnn_states': rnn_states,
        }
         
        if gt_sentences is None:
            return_feature_only=self.archi_kwargs["features_id"]["instruction_generator"]
        else:
            return_feature_only = None 
            input_dict['rnn_states']['gt_sentences'] = gt_sentences
            
        output = self.model.forward(
            **input_dict,
            pipelines={
                "instruction_generator":self.archi_kwargs["pipelines"]["instruction_generator"]
            },
            return_feature_only=return_feature_only,
        )

        return output
    
    def predicate_forward(
        self,
        x,
        gt_sentences=None,
        rnn_states=None,
    ):
        """
        TODO : figure out the inputs necessary 

        """
        batch_size = x.shape[0]

        if rnn_states is None:
            rnn_states = self.model.get_reset_states()

        input_dict = {
            'obs':x,
            'rnn_states': rnn_states,
            # TODO: the rnn_states must contain the sentences_widx/one_hot on which the predicate fn is computed 
        }
         
        return_feature_only = None 
        
        output = self.model.forward(
            **input_dict,
            pipelines={
                "prediction_generator":self.archi_kwargs["pipelines"]["prediction_generator"]
            },
            return_feature_only=return_feature_only,
        )
        
        # TODO: check that this output is conform to what the predicate fn used to return
        decision_logits = output["next_rnn_states"]["DecisionGenerator"]["decision"][0]
        import ipdb; ipdb.set_trace()
        # TODO : find out dimensions :
        # (batch_size, max sentence_lengths, (nbr_distractors+1)=1 most likely, 2 )
        
        sentences_token_idx = None # TODO compute it from provided sentence in ohe or not
        #(batch_size, max_sentence_length)
        eos_mask = (sentences_token_idx==self.vocab_stop_idx)
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
        sentences_lengths = lengths.clamp(max=self.max_sentence_length)
        #(batch_size, )
    
        sentences_lengths = sentences_lengths.reshape(-1,1,1).expand(
            decision_logits.shape[0],
            1,
            decision_logits.shape[2]
        )
    
        final_decision_logits = decision_logits.gather(
            dim=1, 
            index=(sentences_lengths-1),
        ).squeeze(1)
        # (batch_size, (nbr_distractors+1) = 1 / ? (descriptive mode depends on the role of the agent) )
        import ipdb; ipdb.set_trace()
        # TODO check dim 
        output_logits = final_decision_logits.reshape((batch_size, -1))
        
        if output.shape[-1] == 2:
            output_probs = output_logits.softmax(dim=-1)
        else:
            output_probs = output_logits
        #output = output_probs[:,0] > self.predicate_threshold
        output = output_probs[:, 0]

        return output
    
    def speaker_forward(
        self, 
        experiences, 
        sentences=None, 
        multi_round=False, 
        graphtype="straight_through_gumbel_softmax", 
        tau0=0.2,
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
        features = experiences.view(-1, *(experiences.size()[2:]))
        utter_outputs = self._utter(features=features, sentences=None)
        if len(utter_outputs) == 5:
            next_sentences_hidden_states, next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
        else:
            next_sentences_hidden_states = None
            next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
        
        if self.training:
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

    def listener_forward(
        self, 
        sentences, 
        experiences, 
        multi_round=False, 
        graphtype="straight_through_gumbel_softmax", 
        tau0=0.2,
    ):
        """
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        Make sure to shuffle the experiences so that the order does not give away the target. 
        :param multi_round: Boolean defining whether to utter a sentence back or not.
        :param graphtype: String defining the type of symbols used in the output sentence:
                    - `'categorical'`: one-hot-encoded symbols.
                    - `'gumbel_softmax'`: continuous relaxation of a categorical distribution.
                    - `'straight_through_gumbel_softmax'`: improved continuous relaxation...
                    - `'obverter'`: obverter training scheme...
        :param tau0: Float, temperature with which to apply gumbel-softmax estimator.
        """
        batch_size = experiences.size(0)
        '''
        features = self._sense(
            experiences=experiences, 
            sentences=sentences,
            nominal_call=True
        )
        '''
        features = experiences.view(-1, *(experiences.size()[2:]))
        
        if sentences is not None:
            decision_logits, listener_temporal_features = self._reason(sentences=sentences, features=features)
        else:
            decision_logits = None
            listener_temporal_features = None
        
        next_sentences_widx = None 
        next_sentences_logits = None
        next_sentences = None
        temporal_features = None
        
        if multi_round or ("obverter" in graphtype.lower() and sentences is None):
            utter_outputs = self._utter(features=features, sentences=sentences)
            if len(utter_outputs) == 5:
                next_sentences_hidden_states, next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
            else:
                next_sentences_hidden_states = None
                next_sentences_widx, next_sentences_logits, next_sentences, temporal_features = utter_outputs
                        
            if self.training:
                if "gumbel_softmax" in graphtype:    
                    print(f"WARNING: Listener {self.agent_id} is producing messages via a {graphtype}-based graph at the Listener class-level!")
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

        self.output_dict = {
            "output": decision_logits,
            "decision": decision_logits, 
            "sentences_widx":next_sentences_widx, 
            "sentences_logits":next_sentences_logits, 
            "sentences_one_hot":next_sentences,
            #"features":features,
            "temporal_features": temporal_features
        }
        
        if not(multi_round):
            self._reset_rnn_states()

        return self.output_dict 
