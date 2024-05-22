import torch
import torch.nn as nn

from ..networks import choose_architecture, layer_init, hasnan, BetaVAE

from .questioner import Questioner 

from .lstm_cnn_listener import LSTMCNNListener

class LSTMCNNQuestioner(Questioner):
    def __init__(self, obs_shape, vocab_size=100, max_sentence_length=10, agent_id='s0', logger=None, kwargs=None):
        '''
        :param obs_shape: tuple defining the shape of the experience following `(nbr_experiences, sequence_length, *experience_shape)`
                          where, by default, `nbr_experiences=1` (partial observability), and `sequence_length=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        :param agent_id: str defining the ID of the agent over the population.
        :param logger: None or somee kind of logger able to accumulate statistics per agent.
        :param kwargs: Dict of kwargs...
        '''
        super(LSTMCNNQuestioner, self).__init__(
            agent_id=agent_id, 
            obs_shape=obs_shape,
            vocab_size=vocab_size,
            max_sentence_length=max_sentence_length,
            logger=logger, 
            kwargs=kwargs)
        
        self.use_sentences_one_hot_vectors = True 
        self.kwargs = kwargs 

        cnn_input_shape = self.obs_shape[2:]
        MHDPANbrHead=4
        MHDPANbrRecUpdate=1
        MHDPANbrMLPUnit=512
        MHDPAInteractionDim=128
        if 'mhdpa_nbr_head' in self.kwargs: MHDPANbrHead = self.kwargs['mhdpa_nbr_head']
        if 'mhdpa_nbr_rec_update' in self.kwargs: MHDPANbrRecUpdate = self.kwargs['mhdpa_nbr_rec_update']
        if 'mhdpa_nbr_mlp_unit' in self.kwargs: MHDPANbrMLPUnit = self.kwargs['mhdpa_nbr_mlp_unit']
        if 'mhdpa_interaction_dim' in self.kwargs: MHDPAInteractionDim = self.kwargs['mhdpa_interaction_dim']
        
        self.cnn_encoder = choose_architecture(architecture=self.kwargs['architecture'],
                                               kwargs=self.kwargs,
                                               input_shape=cnn_input_shape,
                                               feature_dim=self.kwargs['cnn_encoder_feature_dim'],
                                               nbr_channels_list=self.kwargs['cnn_encoder_channels'],
                                               kernels=self.kwargs['cnn_encoder_kernels'],
                                               strides=self.kwargs['cnn_encoder_strides'],
                                               paddings=self.kwargs['cnn_encoder_paddings'],
                                               fc_hidden_units_list=self.kwargs['cnn_encoder_fc_hidden_units'],
                                               dropout=self.kwargs['dropout_prob'],
                                               MHDPANbrHead=MHDPANbrHead,
                                               MHDPANbrRecUpdate=MHDPANbrRecUpdate,
                                               MHDPANbrMLPUnit=MHDPANbrMLPUnit,
                                               MHDPAInteractionDim=MHDPAInteractionDim)
        
        self.use_feat_converter = self.kwargs['use_feat_converter'] if 'use_feat_converter' in self.kwargs else False 
        if self.use_feat_converter:
            self.feat_converter_input = self.cnn_encoder.get_feature_shape()


        if 'BetaVAE' in self.kwargs['architecture'] or 'MONet' in self.kwargs['architecture']:
            self.VAE_losses = list()
            self.compactness_losses = list()
            self.buffer_cnn_output_dict = dict()
            
            # N.B: with a VAE, we want to learn the weights in any case:
            if 'agent_learning' in self.kwargs:
                assert('transfer_learning' not in self.kwargs['agent_learning'])
            
            self.vae_detached_featout = False
            if self.kwargs['vae_detached_featout']:
                self.vae_detached_featout = True

            self.VAE = self.cnn_encoder

            self.use_feat_converter = True
            self.feat_converter_input = self.cnn_encoder.latent_dim
        else:
            if 'agent_learning' in self.kwargs and 'transfer_learning' in self.kwargs['agent_learning']:
                self.cnn_encoder.detach_conv_maps = True

        self.encoder_feature_shape = self.cnn_encoder.get_feature_shape()
        if self.use_feat_converter:
            self.featout_converter = []
            self.featout_converter.append(nn.Linear(self.feat_converter_input, self.kwargs['cnn_encoder_feature_dim']*2))
            self.featout_converter.append(nn.ReLU())
            self.featout_converter.append(nn.Linear(self.kwargs['cnn_encoder_feature_dim']*2, self.kwargs['feat_converter_output_size'])) 
            self.featout_converter.append(nn.ReLU())
            self.featout_converter =  nn.Sequential(*self.featout_converter)
            self.encoder_feature_shape = self.kwargs['feat_converter_output_size']
        
        self.cnn_encoder_normalization = nn.BatchNorm1d(num_features=self.encoder_feature_shape)

        temporal_encoder_input_dim = self.cnn_encoder.get_feature_shape()
        if self.kwargs['temporal_encoder_nbr_rnn_layers'] > 0:
            self.temporal_feature_encoder = layer_init(nn.LSTM(input_size=temporal_encoder_input_dim,
                                              hidden_size=self.kwargs['temporal_encoder_nbr_hidden_units'],
                                              num_layers=self.kwargs['temporal_encoder_nbr_rnn_layers'],
                                              batch_first=True,
                                              dropout=self.kwargs['dropout_prob'],
                                              bidirectional=False))
        else:
            self.temporal_feature_encoder = None
            print("WARNING: Symbol processing :: the number of hidden units is being reparameterized to fit to convolutional features.")
            self.kwargs['temporal_encoder_nbr_hidden_units'] = self.kwargs['nbr_stimulus']*self.encoder_feature_shape
            self.kwargs['symbol_processing_nbr_hidden_units'] = self.kwargs['temporal_encoder_nbr_hidden_units']


        self.normalization = nn.BatchNorm1d(num_features=self.kwargs['temporal_encoder_nbr_hidden_units'])
        #self.normalization = nn.LayerNorm(normalized_shape=self.kwargs['temporal_encoder_nbr_hidden_units'])

        symbol_decoder_input_dim = self.kwargs['symbol_embedding_size']
        self.symbol_processing = nn.LSTM(input_size=symbol_decoder_input_dim,
                                      hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                                      num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                                      batch_first=True,
                                      dropout=self.kwargs['dropout_prob'],
                                      bidirectional=False)
        # SoS symbol is not part of the vocabulary as it is only prompting the RNNs
        # and is not part of the sentences being uttered.
        # TODO: when applying multi-round, it could be interesting to force SoS 
        # at the beginning of sentences so that agents can align rounds.
        self.sos_symbol = nn.Parameter(torch.zeros(1,1,self.config['symbol_embedding_size']))
        self.symbol_encoder = nn.Linear(self.vocab_size, self.kwargs['symbol_embedding_size'], bias=False)
        
        self.symbol_encoder_dropout = nn.Dropout( p=self.kwargs['embedding_dropout_prob'])
        # EoS symbol is part of the vocabulary:
        self.symbol_decoder = nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], self.vocab_size)

        self.tau_fc = nn.Sequential(
            nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], 1,bias=False),
            nn.Softplus()
        )

        ## Inner Listener:

        self.kwargs['cnn_encoder'] = self.cnn_encoder
        inner_listener = LSTMCNNListener(
            obs_shape, 
            vocab_size, 
            max_sentence_length, 
            agent_id, 
            logger, 
            kwargs=self.kwargs
        )
        self._register_inner_listener(inner_listener)

        self.inner_state = None


    def reset(self):
        if self.inner_listener is not None: 
            self.inner_listener.reset()
        
        self.symbol_processing.apply(layer_init)
        self.symbol_encoder.apply(layer_init)
        
        self._reset_inner_state()
        
    def _compute_tau(self, tau0, h):
        tau = 1.0 / (self.tau_fc(h).squeeze() + tau0)
        return tau

    def _tidyup(self):
        """
        Called at the agent (questioner, in this case) level at the end of the `compute` function.
        In the case of this class, the `embeddings_tf_final_output` is handled in the 
        `reset_inner_state` function, in order to avoid re-computing it at each 
        communication round.
        """
        if isinstance(self.cnn_encoder, BetaVAE):
            self.VAE_losses = list()
            self.compactness_losses.clear()
            self.buffer_cnn_output_dict = dict()

    def _sense(self, experiences, sentences=None):
        '''
        Infers features from the experiences that have been provided.

        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        `experiences[:, 0]` is assumed as the target experience, while the others are distractors, if any. 
        :param sentences: None or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            features: Tensor of shape `(batch_size, *(self.obs_shape[:2]), feature_dim).
        '''
        def _sense(self, experiences, sentences=None):
        r"""
        Infers features from the experiences that have been provided.

        :param experiences: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        Make sure to shuffle the stimuli so that the order does not give away the target. 
        :param sentences: None or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            features: Tensor of shape `(batch_size, -1, feature_dim).
        
        """
        batch_size = experiences.size(0)
        nbr_distractors_po = experiences.size(1)
        experiences = experiences.view(-1, *(experiences.size()[3:]))
        features = []
        total_size = experiences.size(0)
        mini_batch_size = min(self.kwargs['cnn_encoder_mini_batch_size'], total_size)
        for stin in torch.split(experiences, split_size_or_sections=mini_batch_size, dim=0):
            if isinstance(self.cnn_encoder, BetaVAE):
                cnn_output_dict  = self.cnn_encoder.compute_loss(stin)
                if 'VAE_loss' in cnn_output_dict:
                    self.VAE_losses.append(cnn_output_dict['VAE_loss'])
                
                if hasattr(self.cnn_encoder, 'compactness_losses') and self.cnn_encoder.compactness_losses is not None:
                    self.compactness_losses.append(self.cnn_encoder.compactness_losses.cpu())
                
                for key in cnn_output_dict:
                    if key not in self.buffer_cnn_output_dict:
                        self.buffer_cnn_output_dict[key] = list()
                    self.buffer_cnn_output_dict[key].append(cnn_output_dict[key].cpu())

                if self.kwargs['vae_use_mu_value']:
                    featout = self.cnn_encoder.mu 
                else:
                    featout = self.cnn_encoder.z

                if self.vae_detached_featout:
                    featout = featout.detach()

                featout = self.featout_converter(featout)
            else:
                featout = self.cnn_encoder(stin)
                if self.use_feat_converter:
                    featout = self.featout_converter(featout)

            features.append(featout)
        
        self.features = self.cnn_encoder_normalization(torch.cat(features, dim=0))
        
        self.features = self.features.view(batch_size, nbr_distractors_po, self.config['nbr_stimulus'], -1)
        # (batch_size, nbr_distractors+1 / ? (descriptive mode depends on the role of the agent), nbr_stimulus, feature_dim)
        
        # Inner listener:
        setattr(self.inner_listener, 'features', self.features) 


        if isinstance(self.cnn_encoder, BetaVAE):
            self.VAE_losses = torch.cat(self.VAE_losses).contiguous()#.view((batch_size,-1)).mean(dim=-1)
            
            for key in self.buffer_cnn_output_dict:
                self.log_dict[key] = torch.cat(self.buffer_cnn_output_dict[key]).mean()

            self.log_dict['kl_capacity'] = torch.Tensor([100.0*self.cnn_encoder.EncodingCapacity/self.cnn_encoder.maxEncodingCapacity])
            if len(self.compactness_losses):
                self.log_dict['unsup_compactness_loss'] = torch.cat(self.compactness_losses).mean()

        return self.features 

    def _reason(self, sentences, features):
        """
        Reasons about the features and sentences to yield the target-prediction logits.
        
        :param sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        
        :returns:
            - decision_logits: Tensor of shape `(batch_size, self.obs_shape[1])` containing the target-prediction logits.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        """
        self.inner_listener.rnn_states = self.inner_state
        inner_listener_reasoning_output = None
        if sentences is not None:
            inner_listener_reasoning_output, _ = self.inner_listener._reason(
                sentences=sentences, 
                features=features
            )
            self.inner_state = self.inner_listener.rnn_state
        return inner_listener_reasoning_output

    def _utter(self, features, sentences=None):
        '''
        Reasons about the features and the listened sentences, if multi_round, to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: None, or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - word indices: Tensor of shape `(batch_size, max_sentence_length, 1)` of type `long` containing the indices of the words that make up the sentences.
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
            - temporal features: Tensor of shape `(batch_size, (nbr_distractors+1)*temporal_feature_dim)`.
        '''
        batch_size = features.size(0)
        # (batch_size, nbr_distractors+1, nbr_stimulus, kwargs['cnn_encoder_feature_dim'])
        # Forward pass:
        if self.embedding_tf_final_outputs is None:
            if self.temporal_feature_encoder:
                features = features.view(-1, *(features.size()[2:]))
                # (batch_size*(nbr_distractors+1), nbr_stimulus, kwargs['cnn_encoder_feature_dim'])
                rnn_outputs = []
                total_size = features.size(0)
                mini_batch_size = min(self.kwargs['temporal_encoder_mini_batch_size'], total_size)
                for featin in torch.split(features, split_size_or_sections=mini_batch_size, dim=0):
                    outputs, _ = self.temporal_feature_encoder(featin)
                    rnn_outputs.append( outputs)
                outputs = torch.cat(rnn_outputs, dim=0)
                # (batch_size*(nbr_distractors+1), nbr_stimulus, kwargs['temporal_encoder_feature_dim'])
                outputs = outputs.view(batch_size, -1, self.kwargs['nbr_stimulus'], self.kwargs['temporal_encoder_nbr_hidden_units'])
                # (batch_size, (nbr_distractors+1), nbr_stimulus, kwargs['temporal_encoder_feature_dim'])
                
                # Taking only the target features: assumes partial observations anyway...
                # TODO: find a way to compute the sentence while attending other features in case of full observability...
                embedding_tf_final_outputs = outputs[:,0,-1,:].contiguous()
                # (batch_size, kwargs['temporal_encoder_feature_dim'])
                self.embedding_tf_final_outputs = self.normalization(embedding_tf_final_outputs.reshape((-1, self.kwargs['temporal_encoder_nbr_hidden_units'])))
                self.embedding_tf_final_outputs = self.embedding_tf_final_outputs.reshape(batch_size, self.kwargs['nbr_distractors']+1, -1)
                # (batch_size, 1, kwargs['temporal_encoder_nbr_hidden_units'])
            else:
                self.embedding_tf_final_outputs = self.normalization(features.reshape((-1, self.kwargs['temporal_encoder_nbr_hidden_units'])))
                self.embedding_tf_final_outputs = self.embedding_tf_final_outputs.reshape((batch_size, self.kwargs['nbr_distractors']+1, -1))
                # (batch_size, 1, kwargs['temporal_encoder_nbr_hidden_units'])

        sentences_hidden_states = [list() for _ in range(batch_size)]
        sentences_widx = [list() for _ in range(batch_size)]
        sentences_logits = [list() for _ in range(batch_size)]
        sentences_one_hots = [list() for _ in range(batch_size)]
        for b in range(batch_size):
            bemb = self.embedding_tf_final_outputs[b].view((1, 1, -1))
            # (batch_size=1, 1, kwargs['temporal_encoder_nbr_hidden_units'])
            init_rnn_state = bemb
            # (hidden_layer*num_directions=1, batch_size=1, 
            # kwargs['temporal_encoder_nbr_hidden_units']=kwargs['symbol_processing_nbr_hidden_units'])
            rnn_states = (init_rnn_state, torch.zeros_like(init_rnn_state))
            
            # SoS token is given as initial input:
            '''
            # Assuming SoS is part of the vocabulary:
            inputs = self.symbol_encoder.weight[:, self.vocab_start_idx].reshape((1,1,-1))
            '''
            # Assuming SoS is not part of the vocabulary:
            inputs = self.sos_symbol

            #torch.zeros((1, 1, self.kwargs['symbol_embedding_size']))
            if self.embedding_tf_final_outputs.is_cuda: inputs = inputs.cuda()
            # (batch_size=1, 1, kwargs['symbol_embedding_size'])
            
            continuer = True
            sentence_token_count = 0
            token_idx = 0
            while continuer:
                sentence_token_count += 1
                rnn_outputs, next_rnn_states = self.symbol_processing(inputs, rnn_states )
                # (batch_size=1, 1, kwargs['symbol_processing_nbr_hidden_units'])
                # (hidden_layer*num_directions, batch_size=1, kwargs['symbol_processing_nbr_hidden_units'])

                outputs = self.symbol_decoder(rnn_outputs.squeeze(1))
                # (batch_size=1, vocab_size)
                _, prediction = torch.softmax(outputs, dim=1).max(1)                        
                # (batch_size=1)
                prediction = prediction.unsqueeze(1).float()

                sentences_hidden_states[b].append(rnn_outputs.view(1,-1))
                sentences_widx[b].append( prediction)
                sentences_logits[b].append( outputs.view((1,-1)))
                # Counting EoS symbol:
                sentences_one_hots[b].append( nn.functional.one_hot(prediction.squeeze().long(), num_classes=self.vocab_size).view((1,-1)))
                
                # next inputs:
                #inputs = self.symbol_encoder(outputs).unsqueeze(1)
                inputs = self.symbol_encoder.weight[:, prediction.long()].reshape((1,1,-1))
                # (batch_size, 1, kwargs['symbol_embedding_size'])
                inputs = self.symbol_encoder_dropout(inputs)
                # next rnn_states:
                rnn_states = next_rnn_states
                
                stop_word_condition = (prediction == self.vocab_stop_idx)
                if len(sentences_widx[b]) >= self.max_sentence_length or stop_word_condition :
                    continuer = False 
                    #TODO: enforce stop token at the last position, maybe?

                token_idx +=1
            # Embed the sentence:
            # Padding token:
            '''
            # Assumes that the sentences are padded with STOP token:
            while len(sentences_widx[b]) < self.max_sentence_length:
                sentences_widx[b].append((self.vocab_stop_idx)*torch.ones_like(prediction))
            '''
            # Assumes that the sentences are padded with PAD token:
            while len(sentences_widx[b]) < self.max_sentence_length:
                sentences_widx[b].append((self.vocab_pad_idx)*torch.ones_like(prediction))

            sentences_hidden_states[b] = torch.cat(sentences_hidden_states[b], dim=0)
            # (sentence_length<=max_sentence_length, kwargs['symbol_preprocessing_nbr_hidden_units'])
            sentences_widx[b] = torch.cat([ word_idx.view((1,1,-1)) for word_idx in sentences_widx[b]], dim=1)
            # (batch_size=1, max_sentence_length, 1)
            sentences_logits[b] = torch.cat(sentences_logits[b], dim=0)
            # (sentence_length<=max_sentence_length, vocab_size)
            sentences_one_hots[b] = torch.cat(sentences_one_hots[b], dim=0) 
            # (sentence_length<=max_sentence_length, vocab_size)

        sentences_one_hots = nn.utils.rnn.pad_sequence(sentences_one_hots, batch_first=True, padding_value=0.0).float()
        # (batch_size, sentence_length<=max_sentence_length, vocab_size)
        
        sentences_widx = torch.cat(sentences_widx, dim=0)
        # (batch_size, max_sentence_length, 1)
        if self.embedding_tf_final_outputs.is_cuda: sentences_widx = sentences_widx.cuda()


        return sentences_hidden_states, sentences_widx, sentences_logits, sentences_one_hots, self.embedding_tf_final_outputs.squeeze() 
        

