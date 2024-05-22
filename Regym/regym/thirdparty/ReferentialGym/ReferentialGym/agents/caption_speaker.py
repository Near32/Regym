import torch
import torch.nn as nn

from .speaker import Speaker
from ..networks import choose_architecture, layer_init, hasnan


class CaptionSpeaker(Speaker):
    def __init__(self,kwargs, obs_shape, vocab_size=100, max_sentence_length=10):
        '''
        :param obs_shape: tuple defining the shape of the stimulus following `(nbr_distractors+1, nbr_stimulus, *stimulus_shape)`
                          where, by default, `nbr_distractors=0` (partial observability), and `nbr_stimulus=1` (static stimuli). 
        :param vocab_size: int defining the size of the vocabulary of the language.
        :param max_sentence_length: int defining the maximal length of each sentence the speaker can utter.
        '''
        super(CaptionSpeaker, self).__init__(obs_shape,vocab_size,max_sentence_length)
        self.kwargs = kwargs 

        cnn_input_shape = self.obs_shape[2:]
        # add the target-boolean-channel:
        cnn_input_shape[0] += 1

        if self.kwargs['architecture'] == 'CNN':
            self.cnn_encoder = choose_architecture(architecture='CNN',
                                                  input_shape=cnn_input_shape,
                                                  hidden_units_list=None,
                                                  feature_dim=self.kwargs['cnn_encoder_feature_dim'],
                                                  nbr_channels_list=self.kwargs['cnn_encoder_channels'],
                                                  kernels=self.kwargs['cnn_encoder_kernels'],
                                                  strides=self.kwargs['cnn_encoder_strides'],
                                                  paddings=self.kwargs['cnn_encoder_paddings'])
        elif 'ResNet18' in self.kwargs['architecture']:
            self.cnn_encoder = choose_architecture(architecture=self.kwargs['architecture'],
                                                  input_shape=cnn_input_shape,
                                                  feature_dim=self.kwargs['cnn_encoder_feature_dim'])
            
        temporal_encoder_input_dim = self.cnn_encoder.get_feature_shape()
        self.temporal_feature_encoder = layer_init(nn.LSTM(input_size=temporal_encoder_input_dim,
                                          hidden_size=self.kwargs['temporal_encoder_nbr_hidden_units'],
                                          num_layers=self.kwargs['temporal_encoder_nbr_rnn_layers'],
                                          batch_first=True,
                                          dropout=0.0,
                                          bidirectional=False))

        #symbol_decoder_input_dim = self.kwargs['symbol_processing_nbr_hidden_units']+(self.kwargs['nbr_distractors']+1)*self.kwargs['temporal_encoder_nbr_hidden_units']
        symbol_decoder_input_dim = (self.kwargs['nbr_distractors']+1)*self.kwargs['temporal_encoder_nbr_hidden_units']
        self.symbol_processing = layer_init(nn.LSTM(input_size=symbol_decoder_input_dim,
                                      hidden_size=self.kwargs['symbol_processing_nbr_hidden_units'], 
                                      num_layers=self.kwargs['symbol_processing_nbr_rnn_layers'],
                                      batch_first=True,
                                      dropout=0.0,
                                      bidirectional=False))

        self.symbol_decoder = layer_init(nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], self.vocab_size, bias=False))
        self.symbol_encoder = layer_init(nn.Linear(self.vocab_size, self.kwargs['symbol_processing_nbr_hidden_units'], bias=False))
        self.idx_symbol2embed = layer_init(nn.Embedding(self.vocab_size, self.kwargs['symbol_processing_nbr_hidden_units']))

        self.tau_fc = layer_init(nn.Linear(self.kwargs['symbol_processing_nbr_hidden_units'], 1 , bias=False))
        self.reset_weights()

    def reset_weights(self):
        self.temporal_feature_encoder.apply(layer_init)
        self.symbol_processing.apply(layer_init)
        self.symbol_decoder.apply(layer_init)
        self.symbol_encoder.apply(layer_init)

    def _compute_tau(self, tau0, h):
        invtau = 1.0 / (self.tau_fc(h).squeeze() + tau0)
        return invtau

    def _sense(self, stimuli, sentences=None):
        '''
        Infers features from the stimuli that have been provided.

        :param stimuli: Tensor of shape `(batch_size, *self.obs_shape)`. 
                        `stimuli[:, 0]` is assumed as the target stimulus, while the others are distractors, if any. 
        :param sentences: None or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            features: Tensor of shape `(batch_size, *(self.obs_shape[:2]), feature_dim).
        '''
        batch_size = stimuli.size(0)
        stimuli = stimuli.view(-1, *(stimuli.size()[3:]))
        features = []
        total_size = stimuli.size(0)
        mini_batch_size = min(self.kwargs['cnn_encoder_mini_batch_size'], total_size)
        for stin in torch.split(stimuli, split_size_or_sections=mini_batch_size, dim=0):
            features.append( self.cnn_encoder(stin))
        features = torch.cat(features, dim=0)
        features = features.view(batch_size, *(self.obs_shape[:2]), -1)
        # (batch_size, nbr_distractors+1, nbr_stimulus, feature_dim)
        return features 

    def _utter(self, features, sentences=None):
        '''
        Reasons about the features and the listened sentences, if multi_round, to yield the sentences to utter back.
        
        :param features: Tensor of shape `(batch_size, *self.obs_shape[:2], feature_dim)`.
        :param sentences: None, or Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of (potentially one-hot-encoded) symbols.
        
        :returns:
            - logits: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of logits.
            - sentences: Tensor of shape `(batch_size, max_sentence_length, vocab_size)` containing the padded sequence of one-hot-encoded symbols.
        '''
        batch_size = features.size(0)
        # (batch_size, nbr_distractors+1, nbr_stimulus, kwargs['cnn_encoder_feature_dim'])
        # Forward pass:
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
        outputs = outputs.view(batch_size, *(self.obs_shape[:2]), -1)
        # (batch_size, (nbr_distractors+1), nbr_stimulus, kwargs['temporal_encoder_feature_dim'])
        
        embedding_tf_final_outputs = outputs[:,:,-1,:].contiguous()
        # (batch_size, (nbr_distractors+1), kwargs['temporal_encoder_feature_dim'])
        embedding_tf_final_outputs = embedding_tf_final_outputs.view(batch_size,1, -1)
        # (batch_size, 1, (nbr_distractors+1) * kwargs['temporal_encoder_nbr_hidden_units'])
        
        # TODO: implement the sentence consumption strategy using the embedding, rather than the encoder:
        '''
        if sentences is not None:
            # Consume the sentences:
            sentences = sentences.view((-1, self.vocab_size))
            encoded_sentences = self.symbol_encoder(sentences).view((batch_size, self.max_sentence_length, self.kwargs['symbol_processing_nbr_hidden_units'])) 
            
            states = self.rnn_states
            # (batch_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
            # Since we consume the sentence, rather than generating it, we prepend the encoded_sentences with ones:
            inputs = torch.ones((batch_size,1,self.kwargs['symbol_processing_nbr_hidden_units']))
            if encoded_sentences.is_cuda: inputs = inputs.cuda()
            encoded_sentences = torch.cat( [inputs, encoded_sentences], dim=1)
            # Then, as usual, we concatenate this sequence's vectors with repeated temporal feature embedding vectors:
            inputs = torch.cat( [embedding_tf_final_outputs.repeat(1,self.max_sentence_length+1,1), encoded_sentences], dim=-1)
            # (batch_size, max_sentence_length+1, (nbr_distractors+1)*kwargs['temporal_encoder_nbr_hidden_units']+kwargs['symbol_processing_nbr_hidden_units'])
        
            rnn_outputs, self.rnn_states = self.symbol_processing(inputs, states)          
            # (batch_size, max_sentence_length, kwargs['symbol_processing_nbr_hidden_units'])
            # (hidden_layer*num_directions, batch_size, kwargs['symbol_processing_nbr_hidden_units'])
        '''

        # Utter the next sentences:
        next_sentences_one_hots = []
        next_sentences_logits = []
        # ----------------------------------------------------------
        '''
        inputs = torch.zeros((batch_size,1,self.kwargs['symbol_processing_nbr_hidden_units']))
        if embedding_tf_final_outputs.is_cuda: inputs = inputs.cuda()
        inputs = torch.cat( [embedding_tf_final_outputs, inputs], dim=-1)
        '''
        # ----------------------------------------------------------
        inputs = embedding_tf_final_outputs
        # ----------------------------------------------------------
        # (batch_size, 1, (nbr_distractors+1)*kwargs['temporal_encoder_nbr_hidden_units'](=kwargs['symbol_processing_nbr_hidden_units']))
        
        # Utter the next sentences:
        for i in range(self.max_sentence_length):
            hiddens, self.rnn_states = self.symbol_processing(inputs, self.rnn_states)          
            # (batch_size, 1, kwargs['symbol_processing_nbr_hidden_units'])
            
            '''
            inputs = torch.cat([embedding_tf_final_outputs,hiddens], dim=-1)
            '''
            # (batch_size, 1, (nbr_distractors+1)*kwargs['temporal_encoder_nbr_hidden_units']+kwargs['symbol_processing_nbr_hidden_units'])
        
            outputs = self.symbol_decoder(hiddens.squeeze(1))            
            # (batch_size, vocab_size)
            _, prediction = outputs.max(1)                        
            # (batch_size)

            # ----------------------------------------------------------
            inputs = torch.zeros_like(embedding_tf_final_outputs)
            # only the first elements of the inputs are made of the actual word embedding:
            inputs[..., :self.kwargs['symbol_processing_nbr_hidden_units']] = self.idx_symbol2embed(prediction).unsqueeze(1)
            # ----------------------------------------------------------

            next_sentences_logits.append(outputs.unsqueeze(1))
            next_sentences_one_hot = nn.functional.one_hot(prediction, num_classes=self.vocab_size).unsqueeze(1).float()
            # (batch_size, 1, vocab_size)
            next_sentences_one_hots.append(next_sentences_one_hot)
        
        next_sentences_one_hots = torch.cat(next_sentences_one_hots, dim=1)
        # (batch_size, max_sentence_length, vocab_size)
        next_sentences_logits = torch.cat(next_sentences_logits, dim=1)
        # (batch_size, max_sentence_length, vocab_size)
        return next_sentences_logits, next_sentences_one_hots          
        
