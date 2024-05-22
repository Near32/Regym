from typing import Dict, List 

import torch
import torch.nn as nn
import torch.nn.functional as F
from ordered_set import OrderedSet

from Archi.modules.module import Module 
from Archi.modules.utils import (
    layer_init,
    copy_hdict,
    apply_on_hdict,
)

import wandb

    
class LSTMModule(Module):
    def __init__(
        self, 
        state_dim, 
        hidden_units=[256], 
        non_linearities=['None'],
        id='LSTMModule_0',
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
        use_cuda=False,
    ):
        '''
        
        :param state_dim: dimensions of the input.
        :param hidden_units: list of number of neurons per recurrent hidden layers.
        :param non_linearities: list of activation function to use after each hidden layer, e.g. nn.functional.relu. Default [None].

        '''
        
        #assert 'lstm_input' in input_stream_ids
        if input_stream_ids is not None:
            if 'lstm_hidden' not in input_stream_ids:
                input_stream_ids['lstm_hidden'] = f"inputs:{id}:hidden"
            if 'lstm_cell' not in input_stream_ids:
                input_stream_ids['lstm_cell'] = f"inputs:{id}:cell"
            if 'iteration' not in input_stream_ids:
                input_stream_ids['iteration'] = f"inputs:{id}:iteration"


        super(LSTMModule, self).__init__(
            id=id,
            type='LSTMModule',
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        dims = [state_dim] + hidden_units
        # Consider future cases where we may not want to initialize the LSTMCell(s)
        self.layers = nn.ModuleList(
            [
                layer_init(
                    nn.LSTMCell(
                        dim_in, 
                        dim_out,
                    )
                ) 
                for dim_in, dim_out in zip(dims[:-1], dims[1:])
            ]
        )

        self.feature_dim = dims[-1]
        self.non_linearities = non_linearities
        while len(self.non_linearities) < len(self.layers):
            self.non_linearities.append(self.non_linearities[-1])
        for idx, nl in enumerate(self.non_linearities):
            if not isinstance(nl, str):
                raise NotImplementedError
            if nl=='None':
                self.non_linearities[idx] = None
            else:
                nl_cls = getattr(nn, nl, None)
                if nl_cls is None:
                    raise NotImplementedError
                self.non_linearities[idx] = nl_cls()
        

        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()
        
        self.get_reset_states(cuda=self.use_cuda)
        
    def forward(self, inputs):
        '''
        :param inputs: input to LSTM cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, recurrent_neurons = inputs
        hidden_states, cell_states = recurrent_neurons['hidden'], recurrent_neurons['cell']
        iteration = recurrent_neurons.get("iteration", None)
        if iteration is None:
            batch_size = x.shape[0]
            iteration = torch.zeros((batch_size, 1)).to(x.device)
        niteration = [it+1 for it in iteration]

        next_hstates, next_cstates = [], []
        for idx, (layer, hx, cx) in enumerate(zip(self.layers, hidden_states, cell_states) ):
            batch_size = x.size(0)
            if hx.size(0) == 1: # then we have just resetted the values, we need to expand those:
                hx = torch.cat([hx]*batch_size, dim=0)
                cx = torch.cat([cx]*batch_size, dim=0)
            elif hx.size(0) != batch_size:
                raise NotImplementedError("Sizes of the hidden states and the inputs do not coincide.")

            if self.use_cuda:
                x = x.cuda()
                hx = hx.cuda()
                cx = cx.cuda()

            nhx, ncx = layer(x, (hx, cx))
            next_hstates.append(nhx)
            next_cstates.append(ncx)
            # Consider not applying activation functions on last layer's output
            if self.non_linearities[idx] is not None:
                nhx = self.non_linearities[idx](nhx)
        return nhx, {'hidden': next_hstates, 'cell': next_cstates, 'iteration': niteration}
    
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        """
        outputs_stream_dict = {}
        
        lstm_input = input_streams_dict['lstm_input']
        lstm_hidden = input_streams_dict['lstm_hidden']
        lstm_cell = input_streams_dict['lstm_cell']
        iteration = input_streams_dict['iteration']
        
        lstm_output, state_dict = self.forward((
            lstm_input if not isinstance(lstm_input, list) else lstm_input[0],
            {
                'hidden': lstm_hidden,
                'cell': lstm_cell,
                'iteration': iteration,
            }),
        )
        
        outputs_stream_dict[f'output'] = [lstm_output]
        
        outputs_stream_dict[f'hidden'] = state_dict['hidden']
        outputs_stream_dict[f'cell'] = state_dict['cell']
        outputs_stream_dict[f'iteration'] = state_dict['iteration']
        
        for k in list(outputs_stream_dict.keys()):
            if k in self.output_stream_ids:
                outputs_stream_dict[self.output_stream_ids[k]] = outputs_stream_dict[k]

        # Bookkeeping:
        outputs_stream_dict[f'inputs:{self.id}:output'] = [lstm_output]
        outputs_stream_dict[f'inputs:{self.id}:hidden'] = state_dict['hidden']
        outputs_stream_dict[f'inputs:{self.id}:cell'] = state_dict['cell']
        outputs_stream_dict[f'inputs:{self.id}:iteration'] = state_dict['iteration']
        
        return outputs_stream_dict 

    def get_reset_states(self, cuda=False, repeat=1):
        #TODO: repeat into other Moduels:
        if not hasattr(self, 'reset_states'):
            hidden_states, cell_states = [], []
            for layer in self.layers:
                h = torch.zeros(repeat, layer.hidden_size)
                #if cuda:
                #    h = h.cuda()
                hidden_states.append(h)
                cell_states.append(h)
            iteration = torch.zeros((repeat, 1))
            #if cuda:    iteration = iteration.cuda()
            iteration = [iteration]
            self.reset_states = {'hidden': hidden_states, 'cell': cell_states, 'iteration': iteration}
        
        def init(x):
            outx = x.repeat(repeat, *[1 for _ in range(len(x.shape)-1)])
            if cuda:  outx = outx.cuda() 
            return outx
        
        reset_states = copy_hdict(self.reset_states)
        reset_states = apply_on_hdict(
          reset_states,
          fn=init,
        )
        
        return reset_states
      
    def set_reset_states(self, new_reset_states):
        #TODO: repeat into other Moduels 
        repeat = 1
        if 'hidden' not in new_reset_states:
            hidden_states = []
            for layer in self.layers:
                h = torch.zeros(repeat, layer.hidden_size)
                if cuda:
                    h = h.cuda()
                hidden_states.append(h)
            new_reset_states['hidden'] = hidden_states
        else:
            assert len(new_reset_states['hidden']) == len(self.layers)
            for nrs, layer in zip(new_reset_states['hidden'], self.layers):
                assert nrs.shape[0] == 1 and nrs.shape[1] == layer.hidden_size
        if 'cell' not in new_reset_states:
            cell_states = []
            for layer in self.layers:
                h = torch.zeros(repeat, layer.hidden_size)
                if cuda:
                    h = h.cuda()
                cell_states.append(h)
            new_reset_states['cell'] = cell_states
        else:
            assert len(new_reset_states['cell']) == len(self.layers)
            for nrs, layer in zip(new_reset_states['cell'], self.layers):
                assert nrs.shape[0] == 1 and nrs.shape[1] == layer.hidden_size
        if 'iteration' not in new_reset_states:
           iteration = torch.zeros((repeat, 1))
           if cuda:    iteration = iteration.cuda()
           new_reset_states['iteration'] = [iteration]
        else:
            assert len(new_reset_states['iteration']) == 1 
            assert new_reset_states['iteration'][0].shape[0] == 1
            assert new_reset_states['iteration'][0].shape[1] == 1
        self.reset_states = new_reset_states
        return

    def get_feature_shape(self):
        return self.feature_dim


class GRUModule(Module):
    def __init__(
        self, 
        state_dim, 
        hidden_units=[256], 
        non_linearities=['None'],
        id='GRUModule_0',
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
        use_cuda=False,
    ):
        '''
        
        :param state_dim: dimensions of the input.
        :param hidden_units: list of number of neurons per recurrent hidden layers.
        :param non_linearities: list of activation function to use after each hidden layer, e.g. nn.functional.relu. Default [None].

        '''
        
        #assert 'gru_input' in input_stream_ids
        if input_stream_ids is not None:
            if 'gru_hidden' not in input_stream_ids:
                input_stream_ids['gru_hidden'] = f"inputs:{id}:hidden"
            if 'iteration' not in input_stream_ids:
                input_stream_ids['iteration'] = f"inputs:{id}:iteration"


        super(GRUModule, self).__init__(
            id=id,
            type='GRUModule',
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        dims = [state_dim] + hidden_units
        # Consider future cases where we may not want to initialize the LSTMCell(s)
        self.layers = nn.ModuleList(
            [
                layer_init(
                    nn.GRUCell(
                        dim_in, 
                        dim_out,
                    )
                ) 
                for dim_in, dim_out in zip(dims[:-1], dims[1:])
            ]
        )

        self.feature_dim = dims[-1]
        self.non_linearities = non_linearities
        while len(self.non_linearities) < len(self.layers):
            self.non_linearities.append(self.non_linearities[-1])
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def forward(self, inputs):
        '''
        :param inputs: input to LSTM cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, recurrent_neurons = inputs
        hidden_states = recurrent_neurons['hidden']
        
        iteration = recurrent_neurons.get("iteration", None)
        if iteration is None:
            batch_size = x.shape[0]
            iteration = torch.zeros((batch_size, 1)).to(x.device)
        niteration = [it+1 for it in iteration]

        next_hstates = []
        for idx, (layer, hx) in enumerate(zip(self.layers, hidden_states) ):
            batch_size = x.size(0)
            if hx.size(0) == 1: # then we have just resetted the values, we need to expand those:
                hx = torch.cat([hx]*batch_size, dim=0)
            elif hx.size(0) != batch_size:
                raise NotImplementedError("Sizes of the hidden states and the inputs do not coincide.")
            
            if self.use_cuda:
                x = x.cuda()
                hx = hx.cuda()

            nhx = layer(x, hx)
            next_hstates.append(nhx)
            # Consider not applying activation functions on last layer's output
            if self.non_linearities[idx] is not None:
                nhx = self.non_linearities[idx](nhx)
        return nhx, {'hidden': next_hstatesi, 'iteration': niteration}

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        """
        outputs_stream_dict = {}
        
        gru_input = input_streams_dict['gru_input']
        gru_hidden = input_streams_dict['gru_hidden']
        iteration = input_streams_dict['iteration']

        gru_output, state_dict = self.forward((
            gru_input if not isinstance(gru_input, list) else gru_input[0],
            {
                'hidden': gru_hidden,
                'iteration': iteration,
            }),
        )
        
        outputs_stream_dict[f'output'] = gru_output
        
        outputs_stream_dict[f'hidden'] = state_dict['hidden']
        outputs_stream_dict[f'iteration'] = state_dict['iteration']

        for k in outputs_stream_dict.keys():
            if k in self.output_stream_ids:
                outputs_stream_dict[self.output_stream_ids[k]] = outputs_stream_dict[k]

        # Bookkeeping:
        outputs_stream_dict[f'inputs:{self.id}:output'] = gru_output
        outputs_stream_dict[f'inputs:{self.id}:hidden'] = state_dict['hidden']
        outputs_stream_dict[f'inputs:{self.id}:iteration'] = state_dict['iteration']

        return outputs_stream_dict 

    def get_reset_states(self, cuda=False, repeat=1):
        hidden_states = []
        for layer in self.layers:
            h = torch.zeros(repeat, layer.hidden_size)
            if cuda:
                h = h.cuda()
            hidden_states.append(h)
        iteration = torch.zeros((repeat, 1))
        if cuda:    iteration = iteration.cuda()
        iteration = [iteration]
        return {'hidden': hidden_states, 'iteration': iteration}

    def get_feature_shape(self):
        return self.feature_dim


class OracleTHERModule(Module):
    def __init__(
        self,
        _max_sentence_length,
        vocabulary=None,
        _vocab_size=None,
        id='OracleTHERModule_0',
        config={'hidden_units':1024, 'logits_base':1.0,},
        input_stream_ids=None,
        output_stream_ids={},
        use_cuda=False,
    ):
        super(OracleTHERModule, self).__init__(
            id=id,
            type="OracleTHERModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        if vocabulary == 'None':
            vocabulary = 'key ball red green blue purple \
            yellow grey verydark dark neutral light verylight \
            tiny small medium large giant get go fetch go get \
            a fetch a you must fetch a'

        if isinstance(vocabulary, str):
            vocabulary = vocabulary.split(' ')
        
        self.vocabulary = OrderedSet([w.lower() for w in vocabulary])
        self.vocab_size = _vocab_size
        
        #MODIF1 : padding is done with EoS and its index must be 0!
        # Make padding_idx=0:
        #self.vocabulary = ['PAD', 'SoS', 'EoS'] + list(self.vocabulary)
        self.vocabulary = list(self.vocabulary)

        while len(self.vocabulary) < self.vocab_size-2:
            self.vocabulary.append( f"DUMMY{len(self.vocabulary)}")
        self.vocabulary = list(OrderedSet(self.vocabulary))
        #MODIF1:
        self.vocabulary = ['EoS', 'SoS'] + self.vocabulary
        
        self.w2idx = {}
        self.idx2w = {}
        for idx, w in enumerate(self.vocabulary):
            self.w2idx[w] = idx
            self.idx2w[idx] = w
        
        print(type(self))
        print(self.idx2w)

        self.max_sentence_length = _max_sentence_length
        self.voc_size = len(self.vocabulary)
        
        # Dummy weight to avoid optimizer complaints...
        self.dummy = nn.Linear(1,1)
        self.logits_base = self.config.get('logits_base', 1.0)
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()
    
    def reset(self):
        #if self.config.get("semantic_embeddings_prior", False):
        self.semantic_prior = None
        self.semantic_prior_matrix = None
        self.semantic_prior_logits = None
        self.visual_features = None
        self.text_features = None
    
    def get_reset_states(self, cuda=False, repeat=1):
        dummy_goal = torch.zeros((repeat, 1))
        if cuda:    dummy_goal = dummy_goal.cuda()
        return {'achieved_goal': [dummy_goal]}


    def forward(self, x, gt_sentences=None, output_dict=None):
        '''
        If :param gt_sentences: is not `None`,
        then teacher forcing is implemented...
        '''
        if gt_sentences is not None:
            gt_sentences = gt_sentences.long().to(x.device)
        
        batch_size = x.shape[0]
        loss_per_item = []
        
        if x.shape[-1] == self.max_sentence_length:
            predicted_sentences = x.reshape(
                batch_size, self.max_sentence_length,
            ).clone().long() 
        else:
            predicted_sentences = self.w2idx['EoS']*torch.ones(batch_size, self.max_sentence_length, dtype=torch.long).to(x.device)
            predicted_sentences[:,:x.shape[-1],...] = x.reshape(batch_size, -1)
            #predicted_sentences[:, 0] = self.w2idx['EoS']

        for t in range(self.max_sentence_length):
            #predicted_sentences[:, t] = idxs_next_token #.unsqueeze(-1)
            
            # Compute loss:
            if gt_sentences is not None:
                mask = torch.ones_like(gt_sentences[:, t])
                #MODIF1:
                #mask = (gt_sentences[:, t]!=self.w2idx['PAD'])
                mask = mask.float().to(x.device)
                # batch_size x 1
                batched_loss = torch.zeros_like(mask)
                batched_loss *= mask
                loss_per_item.append(batched_loss.unsqueeze(1))
                
        predicted_logits = torch.zeros(
            batch_size, self.max_sentence_length, self.vocab_size,
        ).to(x.device)
        # batch_size x max_sentence_length x vocab_size 
        predicted_logits = predicted_logits.scatter_(
            dim=-1,
            index=predicted_sentences.unsqueeze(-1,
                ).repeat(1,1,self.vocab_size),
            src=torch.ones_like(predicted_logits),
        )
        predicted_logits *= self.logits_base
        hidden_states = torch.zeros(batch_size, self.max_sentence_length, self.config.get('hidden_units', 1024)).to(x.device)
        # batch_size x max_sentence_length x hidden_state_dim=1=dummy

        # Regularize tokens after EoS :
        EoS_count = 0
        for b in range(batch_size):
            end_idx = 0
            for idx_t in range(predicted_sentences.shape[1]):
                if predicted_sentences[b,idx_t] == self.w2idx['EoS']:
                    EoS_count += 1
                    #if not self.predict_PADs:
                    # Whether we predict the PAD tokens or not, 
                    # we still want the output of this module to make 
                    # sense with respect to the EoS token so we filter out
                    # any token that follows EoS token...
                    #MODIF1:
                    #predicted_sentences[b, idx_t+1:] = self.w2idx['PAD']
                    predicted_sentences[b, idx_t+1:] = self.w2idx['EoS']
                    break
                end_idx += 1
        try:
            wandb.log({f"{self.id}/EoSRatioPerBatch":float(EoS_count)/batch_size}, commit=False)
        except Exception as e:
            print(f"WARNING: W&B Logging: {e}")
        if gt_sentences is not None:
            loss_per_item = torch.cat(loss_per_item, dim=-1).mean(-1)
            # batch_size x max_sentence_length
            accuracies = (predicted_sentences==gt_sentences).float().mean(dim=0)
            # Computing accuracy on the tokens that matters the most:
            #MODIF1:
            #mask = (gt_sentences!=self.w2idx['PAD'])
            mask = (gt_sentences!=self.w2idx['EoS'])
            sentence_accuracies = (predicted_sentences==gt_sentences).float().masked_select(mask).mean()
            # BoS Accuracies:
            bos_accuracies = torch.zeros_like(predicted_sentences).float()
            for b in range(batch_size):
                ps = predicted_sentences[b].cpu().detach().tolist()
                for idx_t in range(predicted_sentences.shape[1]):
                    gt_token = gt_sentences[b, idx_t].item()
                    if gt_token in ps:
                        bos_accuracies[b, idx_t] = 1.0
            bos_sentence_accuracies = bos_accuracies.masked_select(mask).mean()
            bos_accuracies = bos_accuracies.mean(dim=0)
            output_dict = {
                'hidden_states':hidden_states, 
                'prediction':predicted_sentences, 
                'prediction_logits':predicted_logits, 
                'loss_per_item':loss_per_item, 
                'accuracies':accuracies, 
                'bos_accuracies':bos_accuracies, 
                'sentence_accuracies':sentence_accuracies,
                'bos_sentence_accuracies':bos_sentence_accuracies,
            }

            return output_dict

        if output_dict is not None:
            output_dict.update({
                'hidden_states':hidden_states, 
                'prediction':predicted_sentences, 
                'prediction_logits':predicted_logits, 
            })

        return predicted_sentences

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        """
        outputs_stream_dict = {}

        for key, experiences in input_streams_dict.items():
            if "gt_sentences" in key:   continue

            output_key = f"processed_{key}"
            if key in self.output_stream_ids:
                output_key = self.output_stream_ids[key]
            
            if isinstance(experiences, list):
                assert len(experiences)==1, f"Provided too many input on id:{key}"
                experiences = experiences[0]
            batch_size = experiences.size(0)

            if self.use_cuda:   experiences = experiences.cuda()

            # GT Sentences ?
            gt_key = f"{key}_gt_sentences"
            gt_sentences = input_streams_dict.get(gt_key, None)
            
            output_dict = {}
            if gt_sentences is None:
                output = self.forward(
                    x=experiences,
                    gt_sentences=gt_sentences,
                    output_dict=output_dict,
                )
                output_dict['prediction'] = output
            else:
                if isinstance(gt_sentences, list):
                    assert len(gt_sentences) == 1
                    gt_sentences = gt_sentences[0]
                output_dict = self.forward(
                    x=experiences,
                    gt_sentences=gt_sentences,
                )
            
            output_sentences = output_dict['prediction']

            outputs_stream_dict[output_key] = [output_sentences]
            
            for okey, ovalue in output_dict.items():
                outputs_stream_dict[f"inputs:{self.id}:{key}_{okey}"] = [ovalue]
                #outputs_stream_dict[f"inputs:{key}_{okey}"] = [ovalue]
        
        return outputs_stream_dict 

class CaptionRNNModule(Module):
    def __init__(
        self,
        max_sentence_length,
        input_dim=64,
        embedding_size=64, 
        hidden_units=256, 
        num_layers=1, 
        vocabulary=None,
        vocab_size=None,
        gate=None, #F.relu, 
        dropout=0.0, 
        rnn_fn="nn.GRU",
        loss_fn="NLL",
        id='CaptionRNNModule_0',
        config={
            "predict_PADs":False,
            "diversity_loss_weighting":False,
            "rectify_contrastive_imbalance":False,
            "semantic_embeddings_prior":False,
            "semantic_prior_mixing":'additive',
            "input_mm_projector_BN": False,
            "semantic_embedding_init": 'none',
            "semantic_embeddings_detach_visual_features": False,
        },
        input_stream_ids=None,
        output_stream_ids={},
        use_cuda=False,
    ):
        super(CaptionRNNModule, self).__init__(
            id=id,
            type="CaptionRNNModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        if vocabulary == 'None':
            vocabulary = 'key ball red green blue purple \
            yellow grey verydark dark neutral light verylight \
            tiny small medium large giant get go fetch go get \
            a fetch a you must fetch a'

        if isinstance(vocabulary, str):
            vocabulary = vocabulary.split(' ')
        
        self.vocabulary = OrderedSet([w.lower() for w in vocabulary])
        self.vocab_size = vocab_size
        
        # MODIF1: 
        # Make padding_idx=0:
        #self.vocabulary = ['PAD', 'SoS', 'EoS'] + list(self.vocabulary)
        self.vocabulary = list(self.vocabulary)
        
        while len(self.vocabulary) < self.vocab_size-2:
            self.vocabulary.append( f"DUMMY{len(self.vocabulary)}")
        self.vocabulary = list(OrderedSet(self.vocabulary))
        #MODIF1:
        self.vocabulary = ['EoS', 'SoS'] + self.vocabulary

        self.w2idx = {}
        self.idx2w = {}
        for idx, w in enumerate(self.vocabulary):
            self.w2idx[w] = idx
            self.idx2w[idx] = w

        self.max_sentence_length = max_sentence_length
        self.voc_size = self.vocab_size #len(self.vocabulary)

        self.embedding_size = embedding_size
        if isinstance(hidden_units, list):  hidden_units=hidden_units[-1]
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        self.gate = gate
        if isinstance(rnn_fn, str):
            rnn_fn = getattr(torch.nn, rnn_fn, None)
            if rnn_fn is None:
                raise NotImplementedError
        
        self.input_dim = input_dim
        self.input_decoder = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, self.hidden_units, bias=False)),
            nn.BatchNorm1d(self.hidden_units),
            nn.ReLU(),
            #nn.BatchNorm1d(self.hidden_units),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units, bias=False)),
            nn.BatchNorm1d(self.hidden_units),
            nn.ReLU(),
            #nn.BatchNorm1d(self.hidden_units),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units, )), #bias=False)),
            #nn.ReLU(),
            #nn.BatchNorm1d(self.hidden_units),
        )
        
        self.rnn_fn = rnn_fn
        self.rnn = rnn_fn(
            input_size=self.embedding_size,
            hidden_size=self.hidden_units, 
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        
        #self.embedding = nn.Embedding(self.voc_size, self.embedding_size, padding_idx=0)
        self.embedding = nn.Embedding(self.voc_size, self.embedding_size)
        self.token_decoder = nn.Sequential(
            layer_init(nn.Linear(self.hidden_units, self.hidden_units, bias=False)),
            nn.BatchNorm1d(self.hidden_units),
            nn.ReLU(),
            #nn.BatchNorm1d(self.hidden_units),
            layer_init(nn.Linear(self.hidden_units, self.hidden_units, bias=False)),
            nn.BatchNorm1d(self.hidden_units),
            nn.ReLU(),
            #nn.BatchNorm1d(self.hidden_units),
            layer_init(nn.Linear(self.hidden_units, self.voc_size, )), #bias=False)),
            #nn.ReLU(),
            #nn.BatchNorm1d(self.voc_size),
        )

        if self.config.get("semantic_embeddings_prior", False):
            self.semantic_embedding = nn.Embedding(self.voc_size, self.embedding_size)
            sem_emb_init = self.config.get("semantic_embedding_init", 'none')
            if sem_emb_init == 'ortho':
                nn.init.orthogonal_(self.semantic_embedding.weight)
            elif sem_emb_init == 'xavier_normal':
                nn.init.xavier_normal_(self.semantic_embedding.weight)
            elif sem_emb_init == 'xavier_uniform':
                nn.init.xavier_uniform_(self.semantic_embedding.weight)
            elif sem_emb_init == 'eye':
                nn.init.eye_(self.semantic_embedding.weight)
            elif sem_emb_init == 'normal':
                nn.init.normal_(self.semantic_embedding.weight)
            elif sem_emb_init == 'uniform':
                nn.init.uniform_(self.semantic_embedding.weight)
            
            self.mm_size = self.hidden_units
            self.sem2mm = nn.Linear(self.embedding_size, self.mm_size, bias=False)
            #self.input2mm = nn.Linear(self.input_dim, self.mm_size, bias=False)
            # Adding BN :
            input2mm = [nn.Linear(self.input_dim, self.mm_size, bias=False)]
            if self.config.get("input_mm_projector_BN", False):
                input2mm += [
                    nn.BatchNorm1d(self.mm_size),
                    nn.Linear(self.mm_size, self.mm_size, bias=False),
                ]
            self.input2mm = nn.Sequential(*input2mm)
        # MODIF: we replace the loss with NLL in order to allow input being logits:
        self.loss_fn = loss_fn
        if 'NLL' in loss_fn:
            self.criterion = nn.NLLLoss(reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def reset(self):
        if self.config.get("semantic_embeddings_prior", False):
            self.semantic_prior = None
            self.semantic_prior_matrix = None
            self.semantic_prior_logits = None
            self.visual_features = None
            self.text_features = None
    
    def _compute_visual_features(self, x):
        # (batch_size x nbr_visual_emb x visual_emb_size)
        batch_size = x.shape[0]
        nbr_visual_emb = x.shape[1]
        x = x.reshape((batch_size*nbr_visual_emb, -1))
        mm_x = self.input2mm(x)
        l2_mm_x = F.normalize(mm_x, p=2.0, dim=-1).reshape(
            (batch_size, nbr_visual_emb, -1),
        )
        # (batch_size x nbr_visual_emb x mm_size )
        self.visual_features = l2_mm_x
        return l2_mm_x
    
    def _compute_text_features(self, emb):
        # (batch_size x nbr_text_emb x emb_size)
        mm_sem_emb = self.sem2mm(emb)
        l2_mm_sem_emb = F.normalize(mm_sem_emb, p=2.0, dim=-1)
        # (batch_size x nbr_text_emb x mm_size)
        self.text_features = l2_mm_sem_emb
        return l2_mm_sem_emb

    def forward(self, x, gt_sentences=None, output_dict=None):
        '''
        If :param gt_sentences: is not `None`,
        then teacher forcing is implemented...
        '''
        if gt_sentences is not None:
            gt_sentences = gt_sentences.long().to(x.device)
        
        batch_size = x.shape[0]
        
        if self.config.get("semantic_embeddings_prior", False):
            prior_x = x
            if len(x.shape)>=4:
                prior_x = prior_x.transpose(-3,-1)
                emb_dim = prior_x.shape[-1]
                prior_x = prior_x.reshape(batch_size, -1, emb_dim)
            else:
                prior_x = prior_x.reshape(batch_size, 1, -1) 
            l2_mm_x = self._compute_visual_features(prior_x).transpose(2,1)
            # (batch_size x mm_size x nbr_visual_emb)
            if self.config.get("semantic_embeddings_detach_visual_features", False):
                l2_mm_x = l2_mm_x.detach()
            # Must be clone for it to be differentiable apparently...
            b_sem_emb = self.semantic_embedding.weight.clone().unsqueeze(0).repeat(batch_size, 1,1)
            # (batch_size x nbr_text_emb x emb_size)
            # = (batch_size x vocab_size x emb_size )
            l2_mm_sem_emb = self._compute_text_features(b_sem_emb) 
            # (batch_size x vocab_size x mm_size)
            self.semantic_prior_matrix = torch.bmm(l2_mm_sem_emb, l2_mm_x)
            # (batch_size x vocab_size x nbr_visual_emb )
            semantic_prior_logits = torch.sum(self.semantic_prior_matrix, dim=-1, keepdim=True)
            # (batch_size x vocab_size x 1)
            self.semantic_prior_logits = semantic_prior_logits
            prior = torch.softmax(
                semantic_prior_logits.squeeze(-1),
                dim=-1,
            )
            # (batch_size x vocab_size)
            self.semantic_prior = prior
            if self.config.get("semantic_prior_mixing_with_detach", False):
                prior = prior.detach()
        else:
            prior = None
            self.semantic_prior = None

        if len(x.shape)>=4:
            # assume input dim corresponds to the number of depth channel
            x = x.transpose(-3,-1)
            assert self.input_dim == x.shape[-1]
            x = x.reshape(batch_size, -1, self.input_dim).sum(dim=1)
        else:
            x = x.reshape(batch_size, -1)
        
        # Input Decoding:
        dx = self.input_decoder(x)

        # batch_size x hidden_units
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(x.device) 
        h_0[0] = dx.reshape(batch_size, -1)
        # (num_layers * num_directions, batch, hidden_size)
        
        if self.rnn_fn==nn.LSTM:
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(x.device) 
            decoder_hidden = (h_0,c_0)
        else:
            decoder_hidden = h_0 
        
        decoder_input = self.embedding(torch.LongTensor([[self.w2idx["SoS"]]]).to(x.device))
        # 1 x embedding_size
        decoder_input = decoder_input.reshape(1, 1, -1).repeat(batch_size, 1, 1)
        # batch_size x 1 x embedding_size
        
        # Compute Contrastive Factors:
        if gt_sentences is not None \
        and self.config.get("rectify_contrastive_imbalance", False):
            # Negative examples :
            negative_mask = (gt_sentences[:,0] == self.w2idx["EoS"])
            # (batch_size, )
            nbr_negatives = negative_mask.sum().item()
            # Positive examples
            positive_mask = (1-negative_mask.float())
            # (batch_size, )
            nbr_positives = batch_size-nbr_negatives

        loss_per_item = []

        #MODIF1:
        #predicted_sentences = self.w2idx['PAD']*torch.ones(batch_size, self.max_sentence_length, dtype=torch.long).to(x.device)
        predicted_sentences = self.w2idx['EoS']*torch.ones(batch_size, self.max_sentence_length, dtype=torch.long).to(x.device)
        predicted_logits = []
        predicted_argmax_logits = []
        hidden_states = []
        for t in range(self.max_sentence_length):
            output, decoder_hidden = self._rnn(decoder_input, h_c=decoder_hidden)
            hidden_states.append(output)
            #token_distribution = F.softmax(self.token_decoder(output), dim=-1) 
            token_unlogit = self.token_decoder(output)
            
            if prior is None:
                token_logit = F.log_softmax(token_unlogit, dim=-1) 
            else:
                token_distr = torch.softmax(token_unlogit, dim=-1)
                semantic_prior_mixing = self.config.get("semantic_prior_mixing", "additive")
                if semantic_prior_mixing == 'additive':
                    eff_token_distr = (prior+token_distr)/2.0
                elif semantic_prior_mixing == 'multiplicative':
                    eff_token_distr = prior*token_distr
                    eff_token_distr = eff_token_distr/(eff_token_distr.sum(dim=-1, keepdim=True)+1.0e-8)
                else:
                    raise NotImplementedError
                # MODIF: need to take into account the prior into the unlogit used in the loss fn:
                token_unlogit = eff_token_distr
                # Note that this is not ideal as two softmax will be applied at the end of the day...
                # It could be good to un-softmax the current token_unlogit, but I do not know it being feasible?
                # Thus, instead, we replace the Cross entropy loss below (criterion) with a NLLLoss that expects logits.
                token_logit = torch.log(eff_token_distr+1.0e-8)

            predicted_logits.append(token_logit)
            #idxs_next_token = torch.argmax(token_distribution, dim=1)
            if self.training:
                # Sampling at training time:
                predicted_distr = torch.distributions.Categorical(logits=token_logit)
                idxs_next_token = predicted_distr.sample()
                logits_next_token = torch.gather(token_logit, dim=-1, index=idxs_next_token.unsqueeze(-1)).squeeze(-1)
            else:
                logits_next_token, idxs_next_token = torch.max(token_logit, dim=1)
            # batch_size x 1
            predicted_sentences[:, t] = idxs_next_token #.unsqueeze(-1)
            predicted_argmax_logits.append(logits_next_token)
            # Compute loss:
            if gt_sentences is not None:
                mask = torch.ones_like(gt_sentences[:, t])
                #MODIF1:
                #if not self.config.get("predict_PADs", False):
                #    mask = (gt_sentences[:, t]!=self.w2idx['PAD'])
                mask = mask.float().to(x.device)
                # batch_size x 1
                if self.config.get("diversity_loss_weighting", False):
                    set_values = OrderedSet(gt_sentences[:,t].cpu().tolist())
                    nbr_div = len(set_values)
                    mask *= max(1.0, nbr_div)
                    try:
                        wandb.log({f"{self.id}/DivPerBatchToken{t}": nbr_div}, commit=False)
                    except Exception as e:
                        print(f"WARNING: W&B Logging: {e}")
                if 'NLL' in self.loss_fn:
                    batched_loss = self.criterion(
                        #input=token_distribution, 
                        # With CrossEntropyLoss, it is expecting unnormalized logit,
                        # and it will perform a log_softmax inside:
                        #input=token_unlogit,
                        # MODIF: we replace it with NLLLoss that expects normalized ones:
                        input=token_logit,
                        target=gt_sentences[:, t].reshape(batch_size),
                    )
                else:
                    batched_loss = self.criterion(
                        #input=token_distribution, 
                        # With CrossEntropyLoss, it is expecting unnormalized logit,
                        # and it will perform a log_softmax inside:
                        input=token_unlogit,
                        # MODIF: we replace it with NLLLoss that expects normalized ones:
                        #input=token_logit,
                        target=gt_sentences[:, t].reshape(batch_size),
                    )
 
                batched_loss *= mask
                if self.config.get("rectify_contrastive_imbalance", False):
                    positive_loss = positive_mask*batched_loss
                    negative_loss = negative_mask*batched_loss
                    batched_loss = positive_loss/max(1,nbr_positives)
                    batched_loss += negative_loss/max(1,nbr_negatives)
                    # (batch_size, )
                loss_per_item.append(batched_loss.unsqueeze(1))
                
            # Preparing next step:
            if gt_sentences is not None:
                # Teacher forcing:
                idxs_next_token = gt_sentences[:, t]
            # batch_size x 1
            decoder_input = self.embedding(idxs_next_token).unsqueeze(1)
            # batch_size x 1 x embedding_size            
        
        predicted_argmax_logits = torch.stack(predicted_argmax_logits, dim=1)
        # batch_size x max_sentence_length x vocab_size 
        predicted_logits = torch.stack(predicted_logits, dim=1)
        # batch_size x max_sentence_length x vocab_size 
        hidden_states = torch.stack(hidden_states, dim=1)
        # batch_size x max_sentence_length x hidden_state_dim

        # Regularize tokens after EoS :
        EoS_count = 0
        predicted_sentences_length = []
        sentences_likelihoods = []
        sentences_perplexities = []
        for b in range(batch_size):
            end_idx = 0
            for idx_t in range(predicted_sentences.shape[1]):
                if predicted_sentences[b,idx_t] == self.w2idx['EoS']:
                    EoS_count += 1
                    #if not self.predict_PADs:
                    # Whether we predict the PAD tokens or not, 
                    # we still want the output of this module to make 
                    # sense with respect to the EoS token so we filter out
                    # any token that follows EoS token...
                    #MODIF1:
                    #predicted_sentences[b, idx_t+1:] = self.w2idx['PAD']
                    predicted_sentences[b, idx_t+1:] = self.w2idx['EoS']
                    break
                end_idx += 1
            predicted_sentences_length.append(end_idx)
            #Compute perplexity: 
            # CSGPU2 cuda drive error technical debt:
            #slhd = torch.prod(torch.pow(predicted_argmax_logits[b,:end_idx+1].exp(), 1.0/(end_idx+1)))
            slhd = torch.pow(predicted_argmax_logits[b,:end_idx+1].exp(), 1.0/(end_idx+1))
            slhd = slhd.cpu().prod().to(slhd.device)

            # unstable : torch.prod(predicted_argmax_logits[b,:end_idx+1].exp(), keepdim=False)
            #perplexity = torch.pow(1.0/slhd, 1.0/(end_idx+1))
            perplexity = 1.0/(slhd+1e-8)
            sentences_likelihoods.append(slhd)
            sentences_perplexities.append(perplexity)
        
        sentences_likelihoods = torch.stack(sentences_likelihoods, dim=-1)
        sentences_perplexities = torch.stack(sentences_perplexities, dim=-1)
        # batch_size 
        
        try:
            wandb.log({f"{self.id}/EoSRatioPerBatch":float(EoS_count)/batch_size}, commit=False)
        except Exception as e:
            print(f"WARNING: W&B Logging: {e}")
        if gt_sentences is not None:
            loss_per_item = torch.cat(loss_per_item, dim=-1).mean(-1)
            # batch_size x max_sentence_length
            accuracies = (predicted_sentences==gt_sentences).float().mean(dim=0)
            # Computing accuracy on the tokens that matters the most:
            #MODIF1:
            mask = (gt_sentences!=self.w2idx['EoS'])
            #mask = (gt_sentences!=self.w2idx['PAD'])
            sentence_accuracies = (predicted_sentences==gt_sentences).float().masked_select(mask).mean()
            # BoS Accuracies:
            bos_accuracies = torch.zeros_like(predicted_sentences).float()
            for b in range(batch_size):
                ps = predicted_sentences[b].cpu().detach().tolist()
                for idx_t in range(predicted_sentences.shape[1]):
                    gt_token = gt_sentences[b, idx_t].item()
                    if gt_token in ps:
                        bos_accuracies[b, idx_t] = 1.0
            bos_sentence_accuracies = bos_accuracies.masked_select(mask).mean()
            bos_accuracies = bos_accuracies.mean(dim=0)
            output_dict = {
                'hidden_states':hidden_states, 
                'prediction':predicted_sentences, 
                'prediction_logits':predicted_logits, 
                'loss_per_item':loss_per_item, 
                'accuracies':accuracies, 
                'bos_accuracies':bos_accuracies, 
                'sentence_accuracies':sentence_accuracies,
                'bos_sentence_accuracies':bos_sentence_accuracies,
            }

            return output_dict
        
        if output_dict is not None:
            output_dict.update({
                'hidden_states':hidden_states, 
                'prediction':predicted_sentences, 
                'prediction_logits':predicted_logits,
                'prediction_likelihoods':sentences_likelihoods,
                'prediction_perplexities':sentences_perplexities, 
            })

        return predicted_sentences

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        """
        outputs_stream_dict = {}

        for key, experiences in input_streams_dict.items():
            if "gt_sentences" in key:   continue

            output_key = f"processed_{key}"
            if key in self.output_stream_ids:
                output_key = self.output_stream_ids[key]
            
            if isinstance(experiences, list):
                assert len(experiences)==1, f"Provided too many input on id:{key}"
                experiences = experiences[0]
            batch_size = experiences.size(0)
            
            if len(experiences.shape)>2 \
            and experiences.shape[-1] != experiences.shape[-2]:
                # if it is not a feature map but it has an extra dimension:
                experiences = experiences.reshape((batch_size, -1))

            if self.use_cuda:   experiences = experiences.cuda()

            # GT Sentences ?
            gt_key = f"{key}_gt_sentences"
            gt_sentences = input_streams_dict.get(gt_key, None)
            
            output_dict = {}
            if gt_sentences is None:
                output = self.forward(
                    x=experiences,
                    gt_sentences=gt_sentences,
                    output_dict=output_dict,
                )
                output_dict['prediction'] = output
            else:
                if isinstance(gt_sentences, list):
                    assert len(gt_sentences) == 1
                    gt_sentences = gt_sentences[0]
                output_dict = self.forward(
                    x=experiences,
                    gt_sentences=gt_sentences,
                )
            
            output_sentences = output_dict['prediction']

            outputs_stream_dict[output_key] = [output_sentences]
            
            for okey, ovalue in output_dict.items():
                outputs_stream_dict[f"inputs:{self.id}:{key}_{okey}"] = [ovalue]
        
        return outputs_stream_dict 

    def _rnn(self, x, h_c):
        batch_size = x.shape[0]
        rnn_outputs, updated_h_c = self.rnn(x, h_c)
        output = rnn_outputs[:,-1,...]
        if self.gate != 'None':
            output = self.gate(output)
        # batch_size x hidden_units 
        return output, updated_h_c
        # batch_size x sequence_length=1 x hidden_units
        # num_layer*num_directions, batch_size, hidden_units
        
    def get_feature_shape(self):
        return self.hidden_units


class EmbeddingRNNModule(Module):
    def __init__(
        self, 
        vocab_size, 
        feature_dim=64,
        embedding_size=64, 
        hidden_units=256, 
        num_layers=1, 
        gate=None, #F.relu, 
        dropout=0.0, 
        rnn_fn="nn.GRU",
        padding_idx=None,
        id='EmbeddingRNNModule_0',
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
        use_cuda=False,
    ):
        super(EmbeddingRNNModule, self).__init__(
            id=id,
            type="EmbeddingRNNModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )
        
        self.voc_size = vocab_size+1
        self.embedding_size = embedding_size
        if isinstance(hidden_units, list):  hidden_units=hidden_units[-1]
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        if padding_idx is None:
            padding_idx = self.voc_size
        self.embedding = nn.Embedding(
            self.voc_size, 
            self.embedding_size, 
            padding_idx=padding_idx,
        )
        
        self.gate = gate
        if isinstance(rnn_fn, str):
            rnn_fn = getattr(torch.nn, rnn_fn, None)
            if rnn_fn is None:
                raise NotImplementedError

        self.rnn = rnn_fn(
            input_size=self.embedding_size,
            hidden_size=hidden_units, 
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        
        self.feature_dim = feature_dim
        if isinstance(self.feature_dim, int):
            self.decoder_mlp = nn.Linear(hidden_units, self.feature_dim)
        else:
            self.feature_dim = hidden_units

        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def forward(self, x):
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)

        batch_size = x.shape[0]
        sentence_length = x.shape[1]
        
        if x.shape[-1] == 1:    x = x.squeeze(-1)
        embeddings = self.embedding(x.long())
        # batch_size x sequence_length x embedding_size

        rnn_outputs, rnn_states = self.rnn(embeddings)
        # batch_size x sequence_length x hidden_units
        # num_layer*num_directions, batch_size, hidden_units
        
        output = rnn_outputs[:,-1,...]
        if self.gate != 'None':
            output = self.gate(output)

        if hasattr(self, 'decoder_mlp'):
            output = self.decoder_mlp(output)

        # batch_size x hidden_units 

        return output
    
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        """
        outputs_stream_dict = {}

        for key, experiences in input_streams_dict.items():
            output_key = f"processed_{key}"
            if key in self.output_stream_ids:
                output_key = self.output_stream_ids[key]
            
            if isinstance(experiences, list):
                assert len(experiences)==1, f"Provided too many input on id:{key}"
                experiences = experiences[0]

            if self.use_cuda:   experiences = experiences.cuda()

            output = self.forward(x=experiences)
            outputs_stream_dict[output_key] = [output]
            
        return outputs_stream_dict 


    def get_feature_shape(self):
        return self.feature_dim

