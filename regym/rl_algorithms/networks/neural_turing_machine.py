from typing import Dict, List, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import layer_init, layer_init_lstm, layer_init_gru

from regym.rl_algorithms.utils import _extract_from_rnn_states, extract_subtree, copy_hdict
from regym.rl_algorithms.networks.bodies import LSTMBody

class BasicHeads(nn.Module):
    def __init__(
        self,
        memory, 
        input_dim=256, 
        nbr_heads=1, 
        is_read=None,
        ):
        super(BasicHeads,self).__init__()

        self.memory = memory
        self.mem_dim = self.memory.mem_dim
        self.nbr_heads = nbr_heads
        self.input_dim = input_dim
        self.is_read = is_read 

        self.generate_ctrl2gate()

    def generate_ctrl2gate(self) :
        if self.is_read is None :
            raise NotImplementedError
        
        if self.is_read :
            # Generate k,beta,g,s,gamma : M + 1 + 1 + 3 + 1 = M+6
            self.head_gate_dim = self.memory.mem_dim+6 
        else :
            # Generate k,beta,g,s,gamma, e, a : M + 1 + 1 + 3 + 1 + M + M = 3*M+6
            self.head_gate_dim = 3*self.memory.mem_dim+6 
        
        self.ctrl2head = layer_init(
            nn.Linear(
                self.input_dim, 
                self.nbr_heads * self.head_gate_dim
            )
        )

    def get_reset_states(self, cuda=False, repeat=1):
        node_id = f"{'read' if self.is_read else 'write'}_prev_w"
        prev_w = torch.zeros((repeat, self.nbr_heads, self.memory.mem_nbr_slots))
        if cuda:
            prev_w = prev_w.cuda()
        return {node_id:{'data': [prev_w]}}
            
    def write(self, memory_state, ctrl_inputs):
        raise NotImplementedError

    def read(self, memory_state, ctrl_inputs):
        raise NotImplementedError

    def forward(self, memory_state, ctrl_inputs):
        # WARNING: it is imperative to make a copy 
        # of the frame_state, otherwise any changes 
        # will be repercuted onto the current frame_state
        x, frame_states = ctrl_inputs[0], copy_hdict(ctrl_inputs[1])
        
        self.ctrl_output = self.ctrl2head(x)
        self.ctrl_output = self.ctrl_output.view((-1,self.nbr_heads,self.head_gate_dim))

        self._generate_addressing()

        # Addressing :
        self.wc = self.memory.content_addressing(memory_state, self.k, self.beta)

        node_id = f"{'read' if self.is_read else 'write'}_prev_w"
        prev_w = extract_subtree(
            in_dict=frame_states,
            node_id=node_id,
        )['data'][0].to(self.wc.device)
        #(batch_size x nbrHeads x nbr_mem_slot )
        
        w = self.memory.location_addressing(memory_state, prev_w, self.wc, self.g, self.s, self.gamma)
        #(batch_size x nbrHeads)
        
        frame_states.update({node_id:
            {'data':[w]}
        })

        return w, frame_states 

    def _generate_addressing(self) :
        self.k = self.ctrl_output[:,:,0:self.mem_dim]
        self.beta = F.softplus(self.ctrl_output[:,:,self.mem_dim:self.mem_dim+1])
        self.g = torch.sigmoid(self.ctrl_output[:,:,self.mem_dim+1:self.mem_dim+2])
        self.s = F.softmax( 
            F.softplus( 
                self.ctrl_output[:,:,self.mem_dim+2:self.mem_dim+5]
            ),
            dim=-1
        )
        self.gamma = 1+F.softplus(self.ctrl_output[:,:,self.mem_dim+5:self.mem_dim+6])    

        if not(self.is_read) :
            self.erase = self.ctrl_output[:,:,self.mem_dim+6:2*self.mem_dim+6]
            self.add = self.ctrl_output[:,:,2*self.mem_dim+6:3*self.mem_dim+6]


class ReadHeads(BasicHeads):
    def __init__(
        self, 
        memory, 
        nbr_heads=1, 
        input_dim=256, 
        ):
        super(ReadHeads,self).__init__(
            memory=memory,
            input_dim=input_dim,
            nbr_heads=nbr_heads,
            is_read=True, 
        )
        
    def read(self, memory_state, ctrl_inputs) :
        w, frame_states = super(ReadHeads, self).forward(memory_state,ctrl_inputs)
        r = self.memory.read(memory_state=memory_state, w=w)
        return r, frame_states


class WriteHeads(BasicHeads):
    def __init__(
        self, 
        memory, 
        nbr_heads=1, 
        input_dim=256,
        ):
        super(WriteHeads,self).__init__(
            memory=memory,
            input_dim=input_dim,
            nbr_heads=nbr_heads,
            is_read=False, 
        )
        
    def write(self, memory_state, ctrl_inputs) :
        w, frame_states = super(WriteHeads,self).forward(memory_state, ctrl_inputs)
        new_memory_state = self.memory.write(memory_state=memory_state, w=w, erase=self.erase, add=self.add)
        return new_memory_state, frame_states

class NTMController(LSTMBody):
    def __init__(
        self, 
        input_dim=32, 
        hidden_units=[512], 
        output_dim=32, 
        mem_nbr_slots=128, 
        mem_dim= 32, 
        nbr_read_heads=1, 
        nbr_write_heads=1,
        classification=False,
        extra_inputs_infos: Optional[Dict]={},
        ):
        """
        :param extra_inputs_infos: Dictionnary containing the shape of the lstm-relevant extra inputs.
        """

        #LSTMinput_size = (input_dim+output_dim)+mem_dim*nbr_read_heads
        LSTMinput_size = input_dim
        # output_dim was added in the context of few-shot learning 
        # where the previous desired output is fed as input alongside the new input.
        # mem_dim*nbr_read_heads are implicit parts that must be taken into account:
        # they are out-of-concern here, though:
        # the NTM module is itself adding them to the input...
        
        super(NTMController, self).__init__(
            state_dim=LSTMinput_size,
            hidden_units=hidden_units,
            gate=None,
            extra_inputs_infos=extra_inputs_infos,
        )

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.mem_nbr_slots = mem_nbr_slots
        self.mem_dim = mem_dim
        self.nbr_read_heads = nbr_read_heads
        self.nbr_write_heads = nbr_write_heads
        self.classification = classification

        self.build_controller()

    def build_controller(self) :
        """

        # LSTMs Controller :
        # input = ( x_t, y_{t-1}, r0_{t-1}, ..., rN_{t-1}) / rX = X-th vector read from the memory.
        LSTMinput_size = (self.input_dim + self.output_dim) + self.mem_dim*self.nbr_read_heads
        # hidden state / output = controller_output_{t}
        LSTMhidden_size = self.hidden_dim
        num_layers = self.nbr_layers
        dropout = 0.5

        self.lstm_body = nn.LSTM(
            input_size=self.LSTMinput_size,
            hidden_size=self.LSTMhidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=False,
            bidirectional=False
        )
        
        # States :
        self.init_controllerStates()
        
        """
        
        # External Outputs :
        self.output_fn = []
        # input = (h_t, r0_{t}, ..., rN_{t})
        self.EXTinput_size = self.hidden_units[-1] + self.mem_dim * self.nbr_read_heads
        self.output_fn.append( 
            layer_init(
                nn.Linear(
                    self.EXTinput_size, 
                    self.output_dim
                )
            )
        )
        
        if self.classification :
            self.output_fn.append(nn.Softmax())
        else :
            self.output_fn.append(nn.Tanh())
        
        self.output_fn = nn.Sequential(*self.output_fn)

    """
    def init_controllerStates(self) :
        self.ControllerStates = [
            torch.zeros((self.nbr_layers,self.batch_size,self.LSTMhidden_size)),
            torch.zeros((self.nbr_layers,self.batch_size,self.LSTMhidden_size)),
        ]
        
        if self.use_cuda :
            self.ControllerStates = [self.ControllerStates[0].cuda(), self.ControllerStates[1].cuda()]
        
        self.LSTMSs_OUTPUTs = list()
        self.LSTMSs_OUTPUTs.append( (0,self.ControllerStates))
    
    def reset(self, batch_size=None) :
        if batch_size is not None :
            self.batch_size = batch_size
        self.init_controllerStates()
    """
    
    """
    def forward(self, x) :
        # Input : batch x seq_len x input_dim
        self.input = x['input']
        # Previous Desired Output : batch x seq_len x output_dim
        self.prev_desired_output = x['prev_desired_output']
        # Previously read vector from the memory : batch x seq_len x nbr_read_head * mem_dim
        self.prev_read_vec = x['prev_read_vec']

        #print(self.input.size(), self.prev_desired_output.size(), self.prev_read_vec.size())
        #print( self.input , self.prev_desired_output , self.prev_read_vec )
        ctrl_input = torch.cat( [self.input, self.prev_desired_output, self.prev_read_vec], dim=2)
        
        # Controller States :
        #   hidden states h_{t-1} : batch x nbr_layers x hidden_dim 
        #   cell states c_{t-1} : batch x nbr_layers x hidden_dim 
        prev_hc = self.LSTMSs_OUTPUTs[-1][1]

        # Computations :
        self.LSTMSs_OUTPUTs.append( self.LSTMs(ctrl_input, prev_hc) )
        
        return self.LSTMSs_OUTPUTs[-1]
    
    """

    def forward_external_output_fn(self, ctrl_output, slots_read) :
        batch_size = slots_read.shape[0]
        rslots_read = slots_read.reshape(batch_size, -1)
        ext_fc_inp = torch.cat( [ctrl_output, rslots_read], dim=-1)
        self.output_fn_output = self.output_fn(ext_fc_inp)
        
        return self.output_fn_output
    
    def forward_controller(self, inputs):
        '''
        :param inputs: input to LSTM cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        # WARNING: it is imperative to make a copy 
        # of the frame_state, otherwise any changes 
        # will be repercuted onto the current frame_state
        x, frame_states = inputs[0], copy_hdict(inputs[1])
        
        recurrent_neurons = extract_subtree(
            in_dict=frame_states,
            node_id='lstm',
        )

        extra_inputs = extract_subtree(
            in_dict=frame_states,
            node_id='extra_inputs',
        )

        extra_inputs = [v[0].to(x.dtype).to(x.device) for v in extra_inputs.values()]
        if len(extra_inputs): x = torch.cat([x]+extra_inputs, dim=-1)

        if next(self.layers[0].parameters()).is_cuda and not(x.is_cuda):    x = x.cuda() 
        hidden_states, cell_states = recurrent_neurons['hidden'], recurrent_neurons['cell']

        next_hstates, next_cstates = [], []
        outputs = []
        for idx, (layer, hx, cx) in enumerate(zip(self.layers, hidden_states, cell_states) ):
            batch_size = x.size(0)
            if hx.size(0) == 1: # then we have just resetted the values, we need to expand those:
                hx = torch.cat([hx]*batch_size, dim=0)
                cx = torch.cat([cx]*batch_size, dim=0)
            elif hx.size(0) != batch_size:
                raise NotImplementedError("Sizes of the hidden states and the inputs do not coincide.")

            if next(layer.parameters()).is_cuda and \
                (hx is not None or not(hx.is_cuda)) and \
                (cx is  not None or not(cx.is_cuda)):
                if hx is not None:  hx = hx.cuda()
                if cx is not None:  cx = cx.cuda() 

            """
            nhx, ncx = layer(x, (hx, cx))
            next_hstates.append(nhx)
            next_cstates.append(ncx)
            """
            # VDN:
            if len(x.shape)==3:
                raise NotImplementedError
                shapex = x.shape
                shapehx = hx.shape
                shapecx = cx.shape 
                x = x.reshape(-1, shapex[-1])
                hx = hx.reshape(-1, shapehx[-1])
                cx = cx.reshape(-1, shapecx[-1])
                nhx, ncx = layer(x, (hx, cx))
                nhx = nhx.reshape(*shapehx[:2], -1)
                ncx = ncx.reshape(*shapecx[:2], -1)
            else:
                nhx, ncx = layer(x, (hx, cx))

            outputs.append([nhx, ncx])
            next_hstates.append(outputs[-1][0])
            next_cstates.append(outputs[-1][1])
            
            # Consider not applying activation functions on last layer's output?
            if self.gate is not None:
                x = self.gate(outputs[-1][0])
            else:
                x = outputs[-1][0]

        frame_states.update({'lstm':
            {'hidden': next_hstates, 
            'cell': next_cstates}
        })

        return x, frame_states
    
    def get_reset_states(self, cuda=False, repeat=1):
        hidden_states, cell_states = [], []
        for layer in self.layers:
            h = torch.zeros(repeat, layer.hidden_size)
            if cuda:
                h = h.cuda()
            hidden_states.append(h)
            cell_states.append(h)
        return {'lstm':{'hidden': hidden_states, 'cell': cell_states}}

    def get_feature_shape(self):
        return self.output_dim


class NTMMemory(nn.Module) :
    def __init__(
        self, 
        mem_nbr_slots, 
        mem_dim, 
        ):
        
        super(NTMMemory,self).__init__()

        self.mem_nbr_slots = mem_nbr_slots
        self.mem_dim = mem_dim
        
        self.initialize_memory()

    def initialize_memory(self) :
        self.init_mem = torch.zeros((self.mem_nbr_slots,self.mem_dim))
        
    def get_reset_states(self, cuda=False, repeat=1):
        memory = []
        h = self.init_mem.clone().repeat(repeat, 1 , 1)
        if cuda:
            h = h.cuda()
        memory.append(h)
        return {'memory': memory}

    def content_addressing(
        self,
        memory,
        k,
        beta
        ):
        batch_size = k.shape[0]
        nbrHeads = k.size()[1]
        eps = 1e-10
        
        memory_bhSMidx = torch.cat([memory.unsqueeze(1)]*nbrHeads, dim=1).to(k.device)
        kmat = torch.cat([k.unsqueeze(2)]*self.mem_nbr_slots, dim=2)
        cossim = F.cosine_similarity( kmat, memory_bhSMidx, dim=3)
        #(batch_size x nbrHeads nbr_mem_slot )
        w = F.softmax( beta * cossim, dim=-1)
        #(batch_size x nbrHeads nbr_mem_slot )
        # beta : (batch_size x nbrHeads x 1)
        return w 

    def location_addressing(
        self,
        memory,
        pw, 
        wc,
        g,
        s,
        gamma
        ):
        batch_size = wc.shape[0]
        nbrHeads = g.size()[1]
        
        # Interpolation : 
        wg =  g*wc + (1-g)*pw
        #(batch_size x nbrHeads nbr_mem_slot )
        
        # Shift :
        ws = torch.zeros((batch_size, nbrHeads, self.mem_nbr_slots)).to(wc.device)
            
        for hidx in range(nbrHeads) :
            res = self._conv_shift(wg[:,hidx], s[:,hidx])
            #(batch_size x nbr_mem_slot )
            ws[:,hidx] = res
        
        # Sharpening :
        gamma_Sidx = torch.cat([gamma]*self.mem_nbr_slots, dim=2)
        wgam = ws ** gamma_Sidx
        sumw = torch.sum(wgam, dim=2, keepdim=True)
        w = wgam / sumw
        return w        

    def _conv_shift(self, wg, s) :
        batch_size = s.shape[0]
        size = s.shape[1]
        seq_len = wg.shape[1]
        
        c = torch.cat([wg[:,-size+1:], wg, wg[:,:size-1]], dim=1)
        #(batch_size x nbr_mem_slot )
        # s : (batch_size x nbr_mem_slot )
        res = []
        for bidx in range(batch_size):
            cr = F.conv1d(c[bidx].reshape(1,1,-1), s[bidx].reshape(1,1,-1)).squeeze(1)
            #(1 x nbr_mem_slot )
            res.append(cr)
        res = torch.cat(res, dim=0)
        #(batch_size x nbr_mem_slot++)
        
        ret = res[:,1:seq_len+1]
        #(batch_size x nbr_mem_slot)
        
        return ret 
    
    def write(
        self, 
        memory_state, 
        w, 
        erase, 
        add,
        ):
        batch_size = w.shape[0]
        memory_state = memory_state.to(w.device)
        nmemory = torch.zeros_like(memory_state)

        for bidx in range(batch_size) :
            for headidx in range(erase.size()[1]) :
                e = torch.ger(w[bidx][headidx], erase[bidx][headidx])
                a = torch.ger(w[bidx][headidx], add[bidx][headidx])
                nmemory[bidx] = memory_state[bidx]*(1-e)+a 
                #(nbr_mem_slots x mem_dim)
        
        return nmemory

    def read(self, memory_state, w):
        nbrHeads = w.size()[1]
        memory_bhSMidx = torch.cat([memory_state.unsqueeze(1)]*nbrHeads, dim=1).to(w.device)
        wb = torch.cat( [w.unsqueeze(3) for i in range(memory_bhSMidx.size()[3])], dim=3)
        reading = torch.sum(wb * memory_bhSMidx, dim=2)
        #(batch_size x nbrHeads x mem_dim)
        return reading


class NTMBody(nn.Module) :
    def __init__(
        self,
        input_dim=32, 
        hidden_units=512, 
        output_dim=32, 
        mem_nbr_slots=128, 
        mem_dim= 32, 
        nbr_read_heads=1, 
        nbr_write_heads=1, 
        classification=False,
        extra_inputs_infos: Optional[Dict]={},
        ):

        super(NTMBody,self).__init__()

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.hidden_dim = hidden_units[-1]
        self.output_dim = output_dim
        self.extra_inputs_infos = extra_inputs_infos

        self.mem_nbr_slots = mem_nbr_slots
        self.mem_dim = mem_dim
        
        self.nbr_read_heads = nbr_read_heads
        self.nbr_write_heads = nbr_write_heads
        
        self.classification = classification
        
        self.build_memory()
        self.build_controller()
        self.build_heads()

    def build_memory(self) :
        self.memory = NTMMemory(
            mem_nbr_slots=self.mem_nbr_slots,
            mem_dim=self.mem_dim,
        )
        
    def build_controller(self) :
        self.controller = NTMController( 
            # taking into account the previously read vec:
            input_dim=self.input_dim+self.mem_dim*self.nbr_read_heads, 
            hidden_units=self.hidden_units, 
            output_dim=self.output_dim, 
            mem_nbr_slots=self.mem_nbr_slots, 
            mem_dim=self.mem_dim, 
            nbr_read_heads=self.nbr_read_heads, 
            nbr_write_heads=self.nbr_write_heads,
            classification=self.classification, 
            extra_inputs_infos=self.extra_inputs_infos,
        )

    def build_heads(self) :
        self.readHeads = ReadHeads(
            memory=self.memory,
            nbr_heads=self.nbr_read_heads, 
            input_dim=self.hidden_dim, 
        )
        
        self.writeHeads = WriteHeads(
            memory=self.memory,
            nbr_heads=self.nbr_write_heads, 
            input_dim=self.hidden_dim, 
        )

    def get_reset_states(self, cuda=False, repeat=1):
        prev_read_vec = []
        h = torch.zeros(repeat, self.nbr_read_heads*self.mem_dim)
        if cuda:
            h = h.cuda()
        prev_read_vec.append(h)

        # As an encapsulating module, it is its responsability
        # to call get_reset_states on the encapsulated elements:
        hdict = {'ntm_body':{'prev_read_vec': prev_read_vec}}
        hdict['ntm_controller'] = self.controller.get_reset_states(repeat=repeat)
        hdict['ntm_memory'] = self.memory.get_reset_states(repeat=repeat)
        hdict['ntm_readheads'] = self.readHeads.get_reset_states(repeat=repeat)
        hdict['ntm_writeheads'] = self.writeHeads.get_reset_states(repeat=repeat)
        return {'ntm':hdict}

    def forward(self, inputs):
        # NTM_input :
        # 'input' : batch_dim x seq_len x self.input_dim
        # 'prev_desired_output' : batch_dim x seq_len x self.output_dim
        # 'prev_read_vec' : batch_dim x seq_len x self.nbr_read_head * self.mem_dim
        #x['prev_read_vec'] = self.read_outputs[-1]
        # Taking into account the previously read vector as a state:
        x, frame_states = inputs[0], copy_hdict(inputs[1])
        batch_size = x.shape[0]

        ntm_state_dict = extract_subtree(
            in_dict=frame_states,
            node_id='ntm',
        )

        prev_read_vec = ntm_state_dict['ntm_body']['prev_read_vec'][0]
        prev_read_vec = prev_read_vec.to(x.dtype).to(x.device)
        x = torch.cat([x, prev_read_vec], dim=-1)

        # Controller Outputs :
        # output : batch_dim x hidden_dim
        # state : ( h, c) 
        controller_inputs = [x, ntm_state_dict['ntm_controller']]
        nx, ntm_state_dict['ntm_controller'] = self.controller.forward_controller(controller_inputs)
        # nx : dim : 

        # Memory Read :
        # batch_dim x nbr_read_heads * mem_dim :
        readHeads_inputs = [nx, ntm_state_dict['ntm_readheads']]
        read_vec, ntm_state_dict['ntm_readheads'] = self.readHeads.read(
            memory_state=ntm_state_dict['ntm_memory']['memory'][0],
            ctrl_inputs=readHeads_inputs,
        )
        
        # Memory Write :
        writeHeads_inputs = [nx, ntm_state_dict['ntm_writeheads']]
        written_memory_state, ntm_state_dict['ntm_writeheads'] =self.writeHeads.write(
            memory_state=ntm_state_dict['ntm_memory']['memory'][0],
            ctrl_inputs=writeHeads_inputs,
        )

        # External Output Function :
        ext_output = self.controller.forward_external_output_fn( 
            nx,
            read_vec,
        )
        
        ntm_state_dict['ntm_body']['prev_read_vec'] = [read_vec.reshape(batch_size, -1)]
        ntm_state_dict['ntm_memory']['memory'] = [written_memory_state]

        frame_states.update({'ntm':ntm_state_dict})

        return ext_output, frame_states 

    def get_feature_shape(self):
        return self.output_dim
