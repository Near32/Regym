from typing import Dict, List 

import math 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from Archi.modules.module import Module 

from Archi.modules.utils import layer_init


class addXYfeatures(nn.Module) :
    def __init__(self) :
        super(addXYfeatures,self).__init__() 
        self.fXY = None

    def forward(self,x, outputFsizes=False) :
        xsize = x.size()
        batch = xsize[0]
        if self.fXY is None:
            # batch x depth x X x Y
            depth = xsize[1]
            sizeX = xsize[2]
            sizeY = xsize[3]
            stepX = 2.0/sizeX
            stepY = 2.0/sizeY

            fx = torch.zeros((1,1,sizeX,1))
            fy = torch.zeros((1,1,1,sizeY))
            
            vx = -1+0.5*stepX
            for i in range(sizeX):
                fx[:,:,i,:] = vx 
                vx += stepX
            vy = -1+0.5*stepY
            for i in range(sizeY):
                fy[:,:,:,i] = vy 
                vy += stepY
            fxy = fx.repeat(1,1,1,sizeY)
            fyx = fy.repeat(1,1,sizeX,1)
            self.fXY = torch.cat( [fxy,fyx], dim=1)
            self.sizeX = sizeX
            self.sizeY = sizeY
            
        fXY = self.fXY.repeat(batch,1,1,1)
        if x.is_cuda : fXY = fXY.cuda()
            
        out = torch.cat( [x,fXY], dim=1)

        if outputFsizes:
            return out, self.sizeX, self.sizeY

        return out 

class addXYRhoThetaFeatures(nn.Module) :
    def __init__(self) :
        super(addXYRhoThetaFeatures,self).__init__() 
        self.fXYRhoTheta = None

    def forward(self,x, outputFsizes=False) :
        xsize = x.size()
        batch = xsize[0]
        if self.fXYRhoTheta is None:
            # batch x depth x X x Y
            depth = xsize[1]
            sizeX = xsize[2]
            sizeY = xsize[3]
            stepX = 2.0/sizeX
            stepY = 2.0/sizeY

            midX = sizeX/2
            midY = sizeY/2
            sizeRho = math.sqrt(midX**2+midY**2)
            sizeTheta = 2*math.pi
            stepX = 2.0/sizeX
            stepY = 2.0/sizeY

            fx = torch.zeros((1,1,sizeX,1))
            fy = torch.zeros((1,1,1,sizeY))
            
            vx = -1+0.5*stepX
            for i in range(sizeX):
                fx[:,:,i,:] = vx 
                vx += stepX
            vy = -1+0.5*stepY
            for i in range(sizeY):
                fy[:,:,:,i] = vy 
                vy += stepY

            fxy = fx.repeat(1,1,1,sizeY).transpose(-1,-2)
            fyx = -fy.repeat(1,1,sizeX,1).transpose(-1,-2)
            
            fRho = (fxy**2+fyx**2).sqrt()/sizeRho
            fTheta = torch.atan2(fyx, fxy)/math.pi
            
            self.fXYRhoTheta = torch.cat( [fxy,fyx, fRho, fTheta], dim=1)
            self.sizeX = sizeX
            self.sizeY = sizeY

        fXYRhoTheta = self.fXYRhoTheta.repeat(batch,1,1,1)
        if x.is_cuda : fXYRhoTheta = fXYRhoTheta.cuda()
            
        out = torch.cat( [x,fXYRhoTheta], dim=1)

        if outputFsizes:
            return out, self.sizeX, self.sizeY
        
        return out 

def conv( sin, sout,k,stride=1,padding=0,batchNorm=True) :
    layers = []
    layers.append( nn.Conv2d( sin,sout, k, stride,padding,bias=not(batchNorm)) )
    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )

    
def deconv( sin, sout,k,stride=1,padding=0,batchNorm=True) :
    layers = []
    layers.append( nn.ConvTranspose2d( sin,sout, k, stride,padding,bias=not(batchNorm)) )
    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )

def coordconv( sin, sout,kernel_size,stride=1,padding=0,batchNorm=False,bias=True, groups=1, dilation=1) :
    layers = []
    layers.append( addXYfeatures() )
    layers.append( nn.Conv2d( sin+2,
                            sout, 
                            kernel_size, 
                            stride,
                            padding, 
                            groups=groups, 
                            bias=(True if bias else not(batchNorm)),
                            dilation=dilation))

    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )

def coorddeconv( sin, sout,kernel_size,stride=2,padding=1,batchNorm=True,bias=False) :
    layers = []
    layers.append( addXYfeatures() )
    layers.append( nn.ConvTranspose2d( sin+2,sout, kernel_size, stride,padding,bias=(True if bias else not(batchNorm) ) ) )
    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )


def coord4conv( sin, sout,kernel_size,stride=1,padding=0,batchNorm=False,bias=True, groups=1, dilation=1) :
    layers = []
    layers.append( addXYRhoThetaFeatures() )
    layers.append( nn.Conv2d( sin+4,
                            sout, 
                            kernel_size, 
                            stride,
                            padding, 
                            groups=groups, 
                            bias=(True if bias else not(batchNorm)),
                            dilation=dilation))

    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )

def coord4deconv( sin, sout,kernel_size,stride=2,padding=1,batchNorm=True,bias=False) :
    layers = []
    layers.append( addXYRhoThetaFeatures() )
    layers.append( nn.ConvTranspose2d( sin+4,sout, kernel_size, stride,padding,bias=(True if bias else not(batchNorm) ) ) )
    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )


class ConvolutionalNetworkModule(Module):
    def __init__(
        self, 
        input_shape, 
        feature_dim=256, 
        channels=[3, 3], 
        kernel_sizes=[1], 
        strides=[1], 
        paddings=[0], 
        fc_hidden_units=None,
        dropout=0.0, 
        non_linearities=[nn.ReLU],
        use_coordconv=None,
        id='CNModule_0', 
        config=None,
        input_stream_ids=None,
        output_stream_ids={},
        use_cuda=False,
    ):
        '''
        Default input channels assume a RGB image (3 channels).

        :param input_shape: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param channels: list of number of channels for each convolutional layer,
                with the initial value being the number of channels of the input.
        :param kernel_sizes: list of kernel sizes for each convolutional layer.
        :param strides: list of strides for each convolutional layer.
        :param paddings: list of paddings for each convolutional layer.
        :param fc_hidden_units: list of number of neurons per fully-connected 
                hidden layer following the convolutional layers.
        :param dropout: dropout probability to use.
        :param non_linearities: list of non-linear nn.Functional functions to use
                after each convolutional layer.
        :param use_coordconv: None or Int specifying the type of coord convolutional layers to use, either 2 or 4.
        '''
        super(ConvolutionalNetworkModule, self).__init__(
            id=id,
            type="ConvolutionalNetworkModule",
            config=config,
            input_stream_ids=input_stream_ids,
            output_stream_ids=output_stream_ids,
        )

        original_conv_fn = nn.Conv2d
        if use_coordconv is not None:
            if isinstance(use_coordconv, bool)\
            and use_coordconv:
                original_conv_fn = coord4conv
            elif isinstance(use_coordconv, bool)\
            and not use_coordconv:
                pass
            elif use_coordconv == 2: 
                original_conv_fn = coordconv
            elif use_coordconv == 4:
                original_conv_fn = coord4conv
            else:
                raise NotImplementedError

        self.dropout = dropout
        self.non_linearities = non_linearities
        if not isinstance(non_linearities, list):
            self.non_linearities = [non_linearities]
        while len(self.non_linearities) <= (len(channels)+len(fc_hidden_units)):
            self.non_linearities.append(self.non_linearities[-1])
        for idx, nl in enumerate(self.non_linearities):
            if not isinstance(nl, str):
                raise NotImplementedError
            nl_cls = getattr(nn, nl, None)
            if nl_cls is None:
                raise NotImplementedError
            self.non_linearities[idx] = nl_cls
        
        self.feature_dim = feature_dim
        if not(isinstance(self.feature_dim, int)):
            self.feature_dim = feature_dim[-1]
        
        self.cnn = []
        dim = input_shape[1] # height
        in_ch = input_shape[0]
        for idx, (cfg, k, s, p) in enumerate(zip(channels, kernel_sizes, strides, paddings)):
            conv_fn = original_conv_fn
            if isinstance(cfg, str) and cfg == 'MP':
                if isinstance(k, str):
                    assert(k=="Full")
                    k = dim
                    channels[idx+1] = in_ch
                layer = nn.MaxPool2d(kernel_size=k, stride=s)
                self.cnn.append(layer)
                # Update of the shape of the input-image, following Conv:
                dim = (dim-k)//s+1
                print(f"Dim: {cfg} x {dim} x {dim}")
            else:
                add_non_lin = True
                add_dp = (self.dropout > 0.0)
                dropout = self.dropout
                add_bn = False
                add_ln = False
                if isinstance(cfg, str) and 'NoNonLin' in cfg:
                    add_non_lin = False
                    cfg = cfg.replace('NoNonLin', '') 
                if isinstance(cfg, str) and 'Coord2' in cfg:
                    conv_fn = coordconv
                    cfg = cfg.replace('Coord2', '') 
                elif isinstance(cfg, str) and 'Coord4' in cfg:
                    conv_fn = coord4conv
                    cfg = cfg.replace('Coord4', '') 
                
                if isinstance(cfg, str) and '_DP' in cfg:
                    add_dp = True
                    cfg = cfg.split('_DP')
                    dropout = float(cfg[-1])
                    cfg = cfg[0] 
                    # Assumes 'YX_DPZ'
                    # where Y may be BN/LN/nothing
                    # and X is an integer
                    # and Z is the float dropout value.
                
                if isinstance(cfg, str) and 'BN' in cfg:
                    add_bn = True
                    cfg = int(cfg[2:])
                    channels[idx] = cfg
                    # Assumes 'BNX' where X is an integer...
                elif isinstance(cfg, str) and 'LN' in cfg:
                    add_ln = True
                    cfg = int(cfg[2:])
                    channels[idx] = cfg
                    # Assumes 'LNX' where X is an integer...
                elif isinstance(cfg, str):
                    cfg = int(cfg)
                    channels[idx] = cfg
                    
                layer = conv_fn(in_ch, cfg, kernel_size=k, stride=s, padding=p, bias=not(add_bn)) 
                layer = layer_init(layer, w_scale=math.sqrt(2))
                in_ch = cfg
                self.cnn.append(layer)
                if add_bn:
                    self.cnn.append(nn.BatchNorm2d(in_ch))
                if add_ln:
                    # Layer Normalization:
                    # solely about the last dimension of the 4D tensor, i.e. channels...
                    # TODO: It might be necessary to have the possibility to apply this 
                    # normalization over the other dimensions, i.e. width x height...
                    self.cnn.append(nn.LayerNorm(in_ch))
                if add_dp:
                    self.cnn.append(nn.Dropout2d(p=dropout))
                if add_non_lin:
                    #self.cnn.append(self.non_linearities[idx](inplace=True))
                    self.cnn.append(self.non_linearities[idx]())
                # Update of the shape of the input-image, following Conv:
                dim = (dim-k+2*p)//s+1
                print(f"Dim: {cfg} x {dim} x {dim}")
        
        if len(self.cnn):
            self.cnn = nn.Sequential(*self.cnn)
        else:
            self.cnn = None 
            dim = 1
            import ipdb; ipdb.set_trace()
            # check that channels is of the expected size
            print(channels[-1])

        self.feat_map_dim = dim 
        self.feat_map_depth = channels[-1]
        
        hidden_units = fc_hidden_units
        if hidden_units is None or fc_hidden_units == []:
            hidden_units = [dim * dim * channels[-1]]
        else:
            hidden_units = [dim * dim * channels[-1]]+hidden_units

        if isinstance(feature_dim, int):
            hidden_units = hidden_units + [feature_dim]
        else:
            hidden_units = hidden_units + feature_dim
        
        if feature_dim != -1 or fc_hidden_units != []:
            self.fcs = [] #nn.ModuleList()
            nbr_fclayers = len(hidden_units[1:])
            self.non_linearities = self.non_linearities[len(channels):]
            for lidx, (nbr_in, nbr_out) in enumerate(zip(hidden_units, hidden_units[1:])):
                add_non_lin = True
                add_bn = False
                if isinstance(nbr_in, str) and 'BN' in nbr_in:
                    cfg = int(nbr_in[2:])
                    nbr_in = cfg
                    # Assumes 'BNX' where X is an integer...
                if isinstance(nbr_out, str) and 'BN' in nbr_out:
                    add_bn = True
                    cfg = int(nbr_out[2:])
                    nbr_out = cfg
                    # Assumes 'BNX' where X is an integer...
                self.fcs.append( layer_init(nn.Linear(nbr_in, nbr_out), w_scale=math.sqrt(2)))
                if add_bn:
                    self.fcs.append(nn.BatchNorm1d(nbr_out))
                #if lidx != (nbr_fclayers-1):
                if add_non_lin \
                and self.non_linearities[lidx] is not None:
                    self.fcs.append(self.non_linearities[lidx]())
                if self.dropout:
                    self.fcs.append( nn.Dropout(p=self.dropout))
            self.fcs = nn.Sequential(*self.fcs)
        else:
            self.feature_dim = (self.feat_map_dim**2)*self.feat_map_depth
            self.fcs = None 

        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def reset(self):
        self.features_map = None
        self.features = None

    def _compute_feat_map(self, x):
        feat_map = x 
        if self.cnn is not None:
            feat_map = self.cnn(x)
        return feat_map 

    def get_feat_map(self):
        return self.features_map
    
    def forward(self, x, non_lin_output=False):
        self.features_map = self._compute_feat_map(x)

        features = self.features_map
        
        if self.fcs is not None:
            features = self.features_map.reshape(self.features_map.shape[0], -1)
            features = self.fcs(features)
            '''
            for idx, fc in enumerate(self.fcs):
                features = fc(features)
                if idx != len(self.fcs)-1 or non_lin_output:
                    features = F.relu(features)
            '''
        self.features = features 

        return features

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.
        
        WARNING: Experiences can be of shape [batch_size (x temporal_dim), depth_dim, h,w].

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
            batch_size = experiences.size(0)

            original_shape = experiences.shape
            if len(original_shape)>4:
                temporal_dims = original_shape[1:-3]
                product_tdims = np.prod(temporal_dims)
                experiences = experiences.view(batch_size*product_tdims, *original_shape[-3:])
            
            if self.use_cuda:   experiences = experiences.cuda()

            features = self.forward(experiences)

            if len(original_shape)>4:
                features = features.reshape(
                    batch_size,
                    *temporal_dims,
                    *features.shape[1:],
                )

            outputs_stream_dict[output_key] = [features]
            
        return outputs_stream_dict 

    def get_feature_shape(self):
        return self.feature_dim


    def get_input_shape(self):
        return self.input_shape

    def get_feature_shape(self):
        return self.feature_dim

    def _compute_feature_shape(self, input_dim=None, nbr_layer=None):
        return self.feat_map_dim, self.feat_map_depth



