from typing import Dict 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from .utils import layer_init, layer_init_lstm, layer_init_gru
from regym.rl_algorithms.utils import _extract_from_rnn_states, extract_subtree

# From : https://github.com/Kaixhin/Raynbow/blob/master/model.py#10
class NoisyLinear(nn.Module):
    def __init__(self, input_shape, output_shape, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(output_shape, input_shape))
        self.weight_sigma = nn.Parameter(torch.empty(output_shape, input_shape))
        self.register_buffer('weight_epsilon', torch.empty(output_shape, input_shape))

        self.bias_mu = nn.Parameter(torch.empty(output_shape))
        self.bias_sigma = nn.Parameter(torch.empty(output_shape))
        self.register_buffer('bias_epsilon', torch.empty(output_shape))

        self._reset_parameters()
        self._reset_noise()

    def _reset_parameters(self):
        mu_range = 1.0/math.sqrt(self.input_shape)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init/math.sqrt(self.input_shape))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init/math.sqrt(self.input_shape))

    def _reset_noise(self):
        epsin = torch.rand(self.input_shape)
        epsout = torch.rand(self.output_shape)

        epsin = epsin.sign().mul_(epsin.abs().sqrt_())
        epsout = epsout.sign().mul_(epsout.abs().sqrt_())

        self.weight_epsilon.data.copy_(epsout.ger(epsin))
        self.bias_epsilon.data.copy_(epsout)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu+self.weight_sigma*self.weight_epsilon, self.bias_mu+self.bias_sigma*self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

def reset_noisy_layer(module):
    if hasattr(module, "_reset_noise"):
        module._reset_noise()

class ConvolutionalBody(nn.Module):
    def __init__(self, input_shape, feature_dim=256, channels=[3, 3], kernel_sizes=[1], strides=[1], paddings=[0], dropout=0.0, non_linearities=[nn.ReLU]):
        '''
        Default input channels assume a RGB image (3 channels).

        :param input_shape: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param channels: list of number of channels for each convolutional layer,
                with the initial value being the number of channels of the input.
        :param kernel_sizes: list of kernel sizes for each convolutional layer.
        :param strides: list of strides for each convolutional layer.
        :param paddings: list of paddings for each convolutional layer.
        :param dropout: dropout probability to use.
        :param non_linearities: list of non-linear nn.Functional functions to use
                after each convolutional layer.
        '''
        super(ConvolutionalBody, self).__init__()
        self.dropout = dropout
        self.non_linearities = non_linearities
        if not isinstance(non_linearities, list):
            self.non_linearities = [non_linearities] * (len(channels) - 1)
        else:
            while len(self.non_linearities) <= (len(channels) - 1):
                self.non_linearities.append(self.non_linearities[0])

        self.feature_dim = feature_dim
        if isinstance(feature_dim, tuple):
            self.feature_dim = feature_dim[-1]

        self.features = []
        # input_shape size: [channels, height, width]
        h_dim = input_shape[1]
        w_dim = input_shape[2]
        in_ch = channels[0]
        for idx, (cfg, k, s, p) in enumerate(zip(channels[1:], kernel_sizes, strides, paddings)):
            if cfg == 'M':
                layer = nn.MaxPool2d(kernel_size=k, stride=s)
                self.features.append(layer)
                # Update of the shape of the input-image, following Conv:
                h_dim = (h_dim-k)//s+1
                w_dim = (w_dim-k)//s+1
                print(f"Dims: Height: {h_dim}\t Width: {w_dim}")
            else:
                layer = nn.Conv2d(in_channels=in_ch, out_channels=cfg, kernel_size=k, stride=s, padding=p) 
                layer = layer_init(layer, w_scale=math.sqrt(2))
                in_ch = cfg
                self.features.append(layer)
                self.features.append(self.non_linearities[idx](inplace=True))
                # Update of the shape of the input-image, following Conv:
                h_dim = (h_dim-k+2*p)//s+1
                w_dim = (w_dim-k+2*p)//s+1
                print(f"Dims: Height: {h_dim}\t Width: {w_dim}")
        self.features = nn.Sequential(*self.features)

        self.feat_map_depth = channels[-1]

        hidden_units = (h_dim * w_dim * channels[-1],)
        if isinstance(feature_dim, tuple):
            hidden_units = hidden_units + feature_dim
        else:
            hidden_units = hidden_units + (self.feature_dim,)

        self.fcs = nn.ModuleList()
        for nbr_in, nbr_out in zip(hidden_units, hidden_units[1:]):
            self.fcs.append( layer_init(nn.Linear(nbr_in, nbr_out), w_scale=math.sqrt(2)))
            if self.dropout:
                self.fcs.append( nn.Dropout(p=self.dropout))

    def _compute_feat_map(self, x):
        return self.features(x)

    def forward(self, x, non_lin_output=True):
        feat_map = self._compute_feat_map(x)

        # View -> Reshape
        #features = feat_map.view(feat_map.size(0), -1)
        features = feat_map.reshape(feat_map.size(0), -1)
        for idx, fc in enumerate(self.fcs):
            features = fc(features)
            if idx != len(self.fcs)-1 or non_lin_output:
                features = F.relu(features)

        return features

    def get_input_shape(self):
        return self.input_shape

    def get_feature_shape(self):
        return self.feature_dim


class addXYfeatures(nn.Module) :
    def __init__(self) :
        super(addXYfeatures,self).__init__() 
        self.fXY = None

    def forward(self,x) :
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
        
        fXY = self.fXY.repeat(batch,1,1,1)
        if x.is_cuda : fXY = fXY.cuda()
            
        out = torch.cat( [x,fXY], dim=1)

        return out 

def conv( sin, sout,k,stride=2,pad=1,batchNorm=True) :
    layers = []
    layers.append( nn.Conv2d( sin,sout, k, stride,pad,bias=not(batchNorm)) )
    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )


def deconv( sin, sout,k,stride=2,pad=1,batchNorm=True) :
    layers = []
    layers.append( nn.ConvTranspose2d( sin,sout, k, stride,pad,bias=not(batchNorm)) )
    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )

def coordconv( sin, sout,kernel_size,stride=2,pad=1,batchNorm=False,bias=False) :
    layers = []
    layers.append( addXYfeatures() )
    layers.append( nn.Conv2d( sin+2,sout, kernel_size, stride,pad,bias=(True if bias else not(batchNorm) ) ) )
    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )

def coorddeconv( sin, sout,kernel_size,stride=2,pad=1,batchNorm=True,bias=False) :
    layers = []
    layers.append( addXYfeatures() )
    layers.append( nn.ConvTranspose2d( sin+2,sout, kernel_size, stride,pad,bias=(True if bias else not(batchNorm) ) ) )
    if batchNorm :
        layers.append( nn.BatchNorm2d( sout) )
    return nn.Sequential( *layers )


class BroadcastingDecoder(nn.Module) :
    def __init__(self, output_shape=[3, 64, 64], 
                       net_depth=3, 
                       kernel_size=3, 
                       stride=1, 
                       padding=1, 
                       latent_dim=32, 
                       conv_dim=64):
        super(BroadcastingDecoder,self).__init__()

        assert(len(output_shape)==3 and output_shape[2]==output_shape[1])
        
        self.output_shape = output_shape
        self.net_depth = net_depth
        self.latent_dim = latent_dim 
        self.conv_dim = conv_dim

        self.dcs = []
        dim = self.output_shape[-1]
        outd = self.conv_dim
        ind= self.latent_dim
        
        for i in range(self.net_depth) :
            
            if i == self.net_depth-1: 
                outd = self.output_shape[0]

            if i == 0: 
                layer = layer_init(coordconv( ind, outd, kernel_size, stride=stride, pad=padding), w_scale=5e-2)
            else:
                layer = layer_init(nn.Conv2d(ind, outd, kernel_size=kernel_size, stride=stride, padding=padding), w_scale=5e-2)
            
            self.dcs.append(layer)

            if i != self.net_depth-1: 
                self.dcs.append( nn.ReLU() )
                #self.dcs.append( nn.LeakyReLU(0.05) )
            
            ind = outd 
            dim = (dim-kernel_size+2*padding)//stride+1
            print('BroadcastingDecoder: layer {} : dim {}.'.format(i, dim))
        
        self.dcs = nn.Sequential( *self.dcs) 
                
    def decode(self, z) :
        z = z.view( z.size(0), z.size(1), 1, 1)
        out = z.repeat(1, 1, self.output_shape[-2], self.output_shape[-1])
        out = self.dcs(out)
        #out = torch.sigmoid(out)
        return out

    def forward(self,z) :
        return self.decode(z)


def permutate_latents(z):
    assert(z.dim() == 2)
    batch_size, latent_dim = z.size()
    pz = list()
    for lz in torch.split(z, split_size_or_sections=1, dim=1):
        b_perm = torch.randperm(batch_size).to(z.device)
        p_lz = lz[b_perm]
        pz.append(p_lz)
    pz = torch.cat(pz, dim=1)
    return pz 


class BetaVAE(nn.Module) :
    def __init__(self, input_shape=[3, 64, 64], 
                       latent_dim=32, 
                       channels=[3, 3], 
                       kernel_sizes=[1], 
                       strides=[1], 
                       paddings=[0], 
                       non_linearities=[F.leaky_relu],
                       beta=1e0, 
                       nbr_attention_slot=None,
                       decoder_conv_dim=32, 
                       pretrained=False, 
                       resnet_encoder=False,
                       resnet_nbr_layer=2,
                       decoder_nbr_layer=4,
                       NormalOutputDistribution=True,
                       EncodingCapacityStep=None,
                       maxEncodingCapacity=1000,
                       nbrEpochTillMaxEncodingCapacity=4,
                       constrainedEncoding=True,
                       observation_sigma=0.05,
                       kwargs=None):
        super(BetaVAE,self).__init__()

        self.kwargs = kwargs

        self.beta = beta
        self.observation_sigma = observation_sigma
        self.latent_dim = latent_dim
        self.nbr_attention_slot = nbr_attention_slot
        self.input_shape = input_shape
        self.NormalOutputDistribution = NormalOutputDistribution

        self.EncodingCapacity = 0.0
        self.EncodingCapacityStep = EncodingCapacityStep
        self.maxEncodingCapacity = maxEncodingCapacity
        self.constrainedEncoding = constrainedEncoding
        self.nbrEpochTillMaxEncodingCapacity = nbrEpochTillMaxEncodingCapacity
        
        self.increaseEncodingCapacity = True
        if self.constrainedEncoding:
            nbritperepoch = 200
            print('ITER PER EPOCH : {}'.format(nbritperepoch))
            nbrepochtillmax = self.nbrEpochTillMaxEncodingCapacity
            nbrittillmax = nbrepochtillmax * nbritperepoch
            print('ITER TILL MAX ENCODING CAPACITY : {}'.format(nbrittillmax))
            self.EncodingCapacityStep = self.maxEncodingCapacity / nbrittillmax        

        '''
        if resnet_encoder:
            self.encoder = ResNetEncoder(input_shape=input_shape, 
                                         latent_dim=latent_dim,
                                         nbr_layer=resnet_nbr_layer,
                                         pretrained=pretrained)
        else:
        '''    
        self.encoder = ConvolutionalBody(input_shape=input_shape,
                                         feature_dim=(256, latent_dim*2), 
                                         channels=channels,#[input_shape[0], 32, 32, 64], 
                                         kernel_sizes=kernel_sizes,#[3, 3, 3], 
                                         strides=strides,#[2, 2, 2],
                                         paddings=paddings,#[0, 0, 0],
                                         non_linearities=non_linearities)#[F.relu])
    
        self.decoder = BroadcastingDecoder(output_shape=input_shape,
                                           net_depth=decoder_nbr_layer, 
                                           kernel_size=3, 
                                           stride=1, 
                                           padding=1, 
                                           latent_dim=latent_dim, 
                                           conv_dim=decoder_conv_dim)

        self.tc_discriminator = FCBody(state_dim=self.get_feature_shape(), 
                                       hidden_units=self.kwargs['vae_tc_discriminator_hidden_units'], 
                                       gate=F.leaky_relu)

    def get_feature_shape(self):
        return self.latent_dim
    
    def get_input_shape(self):
        return self.input_shape
    
    def reparameterize(self, mu,log_var) :
        eps = torch.randn( (mu.size()[0], mu.size()[1]) )
        if mu.is_cuda:  eps = eps.cuda()
        z = mu + eps * torch.exp( log_var ).sqrt()
        return z

    def forward(self,x) :
        z, _, _ = self.encodeZ(x=x)
        return z
    
    def encode(self,x) :
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1 )
        return mu

    def encodeZ(self,x) :
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1 )
        z = self.reparameterize(mu, log_var)        
        return z, mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def _forward(self,x=None,evaluation=False,fixed_latent=None,data=None) :
        if data is None and x is not None :
            if evaluation :
                z, mu, log_var = self.encodeZ(x)
                h = None
                VAE_output = None 
            else :
                h = self.encoder(x)
                mu, log_var = torch.chunk(h, 2, dim=1 )
                z = self.reparameterize(mu, log_var)
                VAE_output = self.decoder(z)
        elif data is not None :
            mu, log_var = data 
            z = self.reparameterize(mu, log_var)
            h = None
            VAE_output = None
            if not(evaluation) :
                VAE_output = self.decoder(z)

        self.batch_size = z.size()[0]
        if fixed_latent is not None :
            idx = fixed_latent[0]
            val = fixed_latent[1]
            mu = mu.cpu().data 
            mu[:,idx] = val
            if next(self.parameters()).is_cuda : mu = mu.cuda()
            z = self.reparameterize(mu, log_var)
            
        return h, z, mu, log_var, VAE_output  

    def compute_modularity(self, x, z):
        if z.size(0) > 1:
            z1, z2 = z.chunk(2, dim=0)
            
            zeros = torch.zeros(z1.size(0)).long().to(z1.device)
            ones = torch.ones(z2.size(0)).long().to(z2.device)
            
            pz = permutate_latents(z2)
            Dz = self.tc_discriminator(z1)
            Dpz = self.tc_discriminator(pz)
            tc_l11 = 0.5*F.cross_entropy(input=Dz, target=zeros, reduction='none')
            # (b1, )
            tc_l12 = 0.5*F.cross_entropy(input=Dpz, target=ones, reduction='none')
            # (b2, )
            
            zeros = torch.zeros(z2.size(0)).long().to(z2.device)
            ones = torch.ones(z1.size(0)).long().to(z1.device)
            
            pz = permutate_latents(z1)
            Dz = self.tc_discriminator(z2)
            Dpz = self.tc_discriminator(pz)

            tc_l21 = 0.5*F.cross_entropy(input=Dz, target=zeros, reduction='none')
            # (b1, )
            tc_l22 = 0.5*F.cross_entropy(input=Dpz, target=ones, reduction='none')
            # (b2, )
            
            tc_loss1 = tc_l11 + tc_l22
            tc_loss2 = tc_l12 + tc_l21
            
            tc_loss = torch.cat([tc_loss1, tc_loss2]).mean()
            # (1, )
            
            probDz = F.softmax(Dz.detach(), dim=1)[...,:1]
            probDpz = F.softmax(Dpz.detach(), dim=1)[...,1:]
            discr_acc = (torch.cat([probDz,probDpz],dim=0) >= 0.5).sum().float().div(2*probDz.size(0))
            
            modularity = discr_acc.cpu().unsqueeze(0)
        else:
            tc_loss = torch.zeros(1).to(z.device)
            modularity = torch.zeros(1).to(z.device)

        return tc_loss, modularity

    def compute_vae_loss(self,x=None,
                         fixed_latent=None,
                         data=None,
                         evaluation=False,
                         observation_sigma=None) :
        if x is None: 
            if self.x is not None:
                x = self.x 
            else:
                raise NotImplementedError

        gtx = x 
        xsize = x.size()
        self.batch_size = xsize[0]
        
        h, z, mu, log_var, VAE_output = self._forward(x=x,fixed_latent=fixed_latent,data=data,evaluation=evaluation)
        
        if evaluation :
            VAE_output = gtx 

        #--------------------------------------------------------------------------------------------------------------
        # VAE loss :
        #--------------------------------------------------------------------------------------------------------------
        # Reconstruction loss :
        if observation_sigma is not None:
            self.observation_sigma = observation_sigma
        if self.NormalOutputDistribution:
            #Normal :
            neg_log_lik = -torch.distributions.Normal(VAE_output, self.observation_sigma).log_prob( gtx)
        else:
            #Bernoulli :
            neg_log_lik = -torch.distributions.Bernoulli( VAE_output ).log_prob( gtx )
        
        reconst_loss = torch.sum( neg_log_lik.view( self.batch_size, -1), dim=1)
        #--------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------
        # KL Divergence :
        true_kl_divergence = 0.5 * (mu**2 + torch.exp(log_var) - log_var -1)
        
        if self.EncodingCapacityStep is None :
            kl_divergence = torch.sum( true_kl_divergence, dim=1)
            kl_divergence_regularized = torch.zeros_like( kl_divergence)
            VAE_loss = reconst_loss + self.beta * kl_divergence
        else:
            kl_divergence_regularized =  torch.abs( torch.sum(true_kl_divergence, dim=1) - self.EncodingCapacity ) 
            kl_divergence =  torch.sum(true_kl_divergence, dim=1 )
            VAE_loss = reconst_loss + self.beta * kl_divergence_regularized
            
            if self.increaseEncodingCapacity and self.training:
                self.EncodingCapacity += self.EncodingCapacityStep
            if self.EncodingCapacity >= self.maxEncodingCapacity :
                self.increaseEncodingCapacity = False 
        #--------------------------------------------------------------------------------------------------------------

        tc_loss, modularity = self.compute_modularity(x, z)

        return VAE_loss, neg_log_lik, kl_divergence_regularized, true_kl_divergence, tc_loss, modularity


def BetaVAEBody(input_shape, feature_dim, channels, kernel_sizes, strides, paddings, kwargs):
    nbr_layer = None
    resnet_encoder = False#('ResNet' in architecture)
    if resnet_encoder:
        nbr_layer = int(architecture[-1])
    pretrained = False #('pretrained' in architecture)
    
    beta = float(kwargs['vae_beta'])
    maxCap = float(kwargs['vae_max_capacity'])
    nbrEpochTillMaxEncodingCapacity = int(kwargs['vae_nbr_epoch_till_max_capacity'])
    constrainedEncoding = bool(kwargs['vae_constrainedEncoding'])
    
    latent_dim = feature_dim
    if 'vae_nbr_latent_dim' in kwargs:
        latent_dim = kwargs['vae_nbr_latent_dim']
    decoder_nbr_layer = 4
    if 'vae_decoder_nbr_layer' in kwargs:
        decoder_nbr_layer = kwargs['vae_decoder_nbr_layer']
    if 'vae_decoder_conv_dim' in kwargs:
        decoder_conv_dim = kwargs['vae_decoder_conv_dim']
    
    model = BetaVAE(input_shape=input_shape, 
                    latent_dim=latent_dim,
                    channels=channels, 
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    paddings=paddings,
                    non_linearities=[F.leaky_relu],
                    beta=beta,
                    resnet_encoder=resnet_encoder,
                    resnet_nbr_layer=nbr_layer,
                    pretrained=pretrained,
                    decoder_nbr_layer=decoder_nbr_layer,
                    decoder_conv_dim=decoder_conv_dim,
                    maxEncodingCapacity=maxCap,
                    nbrEpochTillMaxEncodingCapacity=nbrEpochTillMaxEncodingCapacity,
                    constrainedEncoding=constrainedEncoding,
                    observation_sigma=0.05,
                    kwargs=kwargs)

    return model


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 downsample=None,
                 groups=1,
                 base_width=64, 
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 downsample=None, 
                 groups=1,
                 base_width=64, 
                 dilation=1, 
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, 
                 input_shape,
                 block, 
                 nbr_block_per_layer,
                 stride_per_layer=[1, 2, 2, 2],
                 num_classes=1000, 
                 zero_init_residual=False,
                 groups=1, 
                 width_per_group=64, 
                 replace_stride_with_dilation=[False, False, False],
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.input_shape = input_shape

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.input_shape[0], self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 
                                       64, 
                                       blocks=nbr_block_per_layer[0], 
                                       stride=stride_per_layer[0])
        self.layer2 = self._make_layer(block, 
                                       128, 
                                       blocks=nbr_block_per_layer[1], 
                                       stride=stride_per_layer[1],
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 
                                       256, 
                                       blocks=nbr_block_per_layer[2], 
                                       stride=stride_per_layer[2],
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 
                                       512, 
                                       blocks=nbr_block_per_layer[3], 
                                       stride=stride_per_layer[3],
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


def _resnet(input_shape, block, layers, output_dim, **kwargs):
    return ResNet(input_shape=input_shape, block=block, nbr_block_per_layer=layers, num_classes=output_dim, **kwargs)

def resnet18Input64(input_shape, output_dim, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(input_shape=input_shape, block=BasicBlock, layers=[2, 2, 2, 2], stride_per_layer=[1, 2, 1, 2], output_dim=output_dim, **kwargs)


#class ConvolutionalLstmBody(ConvolutionalBody):
class ConvolutionalLstmBody(nn.Module):
    def __init__(self, input_shape, feature_dim=256, channels=[3, 3],
                 kernel_sizes=[1], strides=[1], paddings=[0],
                 extra_inputs_infos: Dict={},
                 non_linearities=[nn.ReLU], hidden_units=(256,), gate=F.relu):
        '''
        Default input channels assume a RGB image (3 channels).

        :param input_shape: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param channels: list of number of channels for each convolutional layer,
                with the initial value being the number of channels of the input.
        :param kernel_sizes: list of kernel sizes for each convolutional layer.
        :param strides: list of strides for each convolutional layer.
        :param paddings: list of paddings for each convolutional layer.
        :param extra_inputs_infos: Dictionnary containing the shape of the lstm-relevant extra inputs.
        :param non_linearities: list of non-linear nn.Functional functions to use
                after each convolutional layer.
        '''
        super(ConvolutionalLstmBody, self).__init__()
        self.cnn_body = ConvolutionalBody(
            input_shape=input_shape,
            feature_dim=feature_dim,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            non_linearities=non_linearities
        )

        # Use lstm_input_dim instead of cnn_body output feature dimension 
        lstm_input_dim = self.cnn_body.get_feature_shape() # lstm_input_dim if lstm_input_dim != -1 else self.cnn_body.get_feature_shape()
        for key in extra_inputs_infos:
            shape = extra_inputs_infos[key]['shape']
            assert len(shape) == 1 
            lstm_input_dim += shape[-1]

        self.lstm_body = LSTMBody( state_dim=lstm_input_dim, hidden_units=hidden_units, gate=gate)

    def forward(self, inputs):
        '''
        :param inputs: input to LSTM cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, frame_states = inputs[0], inputs[1]
        
        features = self.cnn_body.forward(x)
        
        recurrent_neurons = _extract_from_rnn_states(
            rnn_states_batched=frame_states,
            batch_idx=None,
            map_keys=['hidden', 'cell'],
        )
        
        extra_inputs = extract_subtree(
            in_dict=frame_states,
            node_id='extra_inputs',
        )
        
        extra_inputs = [v[0].to(features.dtype).to(features.device) for v in extra_inputs.values()]
        if len(extra_inputs): features = torch.cat([features]+extra_inputs, dim=-1)

        x, recurrent_neurons['lstm_body'] = self.lstm_body( (features, recurrent_neurons['lstm_body']))
        return x, recurrent_neurons

    def get_reset_states(self, cuda=False, repeat=1):
        return self.lstm_body.get_reset_states(cuda=cuda, repeat=repeat)
    
    def get_input_shape(self):
        #return self.input_shape
        return self.cnn_body.input_shape

    def get_feature_shape(self):
        return self.lstm_body.get_feature_shape()


class ConvolutionalGruBody(ConvolutionalBody):
    def __init__(self, input_shape, feature_dim=256, channels=[3, 3], kernel_sizes=[1], strides=[1], paddings=[0], non_linearities=[F.relu], hidden_units=(256,), gate=F.relu):
        '''
        Default input channels assume a RGB image (3 channels).

        :param input_shape: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param channels: list of number of channels for each convolutional layer,
                with the initial value being the number of channels of the input.
        :param kernel_sizes: list of kernel sizes for each convolutional layer.
        :param strides: list of strides for each convolutional layer.
        :param paddings: list of paddings for each convolutional layer.
        :param non_linearities: list of non-linear nn.Functional functions to use
                after each convolutional layer.
        '''
        super(ConvolutionalGruBody, self).__init__(input_shape=input_shape,
                                                feature_dim=feature_dim,
                                                channels=channels,
                                                kernel_sizes=kernel_sizes,
                                                strides=strides,
                                                paddings=paddings,
                                                non_linearities=non_linearities)

        self.gru_body = GRUBody( state_dim=self.feature_dim, hidden_units=hidden_units, gate=gate)

    def forward(self, inputs):
        '''
        :param inputs: input to GRU cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, recurrent_neurons = inputs
        features = super(ConvolutionalGruBody,self).forward(x)
        x, recurrent_neurons['gru_body'] = self.gru_body( (features, recurrent_neurons['gru_body']))
        return x, recurrent_neurons
        
    def get_reset_states(self, cuda=False, repeat=1):
        return self.gru_body.get_reset_states(cuda=cuda, repeat=repeat)

    def get_input_shape(self):
        return self.input_shape

    def get_feature_shape(self):
        return self.gru_body.get_feature_shape()


class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y

class FCBody(nn.Module):
    def __init__(
        self, 
        state_dim, 
        hidden_units, 
        non_linearities=None, 
        gate=None,
        dropout=0.0,
        use_cuda=False,
        add_non_lin_final_layer=False,
        layer_init_fn=None):
        """
        TODO: gate / nonlinearities hyperparameters...
        """
        super(FCBody, self).__init__()
        
        if isinstance(state_dim,int): state_dim = [state_dim]

        dims = state_dim + hidden_units
        self.dropout = dropout

        if non_linearities is None:
            non_linearities = [nn.ReLU]

        self.non_linearities = non_linearities
        if not isinstance(non_linearities, list):
            self.non_linearities = [non_linearities] * (len(dims) - 1)
        else:
            while len(self.non_linearities) <= (len(dims) - 1):
                self.non_linearities.append(self.non_linearities[0])
        
        self.layers = []
        in_ch = dims[0]
        for idx, cfg in enumerate(dims[1:]):
            add_non_lin = True
            if not(add_non_lin_final_layer) and idx == len(dims)-2:  add_non_lin = False
            add_dp = (self.dropout > 0.0)
            dropout = self.dropout
            add_bn = False
            add_ln = False
            if isinstance(cfg, str) and 'NoNonLin' in cfg:
                add_non_lin = False
                cfg = cfg.replace('NoNonLin', '') 
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
                dims[idx+1] = cfg
                # Assumes 'BNX' where X is an integer...
            elif isinstance(cfg, str) and 'LN' in cfg:
                add_ln = True
                cfg = int(cfg[2:])
                dims[idx+1] = cfg
                # Assumes 'LNX' where X is an integer...
            elif isinstance(cfg, str):
                cfg = int(cfg)
                dims[idx+1] = cfg
                
            layer = nn.Linear(in_ch, cfg, bias=not(add_bn)) 
            if layer_init_fn is not None:
                layer = layer_init_fn(layer)#, w_scale=math.sqrt(2))
            else:
                layer = layer_init(layer, w_scale=math.sqrt(2))
            in_ch = cfg
            self.layers.append(layer)
            if add_bn:
                self.layers.append(nn.BatchNorm1d(in_ch))
            if add_ln:
                # Layer Normalization:
                # solely about the last dimension of the 4D tensor, i.e. channels...
                # TODO: It might be necessary to have the possibility to apply this 
                # normalization over the other dimensions, i.e. width x height...
                self.layers.append(nn.LayerNorm(in_ch))
            if add_dp:
                self.layers.append(nn.Dropout(p=dropout))
            if add_non_lin:
                self.layers.append(self.non_linearities[idx]())
        self.layers = nn.Sequential(*self.layers)

        self.feature_dim = dims[-1]

        self.use_cuda = use_cuda
        if self.use_cuda:
            self = self.cuda()

    def forward(self, x):
        output = self.layers(x)
        return output

    def get_feature_shape(self):
        return self.feature_dim

"""
class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu, layer_fn=nn.Linear):
        super(FCBody, self).__init__()
        if not isinstance(hidden_units, tuple): hidden_units = tuple(hidden_units)
        if isinstance(state_dim,int):   dims = (state_dim, ) + hidden_units
        else:   dims = state_dim + hidden_units
        self.layers = nn.ModuleList([layer_fn(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        if layer_fn == nn.Linear:   self.layers.apply(layer_init)
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

    def get_feature_shape(self):
        return self.feature_dim
"""

class LinearLinearBody(nn.Module):
    def __init__(
        self, 
        state_dim, 
        feature_dim=256, 
        hidden_units=(256,), 
        non_linearities=[nn.ReLU], 
        gate=F.relu,
        dropout=0.0,
        add_non_lin_final_layer=False,
        layer_init_fn=None,
        extra_inputs_infos: Dict={},
        ):
        '''
        
        :param state_dim: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param channels: list of number of channels for each convolutional layer,
                with the initial value being the number of channels of the input.
        :param kernel_sizes: list of kernel sizes for each convolutional layer.
        :param strides: list of strides for each convolutional layer.
        :param paddings: list of paddings for each convolutional layer.
        :param extra_inputs_infos: Dictionnary containing the shape of the lstm-relevant extra inputs.
        :param non_linearities: list of non-linear nn.Functional functions to use
                after each convolutional layer.
        '''
        super(LinearLinearBody, self).__init__()
        self.state_dim = state_dim

        self.linear_body = FCBody(
            state_dim=state_dim,
            hidden_units=[feature_dim],
            non_linearities=non_linearities,
            gate=gate,
            dropout=dropout,
            add_non_lin_final_layer=add_non_lin_final_layer,
            layer_init_fn=layer_init_fn
        )

        final_linear_input_dim = self.linear_body.get_feature_shape() # lstm_input_dim if lstm_input_dim != -1 else self.cnn_body.get_feature_shape()
        # verify featureshape = feature_dim
        for key in extra_inputs_infos:
            shape = extra_inputs_infos[key]['shape']
            assert len(shape) == 1 
            final_linear_input_dim += shape[-1]

        self.final_linear_body = FCBody( 
            state_dim=final_linear_input_dim, 
            hidden_units=hidden_units, 
            gate=gate,
            non_linearities=non_linearities,
            dropout=dropout,
            add_non_lin_final_layer=True,
            layer_init_fn=layer_init_fn,
        )

        self.dummy_lstm_body = LSTMBody( state_dim=final_linear_input_dim, hidden_units=hidden_units, gate=gate)


    def forward(self, inputs):
        '''
        :param inputs: input to LSTM cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, frame_states = inputs[0], inputs[1]
        
        recurrent_neurons = _extract_from_rnn_states(
            rnn_states_batched=frame_states,
            batch_idx=None,
            map_keys=['hidden', 'cell'],
        )
        
        features = self.linear_body(x)
        
        extra_inputs = extract_subtree(
            in_dict=frame_states,
            node_id='extra_inputs',
        )
        
        extra_inputs = [v[0].to(features.dtype).to(features.device) for v in extra_inputs.values()]
        if len(extra_inputs): features = torch.cat([features]+extra_inputs, dim=-1)

        x = self.final_linear_body( features)
        return x, recurrent_neurons

    def get_reset_states(self, cuda=False, repeat=1):
        return self.dummy_lstm_body.get_reset_states(cuda=cuda, repeat=repeat)
    
    def get_input_shape(self):
        return self.state_dim

    def get_feature_shape(self):
        return self.final_linear_body.get_feature_shape()

class LinearLstmBody(nn.Module):
    def __init__(
        self, 
        state_dim, 
        feature_dim=256, 
        hidden_units=(256,), 
        non_linearities=[nn.ReLU], 
        gate=F.relu,
        dropout=0.0,
        add_non_lin_final_layer=False,
        layer_init_fn=None,
        extra_inputs_infos: Dict={},
        ):
        '''
        
        :param state_dim: dimensions of the input.
        :param feature_dim: integer size of the output.
        :param channels: list of number of channels for each convolutional layer,
                with the initial value being the number of channels of the input.
        :param kernel_sizes: list of kernel sizes for each convolutional layer.
        :param strides: list of strides for each convolutional layer.
        :param paddings: list of paddings for each convolutional layer.
        :param extra_inputs_infos: Dictionnary containing the shape of the lstm-relevant extra inputs.
        :param non_linearities: list of non-linear nn.Functional functions to use
                after each convolutional layer.
        '''
        super(LinearLstmBody, self).__init__()
        self.state_dim = state_dim

        self.linear_body = FCBody(
            state_dim=state_dim,
            hidden_units=[feature_dim],
            non_linearities=non_linearities,
            gate=gate,
            dropout=dropout,
            add_non_lin_final_layer=add_non_lin_final_layer,
            layer_init_fn=layer_init_fn
        )

        # Use lstm_input_dim instead of cnn_body output feature dimension 
        lstm_input_dim = self.linear_body.get_feature_shape() # lstm_input_dim if lstm_input_dim != -1 else self.cnn_body.get_feature_shape()
        # verify featureshape = feature_dim
        for key in extra_inputs_infos:
            shape = extra_inputs_infos[key]['shape']
            assert len(shape) == 1 
            lstm_input_dim += shape[-1]

        self.lstm_body = LSTMBody( state_dim=lstm_input_dim, hidden_units=hidden_units, gate=gate)

    def forward(self, inputs):
        '''
        :param inputs: input to LSTM cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, frame_states = inputs[0], inputs[1]
        
        features = self.linear_body(x)
        
        recurrent_neurons = _extract_from_rnn_states(
            rnn_states_batched=frame_states,
            batch_idx=None,
            map_keys=['hidden', 'cell'],
        )
        
        extra_inputs = extract_subtree(
            in_dict=frame_states,
            node_id='extra_inputs',
        )
        
        extra_inputs = [v[0].to(features.dtype).to(features.device) for v in extra_inputs.values()]
        if len(extra_inputs): features = torch.cat([features]+extra_inputs, dim=-1)

        x, recurrent_neurons['lstm_body'] = self.lstm_body( (features, recurrent_neurons['lstm_body']))
        return x, recurrent_neurons

    def get_reset_states(self, cuda=False, repeat=1):
        return self.lstm_body.get_reset_states(cuda=cuda, repeat=repeat)
    
    def get_input_shape(self):
        return self.state_dim

    def get_feature_shape(self):
        return self.lstm_body.get_feature_shape()

class LSTMBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(256), gate=F.relu):
        super(LSTMBody, self).__init__()
        if not isinstance(hidden_units, tuple): hidden_units = tuple(hidden_units)
        if isinstance(state_dim,int):   dims = (state_dim, ) + hidden_units
        else:   dims = state_dim + hidden_units
        self.layers = nn.ModuleList([layer_init_lstm(nn.LSTMCell(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.feature_dim = dims[-1]
        self.gate = gate

    def forward(self, inputs):
        '''
        :param inputs: input to LSTM cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, recurrent_neurons = inputs[0], inputs[1]
        if next(self.layers[0].parameters()).is_cuda and not(x.is_cuda):    x = x.cuda() 
        hidden_states, cell_states = recurrent_neurons['hidden'], recurrent_neurons['cell']

        next_hstates, next_cstates = [], []
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

            nhx, ncx = layer(x, (hx, cx))
            next_hstates.append(nhx)
            next_cstates.append(ncx)
            # Consider not applying activation functions on last layer's output
            if self.gate is not None:
                x = self.gate(nhx)

        return x, {'hidden': next_hstates, 'cell': next_cstates}

    def get_reset_states(self, cuda=False, repeat=1):
        hidden_states, cell_states = [], []
        for layer in self.layers:
            h = torch.zeros(repeat, layer.hidden_size)
            if cuda:
                h = h.cuda()
            hidden_states.append(h)
            cell_states.append(h)
        return {'hidden': hidden_states, 'cell': cell_states}

    def get_feature_shape(self):
        return self.feature_dim


class GRUBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(256), gate=F.relu):
        super(GRUBody, self).__init__()
        if not isinstance(hidden_units, tuple): hidden_units = tuple(hidden_units)
        if isinstance(state_dim,int):   dims = (state_dim, ) + hidden_units
        else:   dims = state_dim + hidden_units
        self.layers = nn.ModuleList([layer_init_gru(nn.GRUCell(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.feature_dim = dims[-1]
        self.gate = gate

    def forward(self, inputs):
        '''
        :param inputs: input to GRU cells. Structured as (feed_forward_input, {hidden: hidden_states, cell: cell_states}).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, recurrent_neurons = inputs
        if next(self.layers[0].parameters()).is_cuda and not(x.is_cuda):    x = x.cuda() 
        hidden_states, cell_states = recurrent_neurons['hidden'], recurrent_neurons['cell']

        next_hstates, next_cstates = [], []
        for idx, (layer, hx, cx) in enumerate(zip(self.layers, hidden_states, cell_states) ):
            batch_size = x.size(0)
            if hx.size(0) == 1: # then we have just resetted the values, we need to expand those:
                hx = torch.cat([hx]*batch_size, dim=0)
                cx = torch.cat([cx]*batch_size, dim=0)
            elif hx.size(0) != batch_size:
                raise NotImplementedError("Sizes of the hidden states and the inputs do not coincide.")

            if next(layer.parameters()).is_cuda and \
                (hx is not None or not(hx.is_cuda)):
                if hx is not None:  hx = hx.cuda()

            nhx = layer(x, hx)
            next_hstates.append(nhx)
            next_cstates.append(nhx)
            # Consider not applying activation functions on last layer's output
            if self.gate is not None:
                x = self.gate(nhx)

        return x, {'hidden': next_hstates, 'cell': next_cstates}

    def get_reset_states(self, cuda=False, repeat=1):
        hidden_states, cell_states = [], []
        for layer in self.layers:
            h = torch.zeros(repeat, layer.hidden_size)
            if cuda:
                h = h.cuda()
            hidden_states.append(h)
            cell_states.append(h)
        return {'hidden': hidden_states, 'cell': cell_states}

    def get_feature_shape(self):
        return self.feature_dim


class EmbeddingRNNBody(nn.Module):
    def __init__(self, 
                 voc_size, 
                 embedding_size=64, 
                 hidden_units=256, 
                 num_layers=1, 
                 gate=F.relu, 
                 dropout=0.0, 
                 rnn_fn=nn.GRU):
        super(EmbeddingRNNBody, self).__init__()
        self.voc_size = voc_size
        self.embedding_size = embedding_size
        if isinstance(hidden_units, list):  hidden_units=hidden_units[-1]
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(self.voc_size, self.embedding_size, padding_idx=0)
        self.rnn = rnn_fn(input_size=self.embedding_size,
                                      hidden_size=hidden_units, 
                                      num_layers=self.num_layers,
                                      batch_first=True,
                                      dropout=dropout,
                                      bidirectional=False)
        self.gate = gate

    def forward(self, inputs):
        x, recurrent_neurons = inputs
        # Embedding:

        embeddings = self.embedding(x)
        # batch_size x sequence_length x embedding_size

        rnn_outputs, rnn_states = self.rnn(embeddings)
        # batch_size x sequence_length x hidden_units
        # num_layer*num_directions, batch_size, hidden_units
        
        output = self.gate(rnn_outputs[:,-1,...])
        # batch_size x hidden_units 

        return output, recurrent_neurons

    def get_feature_shape(self):
        return self.hidden_units


class CaptionRNNBody(nn.Module):
    def __init__(self,
                 vocabulary,
                 max_sentence_length,
                 embedding_size=64, 
                 hidden_units=256, 
                 num_layers=1, 
                 gate=F.relu, 
                 dropout=0.0, 
                 rnn_fn=nn.GRU):
        super(CaptionRNNBody, self).__init__()
        self.vocabulary = set([w.lower() for w in vocabulary])
        # Make padding_idx=0:
        self.vocabulary = ['PAD', 'SoS', 'EoS'] + list(self.vocabulary)
        
        self.w2idx = {}
        self.idx2w = {}
        for idx, w in enumerate(self.vocabulary):
            self.w2idx[w] = idx
            self.idx2w[idx] = w

        self.max_sentence_length = max_sentence_length
        self.voc_size = len(self.vocabulary)

        self.embedding_size = embedding_size
        if isinstance(hidden_units, list):  hidden_units=hidden_units[-1]
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        self.rnn_fn = rnn_fn
        self.rnn = rnn_fn(input_size=self.embedding_size,
                                      hidden_size=self.hidden_units, 
                                      num_layers=self.num_layers,
                                      batch_first=True,
                                      dropout=dropout,
                                      bidirectional=False)
        self.embedding = nn.Embedding(self.voc_size, self.embedding_size, padding_idx=0)
        self.token_decoder = nn.Linear(self.hidden_units, self.voc_size)
        
        self.gate = gate

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.loss = 0

    def forward(self, x, gt_sentences=None):
        '''
        If :param gt_sentences: is not `None`,
        then teacher forcing is implemented...
        '''
        if gt_sentences is not None:
            gt_sentences = gt_sentences.to(x.device)

        # batch_size x hidden_units
        batch_size = x.shape[0]
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(x.device) 
        h_0[0] = x.reshape(batch_size, -1)
        # (num_layers * num_directions, batch, hidden_size)
        
        if self.rnn_fn==nn.LSTM:
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(x.device) 
            decoder_hidden = (h_0,c_0)
        else:
            decoder_hidden = h_0 
        
        decoder_input = self.embedding(torch.LongTensor([self.w2idx["SoS"]]*batch_size).to(x.device)).unsqueeze(1)
        # batch_size x 1 x embedding_size

        loss_per_item = []

        predicted_sentences = self.w2idx['PAD']*torch.ones(batch_size, self.max_sentence_length, dtype=torch.long).to(x.device)
        for t in range(self.max_sentence_length):
            output, decoder_hidden = self._rnn(decoder_input, h_c=decoder_hidden)
            token_distribution = F.softmax(self.token_decoder(output), dim=-1) 
            idxs_next_token = torch.argmax(token_distribution, dim=1)
            # batch_size x 1
            predicted_sentences[:, t] = idxs_next_token
            
            # Compute loss:
            if gt_sentences is not None:
                mask = (gt_sentences[:, t]!=self.w2idx['PAD']).float().to(x.device)
                # batch_size x 1
                batched_loss = self.criterion(token_distribution, gt_sentences[:, t])*mask
                loss_per_item.append(batched_loss.unsqueeze(1))
                
            # Preparing next step:
            if gt_sentences is not None:
                idxs_next_token = gt_sentences[:, t]
            # batch_size x 1
            decoder_input = self.embedding(idxs_next_token).unsqueeze(1)
            # batch_size x 1 x embedding_size            
        
        for b in range(batch_size):
            end_idx = 0
            for idx_t in range(predicted_sentences.shape[1]):
                if predicted_sentences[b,idx_t] == self.w2idx['EoS']:
                    break
                end_idx += 1

        if gt_sentences is not None:
            loss_per_item = torch.cat(loss_per_item, dim=-1).mean(-1)
            # batch_size x max_sentence_length
            accuracies = (predicted_sentences==gt_sentences).float().mean(dim=0)
            mask = (gt_sentences!=self.w2idx['PAD'])
            sentence_accuracies = (predicted_sentences==gt_sentences).float().masked_select(mask).mean()
            return {'prediction':predicted_sentences, 
                    'loss_per_item':loss_per_item, 
                    'accuracies':accuracies, 
                    'sentence_accuracies':sentence_accuracies
                    }

        return predicted_sentences

    def _rnn(self, x, h_c):
        batch_size = x.shape[0]
        rnn_outputs, h_c = self.rnn(x, h_c)
        output = self.gate(rnn_outputs[:,-1,...])
        # batch_size x hidden_units 
        return output, h_c
        # batch_size x sequence_length=1 x hidden_units
        # num_layer*num_directions, batch_size, hidden_units
        
    def get_feature_shape(self):
        return self.hidden_units


class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi


class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi


class DummyBody(nn.Module):
    def __init__(self, state_shape):
        super(DummyBody, self).__init__()
        self.feature_shape = state_shape

    def get_feature_shape(self):
        return self.feature_shape

    def forward(self, x):
        return x
