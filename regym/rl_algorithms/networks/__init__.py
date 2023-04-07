from .utils import hard_update, soft_update, random_sample, layer_init
from .utils import PreprocessFunction, CNNPreprocessFunction, ResizeCNNPreprocessFunction, ResizeCNNInterpolationFunction
from .bodies import FCBody, FCBody2, LSTMBody, GRUBody, EmbeddingRNNBody, CaptionRNNBody 
from .bodies import ConvolutionalBody, BetaVAEBody, resnet18Input64, ConvolutionalLstmBody, ConvolutionalGruBody
from .bodies import LinearLinearBody, LinearLstmBody, LinearLstmBody2
from .bodies import LinearLstmAttentionBody2
from .bodies import NoisyLinear

from .neural_turing_machine import NTMBody
from .differentiable_neural_computer import DNCBody

from .heads import DuelingLayer, EPS
from .heads import SquashedGaussianActorNet, GaussianActorNet, CategoricalActorCriticNet, CategoricalActorCriticVAENet, GaussianActorCriticNet
from .heads import QNet, CategoricalQNet, InstructionPredictor
from .heads import EnsembleQNet

from .archi_predictor import ArchiPredictor
from .archi_predictor_speaker import ArchiPredictorSpeaker


import torch.nn.functional as F 

def choose_architecture(architecture, 
                        hidden_units_list=None,
                        input_shape=None,
                        feature_dim=None, 
                        nbr_channels_list=None, 
                        kernels=None, 
                        strides=None, 
                        paddings=None):
    if 'LSTM-RNN' in architecture:
        return LSTMBody(input_shape[0], hidden_units=hidden_units_list, gate=F.leaky_relu)
    if 'GRU-RNN' in architecture:
        return GRUBody(input_shape[0], hidden_units=hidden_units_list, gate=F.leaky_relu)
    if architecture == 'MLP':
        return FCBody(input_shape[0], hidden_units=hidden_units_list, gate=F.leaky_relu)
    
    if architecture == 'CNN':
        channels = [input_shape[0]] + nbr_channels_list
        phi_body = ConvolutionalBody(input_shape=input_shape,
                                     feature_dim=feature_dim,
                                     channels=channels,
                                     kernel_sizes=kernels,
                                     strides=strides,
                                     paddings=paddings)
    if architecture == 'ResNet18':
        phi_body = resnet18Input64()

    if architecture == 'CNN-RNN':
        channels = [input_shape[0]] + nbr_channels_list
        phi_body = ConvolutionalLstmBody(input_shape=input_shape,
                                     feature_dim=feature_dim,
                                     channels=channels,
                                     kernel_sizes=kernels,
                                     strides=strides,
                                     paddings=paddings,
                                     hidden_units=hidden_units_list)
    return phi_body
