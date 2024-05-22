import torch
import numpy as np
import copy
import random

from ..algorithms.DQN import DQNAlgorithm
from ..algorithms.THER import ther_predictor_loss
from ..algorithms.wrappers import THERAlgorithmWrapper, predictor_based_goal_predicated_reward_fn

from ..networks import CategoricalQNet, InstructionPredictor
from ..networks import FCBody, LSTMBody, GRUBody, EmbeddingRNNBody, CaptionRNNBody 
from ..networks import ConvolutionalBody, BetaVAEBody, resnet18Input64, ConvolutionalGruBody, ConvolutionalLstmBody
from ..networks import NoisyLinear
from ..networks import PreprocessFunction, ResizeCNNPreprocessFunction, ResizeCNNInterpolationFunction

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from .dqn_agent import DQNAgent
from .wrappers import DictHandlingAgentWrapper
from gym.spaces import Dict

from ..algorithms import dqn_ther_loss
from ..algorithms import ddqn_ther_loss

class THERAgent(DQNAgent):
    def clone(self, training=None):
        cloned_algo = self.algorithm.clone()
        clone = THERAgent(name=self.name, algorithm=cloned_algo)

        clone.handled_experiences = self.handled_experiences
        clone.episode_count = self.episode_count
        if training is not None:    clone.training = training
        clone.nbr_steps = self.nbr_steps
        return clone


def build_THER_Agent(task, config, agent_name):
    '''
    :param task: Environment specific configuration
    :param config: Dict containing configuration for ppo agent
    :param agent_name: name of the agent
    :returns: THERAgent adapted to be trained on :param: task under :param: config
    '''

    '''
    Note: Input values are not normalized as RGB values, ever!
    Indeed, they are not RGB values... cf gym_miniworld doc...
    '''

    kwargs = config.copy()
    kwargs['THER_predictor_learning_rate'] = float(kwargs['THER_predictor_learning_rate'])

    kwargs['discount'] = float(kwargs['discount'])
    kwargs['replay_capacity'] = int(float(kwargs['replay_capacity']))
    kwargs['min_capacity'] = int(float(kwargs['min_capacity']))
    
    kwargs['THER_vocabulary'] = set(kwargs['THER_vocabulary'])
    kwargs['THER_max_sentence_length'] = int(kwargs['THER_max_sentence_length'])

    # Default preprocess function:
    kwargs['state_preprocess'] = partial(PreprocessFunction, normalization=False)
    kwargs['goal_preprocess'] = partial(PreprocessFunction, normalization=False)

    if 'None' in kwargs['observation_resize_dim']:  kwargs['observation_resize_dim'] = task.observation_shape[0] if isinstance(task.observation_shape, tuple) else task.observation_shape
    if 'None' in kwargs['goal_resize_dim']:  kwargs['goal_resize_dim'] = task.goal_shape[0] if isinstance(task.goal_shape, tuple) else task.goal_shape
    

    phi_body = None
    input_dim = list(task.observation_shape)
    if kwargs['goal_oriented']:
        goal_input_shape = list(task.goal_shape)
        if 'goal_state_flattening' in kwargs and kwargs['goal_state_flattening']:
            if isinstance(input_dim, int):
                input_dim = input_dim+goal_input_shape
            else:
                input_dim[-1] = input_dim[-1]+goal_input_shape[-1]

    if kwargs['phi_arch'] != 'None':
        output_dim = kwargs['phi_arch_feature_dim']
        if kwargs['phi_arch'] == 'LSTM-RNN':
            phi_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif kwargs['phi_arch'] == 'GRU-RNN':
            phi_body = GRUBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif kwargs['phi_arch'] == 'MLP':
            phi_body = FCBody(input_dim, hidden_units=(output_dim, ), gate=F.leaky_relu)
        elif kwargs['phi_arch'] == 'CNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=False)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            channels = [input_shape[0]] + kwargs['phi_arch_channels']
            kernels = kwargs['phi_arch_kernels']
            strides = kwargs['phi_arch_strides']
            paddings = kwargs['phi_arch_paddings']
            output_dim = kwargs['phi_arch_feature_dim']
            phi_body = ConvolutionalBody(input_shape=input_shape,
                                         feature_dim=output_dim,
                                         channels=channels,
                                         kernel_sizes=kernels,
                                         strides=strides,
                                         paddings=paddings)
        else:
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=False)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            channels = [input_shape[0]] + kwargs['phi_arch_channels']
            kernels = kwargs['phi_arch_kernels']
            strides = kwargs['phi_arch_strides']
            paddings = kwargs['phi_arch_paddings']
            output_dim = kwargs['phi_arch_hidden_units'][-1]
            if kwargs['phi_arch'] == 'CNN-GRU-RNN':
                phi_body = ConvolutionalGruBody(
                    input_shape=input_shape,
                    feature_dim=output_dim,
                    channels=channels,
                    kernel_sizes=kernels,
                    strides=strides,
                    paddings=paddings,
                    hidden_units=kwargs['phi_arch_hidden_units']
                )
            elif kwargs['phi_arch'] == 'CNN-LSTM-RNN':
                phi_body = ConvolutionalLstmBody(
                    input_shape=input_shape,
                    feature_dim=output_dim,
                    channels=channels,
                    kernel_sizes=kernels,
                    strides=strides,
                    paddings=paddings,
                    hidden_units=kwargs['phi_arch_hidden_units']
                )
            else :
                raise NotImplementedError
        
        input_dim = output_dim


    goal_phi_body = None
    if kwargs['goal_oriented']:
        goal_input_shape = task.goal_shape
        if 'goal_state_flattening' in kwargs and kwargs['goal_state_flattening']:
            kwargs['goal_preprocess'] = kwargs['state_preprocess']

        if 'goal_state_shared_arch' in kwargs and kwargs['goal_state_shared_arch']:
            kwargs['goal_preprocess'] = kwargs['state_preprocess']
            if 'preprocessed_observation_shape' in kwargs:
                kwargs['preprocessed_goal_shape'] = kwargs['preprocessed_observation_shape']
                goal_input_shape = kwargs['preprocessed_goal_shape']
            goal_phi_body = None 

        elif kwargs['goal_phi_arch'] != 'None':
            output_dim = 256
            if kwargs['goal_phi_arch'] == 'EmbedLSTM':
                num_layers = len(kwargs['goal_phi_arch_hidden_units'])
                voc_size = task.goal_shape[0]
                goal_phi_body = EmbeddingRNNBody(voc_size=voc_size, 
                                            embedding_size=kwargs['goal_phi_arch_embedding_size'], 
                                            hidden_units=kwargs['goal_phi_arch_hidden_units'], 
                                            num_layers=num_layers,
                                            gate=F.relu, 
                                            dropout=0.0, 
                                            rnn_fn=nn.LSTM)
                output_dim = kwargs['goal_phi_arch_hidden_units'][-1]
            elif kwargs['goal_phi_arch'] == 'EmbedGRU':
                num_layers = len(kwargs['goal_phi_arch_hidden_units'])
                voc_size = task.goal_shape[0]
                goal_phi_body = EmbeddingRNNBody(voc_size=voc_size, 
                                            embedding_size=kwargs['goal_phi_arch_embedding_size'], 
                                            hidden_units=kwargs['goal_phi_arch_hidden_units'], 
                                            num_layers=num_layers,
                                            gate=F.relu, 
                                            dropout=0.0, 
                                            rnn_fn=nn.GRU)
                output_dim = kwargs['goal_phi_arch_hidden_units'][-1]
            elif kwargs['goal_phi_arch'] == 'MLP':
                goal_phi_body = FCBody(goal_input_shape, hidden_units=kwargs['goal_phi_arch_hidden_units'], gate=F.leaky_relu)
            elif kwargs['goal_phi_arch'] == 'CNN':
                # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
                kwargs['goal_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['goal_resize_dim'], normalize_rgb_values=True)
                kwargs['preprocessed_goal_shape'] = [task.goal_shape[-1], kwargs['goal_resize_dim'], kwargs['goal_resize_dim']]
                if 'nbr_frame_stacking' in kwargs:
                    kwargs['preprocessed_goal_shape'][0] *=  kwargs['nbr_frame_stacking']
                input_shape = kwargs['preprocessed_goal_shape']
                channels = [goal_shape[0]] + kwargs['goal_phi_arch_channels']
                kernels = kwargs['goal_phi_arch_kernels']
                strides = kwargs['goal_phi_arch_strides']
                paddings = kwargs['goal_phi_arch_paddings']
                output_dim = kwargs['goal_phi_arch_feature_dim']
                goal_phi_body = ConvolutionalBody(input_shape=input_shape,
                                             feature_dim=output_dim,
                                             channels=channels,
                                             kernel_sizes=kernels,
                                             strides=strides,
                                             paddings=paddings)
            input_dim += output_dim


    critic_body = None
    layer_fn = nn.Linear 
    if kwargs['noisy']:  layer_fn = NoisyLinear
    if kwargs['critic_arch'] != 'None':
        output_dim = 256
        if kwargs['critic_arch'] == 'RNN':
            critic_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif kwargs['critic_arch'] == 'MLP':
            hidden_units=(output_dim,)
            if 'critic_arch_hidden_units' in kwargs:
                hidden_units = tuple(kwargs['critic_arch_hidden_units'])
            critic_body = FCBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu, layer_fn=layer_fn)
        elif kwargs['critic_arch'] == 'CNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            channels = [input_shape[0]] + kwargs['critic_arch_channels']
            kernels = kwargs['critic_arch_kernels']
            strides = kwargs['critic_arch_strides']
            paddings = kwargs['critic_arch_paddings']
            output_dim = kwargs['critic_arch_feature_dim']
            critic_body = ConvolutionalBody(input_shape=input_shape,
                                         feature_dim=output_dim,
                                         channels=channels,
                                         kernel_sizes=kernels,
                                         strides=strides,
                                         paddings=paddings)

    
    assert(task.action_type == 'Discrete')

    obs_shape = list(task.observation_shape)
    if 'preprocessed_observation_shape' in kwargs: obs_shape = kwargs['preprocessed_observation_shape']    
    goal_shape = list(task.goal_shape)
    if 'preprocessed_goal_shape' in kwargs: goal_shape = kwargs['preprocessed_goal_shape']
    if 'goal_state_flattening' in kwargs and kwargs['goal_state_flattening']:
        obs_shape[-1] = obs_shape[-1] + goal_shape[-1] 
    model = CategoricalQNet(state_dim=obs_shape, 
                            action_dim=task.action_dim,
                            phi_body=phi_body,
                            critic_body=critic_body,
                            dueling=kwargs['dueling'],
                            noisy=kwargs['noisy'],
                            goal_oriented=kwargs['goal_oriented'],
                            goal_shape=goal_shape,
                            goal_phi_body=goal_phi_body)

    model.share_memory()


    predictor_input_dim = task.observation_shape
    if 'preprocessed_observation_shape' in kwargs: predictor_input_dim = list(reversed(kwargs['preprocessed_observation_shape']))
    
    if kwargs['predictor_encoder_arch'] == 'LSTM-RNN':
        predictor_encoder = LSTMBody(predictor_input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
    elif kwargs['predictor_encoder_arch'] == 'GRU-RNN':
        predictor_encoder = GRUBody(predictor_input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
    elif kwargs['predictor_encoder_arch'] == 'MLP':
        predictor_encoder = FCBody(predictor_input_dim, hidden_units=(output_dim, ), gate=F.leaky_relu)
    elif kwargs['predictor_encoder_arch'] == 'CNN':
        # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
        #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
        kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=False)
        kwargs['preprocessed_observation_shape'] = [predictor_input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
        if 'nbr_frame_stacking' in kwargs:
            kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
        input_shape = kwargs['preprocessed_observation_shape']
        
        if kwargs['THER_predictor_policy_shared_phi']:
            predictor_encoder = phi_body.cnn_body
            output_dim = predictor_encoder.get_feature_shape()
            assert( output_dim == kwargs['predictor_decoder_arch_hidden_units'][-1])
        else:
            channels = [input_shape[0]] + kwargs['predictor_encoder_arch_channels']
            kernels = kwargs['predictor_encoder_arch_kernels']
            strides = kwargs['predictor_encoder_arch_strides']
            paddings = kwargs['predictor_encoder_arch_paddings']
            output_dim = kwargs['predictor_encoder_arch_feature_dim']
            predictor_encoder = ConvolutionalBody(input_shape=input_shape,
                                         feature_dim=output_dim,
                                         channels=channels,
                                         kernel_sizes=kernels,
                                         strides=strides,
                                         paddings=paddings)

    predictor_decoder = CaptionRNNBody(
        vocabulary=kwargs['THER_vocabulary'],
        max_sentence_length=kwargs['THER_max_sentence_length'],
        embedding_size=kwargs['predictor_decoder_embedding_size'], 
        hidden_units=kwargs['predictor_decoder_arch_hidden_units'], 
        num_layers=1, 
        gate=F.relu, 
        dropout=0.0, 
        rnn_fn=nn.GRU
    )
    predictor_decoder.share_memory()

    predictor = InstructionPredictor(
        encoder=predictor_encoder, 
        decoder=predictor_decoder
    )
    predictor.share_memory()

    loss_fn = dqn_ther_loss.compute_loss
    if kwargs['double'] or kwargs['dueling']:
        loss_fn = ddqn_ther_loss.compute_loss

    dqn_algorithm = DQNAlgorithm(
        kwargs=kwargs, 
        model=model, 
        loss_fn=loss_fn
    )

    assert('use_HER' in kwargs and kwargs['use_HER'])

    goal_predicated_reward_fn = None 
    if 'HER_use_latent' in kwargs and kwargs['HER_use_latent']:
        from ..algorithms.wrappers import latent_based_goal_predicated_reward_fn
        goal_predicated_reward_fn = latent_based_goal_predicated_reward_fn

    if 'THER_use_predictor' in kwargs and kwargs['THER_use_predictor']:
        goal_predicated_reward_fn = partial(predictor_based_goal_predicated_reward_fn, predictor=predictor)
    
    ther_algorithm = THERAlgorithmWrapper(algorithm=dqn_algorithm,
                                        predictor=predictor,
                                        predictor_loss_fn=ther_predictor_loss.compute_loss,
                                        strategy=kwargs['HER_strategy'],
                                        goal_predicated_reward_fn=goal_predicated_reward_fn)

    agent = THERAgent(name=agent_name, algorithm=ther_algorithm)

    if isinstance(getattr(task.env, 'observation_space', None), Dict) or ('use_HER' in kwargs and kwargs['use_HER']):
        agent = DictHandlingAgentWrapper(agent=agent, use_achieved_goal=kwargs['use_HER'])

    print(ther_algorithm.get_models())

    return agent