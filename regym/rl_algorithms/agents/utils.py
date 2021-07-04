from typing import Dict, List

from ..networks import CategoricalQNet, CategoricalActorCriticNet, CategoricalActorCriticVAENet, GaussianActorCriticNet
from ..networks import FCBody, FCBody2, LSTMBody, GRUBody, ConvolutionalBody, BetaVAEBody, resnet18Input64
from ..networks import ConvolutionalGruBody, ConvolutionalLstmBody
from ..networks import LinearLinearBody, LinearLstmBody, LinearLstmBody2
from ..networks import NoisyLinear

from ..networks import PreprocessFunction, ResizeCNNPreprocessFunction, ResizeCNNInterpolationFunction

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def parse_and_check(kwargs: Dict,
                    task: 'regym.environments.Task'):

    # Extra Inputs:
    kwargs['task'] = task

    extra_inputs = kwargs['extra_inputs_infos']
    for key in extra_inputs:
        shape = extra_inputs[key]['shape']
        for idxdim, dimvalue in enumerate(shape):
            if isinstance(dimvalue, str):
                path = dimvalue.split('.')
                if len(path) > 1:
                    pointer = kwargs
                    for el in path:
                        try:
                            if hasattr(pointer, el):
                                pointer = getattr(pointer, el)
                            elif el in pointer:
                                pointer = pointer[el]
                            else:
                                raise RuntimeError
                        except:
                            raise RuntimeError
                else:
                    pointer = path

                try:
                    pointer = int(pointer)
                except Exception as e:
                    print('Exception during parsing and checking:', e)
                    raise e
                shape[idxdim] = pointer

    kwargs['task'] = None
    
    return kwargs


def generate_model(
    task: 'regym.environments.Task', 
    kwargs: Dict,
    head_type: str="CategoricalQNet") -> nn.Module:
    
    phi_body = None
    if isinstance(task.observation_shape, int):
        input_dim = task.observation_shape
    else:
        input_dim = list(task.observation_shape)
    
    """
    # To deprecate: test if breaks without...
    if 'goal_oriented' in kwargs and kwargs['goal_oriented']:
        goal_input_shape = list(task.goal_shape)
        if 'goal_state_flattening' in kwargs and kwargs['goal_state_flattening']:
            if isinstance(input_dim, int):
                input_dim = input_dim+goal_input_shape
            else:
                input_dim[-1] = input_dim[-1]+goal_input_shape[-1]
    """

    if kwargs['phi_arch'] != 'None':
        output_dim = kwargs['phi_arch_feature_dim']
        if kwargs['phi_arch'] == 'LSTM-RNN':
            phi_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif kwargs['phi_arch'] == 'GRU-RNN':
            phi_body = GRUBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif kwargs['phi_arch'] == 'MLP':
            hidden_units=kwargs['phi_arch_hidden_units']
            hidden_units += [output_dim]
            
            extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
            extra_inputs_infos_phi_body = {}
            if extra_inputs_infos != {}:
                for key in extra_inputs_infos:
                    shape = extra_inputs_infos[key]['shape']
                    tll = extra_inputs_infos[key]['target_location']
                    if not isinstance(tll[0], list):
                        tll= [tll]
                    for tl in tll:
                        if 'phi_body' in tl:
                            extra_inputs_infos_phi_body[key] = {
                                'shape':shape, 
                                'target_location':tl
                            }
            if extra_inputs_infos_phi_body == {}:
                phi_body = FCBody(
                    input_dim, 
                    hidden_units=hidden_units,
                )
            else:
                phi_body = FCBody2(
                    input_dim, 
                    hidden_units=hidden_units,
                    extra_inputs_infos=extra_inputs_infos_phi_body
                )

        elif kwargs['phi_arch'] == 'CNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            if isinstance(kwargs['observation_resize_dim'], int):
                input_height, input_width = kwargs['observation_resize_dim'], kwargs['observation_resize_dim']
            else:
                input_height, input_width = kwargs['observation_resize_dim']

            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=input_height, normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], input_height, input_width]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            channels = [input_shape[0]] + kwargs['phi_arch_channels']
            kernels = kwargs['phi_arch_kernels']
            strides = kwargs['phi_arch_strides']
            paddings = kwargs['phi_arch_paddings']
            output_dim = kwargs['phi_arch_feature_dim']
            
            # Selecting Extra Inputs Infos relevant to phi_body:
            extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
            extra_inputs_infos_phi_body = {}
            if extra_inputs_infos != {}:
                for key in extra_inputs_infos:
                    shape = extra_inputs_infos[key]['shape']
                    tll = extra_inputs_infos[key]['target_location']
                    if not isinstance(tll[0], list):
                        tll= [tll]
                    for tl in tll:
                        if 'phi_body' in tl:
                            extra_inputs_infos_phi_body[key] = {
                                'shape':shape, 
                                'target_location':tl
                            }
            import ipdb; ipdb.set_trace()
            phi_body = ConvolutionalBody(
                input_shape=input_shape,
                feature_dim=output_dim,
                channels=channels,
                kernel_sizes=kernels,
                strides=strides,
                paddings=paddings,
                extra_inputs_infos=extra_inputs_infos_phi_body,
            )

        elif kwargs['phi_arch'] == 'ResNet18':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            output_dim = kwargs['phi_arch_feature_dim']
            phi_body = resnet18Input64(input_shape=input_shape, output_dim=output_dim)
        elif kwargs['phi_arch'] == 'CNN-GRU-RNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            channels = [input_shape[0]] + kwargs['phi_arch_channels']
            kernels = kwargs['phi_arch_kernels']
            strides = kwargs['phi_arch_strides']
            paddings = kwargs['phi_arch_paddings']
            output_dim = kwargs['phi_arch_hidden_units'][-1]
            phi_body = ConvolutionalGruBody(input_shape=input_shape,
                                         feature_dim=output_dim,
                                         channels=channels,
                                         kernel_sizes=kernels,
                                         strides=strides,
                                         paddings=paddings,
                                         hidden_units=kwargs['phi_arch_hidden_units'])
        elif kwargs['phi_arch'] == 'CNN-LSTM-RNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            channels = [input_shape[0]] + kwargs['phi_arch_channels']
            kernels = kwargs['phi_arch_kernels']
            strides = kwargs['phi_arch_strides']
            paddings = kwargs['phi_arch_paddings']
            output_dim = kwargs['phi_arch_feature_dim']  # TODO: figure out if this breaks anything else
            
            # Selecting Extra Inputs Infos relevant to phi_body:
            extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
            extra_inputs_infos_phi_body = {}
            if extra_inputs_infos != {}:
                for key in extra_inputs_infos:
                    shape = extra_inputs_infos[key]['shape']
                    tll = extra_inputs_infos[key]['target_location']
                    if not isinstance(tll[0], list):
                        tll= [tll]
                    for tl in tll:
                        if 'phi_body' in tl:
                            extra_inputs_infos_phi_body[key] = {
                                'shape':shape, 
                                'target_location':tl
                            }
            
            phi_body = ConvolutionalLstmBody(input_shape=input_shape,
                                         feature_dim=output_dim,
                                         channels=channels,
                                         kernel_sizes=kernels,
                                         strides=strides,
                                         paddings=paddings,
                                         extra_inputs_infos=extra_inputs_infos_phi_body,
                                         hidden_units=kwargs['phi_arch_hidden_units'])
        input_dim = output_dim


    goal_phi_body = None
    if 'goal_oriented' in kwargs and kwargs['goal_oriented']:
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
            if kwargs['goal_phi_arch'] == 'LSTM-RNN':
                phi_body = LSTMBody(goal_input_shape, hidden_units=(output_dim,), gate=F.leaky_relu)
            elif kwargs['goal_phi_arch'] == 'GRU-RNN':
                phi_body = GRUBody(goal_input_shape, hidden_units=(output_dim,), gate=F.leaky_relu)
            elif kwargs['goal_phi_arch'] == 'MLP':
                phi_body = FCBody(goal_input_shape, hidden_units=(output_dim, ), gate=F.leaky_relu)
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
                phi_body = ConvolutionalBody(input_shape=input_shape,
                                             feature_dim=output_dim,
                                             channels=channels,
                                             kernel_sizes=kernels,
                                             strides=strides,
                                             paddings=paddings)
            elif kwargs['goal_phi_arch'] == 'ResNet18':
                # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
                kwargs['goal_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['goal_resize_dim'], normalize_rgb_values=True)
                kwargs['preprocessed_goal_shape'] = [task.goal_shape[-1], kwargs['goal_resize_dim'], kwargs['goal_resize_dim']]
                if 'nbr_frame_stacking' in kwargs:
                    kwargs['preprocessed_goal_shape'][0] *=  kwargs['nbr_frame_stacking']
                input_shape = kwargs['preprocessed_goal_shape']
                output_dim = kwargs['goal_phi_arch_feature_dim']
                phi_body = resnet18Input64(input_shape=input_shape, output_dim=output_dim)
            elif kwargs['goal_phi_arch'] == 'CNN-GRU-RNN':
                # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
                kwargs['goal_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['goal_resize_dim'], normalize_rgb_values=True)
                kwargs['preprocessed_goal_shape'] = [task.goal_shape[-1], kwargs['goal_resize_dim'], kwargs['goal_resize_dim']]
                if 'nbr_frame_stacking' in kwargs:
                    kwargs['preprocessed_goal_shape'][0] *=  kwargs['nbr_frame_stacking']
                input_shape = kwargs['preprocessed_goal_shape']
                channels = [input_shape[0]] + kwargs['goal_phi_arch_channels']
                kernels = kwargs['goal_phi_arch_kernels']
                strides = kwargs['goal_phi_arch_strides']
                paddings = kwargs['goal_phi_arch_paddings']
                output_dim = kwargs['goal_phi_arch_hidden_units'][-1]
                goal_phi_body = ConvolutionalGruBody(input_shape=input_shape,
                                             feature_dim=output_dim,
                                             channels=channels,
                                             kernel_sizes=kernels,
                                             strides=strides,
                                             paddings=paddings,
                                             hidden_units=kwargs['phi_arch_hidden_units'])
            input_dim += output_dim


    actor_body = None 
    critic_body = None
    layer_fn = nn.Linear
    if 'noisy' in kwargs and kwargs['noisy']:  layer_fn = NoisyLinear
    
    if "actor_arch" in kwargs \
    and kwargs['actor_arch'] != 'None':
        output_dim = 256
        if kwargs['actor_arch'] == 'LSTM-RNN' or kwargs['actor_arch'] == 'GRU-RNN':
            #critic_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
            state_dim = input_dim
            actor_arch_hidden_units = kwargs['actor_arch_hidden_units']

            # Selecting Extra Inputs Infos relevant to phi_body:
            extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
            extra_inputs_infos_actor_body = {}
            if extra_inputs_infos != {}:
                for key in extra_inputs_infos:
                    shape = extra_inputs_infos[key]['shape']
                    tll = extra_inputs_infos[key]['target_location']
                    if not isinstance(tll[0], list):
                        tll= [tll]
                    for tl in tll:
                        if 'actor_body' in tl:
                            extra_inputs_infos_actor_body[key] = {
                                'shape':shape, 
                                'target_location':tl
                            }

            gate = None 
            if 'use_relu_after_rnn' in kwargs \
            and kwargs['use_relu_after_rnn']:
                import ipdb; ipdb.set_trace()
                gate = F.relu

            if kwargs['actor_arch'] == 'LSTM-RNN':
                actor_body = LSTMBody(
                    state_dim=state_dim,
                    hidden_units=actor_arch_hidden_units, 
                    gate=gate,
                    extra_inputs_infos=extra_inputs_infos_actor_body,
                )
            else:
                actor_body = GRUBody(
                    state_dim=state_dim,
                    hidden_units=actor_arch_hidden_units, 
                    gate=gate,
                    extra_inputs_infos=extra_inputs_infos_actor_body,
                )
        elif kwargs['actor_arch'] == 'MLP':
            hidden_units=(output_dim,)
            if 'actor_arch_hidden_units' in kwargs:
                hidden_units = list(kwargs['actor_arch_hidden_units'])
            actor_body = FCBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
        elif kwargs['actor_arch'] == 'CNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            channels = [input_shape[0]] + kwargs['actor_arch_channels']
            kernels = kwargs['actor_arch_kernels']
            strides = kwargs['actor_arch_strides']
            paddings = kwargs['actor_arch_paddings']
            output_dim = kwargs['actor_arch_feature_dim']
            actor_body = ConvolutionalBody(input_shape=input_shape,
                                         feature_dim=output_dim,
                                         channels=channels,
                                         kernel_sizes=kernels,
                                         strides=strides,
                                         paddings=paddings)
        elif kwargs['actor_arch'] == 'MLP-LSTM-RNN':
            # Assuming flatten input:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            state_dim = input_dim
            actor_arch_feature_dim = kwargs['actor_arch_feature_dim']
            actor_arch_hidden_units = kwargs['actor_arch_hidden_units']

            # Selecting Extra Inputs Infos relevant to phi_body:
            extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
            extra_inputs_infos_actor_body = {}
            if extra_inputs_infos != {}:
                for key in extra_inputs_infos:
                    shape = extra_inputs_infos[key]['shape']
                    tll = extra_inputs_infos[key]['target_location']
                    if not isinstance(tll[0], list):
                        tll= [tll]
                    for tl in tll:
                        if 'actor_body' in tl:
                            extra_inputs_infos_actor_body[key] = {
                                'shape':shape, 
                                'target_location':tl
                            }
            
            gate = None 
            if 'use_relu_after_rnn' in kwargs \
            and kwargs['use_relu_after_rnn']:
                import ipdb; ipdb.set_trace()
                gate = F.relu

            actor_body = LinearLstmBody(
                state_dim=state_dim,
                feature_dim=actor_arch_feature_dim, 
                hidden_units=actor_arch_hidden_units, 
                non_linearities=[nn.ReLU], 
                gate=gate,
                dropout=0.0,
                add_non_lin_final_layer=True,
                #layer_init_fn=None,
                extra_inputs_infos=extra_inputs_infos_actor_body,
            )

        elif kwargs['actor_arch'] == 'MLP-MLP-RNN':
            # Assuming flatten input:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            state_dim = input_dim
            actor_arch_feature_dim = kwargs['actor_arch_feature_dim']
            actor_arch_hidden_units = kwargs['actor_arch_hidden_units']

            # Selecting Extra Inputs Infos relevant to phi_body:
            extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
            extra_inputs_infos_actor_body = {}
            if extra_inputs_infos != {}:
                for key in extra_inputs_infos:
                    shape = extra_inputs_infos[key]['shape']
                    tll = extra_inputs_infos[key]['target_location']
                    if not isinstance(tll[0], list):
                        tll= [tll]
                    for tl in tll:
                        if 'actor_body' in tl:
                            extra_inputs_infos_actor_body[key] = {
                                'shape':shape, 
                                'target_location':tl
                            }
                
            actor_body = LinearLinearBody(
                state_dim=state_dim,
                feature_dim=actor_arch_feature_dim, 
                hidden_units=actor_arch_hidden_units, 
                non_linearities=[nn.ReLU], 
                gate=F.relu,
                dropout=0.0,
                add_non_lin_final_layer=True,
                #layer_init_fn=None,
                extra_inputs_infos=extra_inputs_infos_actor_body,
            )
        elif kwargs['actor_arch'] == 'MLP-LSTM-RNN2':
            # Assuming flatten input:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            state_dim = input_dim
            actor_arch_feature_dim = kwargs['actor_arch_feature_dim']
            actor_arch_linear_hidden_units = kwargs['actor_arch_linear_hidden_units']
            actor_arch_linear_post_hidden_units = None
            if 'actor_arch_linear_post_hidden_units' in kwargs:
                actor_arch_linear_post_hidden_units = kwargs['actor_arch_linear_post_hidden_units']
            actor_arch_hidden_units = kwargs['actor_arch_hidden_units']

            # Selecting Extra Inputs Infos relevant to phi_body:
            extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
            extra_inputs_infos_actor_body = {}
            if extra_inputs_infos != {}:
                for key in extra_inputs_infos:
                    shape = extra_inputs_infos[key]['shape']
                    tll = extra_inputs_infos[key]['target_location']
                    if not isinstance(tll[0], list):
                        tll= [tll]
                    for tl in tll:
                        if 'actor_body' in tl:
                            extra_inputs_infos_actor_body[key] = {
                                'shape':shape, 
                                'target_location':tl
                            }
            
            gate = None 
            if 'use_relu_after_rnn' in kwargs \
            and kwargs['use_relu_after_rnn']:
                import ipdb; ipdb.set_trace()
                gate = F.relu

            use_residual_connection = False
            if 'use_residual_connection' in kwargs \
            and kwargs['use_residual_connection']:
                import ipdb; ipdb.set_trace()
                use_residual_connection = kwargs['use_residual_connection']
                
            actor_body = LinearLstmBody2(
                state_dim=state_dim,
                feature_dim=actor_arch_feature_dim, 
                linear_hidden_units=actor_arch_linear_hidden_units,
                linear_post_hidden_units=actor_arch_linear_post_hidden_units,
                hidden_units=actor_arch_hidden_units, 
                non_linearities=[nn.ReLU], 
                gate=gate,
                dropout=0.0,
                add_non_lin_final_layer=True,
                use_residual_connection=use_residual_connection,
                #layer_init_fn=None,
                extra_inputs_infos=extra_inputs_infos_actor_body,
            )

    # CRITIC:

    if kwargs['critic_arch'] != 'None':
        output_dim = 256
        if kwargs['critic_arch'] == 'LSTM-RNN':
            #critic_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
            state_dim = input_dim
            critic_arch_hidden_units = kwargs['critic_arch_hidden_units']

            # Selecting Extra Inputs Infos relevant to phi_body:
            extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
            extra_inputs_infos_critic_body = {}
            if extra_inputs_infos != {}:
                for key in extra_inputs_infos:
                    shape = extra_inputs_infos[key]['shape']
                    tll = extra_inputs_infos[key]['target_location']
                    if not isinstance(tll[0], list):
                        tll= [tll]
                    for tl in tll:
                        if 'critic_body' in tl:
                            extra_inputs_infos_critic_body[key] = {
                                'shape':shape, 
                                'target_location':tl
                            }

            gate = None 
            if 'use_relu_after_rnn' in kwargs \
            and kwargs['use_relu_after_rnn']:
                import ipdb; ipdb.set_trace()
                gate = F.relu

            critic_body = LSTMBody(
                state_dim=state_dim,
                hidden_units=critic_arch_hidden_units, 
                gate=gate,
                extra_inputs_infos=extra_inputs_infos_critic_body,
            )
        elif kwargs['critic_arch'] == 'GRU-RNN':
            state_dim = input_dim
            critic_arch_hidden_units = kwargs['critic_arch_hidden_units']

            # Selecting Extra Inputs Infos relevant to phi_body:
            extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
            extra_inputs_infos_critic_body = {}
            if extra_inputs_infos != {}:
                for key in extra_inputs_infos:
                    shape = extra_inputs_infos[key]['shape']
                    tll = extra_inputs_infos[key]['target_location']
                    if not isinstance(tll[0], list):
                        tll= [tll]
                    for tl in tll:
                        if 'critic_body' in tl:
                            extra_inputs_infos_critic_body[key] = {
                                'shape':shape, 
                                'target_location':tl
                            }
            
            gate = None 
            if 'use_relu_after_rnn' in kwargs \
            and kwargs['use_relu_after_rnn']:
                import ipdb; ipdb.set_trace()
                gate = F.relu

            critic_body = GRUBody(
                state_dim=state_dim,
                hidden_units=critic_arch_hidden_units, 
                gate=gate,
                extra_inputs_infos=extra_inputs_infos_critic_body,
            )
        elif kwargs['critic_arch'] == 'MLP':
            hidden_units=(output_dim,)
            if 'critic_arch_hidden_units' in kwargs:
                hidden_units = list(kwargs['critic_arch_hidden_units'])
            critic_body = FCBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
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
        elif kwargs['critic_arch'] == 'MLP-LSTM-RNN':
            # Assuming flatten input:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            state_dim = input_dim
            critic_arch_feature_dim = kwargs['critic_arch_feature_dim']
            critic_arch_hidden_units = kwargs['critic_arch_hidden_units']

            # Selecting Extra Inputs Infos relevant to phi_body:
            extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
            extra_inputs_infos_critic_body = {}
            if extra_inputs_infos != {}:
                for key in extra_inputs_infos:
                    shape = extra_inputs_infos[key]['shape']
                    tll = extra_inputs_infos[key]['target_location']
                    if not isinstance(tll[0], list):
                        tll= [tll]
                    for tl in tll:
                        if 'critic_body' in tl:
                            extra_inputs_infos_critic_body[key] = {
                                'shape':shape, 
                                'target_location':tl
                            }
            
            gate = None 
            if 'use_relu_after_rnn' in kwargs \
            and kwargs['use_relu_after_rnn']:
                import ipdb; ipdb.set_trace()
                gate = F.relu

            critic_body = LinearLstmBody(
                state_dim=state_dim,
                feature_dim=critic_arch_feature_dim, 
                hidden_units=critic_arch_hidden_units, 
                non_linearities=[nn.ReLU], 
                gate=gate,
                dropout=0.0,
                add_non_lin_final_layer=True,
                #layer_init_fn=None,
                extra_inputs_infos=extra_inputs_infos_critic_body,
            )

        elif kwargs['critic_arch'] == 'MLP-MLP-RNN':
            # Assuming flatten input:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            state_dim = input_dim
            critic_arch_feature_dim = kwargs['critic_arch_feature_dim']
            critic_arch_hidden_units = kwargs['critic_arch_hidden_units']

            # Selecting Extra Inputs Infos relevant to phi_body:
            extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
            extra_inputs_infos_critic_body = {}
            if extra_inputs_infos != {}:
                for key in extra_inputs_infos:
                    shape = extra_inputs_infos[key]['shape']
                    tll = extra_inputs_infos[key]['target_location']
                    if not isinstance(tll[0], list):
                        tll= [tll]
                    for tl in tll:
                        if 'critic_body' in tl:
                            extra_inputs_infos_critic_body[key] = {
                                'shape':shape, 
                                'target_location':tl
                            }
                
            critic_body = LinearLinearBody(
                state_dim=state_dim,
                feature_dim=critic_arch_feature_dim, 
                hidden_units=critic_arch_hidden_units, 
                non_linearities=[nn.ReLU], 
                gate=F.relu,
                dropout=0.0,
                add_non_lin_final_layer=True,
                #layer_init_fn=None,
                extra_inputs_infos=extra_inputs_infos_critic_body,
            )
        elif kwargs['critic_arch'] == 'MLP-LSTM-RNN2':
            # Assuming flatten input:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            state_dim = input_dim
            critic_arch_feature_dim = kwargs['critic_arch_feature_dim']
            critic_arch_linear_hidden_units = kwargs['critic_arch_linear_hidden_units']
            critic_arch_linear_post_hidden_units = None
            if 'critic_arch_linear_post_hidden_units' in kwargs:
                critic_arch_linear_post_hidden_units = kwargs['critic_arch_linear_post_hidden_units']
            critic_arch_hidden_units = kwargs['critic_arch_hidden_units']

            # Selecting Extra Inputs Infos relevant to phi_body:
            extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
            extra_inputs_infos_critic_body = {}
            if extra_inputs_infos != {}:
                for key in extra_inputs_infos:
                    shape = extra_inputs_infos[key]['shape']
                    tll = extra_inputs_infos[key]['target_location']
                    if not isinstance(tll[0], list):
                        tll= [tll]
                    for tl in tll:
                        if 'critic_body' in tl:
                            extra_inputs_infos_critic_body[key] = {
                                'shape':shape, 
                                'target_location':tl
                            }
            
            gate = None 
            if 'use_relu_after_rnn' in kwargs \
            and kwargs['use_relu_after_rnn']:
                import ipdb; ipdb.set_trace()
                gate = F.relu

            use_residual_connection = False
            if 'use_residual_connection' in kwargs \
            and kwargs['use_residual_connection']:
                import ipdb; ipdb.set_trace()
                use_residual_connection = kwargs['use_residual_connection']
            
            critic_body = LinearLstmBody2(
                state_dim=state_dim,
                feature_dim=critic_arch_feature_dim, 
                linear_hidden_units=critic_arch_linear_hidden_units,
                linear_post_hidden_units=critic_arch_linear_post_hidden_units,
                hidden_units=critic_arch_hidden_units, 
                non_linearities=[nn.ReLU], 
                gate=gate,
                dropout=0.0,
                add_non_lin_final_layer=True,
                use_residual_connection=use_residual_connection,
                #layer_init_fn=None,
                extra_inputs_infos=extra_inputs_infos_critic_body,
            )


    use_rnd = False
    if 'use_random_network_distillation' in kwargs and kwargs['use_random_network_distillation']:
        use_rnd = True


    if isinstance(task.observation_shape, int):
        obs_shape = task.observation_shape
    else:
        obs_shape = list(task.observation_shape)
    if 'preprocessed_observation_shape' in kwargs: obs_shape = kwargs['preprocessed_observation_shape']
    if isinstance(task.observation_shape, int):
        goal_shape = task.goal_shape
    else:
        goal_shape = list(task.goal_shape)

    if 'preprocessed_goal_shape' in kwargs: goal_shape = kwargs['preprocessed_goal_shape']
    """
    # depr: goal update
    if 'goal_state_flattening' in kwargs and kwargs['goal_state_flattening']:
        obs_shape[-1] = obs_shape[-1] + goal_shape[-1]
    """
    
    # Selecting Extra Inputs Infos relevant to final_critic_layer:
    extra_inputs_infos = kwargs.get('extra_inputs_infos', {})
    extra_inputs_infos_final_critic_layer = {}
    extra_inputs_infos_final_actor_layer = {}
    if extra_inputs_infos != {}:
        for key in extra_inputs_infos:
            shape = extra_inputs_infos[key]['shape']
            tll = extra_inputs_infos[key]['target_location']
            if not isinstance(tll[0], list):
                tll= [tll]
            for tl in tll:
                if 'final_critic_layer' in tl:
                    extra_inputs_infos_final_critic_layer[key] = {
                        'shape':shape, 
                        'target_location':tl
                    }
                if 'final_actor_layer' in tl:
                    extra_inputs_infos_final_actor_layer[key] = {
                        'shape':shape, 
                        'target_location':tl
                    }
    
    if head_type=='CategoricalQNet':
        model = CategoricalQNet(
            state_dim=obs_shape,
            action_dim=task.action_dim,
            phi_body=phi_body,
            critic_body=critic_body,
            dueling=kwargs['dueling'],
            noisy=kwargs['noisy'] if 'noisy' in kwargs else False,
            goal_oriented=kwargs['goal_oriented'] if 'goal_oriented' in kwargs else False,
            goal_shape=goal_shape,
            goal_phi_body=goal_phi_body,
            extra_inputs_infos=extra_inputs_infos_final_critic_layer
        )
    elif head_type=="CategoricalActorCriticNet":
        model = CategoricalActorCriticNet(
            obs_shape, 
            task.action_dim,
            phi_body=phi_body,
            actor_body=actor_body,
            critic_body=critic_body,
            extra_inputs_infos={
                'critic':extra_inputs_infos_final_critic_layer,
                'actor':extra_inputs_infos_final_actor_layer,
            },
            use_intrinsic_critic=use_rnd
        )
    elif head_type=="GaussianActorCriticNet":
        raise NotImplementedError
        # TODO: implement infos scheme ...
        model = GaussianActorCriticNet(
            task.observation_shape, 
            task.action_dim,
            phi_body=phi_body,
            actor_body=actor_body,
            critic_body=critic_body,
            extra_inputs_infos={
                'critic':extra_inputs_infos_final_critic_layer,
                'actor':extra_inputs_infos_final_actor_layer,
            },
            use_intrinsic_critic=use_rnd
        )
    else:
        raise NotImplementedError

    model.share_memory()
    return model