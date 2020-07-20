import torch
import numpy as np
import copy
import random

from ..algorithms.DDPG import DDPGAlgorithm, ddpg_critic_loss, ddpg_actor_loss
from ..networks import QNet, GaussianActorNet
from ..networks import FCBody, LSTMBody, GRUBody, ConvolutionalBody, BetaVAEBody, resnet18Input64, ConvolutionalGruBody
from ..networks import NoisyLinear
from ..networks import PreprocessFunction, ResizeCNNPreprocessFunction, ResizeCNNInterpolationFunction

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from .agent import Agent
from .wrappers import DictHandlingAgentWrapper
from gym.spaces import Dict
from ..algorithms.wrappers import HERAlgorithmWrapper


class DDPGAgent(Agent):
    def __init__(self, name, algorithm):
        super(DDPGAgent, self).__init__(name=name, algorithm=algorithm)
        self.kwargs = algorithm.kwargs
        
        self.replay_period = int(self.kwargs['replay_period']) if 'replay_period' in self.kwargs else 1
        self.replay_period_count = 0
        
        self.nbr_episode_per_cycle = int(self.kwargs['nbr_episode_per_cycle']) if 'nbr_episode_per_cycle' in self.kwargs else None
        self.nbr_episode_per_cycle_count = 0
        
        self.nbr_training_iteration_per_cycle = int(self.kwargs['nbr_training_iteration_per_cycle']) if 'nbr_training_iteration_per_cycle' in self.kwargs else 1

        self.noisy = self.kwargs['noisy'] if 'noisy' in self.kwargs else False 

        # Number of training steps:
        self.nbr_steps = 0
        
        self.saving_interval = 1e4


    def get_experience_count(self):
        return self.handled_experiences

    def get_update_count(self):
        return self.algorithm.get_update_count()

    def handle_experience(self, s, a, r, succ_s, done, goals=None, infos=None):
        '''
        Note: the batch size may differ from the nbr_actor as soon as some
        actors' episodes end before the others...

        :param s: numpy tensor of states of shape batch x state_shape.
        :param a: numpy tensor of actions of shape batch x action_shape.
        :param r: numpy tensor of rewards of shape batch x reward_shape.
        :param succ_s: numpy tensor of successive states of shape batch x state_shape.
        :param done: list of boolean (batch=nbr_actor) x state_shape.
        :param goals: Dictionnary of goals 'achieved_goal' and 'desired_goal' for each state 's' and 'succ_s'.
        :param infos: Dictionnary of information from the environment.
        '''
        state, r, succ_state, non_terminal = self.preprocess_environment_signals(s, r, succ_s, done)
        a = torch.from_numpy(a)
        # batch x ...

        # We assume that this function has been called directly after take_action:
        # therefore the current prediction correspond to this experience.

        batch_index = -1
        done_actors_among_notdone = []
        for actor_index in range(self.nbr_actor):
            # If this actor is already done with its episode:  
            if self.previously_done_actors[actor_index]:
                continue
            # Otherwise, there is bookkeeping to do:
            batch_index +=1
            
            # Bookkeeping of the actors whose episode just ended:
            if done[actor_index] and not(self.previously_done_actors[actor_index]):
                done_actors_among_notdone.append(batch_index)
                
            exp_dict = {}
            exp_dict['s'] = state[batch_index,...].unsqueeze(0)
            exp_dict['a'] = a[batch_index,...].unsqueeze(0)
            exp_dict['r'] = r[batch_index,...].unsqueeze(0)
            exp_dict['succ_s'] = succ_state[batch_index,...].unsqueeze(0)
            # Watch out for the miss-match: 
            # done is a list of nbr_actor booleans,
            # which is not sync with batch_index, purposefully...
            exp_dict['non_terminal'] = non_terminal[actor_index,...].unsqueeze(0)
            # Watch out for the miss-match: 
            # Similarly, infos is not sync with batch_index, purposefully...
            if infos is not None:
                exp_dict['info'] = infos[actor_index]

            exp_dict.update(Agent._extract_from_prediction(self.current_prediction, batch_index))
            

            if self.recurrent:
                exp_dict['rnn_states'] = Agent._extract_from_rnn_states(self.current_prediction['rnn_states'],batch_index)
                exp_dict['next_rnn_states'] = Agent._extract_from_rnn_states(self.current_prediction['next_rnn_states'],batch_index)
            
            if self.goal_oriented:
                exp_dict['goals'] = Agent._extract_from_hdict(goals, batch_index, goal_preprocessing_fn=self.goal_preprocessing)

            #self.algorithm.storages[actor_index].add(exp_dict)
            self.algorithm.store(exp_dict, actor_index=actor_index)
            self.previously_done_actors[actor_index] = done[actor_index]
            self.handled_experiences +=1

        if len(done_actors_among_notdone):
            # Regularization of the agents' actors:
            done_actors_among_notdone.sort(reverse=True)
            for batch_idx in done_actors_among_notdone:
                self.update_actors(batch_idx=batch_idx)
        

        self.replay_period_count += 1
        period_check = self.replay_period
        period_count_check = self.replay_period_count
        if self.nbr_episode_per_cycle is not None:
            if len(done_actors_among_notdone):
                self.nbr_episode_per_cycle_count += len(done_actors_among_notdone)
            period_check = self.nbr_episode_per_cycle
            period_count_check = self.nbr_episode_per_cycle_count

        if self.training and self.handled_experiences > self.kwargs['min_capacity'] and period_count_check % period_check == 0:
            minibatch_size = self.kwargs['batch_size']
            if self.nbr_episode_per_cycle is None:
                minibatch_size *= self.replay_period
            else:
                self.nbr_episode_per_cycle_count = 1
            for train_it in range(self.nbr_training_iteration_per_cycle):
                self.algorithm.train(minibatch_size=minibatch_size)
            if self.save_path is not None and self.handled_experiences % self.saving_interval == 0: 
                self.save()
        
    def take_action(self, state):
        if self.training:
            self.nbr_steps += state.shape[0]
        
        state = self.state_preprocessing(state, use_cuda=self.algorithm.kwargs['use_cuda'])
        goal = None
        if self.goal_oriented:
            goal = self.goal_preprocessing(self.goals, use_cuda=self.algorithm.kwargs['use_cuda'])

        model_actor = self.algorithm.get_models()['model_actor']
        if 'use_target_to_gather_data' in self.kwargs and self.kwargs['use_target_to_gather_data']:  
            model_actor = self.algorithm.get_models()['target_model_actor'] 

        if self.recurrent:
            self._pre_process_rnn_states()
            self.current_prediction = model_actor(state, rnn_states=self.rnn_states, goal=goal)
        else:
            self.current_prediction = model_actor(state, goal=goal)
        self.current_prediction = self._post_process(self.current_prediction)

        if self.training or self.noisy:
            return self.current_prediction['a'].numpy()
        else:
            #self.algorithm.noise.setSigma(0.5)
            new_action = action.cpu().data.numpy() + self.algorithm.noise.sample()*self.algorithm.model_actor.action_scaler
            return new_action


    def clone(self, training=None):
        cloned_algo = self.algorithm.clone()
        clone = DDPGAgent(name=self.name, algorithm=cloned_algo)

        clone.handled_experiences = self.handled_experiences
        clone.episode_count = self.episode_count
        if training is not None:    clone.training = training
        clone.nbr_steps = self.nbr_steps
        return clone

        
def build_DDPG_Agent(task, config, agent_name):
    '''
    :param task: Environment specific configuration
    :param config: Dict containing configuration for ppo agent
    :param agent_name: name of the agent
    :returns: DeepQNetworkAgent adapted to be trained on :param: task under :param: config
    '''
    kwargs = config.copy()
    kwargs['discount'] = float(kwargs['discount'])
    kwargs['replay_capacity'] = int(float(kwargs['replay_capacity']))
    kwargs['min_capacity'] = int(float(kwargs['min_capacity']))
    
    # Default preprocess function:
    kwargs['state_preprocess'] = partial(PreprocessFunction, normalization=False)
    kwargs['goal_preprocess'] = partial(PreprocessFunction, normalization=False)

    if 'None' in kwargs['observation_resize_dim']:  kwargs['observation_resize_dim'] = task.observation_shape[0] if isinstance(task.observation_shape, tuple) else task.observation_shape
    if 'None' in kwargs['goal_resize_dim']:  kwargs['goal_resize_dim'] = task.goal_shape[0] if isinstance(task.goal_shape, tuple) else task.goal_shape
    
    ##-----------------------------------------------------------------------------------------------------
    ##-----------------------------------------------------------------------------------------------------
    # Actor Model:
    ##-----------------------------------------------------------------------------------------------------
    ##-----------------------------------------------------------------------------------------------------
    
    ##-----------------------------------------------------------------------------------------------------
    ## Phi Body:
    actor_phi_body = None
    input_dim = list(task.observation_shape)
    input_shape = input_dim
    actor_input_dim = None
    if kwargs['goal_oriented']:
        goal_input_shape = list(task.goal_shape)
        if 'goal_state_flattening' in kwargs and kwargs['goal_state_flattening']:
            if isinstance(input_dim, int):
                input_dim = input_dim+goal_input_shape
            else:
                input_dim[-1] = input_dim[-1]+goal_input_shape[-1]

    if kwargs['actor_phi_arch'] != 'None':
        hidden_units = kwargs['actor_phi_arch_hidden_units']
        output_dim = hidden_units[-1]
        if kwargs['actor_phi_arch'] == 'LSTM-RNN':
            actor_phi_body = LSTMBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
        elif kwargs['actor_phi_arch'] == 'GRU-RNN':
            actor_phi_body = GRUBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
        elif kwargs['actor_phi_arch'] == 'MLP':
            actor_phi_body = FCBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu, add_non_lin_final_layer=True)
        elif kwargs['actor_phi_arch'] == 'CNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            channels = [input_shape[0]] + kwargs['actor_phi_arch_channels']
            kernels = kwargs['actor_phi_arch_kernels']
            strides = kwargs['actor_phi_arch_strides']
            paddings = kwargs['actor_phi_arch_paddings']
            output_dim = kwargs['actor_phi_arch_feature_dim']
            actor_phi_body = ConvolutionalBody(
                input_shape=input_shape,
                feature_dim=output_dim,
                channels=channels,
                kernel_sizes=kernels,
                strides=strides,
                paddings=paddings
            )
        elif kwargs['actor_phi_arch'] == 'ResNet18':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            output_dim = kwargs['actor_phi_arch_feature_dim']
            actor_phi_body = resnet18Input64(input_shape=input_shape, output_dim=output_dim)
        elif kwargs['actor_phi_arch'] == 'CNN-GRU-RNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            channels = [input_shape[0]] + kwargs['actor_phi_arch_channels']
            kernels = kwargs['actor_phi_arch_kernels']
            strides = kwargs['actor_phi_arch_strides']
            paddings = kwargs['actor_phi_arch_paddings']
            output_dim = kwargs['actor_phi_arch_hidden_units'][-1]
            actor_phi_body = ConvolutionalGruBody(
                input_shape=input_shape,
                feature_dim=output_dim,
                channels=channels,
                kernel_sizes=kernels,
                strides=strides,
                paddings=paddings,
                hidden_units=kwargs['actor_phi_arch_hidden_units']
            )

        actor_input_dim = output_dim

    ##-----------------------------------------------------------------------------------------------------
    ## goal phi body:
    actor_goal_phi_body = None
    goal_shape = None
    if kwargs['goal_oriented']:
        goal_input_shape = task.goal_shape
        if 'goal_state_flattening' in kwargs and kwargs['goal_state_flattening']:
            kwargs['goal_preprocess'] = kwargs['state_preprocess']

        if 'goal_state_shared_arch' in kwargs and kwargs['goal_state_shared_arch']:
            kwargs['goal_preprocess'] = kwargs['state_preprocess']
            if 'preprocessed_observation_shape' in kwargs:
                kwargs['preprocessed_goal_shape'] = kwargs['preprocessed_observation_shape']
                goal_input_shape = kwargs['preprocessed_goal_shape']
            actor_goal_phi_body = None 

        elif kwargs['actor_goal_phi_arch'] != 'None':
            output_dim = kwargs['actor_goal_phi_arch_feature_dim']
            hidden_units = [*kwargs['actor_goal_phi_arch_hidden_units'], kwargs['actor_goal_phi_arch_feature_dim']]
            if kwargs['actor_goal_phi_arch'] == 'LSTM-RNN':
                actor_goal_phi_body = LSTMBody(goal_input_shape, hidden_units=hidden_units, gate=F.leaky_relu)
            elif kwargs['actor_goal_phi_arch'] == 'GRU-RNN':
                actor_goal_phi_body = GRUBody(goal_input_shape, hidden_units=hidden_units, gate=F.leaky_relu)
            elif kwargs['actor_goal_phi_arch'] == 'MLP':
                actor_goal_phi_body = FCBody(goal_input_shape, hidden_units=hidden_units, gate=F.leaky_relu, add_non_lin_final_layer=True)
            elif kwargs['actor_goal_phi_arch'] == 'CNN':
                # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
                kwargs['goal_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['goal_resize_dim'], normalize_rgb_values=True)
                kwargs['preprocessed_goal_shape'] = [task.goal_shape[-1], kwargs['goal_resize_dim'], kwargs['goal_resize_dim']]
                if 'nbr_frame_stacking' in kwargs:
                    kwargs['preprocessed_goal_shape'][0] *=  kwargs['nbr_frame_stacking']
                input_shape = kwargs['preprocessed_goal_shape']
                channels = [goal_shape[0]] + kwargs['actor_goal_phi_arch_channels']
                kernels = kwargs['actor_goal_phi_arch_kernels']
                strides = kwargs['actor_goal_phi_arch_strides']
                paddings = kwargs['actor_goal_phi_arch_paddings']
                output_dim = kwargs['actor_goal_phi_arch_feature_dim']
                actor_goal_phi_body = ConvolutionalBody(
                    input_shape=input_shape,
                    feature_dim=output_dim,
                    channels=channels,
                    kernel_sizes=kernels,
                    strides=strides,
                    paddings=paddings
                )

            elif kwargs['actor_goal_phi_arch'] == 'ResNet18':
                # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
                kwargs['goal_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['goal_resize_dim'], normalize_rgb_values=True)
                kwargs['preprocessed_goal_shape'] = [task.goal_shape[-1], kwargs['goal_resize_dim'], kwargs['goal_resize_dim']]
                if 'nbr_frame_stacking' in kwargs:
                    kwargs['preprocessed_goal_shape'][0] *=  kwargs['nbr_frame_stacking']
                input_shape = kwargs['preprocessed_goal_shape']
                output_dim = kwargs['actor_goal_phi_arch_feature_dim']
                actor_goal_phi_body = resnet18Input64(input_shape=input_shape, output_dim=output_dim)
            elif kwargs['actor_goal_phi_arch'] == 'CNN-GRU-RNN':
                # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
                kwargs['goal_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['goal_resize_dim'], normalize_rgb_values=True)
                kwargs['preprocessed_goal_shape'] = [task.goal_shape[-1], kwargs['goal_resize_dim'], kwargs['goal_resize_dim']]
                if 'nbr_frame_stacking' in kwargs:
                    kwargs['preprocessed_goal_shape'][0] *=  kwargs['nbr_frame_stacking']
                input_shape = kwargs['preprocessed_goal_shape']
                channels = [input_shape[0]] + kwargs['actor_goal_phi_arch_channels']
                kernels = kwargs['actor_goal_phi_arch_kernels']
                strides = kwargs['actor_goal_phi_arch_strides']
                paddings = kwargs['actor_goal_phi_arch_paddings']
                output_dim = kwargs['actor_goal_phi_arch_hidden_units'][-1]
                actor_goal_phi_body = ConvolutionalGruBody(
                    input_shape=input_shape,
                    feature_dim=output_dim,
                    channels=channels,
                    kernel_sizes=kernels,
                    strides=strides,
                    paddings=paddings,
                    hidden_units=kwargs['actor_goal_phi_arch_hidden_units']
                )

            actor_input_dim += output_dim

    ##-----------------------------------------------------------------------------------------------------
    ## Actor Head Body:
    if kwargs['actor_head_arch'] != 'None':
        hidden_units = kwargs["actor_head_arch_hidden_units"]
        output_dim = hidden_units[-1]
        if kwargs['actor_head_arch'] == 'LSTM-RNN':
            actor_head_body = LSTMBody(actor_input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
        elif kwargs['actor_head_arch'] == 'GRU-RNN':
            actor_head_body = GRUBody(actor_input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
        elif kwargs['actor_head_arch'] == 'MLP':
            actor_head_body = FCBody(actor_input_dim, hidden_units=hidden_units, gate=F.leaky_relu, add_non_lin_final_layer=True)
        
    ##-----------------------------------------------------------------------------------------------------
    ## GaussianActorNet:
    model_actor = GaussianActorNet(
        state_dim=input_shape, 
        action_dim=task.action_dim,
        actor_body=actor_head_body,
        phi_body=actor_phi_body,
        noisy=kwargs['noisy'],
        goal_oriented=kwargs['goal_oriented'],
        goal_shape=goal_shape,
        goal_phi_body=actor_goal_phi_body,
        deterministic=True,
    )
    model_actor.share_memory()

    ##-----------------------------------------------------------------------------------------------------
    ##-----------------------------------------------------------------------------------------------------
    ##-----------------------------------------------------------------------------------------------------

    ##-----------------------------------------------------------------------------------------------------
    ##-----------------------------------------------------------------------------------------------------
    # Critic Model:
    ##-----------------------------------------------------------------------------------------------------
    ##-----------------------------------------------------------------------------------------------------
    
    ##-----------------------------------------------------------------------------------------------------
    ## Phi Body:
    critic_phi_body = None
    input_dim = list(task.observation_shape)
    input_shape = input_dim
    critic_head_input_dim = None
    if kwargs['goal_oriented']:
        goal_input_shape = list(task.goal_shape)
        if 'goal_state_flattening' in kwargs and kwargs['goal_state_flattening']:
            if isinstance(input_dim, int):
                input_dim = input_dim+goal_input_shape
            else:
                input_dim[-1] = input_dim[-1]+goal_input_shape[-1]

    if kwargs['critic_phi_arch'] != 'None':
        hidden_units = kwargs["critic_phi_arch_hidden_units"]
        output_dim = hidden_units[-1]
        if kwargs['critic_phi_arch'] == 'LSTM-RNN':
            critic_phi_body = LSTMBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
        elif kwargs['critic_phi_arch'] == 'GRU-RNN':
            critic_phi_body = GRUBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
        elif kwargs['critic_phi_arch'] == 'MLP':
            critic_phi_body = FCBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu, add_non_lin_final_layer=True)
        elif kwargs['critic_phi_arch'] == 'CNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            channels = [input_shape[0]] + kwargs['critic_phi_arch_channels']
            kernels = kwargs['critic_phi_arch_kernels']
            strides = kwargs['critic_phi_arch_strides']
            paddings = kwargs['critic_phi_arch_paddings']
            output_dim = kwargs['critic_phi_arch_feature_dim']
            critic_phi_body = ConvolutionalBody(
                input_shape=input_shape,
                feature_dim=output_dim,
                channels=channels,
                kernel_sizes=kernels,
                strides=strides,
                paddings=paddings
            )
        elif kwargs['critic_phi_arch'] == 'ResNet18':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            output_dim = kwargs['critic_phi_arch_feature_dim']
            critic_phi_body = resnet18Input64(input_shape=input_shape, output_dim=output_dim)
        elif kwargs['critic_phi_arch'] == 'CNN-GRU-RNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [input_dim[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            channels = [input_shape[0]] + kwargs['critic_phi_arch_channels']
            kernels = kwargs['critic_phi_arch_kernels']
            strides = kwargs['critic_phi_arch_strides']
            paddings = kwargs['critic_phi_arch_paddings']
            output_dim = kwargs['critic_phi_arch_hidden_units'][-1]
            critic_phi_body = ConvolutionalGruBody(
                input_shape=input_shape,
                feature_dim=output_dim,
                channels=channels,
                kernel_sizes=kernels,
                strides=strides,
                paddings=paddings,
                hidden_units=kwargs['critic_phi_arch_hidden_units']
            )

        critic_head_input_dim = output_dim

    ##-----------------------------------------------------------------------------------------------------
    ## goal phi body:
    critic_goal_phi_body = None
    goal_shape = None
    if kwargs['goal_oriented']:
        goal_input_shape = task.goal_shape
        if 'goal_state_flattening' in kwargs and kwargs['goal_state_flattening']:
            kwargs['goal_preprocess'] = kwargs['state_preprocess']

        if 'goal_state_shared_arch' in kwargs and kwargs['goal_state_shared_arch']:
            kwargs['goal_preprocess'] = kwargs['state_preprocess']
            if 'preprocessed_observation_shape' in kwargs:
                kwargs['preprocessed_goal_shape'] = kwargs['preprocessed_observation_shape']
                goal_input_shape = kwargs['preprocessed_goal_shape']
            critic_goal_phi_body = None 

        elif kwargs['critic_goal_phi_arch'] != 'None':
            hidden_units = kwargs["critic_goal_phi_arch_hidden_units"]
            output_dim = hidden_units[-1]
            if kwargs['critic_goal_phi_arch'] == 'LSTM-RNN':
                critic_goal_phi_body = LSTMBody(goal_input_shape, hidden_units=hidden_units, gate=F.leaky_relu)
            elif kwargs['critic_goal_phi_arch'] == 'GRU-RNN':
                critic_goal_phi_body = GRUBody(goal_input_shape, hidden_units=hidden_units, gate=F.leaky_relu)
            elif kwargs['critic_goal_phi_arch'] == 'MLP':
                critic_goal_phi_body = FCBody(goal_input_shape, hidden_units=hidden_units, gate=F.leaky_relu, add_non_lin_final_layer=True)
            elif kwargs['critic_goal_phi_arch'] == 'CNN':
                # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
                kwargs['goal_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['goal_resize_dim'], normalize_rgb_values=True)
                kwargs['preprocessed_goal_shape'] = [task.goal_shape[-1], kwargs['goal_resize_dim'], kwargs['goal_resize_dim']]
                if 'nbr_frame_stacking' in kwargs:
                    kwargs['preprocessed_goal_shape'][0] *=  kwargs['nbr_frame_stacking']
                input_shape = kwargs['preprocessed_goal_shape']
                channels = [goal_shape[0]] + kwargs['critic_goal_phi_arch_channels']
                kernels = kwargs['critic_goal_phi_arch_kernels']
                strides = kwargs['critic_goal_phi_arch_strides']
                paddings = kwargs['critic_goal_phi_arch_paddings']
                output_dim = kwargs['critic_goal_phi_arch_feature_dim']
                critic_goal_phi_body = ConvolutionalBody(
                    input_shape=input_shape,
                    feature_dim=output_dim,
                    channels=channels,
                    kernel_sizes=kernels,
                    strides=strides,
                    paddings=paddings
                )

            elif kwargs['critic_goal_phi_arch'] == 'ResNet18':
                # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
                kwargs['goal_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['goal_resize_dim'], normalize_rgb_values=True)
                kwargs['preprocessed_goal_shape'] = [task.goal_shape[-1], kwargs['goal_resize_dim'], kwargs['goal_resize_dim']]
                if 'nbr_frame_stacking' in kwargs:
                    kwargs['preprocessed_goal_shape'][0] *=  kwargs['nbr_frame_stacking']
                input_shape = kwargs['preprocessed_goal_shape']
                output_dim = kwargs['critic_goal_phi_arch_feature_dim']
                critic_goal_phi_body = resnet18Input64(input_shape=input_shape, output_dim=output_dim)
            elif kwargs['critic_goal_phi_arch'] == 'CNN-GRU-RNN':
                # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
                kwargs['goal_preprocess'] = partial(ResizeCNNInterpolationFunction, size=kwargs['goal_resize_dim'], normalize_rgb_values=True)
                kwargs['preprocessed_goal_shape'] = [task.goal_shape[-1], kwargs['goal_resize_dim'], kwargs['goal_resize_dim']]
                if 'nbr_frame_stacking' in kwargs:
                    kwargs['preprocessed_goal_shape'][0] *=  kwargs['nbr_frame_stacking']
                input_shape = kwargs['preprocessed_goal_shape']
                channels = [input_shape[0]] + kwargs['critic_goal_phi_arch_channels']
                kernels = kwargs['critic_goal_phi_arch_kernels']
                strides = kwargs['critic_goal_phi_arch_strides']
                paddings = kwargs['critic_goal_phi_arch_paddings']
                output_dim = kwargs['critic_goal_phi_arch_hidden_units'][-1]
                actor_goal_phi_body = ConvolutionalGruBody(
                    input_shape=input_shape,
                    feature_dim=output_dim,
                    channels=channels,
                    kernel_sizes=kernels,
                    strides=strides,
                    paddings=paddings,
                    hidden_units=kwargs['critic_goal_phi_arch_hidden_units']
                )

            critic_head_input_dim += output_dim


    ##-----------------------------------------------------------------------------------------------------
    ## Action Phi Body:
    critic_action_phi_body = None
    input_dim = task.action_dim
    input_shape = input_dim
    if kwargs['critic_action_phi_arch'] != 'None':
        hidden_units = kwargs["critic_action_phi_arch_hidden_units"]
        output_dim = hidden_units[-1]
        if kwargs['critic_action_phi_arch'] == 'LSTM-RNN':
            critic_action_phi_body = LSTMBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
        elif kwargs['critic_action_phi_arch'] == 'GRU-RNN':
            critic_action_phi_body = GRUBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
        elif kwargs['critic_action_phi_arch'] == 'MLP':
            critic_action_phi_body = FCBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu, add_non_lin_final_layer=True)
    else:
        output_dim = input_dim

    critic_head_input_dim += output_dim
    ##-----------------------------------------------------------------------------------------------------
    ## Critic Head Body:
    if kwargs['critic_head_arch'] != 'None':
        hidden_units = kwargs["critic_head_arch_hidden_units"]
        output_dim = hidden_units[-1]
        if kwargs['critic_head_arch'] == 'LSTM-RNN':
            critic_head_body = LSTMBody(critic_head_input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
        elif kwargs['critic_head_arch'] == 'GRU-RNN':
            critic_head_body = GRUBody(critic_head_input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
        elif kwargs['critic_head_arch'] == 'MLP':
            critic_head_body = FCBody(critic_head_input_dim, hidden_units=hidden_units, gate=F.leaky_relu, add_non_lin_final_layer=True)
        
    ##-----------------------------------------------------------------------------------------------------
    ## QNet:
    model_critic = QNet(
        state_dim=input_shape, 
        action_dim=task.action_dim,
        critic_body=critic_head_body,
        action_phi_body=critic_action_phi_body,
        phi_body=critic_phi_body,
        noisy=kwargs['noisy'],
        goal_oriented=kwargs['goal_oriented'],
        goal_shape=goal_shape,
        goal_phi_body=critic_goal_phi_body
    )
    model_critic.share_memory()

    ##-----------------------------------------------------------------------------------------------------
    ##-----------------------------------------------------------------------------------------------------
    ##-----------------------------------------------------------------------------------------------------


    ddpg_algorithm = DDPGAlgorithm(
        kwargs, 
        model_actor=model_actor,
        model_critic=model_critic, 
        actor_loss_fn=ddpg_actor_loss.compute_loss,
        critic_loss_fn=ddpg_critic_loss.compute_loss,
    )

    if 'use_HER' in kwargs and kwargs['use_HER']:
        from ..algorithms.wrappers import latent_based_goal_predicated_reward_fn
        goal_predicated_reward_fn = None 
        if 'HER_use_latent' in kwargs and kwargs['HER_use_latent']:
            goal_predicated_reward_fn = latent_based_goal_predicated_reward_fn

        ddpg_algorithm = HERAlgorithmWrapper(algorithm=ddpg_algorithm, 
                                            strategy=kwargs['HER_strategy'],
                                            goal_predicated_reward_fn=goal_predicated_reward_fn)

    agent = DDPGAgent(name=agent_name, algorithm=ddpg_algorithm)

    if isinstance(getattr(task.env, 'observation_space', None), Dict) or ('use_HER' in kwargs and kwargs['use_HER']):
        agent = DictHandlingAgentWrapper(agent=agent, use_achieved_goal=kwargs['use_HER'])

    print(ddpg_algorithm.get_models())

    return agent