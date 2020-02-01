import torch
import numpy as np
import copy
import random

#from ..replay_buffers import EXP
#from ..networks import LeakyReLU, DQN, DuelingDQN
#from ..networks import PreprocessFunction
#from ..DQN import DeepQNetworkAlgorithm, DoubleDeepQNetworkAlgorithm

from ..algorithms.DQN import DQNAlgorithm
from ..networks import CategoricalQNet
from ..networks import FCBody, LSTMBody, GRUBody, ConvolutionalBody, BetaVAEBody, resnet18Input64, ConvolutionalGruBody
from ..networks import NoisyLinear
from ..networks import PreprocessFunction, ResizeCNNPreprocessFunction, ResizeCNNInterpolationFunction

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from .agent import Agent


class DQNAgent(Agent):
    def __init__(self, name, algorithm):
        super(DQNAgent, self).__init__(name=name, algorithm=algorithm)
        self.kwargs = algorithm.kwargs
        self.epsend = float(self.kwargs['epsend'])
        self.epsstart = float(self.kwargs['epsstart'])
        self.epsdecay = float(self.kwargs['epsdecay'])
        self.replay_period = int(self.kwargs['replay_period']) if 'replay_period' in self.kwargs else 1
        self.replay_period_count = 0
        self.noisy = self.kwargs['noisy'] if 'noisy' in self.kwargs else False 

        self.nbr_steps = 0
        self.saving_interval = 1e4

    def get_experience_count(self):
        return self.handled_experiences

    def get_update_count(self):
        return self.algorithm.get_update_count()

    def handle_experience(self, s, a, r, succ_s, done):
        '''
        Note: the batch size may differ from the nbr_actor as soon as some
        actors' episodes end before the others...

        :param s: numpy tensor of states of shape batch x state_shape.
        :param a: numpy tensor of actions of shape batch x action_shape.
        :param r: numpy tensor of rewards of shape batch x reward_shape.
        :param succ_s: numpy tensor of successive states of shape batch x state_shape.
        :param done: list of boolean (batch=nbr_actor) x state_shape.
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
            # Watch out for the miss-match: done is a list of nbr_actor booleans,
            # which is not sync with batch_index, purposefully...
            exp_dict['non_terminal'] = non_terminal[actor_index,...].unsqueeze(0)

            exp_dict.update(Agent._extract_from_prediction(self.current_prediction, batch_index))
            
            if self.recurrent:
                exp_dict['rnn_states'] = Agent._extract_from_rnn_states(self.current_prediction['rnn_states'],batch_index)
                exp_dict['next_rnn_states'] = Agent._extract_from_rnn_states(self.current_prediction['next_rnn_states'],batch_index)
            
            #self.algorithm.storages[actor_index].add(exp_dict)
            self.algorithm.store(exp_dict, actor_index=actor_index)
            self.previously_done_actors[actor_index] = done[actor_index]
            self.handled_experiences +=1

        if len(done_actors_among_notdone):
            # Regularization of the agents' actors:
            done_actors_among_notdone.sort(reverse=True)
            for batch_idx in done_actors_among_notdone:
                self.update_actors(batch_idx=batch_idx)
        

        if self.training and self.handled_experiences > self.kwargs['min_capacity'] and self.replay_period_count % self.replay_period == 0:
            self.replay_period_count = 0 
            self.algorithm.train(minibatch_size=self.replay_period*self.kwargs['batch_size'])
            if self.save_path is not None and self.handled_experiences % self.saving_interval == 0: 
                self.save()
        else:
            self.replay_period_count += 1

    def take_action(self, state):
        self.nbr_steps += state.shape[0]
        self.eps = self.epsend + (self.epsstart-self.epsend) * np.exp(-1.0 * self.nbr_steps / self.epsdecay)
        
        state = self.state_preprocessing(state, use_cuda=self.algorithm.kwargs['use_cuda'])
        
        if self.recurrent:
            self._pre_process_rnn_states()
            self.current_prediction = self.algorithm.model(state, rnn_states=self.rnn_states)
        else:
            self.current_prediction = self.algorithm.model(state)
        self.current_prediction = self._post_process(self.current_prediction)

        sample = np.random.random()
        if self.noisy or sample > self.eps:
            return self.current_prediction['a'].numpy()
        else:
            random_actions = [random.randrange(self.algorithm.model.action_dim) for _ in range(state.shape[0])]
            random_actions = np.reshape(np.array(random_actions), (state.shape[0],1))
            return random_actions

    def clone(self, training=None):
        storages = self.algorithm.storages
        self.algorithm.storages = None
        
        clone = DQNAgent(name=self.name, algorithm=copy.deepcopy(self.algorithm))
        
        self.algorithm.storages = storages

        clone.handled_experiences = self.handled_experiences
        clone.episode_count = self.episode_count
        if training is not None:    clone.training = training
        clone.nbr_steps = self.nbr_steps
        return clone

    def save(self):
        storages = self.algorithm.storages
        self.algorithm.storages = None
        torch.save(self, self.save_path)
        self.algorithm.storages = storages

def build_DQN_Agent(task, config, agent_name):
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
    kwargs['state_preprocess'] = PreprocessFunction
    
    input_dim = task.observation_shape
    if kwargs['phi_arch'] != 'None':
        output_dim = 256
        if kwargs['phi_arch'] == 'LSTM-RNN':
            phi_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif kwargs['phi_arch'] == 'GRU-RNN':
            phi_body = GRUBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif kwargs['phi_arch'] == 'MLP':
            phi_body = FCBody(input_dim, hidden_units=(output_dim, output_dim), gate=F.leaky_relu)
        elif kwargs['phi_arch'] == 'CNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=config['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [task.observation_shape[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
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
        elif kwargs['phi_arch'] == 'ResNet18':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=config['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [task.observation_shape[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
            if 'nbr_frame_stacking' in kwargs:
                kwargs['preprocessed_observation_shape'][0] *=  kwargs['nbr_frame_stacking']
            input_shape = kwargs['preprocessed_observation_shape']
            output_dim = kwargs['phi_arch_feature_dim']
            phi_body = resnet18Input64(input_shape=input_shape, output_dim=output_dim)
        elif kwargs['phi_arch'] == 'CNN-GRU-RNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=config['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [task.observation_shape[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
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
        input_dim = output_dim
    else:
        phi_body = None


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
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=config['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [task.observation_shape[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
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
    else:
        critic_body = None

    assert(task.action_type == 'Discrete')
    obs_shape = task.observation_shape
    if 'preprocessed_observation_shape' in kwargs: obs_shape = kwargs['preprocessed_observation_shape']    
    model = CategoricalQNet(state_dim=obs_shape, 
                            action_dim=task.action_dim,
                            phi_body=phi_body,
                            critic_body=critic_body,
                            dueling=kwargs['dueling'],
                            noisy=kwargs['noisy'])

    model.share_memory()
    dqn_algorithm = DQNAlgorithm(kwargs, model)

    return DQNAgent(name=agent_name, algorithm=dqn_algorithm)
