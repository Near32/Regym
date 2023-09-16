import torch
import numpy as np
import copy

from ..networks import CategoricalActorCriticNet, CategoricalActorCriticVAENet, GaussianActorCriticNet
from ..networks import FCBody, LSTMBody, GRUBody, ConvolutionalBody, BetaVAEBody, resnet18Input64, ConvolutionalGruBody
from ..networks import PreprocessFunction, ResizeCNNPreprocessFunction, ResizeCNNInterpolationFunction
from ..algorithms.A2C import A2CAlgorithm

import torch.nn.functional as F
import numpy as np
from functools import partial


class A2CAgent(object):

    def __init__(self, name, algorithm):
        self.training = True
        self.algorithm = algorithm
        self.state_preprocessing = self.algorithm.kwargs['state_preprocess']
        self.handled_experiences = 0
        self.name = name
        self.save_path = None
        self.episode_count = 0

        self.nbr_actor = self.algorithm.nbr_actor
        self.previously_done_actors = [False]*self.nbr_actor

        self.use_rnd = self.algorithm.use_rnd

        self.recurrent = False
        self.rnn_states = None
        self.rnn_keys = [key for key, value in self.algorithm.kwargs.items() if isinstance(value, str) and 'RNN' in value]
        if len(self.rnn_keys):
            self.recurrent = True
            self._reset_rnn_states()

    def get_intrinsic_reward(self, actor_idx):
        if len(self.algorithm.storages[actor_idx].int_r):
            #return self.algorithm.storages[actor_idx].int_r[-1] / (self.algorithm.int_reward_std+1e-8)
            return self.algorithm.storages[actor_idx].int_r[-1] / (self.algorithm.int_return_std+1e-8)
        else:
            return 0.0

    def set_nbr_actor(self, nbr_actor):
        if nbr_actor != self.nbr_actor:
            self.nbr_actor = nbr_actor
            self.reset_actors(init=True)
            self.algorithm.reset_storages(nbr_actor=self.nbr_actor)

    def reset_actors(self, indices=None, init=False):
        '''
        In case of a multi-actor process, this function is called to reset
        the actors' internal values.
        '''
        if indices is None: indices = range(self.nbr_actor)
        
        if init:
            self.previously_done_actors = [False]*self.nbr_actor
        else:
            for idx in indices: self.previously_done_actors[idx] = False

        if self.recurrent:
            self._reset_rnn_states(indices=indices)

    def update_actors(self, batch_idx):
        '''
        In case of a multi-actor process, this function is called to handle
        the (dynamic) number of environment that are being used.
        More specifically, it regularizes the rnn_states when
        an actor's episode ends.
        It is assumed that update can only go by removing stuffs...
        Indeed, actors all start at the same time, and thus self.reset_actors()
        ought to be called at that time.
        Note: since the number of environment that are running changes, 
        the size of the rnn_states on the batch dimension will change too.
        Therefore, we cannot identify an rnn_state by the actor/environment index.
        Thus, it is a batch index that is requested, that would truly be in touch
        with the current batch dimension.
        :param batch_idx: index of the actor whose episode just finished.
        '''
        if self.recurrent:
            self.remove_from_rnn_states(batch_idx=batch_idx)

    def _reset_rnn_states(self, indices=None):
        # TODO: account for the indices in rnn states:
        if indices is None: indices = [i for i in range(self.nbr_actor)]

        self.rnn_states = {k: None for k in self.rnn_keys}
        for k in self.rnn_states:
            if 'phi' in k:
                self.rnn_states[k] = self.algorithm.model.network.phi_body.get_reset_states(cuda=self.algorithm.kwargs['use_cuda'], repeat=self.nbr_actor)
            if 'critic' in k:
                self.rnn_states[k] = self.algorithm.model.network.critic_body.get_reset_states(cuda=self.algorithm.kwargs['use_cuda'], repeat=self.nbr_actor)
            if 'actor' in k:
                self.rnn_states[k] = self.algorithm.model.network.actor_body.get_reset_states(cuda=self.algorithm.kwargs['use_cuda'], repeat=self.nbr_actor)

    def remove_from_rnn_states(self, batch_idx):
        '''
        Remove a row(=batch) of data from the rnn_states.
        :param batch_idx: index on the batch dimension that specifies which row to remove.
        '''
        for recurrent_submodule_name in self.rnn_states:
            for idx in range(len(self.rnn_states[recurrent_submodule_name]['hidden'])):
                self.rnn_states[recurrent_submodule_name]['hidden'][idx] = torch.cat(
                    [self.rnn_states[recurrent_submodule_name]['hidden'][idx][:batch_idx,...], 
                     self.rnn_states[recurrent_submodule_name]['hidden'][idx][batch_idx+1:,...]],
                     dim=0)
                self.rnn_states[recurrent_submodule_name]['cell'][idx] = torch.cat(
                    [self.rnn_states[recurrent_submodule_name]['cell'][idx][:batch_idx,...], 
                     self.rnn_states[recurrent_submodule_name]['cell'][idx][batch_idx+1:,...]],
                     dim=0)
        
    def _pre_process_rnn_states(self):
        if self.rnn_states is None: self._reset_rnn_states()

        if self.algorithm.kwargs['use_cuda']:
            for recurrent_submodule_name in self.rnn_states:
                for idx in range(len(self.rnn_states[recurrent_submodule_name]['hidden'])):
                    self.rnn_states[recurrent_submodule_name]['hidden'][idx] = self.rnn_states[recurrent_submodule_name]['hidden'][idx].cuda()
                    self.rnn_states[recurrent_submodule_name]['cell'][idx]   = self.rnn_states[recurrent_submodule_name]['cell'][idx].cuda()

    @staticmethod
    def _extract_from_rnn_states(rnn_states_batched: dict, batch_idx: int):
        rnn_states = {k: {'hidden':[], 'cell':[]} for k in rnn_states_batched}
        for recurrent_submodule_name in rnn_states_batched:
            for idx in range(len(rnn_states_batched[recurrent_submodule_name]['hidden'])):
                rnn_states[recurrent_submodule_name]['hidden'].append( rnn_states_batched[recurrent_submodule_name]['hidden'][idx][batch_idx,...].unsqueeze(0))
                rnn_states[recurrent_submodule_name]['cell'].append( rnn_states_batched[recurrent_submodule_name]['cell'][idx][batch_idx,...].unsqueeze(0))
        return rnn_states

    def _post_process(self, prediction):
        if self.recurrent:
            for recurrent_submodule_name in self.rnn_states:
                for idx in range(len(self.rnn_states[recurrent_submodule_name]['hidden'])):
                    self.rnn_states[recurrent_submodule_name]['hidden'][idx] = prediction['next_rnn_states'][recurrent_submodule_name]['hidden'][idx].cpu()
                    self.rnn_states[recurrent_submodule_name]['cell'][idx]   = prediction['next_rnn_states'][recurrent_submodule_name]['cell'][idx].cpu()

            for k, v in prediction.items():
                if isinstance(v, dict):
                    for vk in v:
                        hs, cs = v[vk]['hidden'], v[vk]['cell']
                        for idx in range(len(hs)):
                            hs[idx] = hs[idx].detach().cpu()
                            cs[idx] = cs[idx].detach().cpu()
                        prediction[k][vk] = {'hidden': hs, 'cell': cs}
                else:
                    prediction[k] = v.detach().cpu()
        else:
            prediction = {k: v.detach().cpu() for k, v in prediction.items()}

        return prediction

    @staticmethod
    def _extract_from_prediction(prediction: dict, batch_idx: int):
        out_pred = dict()
        for k, v in prediction.items():
            if isinstance(v, dict):
                continue
            out_pred[k] = v[batch_idx,...].unsqueeze(0)
        return out_pred

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

            exp_dict.update(A2CAgent._extract_from_prediction(self.current_prediction, batch_index))
            
            if self.use_rnd:
                int_reward, target_int_f = self.algorithm.compute_intrinsic_reward(exp_dict['succ_s'])
                rnd_dict = {'int_r':int_reward, 'target_int_f':target_int_f}
                exp_dict.update(rnd_dict)

            if self.recurrent:
                exp_dict['rnn_states'] = A2CAgent._extract_from_rnn_states(self.current_prediction['rnn_states'],batch_index)
                exp_dict['next_rnn_states'] = A2CAgent._extract_from_rnn_states(self.current_prediction['next_rnn_states'],batch_index)
            
            self.algorithm.storages[actor_index].add(exp_dict)
            self.previously_done_actors[actor_index] = done[actor_index]
            self.handled_experiences +=1

        if len(done_actors_among_notdone):
            # Regularization of the agents' actors:
            done_actors_among_notdone.sort(reverse=True)
            for batch_idx in done_actors_among_notdone:
                self.update_actors(batch_idx=batch_idx)
        
        if self.training and self.handled_experiences >= self.algorithm.kwargs['horizon']*self.nbr_actor:
            self.algorithm.train()
            self.handled_experiences = 0
            if self.save_path is not None: torch.save(self, self.save_path)

    def preprocess_environment_signals(self, state, reward, succ_state, done):
        non_terminal = torch.from_numpy(1 - np.array(done)).type(torch.FloatTensor)
        state = self.state_preprocessing(state, use_cuda=False)
        succ_state = self.state_preprocessing(succ_state, use_cuda=False)
        if isinstance(reward, np.ndarray): r = torch.from_numpy(reward).type(torch.FloatTensor)
        else: r = torch.ones(1).type(torch.FloatTensor)*reward
        return state, r, succ_state, non_terminal

    def take_action(self, state):
        state = self.state_preprocessing(state, use_cuda=self.algorithm.kwargs['use_cuda'])

        if self.recurrent:
            self._pre_process_rnn_states()
            self.current_prediction = self.algorithm.model(state, rnn_states=self.rnn_states)
        else:
            self.current_prediction = self.algorithm.model(state)
        self.current_prediction = self._post_process(self.current_prediction)

        return self.current_prediction['a'].numpy()

    def clone(self, training=None):
        clone = A2CAgent(name=self.name, algorithm=copy.deepcopy(self.algorithm))
        clone.training = training

        return clone


def build_A2C_Agent(task, config, agent_name):
    '''
    :param task: Environment specific configuration
    :param config: Dict containing configuration for a2c agent
    :param agent_name: name of the agent
    :returns: A2CAgent adapted to be trained on :param: task under :param: config
    '''
    kwargs = config.copy()
    kwargs['discount'] = float(kwargs['discount'])

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
        elif 'VAE' in kwargs['phi_arch']:
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
            phi_body = BetaVAEBody(input_shape=input_shape,
                                   feature_dim=output_dim,
                                   channels=channels,
                                   kernel_sizes=kernels,
                                   strides=strides,
                                   paddings=paddings,
                                   kwargs=kwargs)
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

    if kwargs['actor_arch'] != 'None':
        output_dim = 256
        if kwargs['actor_arch'] == 'RNN':
            actor_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif kwargs['actor_arch'] == 'MLP':
            hidden_units=(output_dim,)
            if 'actor_arch_hidden_units' in kwargs:
                hidden_units = tuple(kwargs['actor_arch_hidden_units'])
            actor_body = FCBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
        elif kwargs['actor_arch'] == 'CNN':
            # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
            #kwargs['state_preprocess'] = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
            kwargs['state_preprocess'] = partial(ResizeCNNInterpolationFunction, size=config['observation_resize_dim'], normalize_rgb_values=True)
            kwargs['preprocessed_observation_shape'] = [task.observation_shape[-1], kwargs['observation_resize_dim'], kwargs['observation_resize_dim']]
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
    else:
        actor_body = None

    if kwargs['critic_arch'] != 'None':
        output_dim = 256
        if kwargs['critic_arch'] == 'RNN':
            critic_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif kwargs['critic_arch'] == 'MLP':
            hidden_units=(output_dim,)
            if 'actor_arch_hidden_units' in kwargs:
                hidden_units = tuple(kwargs['actor_arch_hidden_units'])
            critic_body = FCBody(input_dim, hidden_units=hidden_units, gate=F.leaky_relu)
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

    use_rnd = False
    if 'use_random_network_distillation' in kwargs and kwargs['use_random_network_distillation']:
        use_rnd = True

    if task.action_type == 'Discrete':
        if task.observation_type == 'Discrete':
            model = CategoricalActorCriticNet(task.observation_shape, task.action_dim,
                                              phi_body=phi_body,
                                              actor_body=actor_body,
                                              critic_body=critic_body,
                                              use_intrinsic_critic=use_rnd)
        
        elif task.observation_type == 'Continuous':
            obs_shape = task.observation_shape
            if 'preprocessed_observation_shape' in kwargs: obs_shape = kwargs['preprocessed_observation_shape']
            if 'use_vae' in kwargs and kwargs['use_vae']:
                model = CategoricalActorCriticVAENet(obs_shape, 
                                                     task.action_dim,
                                                     phi_body=phi_body,
                                                     actor_body=actor_body,
                                                     critic_body=critic_body,
                                                     use_intrinsic_critic=use_rnd)
            else:
                model = CategoricalActorCriticNet(obs_shape, task.action_dim,
                                                  phi_body=phi_body,
                                                  actor_body=actor_body,
                                                  critic_body=critic_body,
                                                  use_intrinsic_critic=use_rnd)

    if task.action_type is 'Continuous' and task.observation_type is 'Continuous':
        model = GaussianActorCriticNet(task.observation_shape, task.action_dim,
                                       phi_body=phi_body,
                                       actor_body=actor_body,
                                       critic_body=critic_body,
                                       use_intrinsic_critic=use_rnd)

    target_intr_model = None
    predict_intr_model = None
    if use_rnd:
        if kwargs['rnd_arch'] == 'MLP':
            target_intr_model = FCBody(task.observation_shape, hidden_units=kwargs['rnd_feature_net_fc_arch_hidden_units'], gate=F.leaky_relu)
            predict_intr_model = FCBody(task.observation_shape, hidden_units=kwargs['rnd_feature_net_fc_arch_hidden_units'], gate=F.leaky_relu)
        elif 'CNN' in kwargs['rnd_arch']:
            input_shape = kwargs['preprocessed_observation_shape']
            channels = [input_shape[0]] + kwargs['rnd_arch_channels']
            kernels = kwargs['rnd_arch_kernels']
            strides = kwargs['rnd_arch_strides']
            paddings = kwargs['rnd_arch_paddings']
            output_dim = kwargs['rnd_arch_feature_dim']
            target_intr_model = ConvolutionalBody(input_shape=input_shape,
                                                  feature_dim=output_dim,
                                                  channels=channels,
                                                  kernel_sizes=kernels,
                                                  strides=strides,
                                                  paddings=paddings)
            output_dim = (256,256,)+(output_dim,)
            predict_intr_model = ConvolutionalBody(input_shape=input_shape,
                                                  feature_dim=output_dim,
                                                  channels=channels,
                                                  kernel_sizes=kernels,
                                                  strides=strides,
                                                  paddings=paddings)
        target_intr_model.share_memory()
        predict_intr_model.share_memory()

    model.share_memory()    
    a2c_algorithm = A2CAlgorithm(kwargs, model, target_intr_model=target_intr_model, predict_intr_model=predict_intr_model)

    return A2CAgent(name=agent_name, algorithm=a2c_algorithm)
