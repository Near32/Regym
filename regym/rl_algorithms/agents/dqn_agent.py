from typing import Dict, Any
import torch
import numpy as np
import copy
import random
from collections.abc import Iterable

from ..algorithms.DQN import DQNAlgorithm, dqn_loss, ddqn_loss
from ..networks import CategoricalQNet
from ..networks import FCBody, LSTMBody, GRUBody, ConvolutionalBody, BetaVAEBody, resnet18Input64
from ..networks import ConvolutionalGruBody, ConvolutionalLstmBody
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
from regym.rl_algorithms.utils import _extract_from_rnn_states, copy_hdict


class DQNAgent(Agent):
    def __init__(self, name, algorithm):
        super(DQNAgent, self).__init__(name=name, algorithm=algorithm)
        self.kwargs = algorithm.kwargs
        self.epsend = float(self.kwargs['epsend'])
        self.epsstart = float(self.kwargs['epsstart'])
        self.epsdecay = float(self.kwargs['epsdecay'])
        self.epsdecay_strategy = self.kwargs['epsdecay_strategy'] if 'epsdecay_strategy' in self.kwargs else 'exponential'
        self.eps = None

        self.replay_period = int(self.kwargs['replay_period']) if 'replay_period' in self.kwargs else 1
        self.replay_period_count = 0

        self.nbr_episode_per_cycle = int(self.kwargs['nbr_episode_per_cycle']) if 'nbr_episode_per_cycle' in self.kwargs else None
        self.nbr_episode_per_cycle_count = 0

        self.nbr_training_iteration_per_cycle = int(self.kwargs['nbr_training_iteration_per_cycle']) if 'nbr_training_iteration_per_cycle' in self.kwargs else 1

        self.noisy = self.kwargs['noisy'] if 'noisy' in self.kwargs else False

        # Number of training steps:
        self.nbr_steps = 0

        self.saving_interval = float(self.kwargs['saving_interval']) if 'saving_interval' in self.kwargs else 1e4
        
        self.previous_save_quotient = -1

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
                exp_dict['rnn_states'] = _extract_from_rnn_states(
                    self.current_prediction['rnn_states'],
                    batch_index,
                    post_process_fn=(lambda x: x.detach().cpu())
                )
                exp_dict['next_rnn_states'] = _extract_from_rnn_states(
                    self.current_prediction['next_rnn_states'],
                    batch_index,
                    post_process_fn=(lambda x: x.detach().cpu())
                )

            if self.goal_oriented:
                exp_dict['goals'] = Agent._extract_from_hdict(goals, batch_index, goal_preprocessing_fn=self.goal_preprocessing)

            self.algorithm.store(exp_dict, actor_index=actor_index)
            self.previously_done_actors[actor_index] = done[actor_index]
            #with self.handled_experiences.get_lock():
            self.handled_experiences.value +=1

        if len(done_actors_among_notdone):
            # Regularization of the agents' actors:
            done_actors_among_notdone.sort(reverse=True)
            for batch_idx in done_actors_among_notdone:
                self.update_actors(batch_idx=batch_idx)


        self.replay_period_count += 1
        if self.nbr_episode_per_cycle is not None:
            if len(done_actors_among_notdone):
                self.nbr_episode_per_cycle_count += len(done_actors_among_notdone)
        
        if not(self.async_actor):
            self.train()

    def train(self):
        nbr_updates = 0

        period_check = self.replay_period
        period_count_check = self.replay_period_count
        if self.nbr_episode_per_cycle is not None:
            period_check = self.nbr_episode_per_cycle
            period_count_check = self.nbr_episode_per_cycle_count

        if self.training \
        and self.handled_experiences.value > self.kwargs['min_capacity'] \
        and (period_count_check % period_check == 0 or not(self.async_actor)):
            minibatch_size = self.kwargs['batch_size']
            if self.nbr_episode_per_cycle is None:
                minibatch_size *= self.replay_period
            else:
                self.nbr_episode_per_cycle_count = 1
                
            for train_it in range(self.nbr_training_iteration_per_cycle):
                self.algorithm.train(minibatch_size=minibatch_size)
            nbr_updates = self.nbr_training_iteration_per_cycle
            # Update actor's models:
            if self.async_learner\
            and (self.algorithm.get_update_count() // self.actor_models_update_optimization_interval) != \
            self.previous_actor_models_update_quotient:
                self.previous_actor_models_update_quotient = self.algorithm.get_update_count() // self.actor_models_update_optimization_interval
                self.actor_learner_shared_dict["models"] = self.algorithm.get_models()
                self.actor_learner_shared_dict["models_update_required"] = True
            
            if self.save_path is not None \
            and (self.handled_experiences.value // self.saving_interval) != self.previous_save_quotient:
                self.previous_save_quotient = self.handled_experiences.value // self.saving_interval
                self.save()

        return nbr_updates

    def take_action(self, state):
        if self.async_actor:
            # Update the algorithm's model if needs be:
            if self.actor_learner_shared_dict["models_update_required"]:
                self.actor_learner_shared_dict["models_update_required"] = False
                if "models" in self.actor_learner_shared_dict.keys():
                    self.algorithm.set_models(self.actor_learner_shared_dict["models"])
                else:
                    raise NotImplementedError 

        if self.training:
            self.nbr_steps += state.shape[0]
        self.eps = self.algorithm.get_epsilon(nbr_steps=self.nbr_steps, strategy=self.epsdecay_strategy)

        state = self.state_preprocessing(state, use_cuda=self.algorithm.kwargs['use_cuda'])
        goal = None
        if self.goal_oriented:
            goal = self.goal_preprocessing(self.goals, use_cuda=self.algorithm.kwargs['use_cuda'])

        model = self.algorithm.get_models()['model']
        if 'use_target_to_gather_data' in self.kwargs and self.kwargs['use_target_to_gather_data']:
            model = self.algorithm.get_models()['target_model']

        self.current_prediction = self.query_model(model, state, goal)
        
        # Post-process and update the rnn_states from the current prediction:
        # self.rnn_states <-- self.current_prediction['next_rnn_states']
        # WARNING: _post_process affects self.rnn_states. It is imperative to
        # manipulate a copy of it outside of the agent's manipulation, e.g.
        # when feeding it to the models.
        self.current_prediction = self._post_process(self.current_prediction)

        sample = np.random.random()
        if self.noisy or sample > self.eps:
            return self.current_prediction['a'].numpy()
        else:
            random_actions = [random.randrange(model.action_dim) for _ in range(state.shape[0])]
            random_actions = np.reshape(np.array(random_actions), (state.shape[0],1))
            return random_actions

    def query_model(self, model, state, goal):
        if self.recurrent:
            self._pre_process_rnn_states()
            # WARNING: it is imperative to make a copy 
            # of the self.rnn_states, otherwise it will be 
            # referenced in the (self.)current_prediction
            # and any subsequent update of rnn_states will 
            # also update the current_prediction, e.g. the call
            # to _post_process in line 163 affects self.rnn_states
            # and therefore might affect current_prediction's rnn_states...
            rnn_states_input = copy_hdict(self.rnn_states)
            current_prediction = model(state, rnn_states=rnn_states_input, goal=goal)
        else:
            current_prediction = model(state, goal=goal)
        return current_prediction

    def clone(self, training=None, with_replay_buffer=False, clone_proxies=False):
        cloned_algo = self.algorithm.clone(
            with_replay_buffer=with_replay_buffer,
            clone_proxies=clone_proxies
        )
        clone = DQNAgent(name=self.name, algorithm=cloned_algo)

        clone.actor_learner_shared_dict = self.actor_learner_shared_dict
        clone.handled_experiences = self.handled_experiences
        clone.episode_count = self.episode_count
        if training is not None:    clone.training = training
        clone.nbr_steps = self.nbr_steps

        # Goes through all variables 'Proxy' (dealing with multiprocessing)
        # contained in this class and removes them from clone
        if not(clone_proxies):
            proxy_key_values = [
                (key, value) 
                for key, value in clone.__dict__.items() 
                if ('Proxy' in str(type(value)))
            ]
            for key, value in proxy_key_values:
                setattr(clone, key, None)

        return clone

    def get_async_actor(self, training=None, with_replay_buffer=False):
        self.async_learner = True
        self.async_actor = False 

        cloned_algo = self.algorithm.async_actor()
        clone = DQNAgent(name=self.name, algorithm=cloned_algo)
        clone.async_learner = False 
        clone.async_actor = True 

        clone.actor_learner_shared_dict = self.actor_learner_shared_dict
        clone.handled_experiences = self.handled_experiences
        clone.episode_count = self.episode_count
        if training is not None:    clone.training = training
        clone.nbr_steps = self.nbr_steps
        return clone


def generate_model(task: 'regym.environments.Task', kwargs: Dict) -> nn.Module:
    phi_body = None
    input_dim = list(task.observation_shape)
    if 'goal_oriented' in kwargs and kwargs['goal_oriented']:
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
            phi_body = ConvolutionalBody(input_shape=input_shape,
                                         feature_dim=output_dim,
                                         channels=channels,
                                         kernel_sizes=kernels,
                                         strides=strides,
                                         paddings=paddings)
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
                    tl = extra_inputs_infos[key]['target_location']
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


    # TODO: remove this! We needed to relax this condition for MineRL
    # assert(task.action_type == 'Discrete')
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
                            goal_oriented=kwargs['goal_oriented'] if 'goal_oriented' in kwargs else False,
                            goal_shape=goal_shape,
                            goal_phi_body=goal_phi_body)

    model.share_memory()
    return model


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
    kwargs['state_preprocess'] = partial(PreprocessFunction, normalization=False)
    kwargs['goal_preprocess'] = partial(PreprocessFunction, normalization=False)

    if not isinstance(kwargs['observation_resize_dim'], int):  kwargs['observation_resize_dim'] = task.observation_shape[0] if isinstance(task.observation_shape, tuple) else task.observation_shape
    #if 'None' in kwargs['goal_resize_dim']:  kwargs['goal_resize_dim'] = task.goal_shape[0] if isinstance(task.goal_shape, tuple) else task.goal_shape

    model = generate_model(task, kwargs)

    loss_fn = dqn_loss.compute_loss
    if kwargs['double'] or kwargs['dueling']:
        loss_fn = ddqn_loss.compute_loss

    dqn_algorithm = DQNAlgorithm(kwargs, model, loss_fn=loss_fn)

    if 'use_HER' in kwargs and kwargs['use_HER']:
        from ..algorithms.wrappers import latent_based_goal_predicated_reward_fn
        goal_predicated_reward_fn = None
        if 'HER_use_latent' in kwargs and kwargs['HER_use_latent']:
            goal_predicated_reward_fn = latent_based_goal_predicated_reward_fn

        dqn_algorithm = HERAlgorithmWrapper(algorithm=dqn_algorithm,
                                            strategy=kwargs['HER_strategy'],
                                            goal_predicated_reward_fn=goal_predicated_reward_fn)

    agent = DQNAgent(name=agent_name, algorithm=dqn_algorithm)

    if isinstance(getattr(task.env, 'observation_space', None), Dict) or ('use_HER' in kwargs and kwargs['use_HER']):
        agent = DictHandlingAgentWrapper(agent=agent, use_achieved_goal=kwargs['use_HER'])

    print(dqn_algorithm.get_models())

    return agent
