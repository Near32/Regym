from typing import Dict, Any 

import ray
import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
import copy

from ..algorithms.PPO import PPOAlgorithm, ppo_loss, rnd_loss
from ..networks import PreprocessFunction, ResizeCNNPreprocessFunction, ResizeCNNInterpolationFunction
from regym.rl_algorithms.networks import ConvolutionalBody, FCBody

from regym.rl_algorithms.agents.agent import Agent, ExtraInputsHandlingAgent
from .wrappers import DictHandlingAgentWrapper
from gym.spaces import Dict as gymDict

from regym.rl_algorithms.utils import _extract_from_rnn_states, copy_hdict
from regym.rl_algorithms.utils import apply_on_hdict, _concatenate_list_hdict
from regym.rl_algorithms.utils import recursive_inplace_update
from regym.rl_algorithms.agents.utils import generate_model, parse_and_check

import wandb


class PPOAgent(ExtraInputsHandlingAgent, Agent):
    def __init__(self, name, algorithm, extra_inputs_infos):
        ExtraInputsHandlingAgent.__init__(
            self,
            name=name,
            algorithm=algorithm,
            extra_inputs_infos=extra_inputs_infos
        )
        
        Agent.__init__(
            self,
            name=name, 
            algorithm=algorithm
        )

        self.kwargs = algorithm.kwargs

        self.use_rnd = self.algorithm.use_rnd

        # Number of interaction/step with/in the environment:
        self.nbr_steps = 0

        self.saving_interval = float(self.kwargs['saving_interval']) if 'saving_interval' in self.kwargs else 1e5
        
        self.previous_save_quotient = 0

    def get_experience_count(self):
        return self.handled_experiences

    def get_update_count(self):
        return self.algorithm.unwrapped.get_update_count()
        #return self.handled_experiences // (self.algorithm.kwargs['horizon']*self.nbr_actor)

    def get_obs_count(self):
        return self.algorithm.unwrapped.get_obs_count()

    def get_intrinsic_reward(self, actor_idx):
        if len(self.algorithm.storages[actor_idx].int_r):
            #return self.algorithm.storages[actor_idx].int_r[-1] / (self.algorithm.int_reward_std+1e-8)
            return self.algorithm.storages[actor_idx].int_r[-1] / (self.algorithm.int_return_std+1e-8)
        else:
            return 0.0

    def _handle_experience(self, s, a, r, succ_s, done, goals=None, infos=None, succ_infos=None, prediction=None):
        '''
        Note: the batch size may differ from the nbr_actor as soon as some
        actors' episodes end before the others...

        :param s: numpy tensor of states of shape batch x state_shape.
        :param a: numpy tensor of actions of shape batch x action_shape.
        :param r: numpy tensor of rewards of shape batch x reward_shape.
        :param succ_s: numpy tensor of successive states of shape batch x state_shape.
        :param done: list of boolean (batch=nbr_actor) x state_shape.
        :param infos: Dictionnary of information from the environment.
        :param prediction: Dictionnary of tensors containing the model's output at the current state.
        '''
        torch.set_grad_enabled(False)

        if "sad" in self.kwargs \
        and self.kwargs["sad"]:
            a = a["action"]

        if prediction is None:  prediction = copy.deepcopy(self.current_prediction)

        state, r, succ_state, non_terminal = self.preprocess_environment_signals(s, r, succ_s, done)
        a = torch.from_numpy(a)
        # batch x ...

        batch_size = a.shape[0]

        if "vdn" in self.kwargs \
        and self.kwargs["vdn"]:
            # Add a player dimension to each element:
            # Assume inputs have shape : [batch_size*nbr_players, ...],
            # i.e. [batch_for_p0; batch_for_p1, ...]
            nbr_players = self.kwargs["vdn_nbr_players"]
            batch_size = state.shape[0] // nbr_players
            
            new_state = []
            for bidx in range(batch_size):
                bidx_states = torch.stack(
                    [
                        state[pidx*batch_size+bidx].unsqueeze(0) 
                        for pidx in range(nbr_players)
                    ], 
                    dim=1
                )
                new_state.append(bidx_states)
            state = torch.cat(new_state, dim=0)
            
            new_a = []
            for bidx in range(batch_size):
                bidx_as = torch.stack(
                    [
                        a[pidx*batch_size+bidx].unsqueeze(0) 
                        for pidx in range(nbr_players)
                    ], 
                    dim=1
                )
                new_a.append(bidx_as)
            a = torch.cat(new_a, dim=0)

            new_r = []
            for bidx in range(batch_size):
                bidx_rs = torch.stack(
                    [
                        r[pidx*batch_size+bidx].unsqueeze(0) 
                        for pidx in range(nbr_players)
                    ], 
                    dim=1
                )
                new_r.append(bidx_rs)
            r = torch.cat(new_r, dim=0)

            '''
            non_terminal = torch.cat([non_terminal]*2, dim=0)
            new_nt = []
            for bidx in range(batch_size):
                bidx_nts = torch.stack([non_terminal[pidx*batch_size+bidx].unsqueeze(0) for pidx in range(nbr_players)], dim=1)
                new_nt.append(bidx_nts)
            non_terminal = torch.cat(new_nt, dim=0)            
            '''
            
            new_succ_state = []
            for bidx in range(batch_size):
                bidx_succ_states = torch.stack(
                    [
                        succ_state[pidx*batch_size+bidx].unsqueeze(0) 
                        for pidx in range(nbr_players)
                    ], 
                    dim=1
                )
                new_succ_state.append(bidx_succ_states)
            succ_state = torch.cat(new_succ_state, dim=0)
            
            # BEWARE: reshaping might not give the expected ordering due to the dimensions' ordering...
            #hdict_reshape_fn = lambda x: x.reshape(batch_size, nbr_players, *x.shape[1:])
            # The above fails to capture the correct ordering:
            # [ batch0=[p0_exp1, p0_exp2 ; .. ]] instead of 
            # [ batch0=[p0_exp1, p1_exp1 ; .. ]], if only two players are considered...  
            def reshape_fn(x):
                new_x = []
                for bidx in range(batch_size):
                    bidx_x = torch.stack(
                        [
                            x[pidx*batch_size+bidx].unsqueeze(0) 
                            for pidx in range(nbr_players)
                        ], 
                        dim=1
                    )
                    new_x.append(bidx_x)
                return torch.cat(new_x, dim=0)
                
            for k, t in prediction.items():
                if isinstance(t, torch.Tensor):
                    #prediction[k] = t.reshape(batch_size, nbr_players, *t.shape[1:])
                    prediction[k] = reshape_fn(prediction[k])
                elif isinstance(t, dict):
                    prediction[k] = apply_on_hdict(
                        hdict=t,
                        fn=reshape_fn, #hdict_reshape_fn,
                    )
                else:
                    raise NotImplementedError
            
            """
            # not used...
            # Infos: list of batch_size * nbr_players dictionnaries:
            new_infos = []
            for bidx in range(batch_size):
                bidx_infos = [infos[pidx*batch_size+bidx] for pidx in range(nbr_players)]
                bidx_info = _concatenate_list_hdict(
                    lhds=bidx_infos,
                    concat_fn=partial(np.stack, axis=1),   #new player dimension
                    preprocess_fn=(lambda x: x),
                )
                new_infos.append(bidx_info)
            infos = new_infos
            
            # Goals:
            if self.goal_oriented:
                raise NotImplementedError
            """

        # We assume that this function has been called directly after take_action:
        # therefore the current prediction correspond to this experience.

        # Update the next_rnn_states with relevant infos, before extraction:
        if succ_infos is not None \
        and hasattr(self, '_build_dict_from'):
            hdict = self._build_dict_from(lhdict=succ_infos)
            recursive_inplace_update(prediction['next_rnn_states'], hdict)
             
        done_actors_among_notdone = []
        for actor_index in range(self.nbr_actor):
            # If this actor is already done with its episode:  
            if self.previously_done_actors[actor_index]:
                continue
            # Otherwise, there is bookkeeping to do:
            
            # Bookkeeping of the actors whose episode just ended:
            if done[actor_index]:
                done_actors_among_notdone.append(actor_index)
                
            exp_dict = {}
            exp_dict['s'] = state[actor_index,...].unsqueeze(0)
            exp_dict['a'] = a[actor_index,...].unsqueeze(0)
            exp_dict['r'] = r[actor_index,...].unsqueeze(0)
            exp_dict['succ_s'] = succ_state[actor_index,...].unsqueeze(0)
            exp_dict['non_terminal'] = non_terminal[actor_index,...].unsqueeze(0)
            if infos is not None:
                exp_dict['info'] = infos[actor_index]
            if succ_infos is not None:
                exp_dict['succ_info'] = succ_infos[actor_index]

            
            #########################################################################
            #########################################################################
            # Exctracts tensors at root level:
            exp_dict.update(Agent._extract_from_prediction(prediction, actor_index))
            #########################################################################
            #########################################################################
            
            if self.use_rnd:
                with torch.no_grad():
                    int_reward, target_int_f = self.algorithm.compute_intrinsic_reward(exp_dict['succ_s'])
                rnd_dict = {'int_r':int_reward, 'target_int_f':target_int_f}
                exp_dict.update(rnd_dict)

            # Extracts remaining info:
            if self.recurrent:
                exp_dict['rnn_states'] = _extract_from_rnn_states(
                    prediction['rnn_states'],
                    actor_index,
                    post_process_fn=(lambda x: x.detach().cpu())
                )
                exp_dict['next_rnn_states'] = _extract_from_rnn_states(
                    prediction['next_rnn_states'],
                    actor_index,
                    post_process_fn=(lambda x: x.detach().cpu())
                )

            self.algorithm.storages[actor_index].add(exp_dict)
            self.previously_done_actors[actor_index] = done[actor_index]
            self.handled_experiences +=1

        #if len(done_actors_among_notdone):
        #    # Regularization of the agents' actors:
        #    done_actors_among_notdone.sort(reverse=True)
        #    #for batch_idx in done_actors_among_notdone:
        #    #    self.update_actors(batch_idx=batch_idx)

        if not(self.async_actor):
            self.train()

        #if self.training \
        #and self.handled_experiences % self.algorithm.kwargs['horizon']*self.nbr_actor == 0: 
            #self.algorithm.train()
            #if self.save_path is not None: torch.save(self, self.save_path)

    def train(self):
        nbr_updates = 0

        if self.training \
        and self.algorithm.stored_experiences() > self.algorithm.kwargs['horizon']*self.nbr_actor:
            self.algorithm.train()
            
            if isinstance(self.actor_learner_shared_dict, ray.actor.ActorHandle):
                actor_learner_shared_dict = ray.get(self.actor_learner_shared_dict.get.remote())
            else:
                actor_learner_shared_dict = self.actor_learner_shared_dict.get()
            nbr_update_remaining = sum(actor_learner_shared_dict["models_update_required"])
            wandb.log({
                f'PerUpdate/ActorLearnerSynchroRemainingUpdates':
                nbr_update_remaining
                }, 
                #self.algorithm.unwrapped.get_update_count()
            )
            
            # Update actor's models:
            if self.async_learner\
            and (self.handled_experiences // self.actor_models_update_steps_interval) != self.previous_actor_models_update_quotient:
                self.previous_actor_models_update_quotient = self.handled_experiences // self.actor_models_update_steps_interval
                new_models_cpu = {k:copy.deepcopy(m).cpu() for k,m in self.algorithm.get_models().items()}
                
                if isinstance(self.actor_learner_shared_dict, ray.actor.ActorHandle):
                    actor_learner_shared_dict = ray.get(self.actor_learner_shared_dict.get.remote())
                else:
                    actor_learner_shared_dict = self.actor_learner_shared_dict.get()
                
                actor_learner_shared_dict["models"] = new_models_cpu
                actor_learner_shared_dict["models_update_required"] = [True]*len(actor_learner_shared_dict["models_update_required"])
                
                if isinstance(self.actor_learner_shared_dict, ray.actor.ActorHandle):
                    self.actor_learner_shared_dict.set.remote(actor_learner_shared_dict)
                else:
                    self.actor_learner_shared_dict.set(actor_learner_shared_dict)

            obs_count = self.algorithm.unwrapped.get_obs_count()
            if self.async_learner\
            and self.save_path is not None \
            and (obs_count // self.saving_interval) != self.previous_save_quotient:
                self.previous_save_quotient = obs_count // self.saving_interval
                original_save_path = self.save_path
                self.save_path = original_save_path.split(".agent")[0]+"."+str(int(self.previous_save_quotient))+".agent"
                self.save()
                self.save_path = original_save_path

        return 

    def _take_action(self, state, infos=None, as_logit=False, training=False):
        torch.set_grad_enabled(training)
        
        if self.async_actor:
            # Update the algorithm's model if needs be:
            if isinstance(self.actor_learner_shared_dict, ray.actor.ActorHandle):
                actor_learner_shared_dict = ray.get(self.actor_learner_shared_dict.get.remote())
            else:
                actor_learner_shared_dict = self.actor_learner_shared_dict.get()
            if actor_learner_shared_dict["models_update_required"][self.async_actor_idx]:
                actor_learner_shared_dict["models_update_required"][self.async_actor_idx] = False
                
                if isinstance(self.actor_learner_shared_dict, ray.actor.ActorHandle):
                    self.actor_learner_shared_dict.set.remote(actor_learner_shared_dict)
                else:
                    self.actor_learner_shared_dict.set(actor_learner_shared_dict)
                
                if "models" in actor_learner_shared_dict.keys():
                    new_models = actor_learner_shared_dict["models"]
                    self.algorithm.set_models(new_models)
                else:
                    raise NotImplementedError 

        state = self.state_preprocessing(state, use_cuda=self.algorithm.kwargs['use_cuda'])
        '''
        wandb.log({
            "Model/StateMean": state.mean(),
            "Model/StateStd": state.std(),
            "Model/Min": state.min(),
            "Model/Max": state.max(),
            },
            commit=False,
        )
        '''
        model = self.algorithm.unwrapped.get_models()['model']
        model = model.train(mode=self.training)

        if self.training:
            self.nbr_steps += state.shape[0]
        
        self.current_prediction = self.query_model(model, state)
        # Post-process and update the rnn_states from the current prediction:
        # self.rnn_states <-- self.current_prediction['next_rnn_states']
        # WARNING: _post_process affects self.rnn_states. It is imperative to
        # manipulate a copy of it outside of the agent's manipulation, e.g.
        # when feeding it to the models.
        self.current_prediction = self._post_process(self.current_prediction)
        
        if as_logit:
            return self.current_prediction['log_a']

        #action = self.current_prediction['a'].numpy()
        actions = self.current_prediction['a'].reshape((-1,1)).numpy()
        greedy_action = self.current_prediction['greedy_action'].reshape((-1,1)).numpy()

        if not(self.training):
            return greedy_action

        if "sad" in self.kwargs \
        and self.kwargs["sad"]:
            action_dict = {
                'action': actions,
                'greedy_action': greedy_action,
            }
            return action_dict 

        return actions

    def query_action(self, state, infos=None, as_logit=False, training=False):
        """
        Query's the model in training mode...
        """
        torch.set_grad_enabled(training)

        if self.async_actor:
            # Update the algorithm's model if needs be:
            if isinstance(self.actor_learner_shared_dict, ray.actor.ActorHandle):
                actor_learner_shared_dict = ray.get(self.actor_learner_shared_dict.get.remote())
            else:
                actor_learner_shared_dict = self.actor_learner_shared_dict.get()
            if actor_learner_shared_dict["models_update_required"][self.async_actor_idx]:
                actor_learner_shared_dict["models_update_required"][self.async_actor_idx] = False
                
                if isinstance(self.actor_learner_shared_dict, ray.actor.ActorHandle):
                    self.actor_learner_shared_dict.set.remote(actor_learner_shared_dict)
                else:
                    self.actor_learner_shared_dict.set(actor_learner_shared_dict)
                
                if "models" in actor_learner_shared_dict.keys():
                    new_models = actor_learner_shared_dict["models"]
                    self.algorithm.unwrapped.set_models(new_models)
                else:
                    raise NotImplementedError 

        state = self.state_preprocessing(state, use_cuda=self.algorithm.unwrapped.kwargs['use_cuda'], training=training)
        
        model = self.algorithm.unwrapped.get_models()['model']
        if not(model.training):  model = model.train(mode=True)

        current_prediction = self.query_model(model, state)
        
        # 1) Post-process and update the rnn_states from the current prediction:
        # self.rnn_states <-- self.current_prediction['next_rnn_states']
        # WARNING: _post_process affects self.rnn_states. It is imperative to
        # manipulate a copy of it outside of the agent's manipulation, e.g.
        # when feeding it to the models.
        # 2) All the elements from the prediction dictionnary are being detached+cpued from the graph.
        # Thus, here, we only want to update the rnn state:
        
        if as_logit:
            self._keep_grad_update_rnn_states(
                next_rnn_states_dict=current_prediction['next_rnn_states'],
                rnn_states_dict=self.rnn_states
            )
            return current_prediction
        else:
            current_prediction = self._post_process(current_prediction)
        
        action = current_prediction['a'].reshape((-1,1)).numpy()
        greedy_action = self.current_prediction['greedy_action'].reshape((-1,1)).numpy()

        if not(self.training):
            return action

        if "sad" in self.kwargs \
        and self.kwargs["sad"]:
            action_dict = {
                'action': actions,
                'greedy_action': greedy_action,
            }
            return action_dict 

        return actions

    def query_model(self, model, state, goal=None):
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
            current_prediction = model(state, rnn_states=rnn_states_input)
        else:
            current_prediction = model(state, goal=goal)
        return current_prediction

    def clone(self, training=None, with_replay_buffer=False, clone_proxies=False, minimal=False):
        cloned_algo = self.algorithm.clone(
            with_replay_buffer=with_replay_buffer,
            clone_proxies=clone_proxies,
            minimal=minimal
        )
        
        clone = PPOAgent(
            name=self.name, 
            algorithm=cloned_algo,
            extra_inputs_infos=copy.deepcopy(self.extra_inputs_infos)
        )
        clone.save_path = self.save_path

        clone.actor_learner_shared_dict = self.actor_learner_shared_dict
        clone._handled_experiences = self._handled_experiences
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
        clone = PPOAgent(
            name=self.name, 
            algorithm=cloned_algo,
            extra_inputs_infos=copy.deepcopy(self.extra_inputs_infos)
        )
        
        clone.async_learner = False 
        clone.async_actor = True 

        ######################################
        ######################################
        # Update actor_learner_shared_dict:
        ######################################
        if isinstance(self.actor_learner_shared_dict, ray.actor.ActorHandle):
            actor_learner_shared_dict = ray.get(self.actor_learner_shared_dict.get.remote())
        else:
            actor_learner_shared_dict = self.actor_learner_shared_dict.get()
        # Increase the size of the list of toggle booleans:
        actor_learner_shared_dict["models_update_required"] += [False]
        
        # Update the (Ray)SharedVariable            
        if isinstance(self.actor_learner_shared_dict, ray.actor.ActorHandle):
            self.actor_learner_shared_dict.set.remote(actor_learner_shared_dict)
        else:
            self.actor_learner_shared_dict.set(actor_learner_shared_dict)
        
        ######################################
        # Update the async_actor index:
        clone.async_actor_idx = len(actor_learner_shared_dict["models_update_required"])-1

        ######################################
        ######################################
        
        clone.actor_learner_shared_dict = self.actor_learner_shared_dict
        clone._handled_experiences = self._handled_experiences
        clone.episode_count = self.episode_count
        if training is not None:    clone.training = training
        clone.nbr_steps = self.nbr_steps
        return clone



def build_PPO_Agent(task, config, agent_name):
    '''
    :param task: Environment specific configuration
    :param config: Dict containing configuration for ppo agent
    :param agent_name: name of the agent
    :returns: PPOAgent adapted to be trained on :param: task under :param: config
    '''
    kwargs = config.copy()
    kwargs['discount'] = float(kwargs['discount'])

    # Default preprocess function:
    kwargs['state_preprocess'] = PreprocessFunction
    

    if task.action_type == 'Discrete':
        if task.observation_type == 'Discrete':
            head_type = "CategoricalActorCriticNet"
        elif task.observation_type == 'Continuous':
            if 'use_vae' in kwargs and kwargs['use_vae']:
                head_type = "CategoricalActorCriticVAENet"
                raise NotImplementedError
            else:
                head_type = "CategoricalActorCriticNet"

    if task.action_type is 'Continuous' and task.observation_type is 'Continuous':
        head_type = "GaussianActorCriticNet"

    kwargs = parse_and_check(kwargs, task)
    model = generate_model(task, kwargs, head_type=head_type)
    
    print(model)

    use_rnd = False
    if 'use_random_network_distillation' in kwargs and kwargs['use_random_network_distillation']:
        use_rnd = True

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
        
        print(target_intr_model)
        target_intr_model.share_memory()
        print(predict_intr_model)
        predict_intr_model.share_memory()

    loss_fn = ppo_loss.compute_loss
    if use_rnd:
        loss_fn = rnd_loss.compute_loss
    
    ppo_algorithm = PPOAlgorithm(
        kwargs, 
        model, 
        name=f"{agent_name}_algo",
        target_intr_model=target_intr_model, 
        predict_intr_model=predict_intr_model,
        loss_fn=loss_fn,
    )

    agent = PPOAgent(
        name=agent_name, 
        algorithm=ppo_algorithm,
        extra_inputs_infos=kwargs['extra_inputs_infos'],
    )

    """
    if isinstance(getattr(task.env, 'observation_space', None), gymDict):
        agent = DictHandlingAgentWrapper(agent=agent, use_achieved_goal=False)
    """

    print(agent)

    return agent 
