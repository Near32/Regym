from typing import Dict
from functools import partial

import copy
import torch
import torch.nn as nn
import ray

from regym.rl_algorithms.agents.r2d2_agent import R2D2Agent
from regym.rl_algorithms.agents.utils import generate_model, parse_and_check, build_ther_predictor
from regym.rl_algorithms.algorithms.RecurrentPPO import RecurrentPPOAlgorithm
from regym.rl_algorithms.networks import PreprocessFunction, ResizeCNNPreprocessFunction, ResizeCNNInterpolationFunction

from regym.rl_algorithms.algorithms.wrappers import HERAlgorithmWrapper2, THERAlgorithmWrapper2, predictor_based_goal_predicated_reward_fn2
from regym.rl_algorithms.algorithms.wrappers import ETHERAlgorithmWrapper
from regym.rl_algorithms.networks import ArchiPredictor, ArchiPredictorSpeaker


class RecurrentPPOAgent(R2D2Agent):
    def __init__(self, name, algorithm, extra_inputs_infos):
        R2D2Agent.__init__(
            self,
            name=name,
            algorithm=algorithm,
            extra_inputs_infos=extra_inputs_infos
        )

    def train(self):
        '''
        Trains like PPOAgent.
        '''
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

    def clone(self, training=None, with_replay_buffer=False, clone_proxies=False, minimal=False):
        cloned_algo = self.algorithm.clone(
            with_replay_buffer=with_replay_buffer,
            clone_proxies=clone_proxies,
            minimal=minimal
        )

        clone = RecurrentPPOAgent(
            name=self.name,
            algorithm=cloned_algo,
            extra_inputs_infos=copy.deepcopy(self.extra_inputs_infos)
        )

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
        clone = RecurrentPPOAgent(
            name=self.name,
            algorithm=cloned_algo,
            extra_inputs_infos=copy.deepcopy(self.extra_inputs_infos)
        )
        clone.save_path = self.save_path
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
        if training is not None:    
            clone.training = training
        else:
            clone.training = self.training
        clone.nbr_steps = self.nbr_steps
        return clone


def build_RecurrentPPO_Agent(
    task: 'regym.environments.Task',
    config: Dict,
    agent_name: str,
):
    '''
    TODO: say that config is the same as DQN agent except for
    - expert_demonstrations: ReplayStorage object with expert demonstrations
    - demo_ratio: [0, 1] Probability of sampling from expert_demonstrations
                  instead of sampling from replay buffer of gathered
                  experiences. Should be small (i.e 1/256)
    - sequence_length:  TODO

    :returns: RecurrentPPO agent
    '''

    kwargs = config.copy()
    kwargs['discount'] = float(kwargs['discount'])
    kwargs['replay_capacity'] = int(float(kwargs['replay_capacity']))
    kwargs['min_capacity'] = int(float(kwargs['min_capacity']))

    # Default preprocess function:
    kwargs['state_preprocess'] = partial(PreprocessFunction, normalization=False)
    kwargs['goal_preprocess'] = partial(PreprocessFunction, normalization=False)

    if 'observation_resize_dim' in kwargs\
    and not isinstance(kwargs['observation_resize_dim'], int):  
        kwargs['observation_resize_dim'] = task.observation_shape[0] if isinstance(task.observation_shape, tuple) else task.observation_shape
    #if 'None' in kwargs['goal_resize_dim']:  kwargs['goal_resize_dim'] = task.goal_shape[0] if isinstance(task.goal_shape, tuple) else task.goal_shape

    if task.action_type == 'Discrete':
        if task.observation_type == 'Discrete':
            head_type = "CategoricalActorCriticNet"
        elif task.observation_type == 'Continuous':
            if 'use_vae' in kwargs and kwargs['use_vae']:
                head_type = "CategoricalActorCriticVAENet"
                raise NotImplementedError
            else:
                head_type = "CategoricalActorCriticNet"

    if task.action_type == 'Continuous' and task.observation_type == 'Continuous':
        head_type = "GaussianActorCriticNet"

    kwargs = parse_and_check(kwargs, task)
    model = generate_model(task, kwargs, head_type=head_type)

    print(model)

    algorithm = RecurrentPPOAlgorithm(
        kwargs=kwargs,
        model=model,
        name=f"{agent_name}_algo",
    )

    if kwargs.get('use_HER', False):
        from regym.rl_algorithms.algorithms.wrappers import latent_based_goal_predicated_reward_fn2
        from regym.rl_algorithms.algorithms.wrappers import predictor_based_goal_predicated_reward_fn2
        from regym.rl_algorithms.algorithms.wrappers import batched_predictor_based_goal_predicated_reward_fn2
        
        goal_predicated_reward_fn = kwargs.get(
            "HER_goal_predicated_reward_fn",
            None,
        )

        if kwargs.get('HER_use_latent', False):
            goal_predicated_reward_fn = latent_based_goal_predicated_reward_fn2
        elif kwargs.get('use_THER', False):
            goal_predicated_reward_fn = predictor_based_goal_predicated_reward_fn2
        elif goal_predicated_reward_fn is None:
            raise NotImplementedError("if only using HER, then need HER_use_latent=True")

        wrapper = HERAlgorithmWrapper2 
        wrapper_kwargs = {
            "algorithm":algorithm,
            "strategy":kwargs['HER_strategy'],
            "goal_predicated_reward_fn":goal_predicated_reward_fn,
            "extra_inputs_infos":kwargs['extra_inputs_infos'],
            "_extract_goal_from_info_fn":kwargs.get("HER_extract_goal_from_info_fn", None),
            "target_goal_key_from_info":kwargs["HER_target_goal_key_from_info"],
            "achieved_latent_goal_key_from_info":kwargs.get("HER_achieved_latent_goal_key_from_info", None),
            "target_latent_goal_key_from_info":kwargs.get("HER_target_latent_goal_key_from_info", None),
            "filtering_fn":kwargs["HER_filtering_fn"],
        }

        if kwargs.get('use_THER', False):
            # THER would use the predictor to provide the achieved goal, so the key is not necessary:
            # WARNING: if you want to use this functionality, 
            # then you need to be careful for the predicated fn, maybe ? not tested yet... 
            # 
            wrapper_kwargs["achieved_goal_key_from_info"] = kwargs.get("HER_achieved_goal_key_from_info", None)
            from regym.rl_algorithms.algorithms.THER import ther_predictor_loss

            wrapper = THERAlgorithmWrapper2 
            kwargs['THER_predictor_learning_rate'] = float(kwargs['THER_predictor_learning_rate'])
            kwargs['discount'] = float(kwargs['discount'])
            kwargs['replay_capacity'] = int(float(kwargs['replay_capacity']))
            kwargs['min_capacity'] = int(float(kwargs['min_capacity']))
            kwargs['THER_vocabulary'] = set(kwargs['THER_vocabulary'])
            kwargs['THER_max_sentence_length'] = int(kwargs['THER_max_sentence_length'])
                
            if 'ArchiModel' in kwargs.keys():
                # The predictor corresponds to the instruction generator pipeline:
                assert "instruction_generator" in kwargs['ArchiModel']['pipelines']
                if kwargs.get("THER_predict_PADs", False):
                    model.modules["InstructionGenerator"].predict_PADs = kwargs["THER_predict_PADs"]
                    print("WARNING : RecurrentPPO Agent with THER : THER predictor DOES predict PAD tokens.")
                else:
                    print("WARNING : RecurrentPPO Agent with THER : THER predictor does NOT predict PAD tokens.")
                if kwargs.get("use_ETHER", False):
                    predictor = ArchiPredictorSpeaker(model=model, **kwargs["ArchiModel"])
                else:
                    predictor = ArchiPredictor(model=model, **kwargs["ArchiModel"])
            else:
                predictor = build_ther_predictor(kwargs, task)
            
            print(predictor)
            for np, p in predictor.named_parameters():
                print(np, p.shape)

            wrapper_kwargs['predictor'] = predictor
            wrapper_kwargs['predictor_loss_fn'] = ther_predictor_loss.compute_loss
            wrapper_kwargs['feedbacks'] = {"failure":kwargs['THER_feedbacks_failure_reward'], "success":kwargs['THER_feedbacks_success_reward']}
            wrapper_kwargs['relabel_terminal'] = kwargs['THER_relabel_terminal']
            wrapper_kwargs['filter_predicate_fn'] = kwargs['THER_filter_predicate_fn']
            wrapper_kwargs['filter_out_timed_out_episode'] = kwargs['THER_filter_out_timed_out_episode']
            wrapper_kwargs['timing_out_episode_length_threshold'] = kwargs['THER_timing_out_episode_length_threshold']
            wrapper_kwargs['episode_length_reward_shaping'] = kwargs['THER_episode_length_reward_shaping']
            wrapper_kwargs['train_contrastively'] = kwargs['THER_train_contrastively']
            wrapper_kwargs['contrastive_training_nbr_neg_examples'] = kwargs['THER_contrastive_training_nbr_neg_examples']
        
            if 'THER_use_predictor' in kwargs and kwargs['THER_use_predictor']:
                wrapper_kwargs['goal_predicated_reward_fn'] = partial(
                    #predictor_based_goal_predicated_reward_fn2, 
                    batched_predictor_based_goal_predicated_reward_fn2, 
                    predictor=predictor,
                )

            if kwargs.get("use_ETHER", False):
                wrapper = ETHERAlgorithmWrapper
        else:
            wrapper_kwargs["achieved_goal_key_from_info"] = kwargs["HER_achieved_goal_key_from_info"]

        algorithm = wrapper(**wrapper_kwargs)
    
    agent = RecurrentPPOAgent(
        name=agent_name,
        algorithm=algorithm,
        extra_inputs_infos=kwargs['extra_inputs_infos'],
    )

    return agent


