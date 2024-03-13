from typing import Dict
from functools import partial

import copy
import torch
import torch.nn as nn
import ray

from regym.rl_algorithms.agents.agent import ExtraInputsHandlingAgent
from regym.rl_algorithms.agents.dqn_agent import DQNAgent
from regym.rl_algorithms.agents.utils import generate_model, parse_and_check, build_ther_predictor
from regym.rl_algorithms.algorithms.R2D2 import R2D2Algorithm
from regym.rl_algorithms.networks import PreprocessFunction, ResizeCNNPreprocessFunction, ResizeCNNInterpolationFunction

from regym.rl_algorithms.algorithms.wrappers import HERAlgorithmWrapper2, THERAlgorithmWrapper2, predictor_based_goal_predicated_reward_fn2
from regym.rl_algorithms.algorithms.wrappers import (
    ETHERAlgorithmWrapper,
    OnlineReferentialGameAlgorithmWrapper,
    ELAAlgorithmWrapper,
    RewardPredictionAlgorithmWrapper,
)
from regym.rl_algorithms.networks import (
    ArchiPredictor, 
    ArchiPredictorSpeaker,
    ArchiPredictorListener,
    ArchiRewardPredictor,
)

import wandb 


class R2D2Agent(ExtraInputsHandlingAgent, DQNAgent):
    def __init__(self, name, algorithm, extra_inputs_infos):
        # Both init will call the self's reset_rnn_states following self.mro's order, i.e. ExtraInputs's one first.
        ExtraInputsHandlingAgent.__init__(
            self,
            name=name,
            algorithm=algorithm,
            extra_inputs_infos=extra_inputs_infos
        )
        DQNAgent.__init__(
            self,
            name=name,
            algorithm=algorithm
        )

    def _take_action(self, state, infos=None, as_logit=False, training=False):
        return DQNAgent.take_action(self, state=state, infos=infos, as_logit=as_logit, training=training)

    def _query_action(self, state, infos=None, as_logit=False, training=False):
        return DQNAgent.query_action(self, state=state, infos=infos, as_logit=as_logit, training=training)

    def _handle_experience(self, s, a, r, succ_s, done, goals=None, infos=None, succ_infos=None):
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
        DQNAgent.handle_experience(
            self,
            s=s,
            a=a,
            r=r,
            succ_s=succ_s,
            done=done,
            goals=goals,
            infos=infos,
            succ_infos=succ_infos,
        )

    def clone(self, training=None, with_replay_buffer=False, clone_proxies=False, minimal=False):
        '''
        TODO: test
        '''
        cloned_algo = self.algorithm.clone(
            with_replay_buffer=with_replay_buffer,
            clone_proxies=clone_proxies,
            minimal=minimal
        )

        clone = R2D2Agent(
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
        '''
        TODO: test
        '''
        self.async_learner = True
        self.async_actor = False

        cloned_algo = self.algorithm.async_actor()
        clone = R2D2Agent(
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


def build_R2D2_Agent(task: 'regym.environments.Task',
                     config: Dict,
                     agent_name: str):
    '''
    TODO: say that config is the same as DQN agent except for
    - expert_demonstrations: ReplayStorage object with expert demonstrations
    - demo_ratio: [0, 1] Probability of sampling from expert_demonstrations
                  instead of sampling from replay buffer of gathered
                  experiences. Should be small (i.e 1/256)
    - sequence_length:  TODO

    :returns: R2D2 agent
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

    kwargs = parse_and_check(kwargs, task)

    model = generate_model(task, kwargs)

    print(model)

    algorithm = R2D2Algorithm(
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
                    print("WARNING : R2D2 Agent with THER : THER predictor DOES predict PAD tokens.")
                else:
                    print("WARNING : R2D2 Agent with THER : THER predictor does NOT predict PAD tokens.")
                if kwargs.get("use_ETHER", False):
                    if 'obverter' in kwargs['ETHER_rg_graphtype']:
                        predictor = ArchiPredictorListener(
                            model=model, 
                            trainable=not(kwargs.get("ETHER_rg_freeze_speaker", False)),
                            **kwargs["ArchiModel"],
                            pipeline_name={
                                "speaker":"instruction_generator",
                                "listener":"decision_generator",
                            },
                            generator_name={
                                "speaker":"DecisionGenerator",
                                "listener":"DecisionGenerator",
                            },
                        )
                    else:
                        predictor = ArchiPredictorSpeaker(
                            model=model, 
                            trainable=not(kwargs.get("ETHER_rg_freeze_speaker", False)),
                            **kwargs["ArchiModel"],
                            # TODO: maybe provide hyperparameter 
                            pipeline_name="instruction_generator", #"caption_generator",
                            generator_name="InstructionGenerator", #"CaptionGenerator",
                        )
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
            wrapper_kwargs['episode_length_reward_shaping_type'] = kwargs['THER_episode_length_reward_shaping_type']
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
    
    if kwargs.get('use_RP', False):
        reward_predictor = ArchiRewardPredictor(
            model=model,
            **kwargs['ArchiModel'],
        )
        algorithm = RewardPredictionAlgorithmWrapper(
            algorithm=algorithm,
            predictor=reward_predictor,
        )

    if kwargs.get('use_ELA', False):
        # The predictor corresponds to the caption generator pipeline:
        assert "caption_generator" in kwargs['ArchiModel']['pipelines']
        caption_predictor = ArchiPredictorSpeaker(
            model=model, 
            **kwargs["ArchiModel"],
            pipeline_name="caption_generator",
            generator_name="CaptionGenerator",
        )
        algorithm = ELAAlgorithmWrapper(
            algorithm=algorithm,
            predictor=caption_predictor,
            extrinsic_weight=kwargs['ELA_reward_extrinsic_weight'], 
            intrinsic_weight=kwargs['ELA_reward_intrinsic_weight'], 
            feedbacks={
                "failure":kwargs['ELA_feedbacks_failure_reward'], 
                "success":kwargs['ELA_feedbacks_success_reward'],
            },
        )
    
    if kwargs.get("use_ORG", False):
        predictor = ArchiPredictorSpeaker(
            model=model, 
            **kwargs["ArchiModel"],
            pipeline_name="instruction_generator" if kwargs["ORG_with_Oracle"] else "caption_generator",
            generator_name="InstructionGenerator" if kwargs["ORG_with_Oracle"] else "CaptionGenerator",
            trainable=False,
        )
        algorithm = OnlineReferentialGameAlgorithmWrapper(
          algorithm=algorithm,
          predictor=predictor,
        )
        if not kwargs["ORG_with_Oracle"]:
            predictor.set_postprocess_fn(
                partial(kwargs.get("ORG_postprocess_fn", None),
                    algorithm=algorithm,
                )
            )
            
    agent = R2D2Agent(
        name=agent_name,
        algorithm=algorithm,
        extra_inputs_infos=kwargs['extra_inputs_infos'],
    )

    return agent


