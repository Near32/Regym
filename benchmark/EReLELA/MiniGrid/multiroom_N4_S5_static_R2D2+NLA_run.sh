WANDB_CACHE_DIR=./wandb_cache/ xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m ipdb -c c ../benchmark_wandb_erelela.py \
--seed=10 --env_seed=12 --static_envs=True \
--use_cuda=True \
--project=EReLELA-MultiRoom-Benchmark \
--success_threshold=0.999 \
--config=multiroom_N4_S5_minigrid_wandb_benchmark_ETHER+R2D2+RP+ELA+SharedObsEncoder_config.yaml \
--language_guided_curiosity=True \
--language_guided_curiosity_descr_type='descr' \
--language_guided_curiosity_extrinsic_weight=2.0 \
--language_guided_curiosity_intrinsic_weight=0.1 \
--language_guided_curiosity_densify=False \
--language_guided_curiosity_non_episodic_dampening_rate=1.0 \
--coverage_manipulation_metric=True \
--MiniWorld_entity_visibility_oracle=False \
--MiniWorld_entity_visibility_oracle_language_specs='none' \
--MiniWorld_entity_visibility_oracle_too_far_threshold=-1.0 \
--MiniWorld_entity_visibility_oracle_include_discrete_depth=True \
--MiniWorld_entity_visibility_oracle_include_depth_precision=-1 \
--MiniWorld_entity_visibility_oracle_top_view=False \
--PER_alpha=0.5 --PER_beta=1.0 \
--PER_use_rewards_in_priority=False \
--sequence_replay_PER_eta=0.9 \
--PER_compute_initial_priority=True \
--use_ETHER=False --use_THER=False \
--use_RP=False --RP_use_RP=True \
--use_ELA=True --ELA_use_ELA=False \
--use_HER=False --goal_oriented=False \
--ETHER_use_ETHER=False --THER_use_THER=False \
--ETHER_rg_use_cuda=True \
--ETHER_rg_sanity_check_compactness_ambiguity_metric=False \
--ETHER_rg_shared_architecture=False \
--ETHER_rg_with_logits_mdl_principle=True \
--ETHER_rg_logits_mdl_principle_factor=1.0e-3 \
--ETHER_rg_logits_mdl_principle_accuracy_threshold=10.0 \
--ETHER_rg_agent_loss_type=Impatient+Hinge \
--ETHER_use_supervised_training=False \
--ETHER_rg_use_semantic_cooccurrence_grounding=True \
--ETHER_rg_semantic_cooccurrence_grounding_lambda=1.0 \
--ETHER_rg_semantic_cooccurrence_grounding_noise_magnitude=0.2 \
--ETHER_lock_test_storage=True --ETHER_rg_filter_out_non_unique=False\
--ETHER_rg_with_color_jitter_augmentation=False --ETHER_rg_with_gaussian_blur_augmentation=True \
--ETHER_rg_egocentric=False \
--ETHER_rg_object_centric_version=2 --ETHER_rg_descriptive_version=1 \
--ETHER_rg_learning_rate=6.25e-5 --ETHER_rg_weight_decay=0.0 \
--ETHER_rg_vocab_size=64 --ETHER_rg_training_period=4096 \
--ETHER_rg_descriptive=False --ETHER_rg_use_curriculum_nbr_distractors=False \
--ETHER_rg_nbr_epoch_per_update=1 --ETHER_rg_accuracy_threshold=99 \
--ETHER_rg_nbr_train_distractors=7 --ETHER_rg_nbr_test_distractors=7 \
--ETHER_replay_capacity=2048 --ETHER_test_replay_capacity=512 \
--ETHER_rg_distractor_sampling=similarity-90 \
--RP_use_PER=True \
--RP_lock_test_storage=False \
--RP_predictor_learning_rate=6.25e-5 \
--RP_gradient_clip=5.0 \
--RP_replay_capacity=16384 --RP_min_capacity=32 \
--RP_predictor_nbr_minibatches=4 --RP_predictor_batch_size=256 \
--RP_predictor_test_train_split_interval=3 --RP_test_replay_capacity=1024 \
--RP_test_min_capacity=32 --RP_replay_period=1024 \
--RP_nbr_training_iteration_per_update=8 \
--RP_predictor_accuracy_threshold=90 \
--ELA_rg_use_cuda=True \
--ELA_rg_sanity_check_compactness_ambiguity_metric=False \
--ELA_rg_shared_architecture=False \
--ELA_rg_with_logits_mdl_principle=True \
--ELA_rg_logits_mdl_principle_factor=1.0e-3 \
--ELA_rg_logits_mdl_principle_accuracy_threshold=80.0 \
--ELA_rg_agent_loss_type=Impatient+Hinge \
--ELA_rg_use_semantic_cooccurrence_grounding=False \
--ELA_rg_semantic_cooccurrence_grounding_lambda=1.0 \
--ELA_rg_semantic_cooccurrence_grounding_noise_magnitude=0.2 \
--ELA_lock_test_storage=True \
--ELA_rg_color_jitter_prob=0.0 \
--ELA_rg_gaussian_blur_prob=0.5 \
--ELA_rg_egocentric_prob=0.5 \
--ELA_rg_object_centric_version=2 --ELA_rg_descriptive_version=1 \
--ELA_rg_learning_rate=6.25e-5 --ELA_rg_weight_decay=0.0 \
--ELA_rg_l1_weight_decay=0.0 --ELA_rg_l2_weight_decay=0.0 \
--ELA_rg_vocab_size=64 --ELA_rg_max_sentence_length=64 \
--ELA_rg_training_period=4096 \
--ELA_rg_descriptive=False --ELA_rg_use_curriculum_nbr_distractors=False \
--ELA_rg_nbr_epoch_per_update=2 --ELA_rg_accuracy_threshold=95 \
--ELA_rg_nbr_train_distractors=15 --ELA_rg_nbr_test_distractors=7 \
--ELA_replay_capacity=4096 --ELA_test_replay_capacity=1024 \
--ELA_rg_distractor_sampling=uniform \
--ELA_reward_extrinsic_weight=1.0 --ELA_reward_intrinsic_weight=1.0 \
--ELA_feedbacks_failure_reward=0.0 --ELA_feedbacks_success_reward=1 \
--THER_use_PER=True --THER_observe_achieved_goal=False \
--THER_lock_test_storage=True \
--THER_feedbacks_failure_reward=0 --THER_feedbacks_success_reward=1 \
--THER_episode_length_reward_shaping=True \
--THER_replay_capacity=1e2 --THER_min_capacity=4 \
--THER_predictor_nbr_minibatches=1 --THER_predictor_batch_size=32 \
--THER_predictor_test_train_split_interval=5 --THER_test_replay_capacity=1e2 \
--THER_test_min_capacity=4 --THER_replay_period=1028 \
--THER_train_on_success=False --THER_nbr_training_iteration_per_update=128 \
--THER_predict_PADs=False --THER_predictor_accuracy_threshold=0.95 \
--THER_predictor_accuracy_safe_to_relabel_threshold=0.2 --THER_filter_predicate_fn=True \
--THER_relabel_terminal=False --THER_filter_out_timed_out_episode=True \
--THER_train_contrastively=False --THER_contrastive_training_nbr_neg_examples=0 \
--THER_timing_out_episode_length_threshold=200 \
--BabyAI_Bot_action_override=False \
--n_step=3 --nbr_actor=32 \
--epsstart=1.0 --epsend=0.1 \
--epsdecay=100000 --eps_greedy_alpha=2.0 \
--nbr_minibatches=1 --batch_size=64 \
--min_capacity=4e3 --replay_capacity=20e3 --learning_rate=6.25e-5 \
--sequence_replay_burn_in_ratio=0.5 --weights_entropy_lambda=0.0 \
--sequence_replay_unroll_length=20 --sequence_replay_overlap_length=10 \
--sequence_replay_use_online_states=True --sequence_replay_use_zero_initial_states=False \
--sequence_replay_store_on_terminal=False --HER_target_clamping=False \
--adam_weight_decay=0.0 --ther_adam_weight_decay=0.0 \
--nbr_training_iteration_per_cycle=2 --nbr_episode_per_cycle=0 \
--single_pick_episode=False \
--terminate_on_completion=True \
--allow_carrying=False \
--time_limit=200 \
--benchmarking_record_episode_interval=8 \
--benchmarking_interval=1.0e4 \
--train_observation_budget=2.0e6

#--train_observation_budget=300000 


