WANDB_CACHE_DIR=./wandb_cache/ xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m ipdb -c c benchmark_wandb_ether.py \
--seed=20 \
--project=ETHER \
--success_threshold=0.5 \
--use_cuda=True \
--config=room12x5Objs_miniworld_wandb_benchmark_OracleETHER+R2D2+RP+SharedObsEncoder_config.yaml \
--language_guided_curiosity=False \
--coverage_manipulation_metric=True \
--MiniWorld_entity_visibility_oracle=True \
--MiniWorld_entity_visibility_oracle_top_view=False \
--use_ETHER=True --use_THER=True \
--use_RP=False --RP_use_RP=True \
--use_ELA=False --ELA_use_ELA=True \
--use_HER=False --goal_oriented=False \
--ETHER_use_ETHER=True \
--THER_use_THER=True \
--THER_use_THER_predictor_supervised_training=False \
--THER_use_THER_predictor_supervised_training_data_collection=True \
--ETHER_with_Oracle=True \
--ETHER_rg_max_sentence_length=20 \
--ETHER_use_supervised_training=False \
--ETHER_rg_sanity_check_compactness_ambiguity_metric=False \
--ETHER_rg_shared_architecture=False \
--ETHER_rg_with_logits_mdl_principle=True \
--ETHER_rg_logits_mdl_principle_factor=1.0e-4 \
--ETHER_rg_logits_mdl_principle_accuracy_threshold=40.0 \
--ETHER_rg_agent_loss_type=Impatient+Hinge \
--ETHER_use_continuous_feedback=True \
--ETHER_rg_agent_nbr_latent_dim=1024 \
--ETHER_rg_normalize_features=False \
--ETHER_listener_based_predicated_reward_fn=True \
--ETHER_rg_with_semantic_grounding_metric=True --MiniWorld_symbolic_image=True \
--ETHER_rg_homoscedastic_multitasks_loss=False \
--semantic_embedding_init='none' \
--semantic_prior_mixing='multiplicative' \
--semantic_prior_mixing_with_detach=False \
--ETHER_rg_use_semantic_cooccurrence_grounding=False \
--ETHER_rg_semantic_cooccurrence_grounding_semantic_level=False \
--ETHER_rg_semantic_cooccurrence_grounding_semantic_level_ungrounding=False \
--ETHER_rg_semantic_cooccurrence_grounding_sentence_level=True \
--ETHER_rg_semantic_cooccurrence_grounding_sentence_level_ungrounding=False \
--ETHER_rg_semantic_cooccurrence_grounding_sentence_level_lambda=100.0 \
--ETHER_rg_semantic_cooccurrence_grounding_lambda=1.0 \
--ETHER_rg_semantic_cooccurrence_grounding_noise_magnitude=0.2 \
--ETHER_lock_test_storage=True --ETHER_rg_filter_out_non_unique=False\
--ETHER_rg_color_jitter_prob=0 \
--ETHER_rg_gaussian_blur_prob=0 \
--ETHER_rg_egocentric_prob=0 \
--ETHER_rg_egocentric_tr_degrees=30 --ETHER_rg_egocentric_tr_xy=10 \
--ETHER_rg_object_centric_version=2 --ETHER_rg_descriptive_version=1 \
--ETHER_rg_learning_rate=6.25e-5 --ETHER_rg_weight_decay=0.0 \
--ETHER_rg_l2_weight_decay=0.0 --ETHER_rg_l1_weight_decay=0.0 \
--ETHER_rg_vocab_size=64 --ETHER_rg_training_period=4096 \
--ETHER_rg_descriptive=True --ETHER_rg_use_curriculum_nbr_distractors=False \
--ETHER_rg_nbr_epoch_per_update=256 --ETHER_rg_accuracy_threshold=20 \
--ETHER_rg_nbr_train_distractors=15 --ETHER_rg_nbr_test_distractors=7 \
--ETHER_replay_capacity=4096 --ETHER_test_replay_capacity=1024 \
--ETHER_rg_distractor_sampling=uniform \
--ETHER_rg_use_cuda=True \
--ETHER_rg_metric_fast=True --ETHER_rg_parallel_TS_worker=8 \
--ETHER_rg_metric_epoch_period=8 --ETHER_rg_dis_metric_epoch_period=0 \
--ETHER_rg_metric_batch_size=32 \
--ETHER_rg_nbr_train_points=1024 --ETHER_rg_nbr_eval_points=256 \
--ETHER_rg_metric_resampling=False --ETHER_rg_dis_metric_resampling=False \
--ETHER_rg_metric_active_factors_only=True \
--RP_use_PER=True \
--RP_lock_test_storage=False \
--RP_predictor_learning_rate=6.25e-5 \
--RP_gradient_clip=5.0 \
--RP_replay_capacity=16384 --RP_min_capacity=32 \
--RP_predictor_nbr_minibatches=4 --RP_predictor_batch_size=256 \
--RP_predictor_test_train_split_interval=3 --RP_test_replay_capacity=1024 \
--RP_test_min_capacity=32 --RP_replay_period=1024 \
--RP_nbr_training_iteration_per_update=128 \
--RP_predictor_accuracy_threshold=90 \
--ELA_rg_sanity_check_compactness_ambiguity_metric=False \
--ELA_rg_shared_architecture=False \
--ELA_rg_with_logits_mdl_principle=True \
--ELA_rg_logits_mdl_principle_factor=1.0e-3 \
--ELA_rg_logits_mdl_principle_accuracy_threshold=10.0 \
--ELA_rg_agent_loss_type=Impatient+Hinge \
--ELA_rg_use_semantic_cooccurrence_grounding=False \
--ELA_rg_semantic_cooccurrence_grounding_lambda=1.0 \
--ELA_rg_semantic_cooccurrence_grounding_noise_magnitude=0.2 \
--ELA_lock_test_storage=True \
--ELA_rg_with_color_jitter_augmentation=False \
--ELA_rg_with_gaussian_blur_augmentation=True \
--ELA_rg_egocentric=False \
--ELA_rg_object_centric_version=2 --ELA_rg_descriptive_version=1 \
--ELA_rg_learning_rate=6.25e-5 --ELA_rg_weight_decay=0.0 \
--ELA_rg_vocab_size=64 --ELA_rg_training_period=4096 \
--ELA_rg_descriptive=False --ELA_rg_use_curriculum_nbr_distractors=False \
--ELA_rg_nbr_epoch_per_update=2 --ELA_rg_accuracy_threshold=90 \
--ELA_rg_nbr_train_distractors=7 --ELA_rg_nbr_test_distractors=7 \
--ELA_replay_capacity=8192 --ELA_test_replay_capacity=2048 \
--ELA_rg_distractor_sampling=uniform \
--ELA_reward_extrinsic_weight=0.0 --ELA_reward_intrinsic_weight=1.0 \
--ELA_feedbacks_failure_reward=-0.1 --ELA_feedbacks_success_reward=1 \
--THER_use_PER=True --THER_observe_achieved_goal=True \
--THER_lock_test_storage=True \
--THER_feedbacks_failure_reward=-1 --THER_feedbacks_success_reward=1 \
--THER_episode_length_reward_shaping=True \
--THER_replay_capacity=1024 --THER_min_capacity=12 \
--THER_predictor_nbr_minibatches=1 --THER_predictor_batch_size=32 \
--THER_predictor_test_train_split_interval=5 --THER_test_replay_capacity=256 \
--THER_test_min_capacity=4 --THER_replay_period=16384 \
--THER_train_on_success=False --THER_nbr_training_iteration_per_update=128 \
--THER_predict_PADs=False --THER_predictor_accuracy_threshold=0.95 \
--THER_predictor_accuracy_safe_to_relabel_threshold=0.0 --THER_filter_predicate_fn=False \
--THER_relabel_terminal=True --THER_filter_out_timed_out_episode=False \
--THER_train_contrastively=False --THER_contrastive_training_nbr_neg_examples=0 \
--THER_timing_out_episode_length_threshold=40 \
--BabyAI_Bot_action_override=False \
--n_step=3 --nbr_actor=32 --eps_greedy_alpha=2.0 \
--nbr_minibatches=1 --batch_size=64 \
--min_capacity=4e3 --min_handled_experiences=1.7e4 --replay_capacity=5e3 --learning_rate=6.25e-5 \
--sequence_replay_burn_in_ratio=0.5 --weights_entropy_lambda=0.0 \
--sequence_replay_unroll_length=20 --sequence_replay_overlap_length=10 \
--sequence_replay_use_online_states=True --sequence_replay_use_zero_initial_states=False \
--sequence_replay_store_on_terminal=False --HER_target_clamping=False \
--adam_weight_decay=0.0 --ther_adam_weight_decay=0.0 \
--nbr_training_iteration_per_cycle=1 --nbr_episode_per_cycle=0 \
--single_pick_episode=True \
--time_limit=40 \
--train_observation_budget=1.0e6

#--train_observation_budget=300000 
# WARNING: if THER_contrastive_training_nbr_neg_examples != 0 then THER_train_contrastively is toggled to True. 
# WARNING: THER_filter_out_timed_out_episode is only filtering out for relabelling but not the actual trajectory : it is going to make it to the ReplayBuffer.
