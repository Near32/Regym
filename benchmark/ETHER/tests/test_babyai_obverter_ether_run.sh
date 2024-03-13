WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c ../benchmark_wandb_ether.py \
--seed=10 \
--project=ETHER-Obverter \
--success_threshold=0.001 \
--use_cuda=True \
--config=../BabyAI/babyAI_wandb_benchmark_obverter_ETHER_config.yaml \
--use_ETHER=True --use_THER=True \
--use_RP=False --RP_use_RP=False \
--use_ELA=False --ELA_use_ELA=False \
--use_HER=False --goal_oriented=False \
--HER_strategy=final-1 \
--HER_target_clamping=True \
--ETHER_use_ETHER=True \
--THER_use_THER=True \
--ETHER_use_supervised_training=False \
--THER_use_THER_predictor_supervised_training=False \
--THER_use_THER_predictor_supervised_training_data_collection=True \
--ETHER_with_Oracle=False \
--ETHER_with_Oracle_type='goal-only' \
--ETHER_with_Oracle_listener=False \
--ETHER_rg_use_aita_sampling=False \
--ETHER_rg_aita_update_epoch_period=8 \
--ETHER_rg_aita_levenshtein_comprange=1.0 \
--ETHER_rg_max_sentence_length=20 \
--ETHER_rg_sanity_check_compactness_ambiguity_metric=False \
--ETHER_rg_shared_architecture=False \
--ETHER_rg_with_logits_mdl_principle=False \
--ETHER_rg_logits_mdl_principle_factor=1.0e-5 \
--ETHER_rg_logits_mdl_principle_accuracy_threshold=10.0 \
--ETHER_rg_agent_loss_type="NLL" \
--ETHER_use_continuous_feedback=False \
--ETHER_rg_agent_nbr_latent_dim=1024 \
--ETHER_rg_normalize_features=False \
--ETHER_listener_based_predicated_reward_fn=True \
--ETHER_rg_with_semantic_grounding_metric=False --MiniWorld_symbolic_image=False \
--ETHER_rg_homoscedastic_multitasks_loss=False \
--semantic_embedding_init='none' \
--semantic_prior_mixing='multiplicative' \
--semantic_prior_mixing_with_detach=False \
--ETHER_rg_use_semantic_cooccurrence_grounding=False \
--ETHER_rg_semantic_cooccurrence_grounding_semantic_level=False \
--ETHER_rg_semantic_cooccurrence_grounding_semantic_level_ungrounding=False \
--ETHER_rg_semantic_cooccurrence_grounding_sentence_level=True \
--ETHER_rg_semantic_cooccurrence_grounding_sentence_level_ungrounding=False \
--ETHER_rg_semantic_cooccurrence_grounding_lambda=100.0 \
--ETHER_rg_semantic_cooccurrence_grounding_noise_magnitude=0.2 \
--ETHER_lock_test_storage=True --ETHER_rg_filter_out_non_unique=False\
--ETHER_rg_color_jitter_prob=0.0 \
--ETHER_rg_gaussian_blur_prob=0.5 \
--ETHER_rg_egocentric_prob=0.0 \
--ETHER_rg_egocentric_tr_degrees=30 --ETHER_rg_egocentric_tr_xy=10 \
--ETHER_rg_object_centric_version=2 \
--ETHER_rg_distractor_sampling_scheme_version=2 \
--ETHER_rg_descriptive_version=2 \
--ETHER_rg_learning_rate=3.0e-4 --ETHER_rg_weight_decay=0.0 \
--ETHER_rg_l2_weight_decay=0.0 --ETHER_rg_l1_weight_decay=0.0 \
--ETHER_rg_vocab_size=64 --ETHER_rg_training_period=512 \
--ETHER_rg_graphtype="obverter" \
--ETHER_rg_use_obverter_sampling=False \
--ETHER_rg_obverter_sampling_round_alternation_only=True \
--ETHER_rg_obverter_sampling_repeat_experiences=False \
--ETHER_rg_obverter_nbr_games_per_round=8 \
--ETHER_rg_obverter_BN_in_decision_head=False \
--ETHER_rg_obverter_DP_in_decision_head=0.0 \
--ETHER_rg_obverter_threshold_to_stop_message_generation=0.75 \
--ETHER_rg_descriptive=True --ETHER_rg_object_centric=False \
--ETHER_rg_use_curriculum_nbr_distractors=False \
--ETHER_rg_nbr_epoch_per_update=64 --ETHER_rg_accuracy_threshold=99 \
--ETHER_rg_nbr_train_distractors=0 --ETHER_rg_nbr_test_distractors=0 \
--ETHER_replay_capacity=1024 --ETHER_test_replay_capacity=1024 \
--ETHER_rg_distractor_sampling="episodic-dissimilarity" \
--ETHER_rg_use_cuda=True \
--ETHER_rg_metric_fast=True --ETHER_rg_parallel_TS_worker=8 \
--ETHER_rg_metric_epoch_period=8 --ETHER_rg_dis_metric_epoch_period=8 \
--ETHER_rg_metric_batch_size=32 \
--ETHER_rg_nbr_train_points=2048 --ETHER_rg_nbr_eval_points=1024 \
--ETHER_rg_metric_resampling=False --ETHER_rg_dis_metric_resampling=False \
--ETHER_rg_metric_active_factors_only=True \
--THER_use_PER=True --THER_describe_achieved_goal=False \
--THER_lock_test_storage=True \
--THER_feedbacks_failure_reward=-1 --THER_feedbacks_success_reward=1 \
--THER_episode_length_reward_shaping=True \
--THER_replay_capacity=1e2 --THER_min_capacity=12 \
--THER_predictor_nbr_minibatches=1 --THER_predictor_batch_size=32 \
--THER_predictor_test_train_split_interval=5 --THER_test_replay_capacity=1e2 \
--THER_test_min_capacity=4 --THER_replay_period=1024 \
--THER_train_on_success=False --THER_nbr_training_iteration_per_update=128 \
--THER_predict_PADs=False --THER_predictor_accuracy_threshold=0.95 \
--THER_predictor_accuracy_safe_to_relabel_threshold=0.2 --THER_filter_predicate_fn=False \
--THER_relabel_terminal=False --THER_filter_out_timed_out_episode=False \
--THER_train_contrastively=False --THER_contrastive_training_nbr_neg_examples=0 \
--BabyAI_Bot_action_override=False \
--n_step=3 --nbr_actor=8 --eps_greedy_alpha=2.0 \
--nbr_minibatches=1 --batch_size=64 \
--min_capacity=4e3 --min_handled_experiences=4e3 --replay_capacity=5e3 --learning_rate=6.25e-5 \
--sequence_replay_burn_in_ratio=0.5 --weights_entropy_lambda=0.0 \
--sequence_replay_unroll_length=20 --sequence_replay_overlap_length=10 \
--sequence_replay_use_online_states=True --sequence_replay_use_zero_initial_states=False \
--sequence_replay_store_on_terminal=False \
--adam_weight_decay=0.0 --ther_adam_weight_decay=0.0 \
--nbr_training_iteration_per_cycle=40 --nbr_episode_per_cycle=16 \
--single_pick_episode=False --THER_timing_out_episode_length_threshold=40 \
--time_limit=40 \
--train_observation_budget=200000


