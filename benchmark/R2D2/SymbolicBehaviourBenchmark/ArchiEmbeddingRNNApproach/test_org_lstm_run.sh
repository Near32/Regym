#/bin/bash
#CUDA_LAUNCH_BLOCKING=1
#python benchmark_selfplay_s2b.py \
WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c benchmark_selfplay_s2b.py \
--success_threshold=0.5 \
--use_cuda=True \
--seed=30 \
--saving_interval=1e20 \
--yaml_config=s2b_descr+feedback_comp_foc_1shot_r2d2_org_lstm_benchmark_config.yaml \
--descriptive=True \
--max_nbr_values_per_latent=5 \
--min_nbr_values_per_latent=2 \
--sampling_strategy=component-focused-2shots \
--nbr_distractors=0 \
--nbr_latents=3 \
--nbr_object_centric_samples=1 \
--provide_listener_feedback=True \
--sad=True --vdn=True \
--use_ORG=True \
--ORG_use_ORG=True \
--ORG_with_Oracle=False \
--ORG_with_Oracle_listener=False \
--ORG_rg_use_aita_sampling=False \
--ORG_rg_aita_update_epoch_period=256 \
--ORG_rg_aita_levenshtein_comprange=1.0 \
--ORG_rg_max_sentence_length=20 \
--ORG_use_supervised_training=False \
--ORG_with_compactness_ambiguity_metric=False \
--ORG_rg_sanity_check_compactness_ambiguity_metric=False \
--ORG_rg_reset_listener_each_training=True \
--ORG_rg_shared_architecture=False \
--ORG_rg_with_logits_mdl_principle=True \
--ORG_rg_logits_mdl_principle_factor=1.0e-4 \
--ORG_rg_logits_mdl_principle_accuracy_threshold=40.0 \
--ORG_rg_agent_loss_type=Impatient+Hinge \
--ORG_rg_agent_nbr_latent_dim=1024 \
--ORG_rg_normalize_features=False \
--ORG_rg_with_semantic_grounding_metric=False \
--ORG_rg_homoscedastic_multitasks_loss=False \
--ORG_rg_use_semantic_cooccurrence_grounding=False \
--ORG_rg_semantic_cooccurrence_grounding_semantic_level=False \
--ORG_rg_semantic_cooccurrence_grounding_semantic_level_ungrounding=False \
--ORG_rg_semantic_cooccurrence_grounding_sentence_level=True \
--ORG_rg_semantic_cooccurrence_grounding_sentence_level_ungrounding=False \
--ORG_rg_semantic_cooccurrence_grounding_sentence_level_lambda=100.0 \
--ORG_rg_semantic_cooccurrence_grounding_lambda=1.0 \
--ORG_rg_semantic_cooccurrence_grounding_noise_magnitude=0.2 \
--ORG_rg_color_jitter_prob=0 \
--ORG_rg_gaussian_blur_prob=0 \
--ORG_rg_egocentric_prob=0 \
--ORG_rg_egocentric_tr_degrees=30 \
--ORG_rg_egocentric_tr_xy=10 \
--ORG_rg_object_centric_version=2 \
--ORG_rg_distractor_sampling_scheme_version=2 \
--ORG_rg_descriptive_version=1 \
--ORG_rg_learning_rate=6.25e-5 --ORG_rg_weight_decay=0.0 \
--ORG_rg_l2_weight_decay=0.0 --ORG_rg_l1_weight_decay=0.0 \
--ORG_rg_vocab_size=64 --ORG_rg_training_period=256 \
--ORG_rg_descriptive=True --ORG_rg_object_centric=True \
--ORG_rg_use_curriculum_nbr_distractors=False \
--ORG_rg_batch_size=32 \
--ORG_rg_nbr_epoch_per_update=256 --ORG_rg_accuracy_threshold=101 \
--ORG_rg_nbr_train_distractors=3 --ORG_rg_nbr_test_distractors=0 \
--ORG_rg_distractor_sampling=uniform \
--ORG_rg_distractor_sampling_with_replacement=True \
--ORG_rg_use_cuda=True \
--ORG_rg_metric_fast=True --ORG_rg_parallel_TS_worker=8 \
--ORG_rg_metric_epoch_period=32 --ORG_rg_dis_metric_epoch_period=0 \
--ORG_rg_metric_batch_size=32 \
--ORG_rg_nbr_train_points=256 --ORG_rg_nbr_eval_points=128 \
--ORG_rg_metric_resampling=False --ORG_rg_dis_metric_resampling=False \
--ORG_rg_metric_active_factors_only=True \
--nbr_episode_per_cycle=32 \
--nbr_training_iteration_per_cycle=4 \
--min_handled_experiences=1e1 \
--nbr_minibatches=4 \
--batch_size=64 \
--learning_rate=6.25e-05 \
--critic_arch_feature_dim=256 \
--tau=None \
--inverted_tau=2500 \
--n_step=7 \
--nbr_actor=32 \
--replay_capacity=1e4 \
--min_capacity=1e1 \
--sequence_replay_burn_in_ratio=0.5 \
--sequence_replay_unroll_length=40 \
--r2d2_use_value_function_rescaling=False \
--use_rule_based_agent=False \
--use_speaker_rule_based_agent=False \
--node_id_to_extract=hidden \
--speaker_rec_adaptive_period=True \
--speaker_rec_biasing=True \
--speaker_rec=True \
--speaker_rec_period=2 \
--speaker_rec_max_adaptive_period=1000 \
--speaker_rec_lambda=1.0e0 \
--listener_comm_rec_adaptive_period=True \
--listener_rec_adaptive_period=True \
--listener_comm_rec_biasing=True \
--listener_comm_rec=True \
--listener_rec_biasing=True \
--listener_rec=True \
--listener_rec_period=2 \
--listener_comm_rec_period=2 \
--listener_rec_max_adaptive_period=1000 \
--listener_comm_rec_lambda=1.0e0 \
--listener_rec_lambda=1.0e0 \
--rec_threshold=0.02 \
--train_observation_budget=1.0e7 

