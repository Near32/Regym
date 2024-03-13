WANDB_CACHE_DIR=./wandb_cache/ xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m ipdb -c c ../benchmark_wandb_erelela.py \
--seed=50 --env_seed=12 --static_envs=False \
--with_early_stopping=False \
--use_cuda=True \
--project=EReLELA-MultiRoom-Benchmark \
--success_threshold=0.01 \
--config=multiroom_N7_S4_minigrid_wandb_benchmark_POMDPERELELA_config.yaml \
--language_guided_curiosity=False \
--language_guided_curiosity_descr_type='descr' \
--language_guided_curiosity_extrinsic_weight=10.0 \
--language_guided_curiosity_intrinsic_weight=0.1 \
--language_guided_curiosity_binary_reward=False \
--language_guided_curiosity_densify=False \
--language_guided_curiosity_non_episodic_dampening_rate=0.0 \
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
--use_ELA=True --ELA_use_ELA=True \
--use_HER=False --goal_oriented=False \
--ETHER_use_ETHER=False --THER_use_THER=False \
--ELA_with_rg_training=True \
--ELA_rg_use_cuda=True \
--ELA_rg_graphtype='straight_through_gumbel_softmax' \
--ELA_rg_obverter_threshold_to_stop_message_generation=0.9 \
--ELA_rg_obverter_nbr_games_per_round=32 \
--ELA_rg_obverter_sampling_round_alternation_only=False --ELA_rg_use_obverter_sampling=False \
--ELA_rg_compactness_ambiguity_metric_language_specs=emergent+natural+color+shape+shuffled-emergent+shuffled-natural+shuffled-color+shuffled-shape \
--ELA_rg_sanity_check_compactness_ambiguity_metric=False \
--ELA_rg_shared_architecture=True \
--ELA_rg_with_logits_mdl_principle=False \
--ELA_rg_logits_mdl_principle_factor=1.0e-3 \
--ELA_rg_logits_mdl_principle_accuracy_threshold=60.0 \
--ELA_rg_agent_loss_type=Impatient+Hinge \
--ELA_rg_use_semantic_cooccurrence_grounding=False \
--ELA_rg_semantic_cooccurrence_grounding_lambda=1.0 \
--ELA_rg_semantic_cooccurrence_grounding_noise_magnitude=0.2 \
--ELA_lock_test_storage=False \
--ELA_rg_color_jitter_prob=0.0 \
--ELA_rg_gaussian_blur_prob=0.5 \
--ELA_rg_egocentric_prob=0.0 \
--ELA_rg_object_centric_version=2 --ELA_rg_descriptive_version=1 \
--ELA_rg_learning_rate=3e-4 --ELA_rg_weight_decay=0.0 \
--ELA_rg_l1_weight_decay=0.0 --ELA_rg_l2_weight_decay=0.0 \
--ELA_rg_vocab_size=64 --ELA_rg_max_sentence_length=128 \
--ELA_rg_training_period=32768 \
--ELA_rg_descriptive=True --ELA_rg_use_curriculum_nbr_distractors=False \
--ELA_rg_nbr_epoch_per_update=32 --ELA_rg_accuracy_threshold=65 \
--ELA_rg_nbr_train_distractors=3 --ELA_rg_nbr_test_distractors=3 \
--ELA_replay_capacity=8192 --ELA_test_replay_capacity=2048 \
--ELA_rg_distractor_sampling=uniform \
--ELA_reward_extrinsic_weight=10.0 --ELA_reward_intrinsic_weight=0.1 \
--ELA_feedbacks_failure_reward=0.0 --ELA_feedbacks_success_reward=1 \
--BabyAI_Bot_action_override=False \
--n_step=3 --nbr_actor=32 \
--epsstart=1.0 --epsend=0.1 \
--epsdecay=100000 --eps_greedy_alpha=2.0 \
--nbr_minibatches=1 --batch_size=64 \
--min_capacity=4e3 --min_handled_experiences=28e3 --replay_capacity=20e3 --learning_rate=6.25e-5 \
--sequence_replay_burn_in_ratio=0.5 --weights_entropy_lambda=0.0 \
--sequence_replay_unroll_length=20 --sequence_replay_overlap_length=10 \
--sequence_replay_use_online_states=True --sequence_replay_use_zero_initial_states=False \
--sequence_replay_store_on_terminal=False --HER_target_clamping=False \
--adam_weight_decay=0.0 --ther_adam_weight_decay=0.0 \
--nbr_training_iteration_per_cycle=2 --nbr_episode_per_cycle=0 \
--single_pick_episode=False \
--terminate_on_completion=True \
--allow_carrying=False \
--time_limit=0 \
--benchmarking_record_episode_interval=4 \
--benchmarking_interval=1.0e4 \
--train_observation_budget=1.0e6

#--train_observation_budget=300000 
#--project=EReLELA-MultiRoom-ELA-Test \


