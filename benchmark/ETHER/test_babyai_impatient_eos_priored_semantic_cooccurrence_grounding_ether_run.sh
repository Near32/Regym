WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c benchmark_wandb_ether.py --seed=10 \
--config=babyAI_wandb_benchmark_ETHER_config.yaml \
--n_step=3 --nbr_actor=8 --eps_greedy_alpha=2.0 \
--nbr_minibatches=1 --batch_size=64 \
--min_capacity=5e3 --replay_capacity=6e3 --learning_rate=6.25e-5 \
--sequence_replay_burn_in_ratio=0.5 --weights_entropy_lambda=0.0 \
--sequence_replay_unroll_length=20 --sequence_replay_overlap_length=10 \
--sequence_replay_use_online_states=True --sequence_replay_use_zero_initial_states=False \
--sequence_replay_store_on_terminal=False --HER_target_clamping=False \
--adam_weight_decay=0.0 --ther_adam_weight_decay=0.0 \
--nbr_training_iteration_per_cycle=40 --nbr_episode_per_cycle=16 \
--ETHER_use_ETHER=True --THER_use_THER=True \
--ETHER_rg_shared_architecture=False \
--ETHER_rg_with_logits_mdl_principle=True \
--ETHER_rg_logits_mdl_principle_factor=1.0e-5 \
--ETHER_rg_logits_mdl_principle_accuracy_threshold=10.0 \
--ETHER_rg_agent_loss_type=Impatient+Hinge \
--ETHER_use_supervised_training=True \
--ETHER_use_continuous_feedback=False \
--ETHER_listener_based_predicated_reward_fn=True \
--ETHER_rg_use_semantic_cooccurrence_grounding=True \
--ETHER_rg_semantic_cooccurrence_grounding_lambda=1.0 \
--ETHER_rg_semantic_cooccurrence_grounding_noise_magnitude=0.2 \
--ETHER_lock_test_storage=True --ETHER_rg_filter_out_non_unique=False\
--ETHER_rg_with_color_jitter_augmentation=False --ETHER_rg_with_gaussian_blur_augmentation=True \
--ETHER_rg_egocentric=False \
--ETHER_rg_object_centric_version=2 --ETHER_rg_descriptive_version=1 \
--ETHER_rg_learning_rate=3e-4 --ETHER_rg_weight_decay=0.0 \
--ETHER_rg_vocab_size=64 --ETHER_rg_training_period=4096 \
--ETHER_rg_descriptive=False --ETHER_rg_use_curriculum_nbr_distractors=False \
--ETHER_rg_nbr_epoch_per_update=32 --ETHER_rg_accuracy_threshold=99 \
--ETHER_rg_nbr_train_distractors=7 --ETHER_rg_nbr_test_distractors=7 \
--ETHER_replay_capacity=1024 --ETHER_test_replay_capacity=512 \
--ETHER_rg_distractor_sampling=uniform \
--THER_use_PER=True --THER_observe_achieved_goal=False \
--THER_lock_test_storage=True \
--THER_feedbacks_failure_reward=-1 --THER_feedbacks_success_reward=1 \
--THER_episode_length_reward_shaping=True \
--THER_replay_capacity=1e2 --THER_min_capacity=12 \
--THER_predictor_nbr_minibatches=1 --THER_predictor_batch_size=32 \
--THER_predictor_test_train_split_interval=5 --THER_test_replay_capacity=1e2 \
--THER_test_min_capacity=4 --THER_replay_period=1024 \
--THER_train_on_success=False --THER_nbr_training_iteration_per_update=128 \
--THER_predict_PADs=False --THER_predictor_accuracy_threshold=0.95 \
--THER_predictor_accuracy_safe_to_relabel_threshold=0.2 --THER_filter_predicate_fn=True \
--THER_relabel_terminal=False --THER_filter_out_timed_out_episode=False \
--THER_train_contrastively=False --THER_contrastive_training_nbr_neg_examples=0 \
--single_pick_episode=True --THER_timing_out_episode_length_threshold=40 \
--BabyAI_Bot_action_override=False \
--train_observation_budget=200000

#--ETHER_train_dataset_length=1024 --ETHER_test_dataset_length=512 \
#3072
#PYOPENGL_PLATFORM=egl 

