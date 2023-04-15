WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c benchmark_wandb_ther.py \
--config=babyAI_wandb_benchmark_SharedFiLMedTHER_config.yaml \
--seed=30 \
--THER_contrastive_training_nbr_neg_examples=8 \
--THER_episode_length_reward_shaping=True \
--THER_filter_out_timed_out_episode=True \
--THER_filter_predicate_fn=True \
--THER_lock_test_storage=False \
--THER_min_capacity=16 \
--THER_nbr_training_iteration_per_update=32 \
--THER_observe_achieved_goal=False \
--THER_feedbacks_failure_reward=0 --THER_feedbacks_success_reward=1 \
--THER_predict_PADs=False \
--THER_predictor_accuracy_safe_to_relabel_threshold=0.2 \
--THER_predictor_accuracy_threshold=0.95 \
--THER_predictor_nbr_minibatches=4 --THER_predictor_batch_size=128 \
--THER_predictor_test_train_split_interval=5 \
--THER_relabel_terminal=False \
--THER_replay_capacity=1000 \
--THER_replay_period=256 \
--THER_test_min_capacity=64 \
--THER_test_replay_capacity=256 \
--THER_timing_out_episode_length_threshold=40 \
--THER_train_contrastively=True \
--THER_train_on_success=False \
--THER_use_PER=True --THER_use_THER=True \
--learning_rate=6.25e-05 --min_capacity=5000 \
--nbr_minibatches=4 --batch_size=128 \
--adam_weight_decay=0 \
--ther_adam_weight_decay=0 \
--tau=0.0004 \
--weights_entropy_lambda=0 \
--n_step=3 --nbr_actor=32 \
--replay_capacity=10000 \
--r2d2_use_value_function_rescaling=False \
--sequence_replay_burn_in_ratio=0.5 \
--sequence_replay_overlap_length=10 \
--sequence_replay_unroll_length=20 \
--single_pick_episode=True \
--train_observation_budget=200000 \


