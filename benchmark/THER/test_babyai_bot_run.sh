WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c benchmark_wandb_ther.py --seed=10 \
--config=babyAI_wandb_benchmark_OracleTHER_config.yaml \
--n_step=3 --nbr_actor=32 --nbr_minibatches=4 --batch_size=64 \
--min_capacity=1e3 --replay_capacity=2e3 --tau=1.0e-3 --learning_rate=6.25e-5 \
--sequence_replay_burn_in_ratio=0.0 --weights_entropy_lambda=0.0 \
--adam_weight_decay=0.0 --ther_adam_weight_decay=0.0 \
--THER_use_THER=True --THER_use_PER=True --THER_observe_achieved_goal=True \
--THER_feedbacks_failure_reward=0 --THER_feedbacks_success_reward=1 \
--THER_episode_length_reward_shaping=True \
--THER_replay_capacity=1e2 --THER_min_capacity=12 \
--THER_predictor_nbr_minibatches=1 --THER_predictor_batch_size=32 \
--THER_predictor_test_train_split_interval=5 --THER_test_replay_capacity=1e2 \
--THER_test_min_capacity=4 --THER_replay_period=1024 \
--THER_train_on_success=False --THER_nbr_training_iteration_per_update=1 \
--THER_predict_PADs=False --THER_predictor_accuracy_threshold=0.95 \
--THER_predictor_accuracy_safe_to_relabel_threshold=0.2 --THER_filter_predicate_fn=False \
--THER_relabel_terminal=False --THER_filter_out_timed_out_episode=True \
--THER_train_contrastively=False --THER_contrastive_training_nbr_neg_examples=0 \
--single_pick_episode=False --THER_timing_out_episode_length_threshold=40 \
--BabyAI_Bot_action_override=True \
--train_observation_budget=2.0e7


