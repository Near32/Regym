WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c benchmark_wandb_ther.py --seed=10 \
--config=babyAI_wandb_benchmark_OracleTHER_config.yaml \
--n_step=3 --nbr_actor=32 --nbr_minibatches=1 --batch_size=64 \
--min_capacity=2e3 --replay_capacity=5e3 --learning_rate=1e-4 \
--sequence_replay_burn_in_ratio=0.0 --weights_entropy_lambda=0.0 \
--sequence_replay_unroll_length=20 --sequence_replay_overlap_length=10 \
--sequence_replay_use_online_states=True --sequence_replay_use_zero_initial_states=False \
--sequence_replay_store_on_terminal=False --HER_target_clamping=False \
--adam_weight_decay=0.0 --ther_adam_weight_decay=0.0 \
--THER_use_THER=True --THER_use_PER=True --THER_observe_achieved_goal=TRUe \
--THER_feedbacks_failure_reward=-1 --THER_feedbacks_success_reward=20 \
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
--train_observation_budget=2.0e7

#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_AgnosticTHER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=5e3 --replay_capacity=1e4 --learning_rate=3e-4 --weights_entropy_lambda=0.0 --THER_replay_capacity=1e3 --THER_min_capacity=32 --THER_test_replay_capacity=100 --THER_test_min_capacity=10 --THER_replay_period=4096 --THER_train_on_success=False --THER_nbr_training_iteration_per_update=128 --THER_predict_PADs=False --THER_predictor_accuracy_safe_to_relabel_threshold=0.5 --THER_relabel_terminal=False 
#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_AgnosticTHER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=5e3 --replay_capacity=1e4 --learning_rate=3e-4 --weights_entropy_lambda=0.0 --THER_replay_capacity=1e3 --THER_min_capacity=32 --THER_test_replay_capacity=100 --THER_test_min_capacity=10 --THER_replay_period=16384 --THER_train_on_success=False --THER_nbr_training_iteration_per_update=128 --THER_predict_PADs=False
#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_AgnosticTHER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=5e3 --replay_capacity=1e4 --learning_rate=3e-4 --weights_entropy_lambda=0.0 --THER_replay_capacity=1e3 --THER_min_capacity=256 --THER_test_replay_capacity=100 --THER_replay_period=128 --THER_nbr_training_iteration_per_update=16
#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_THER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=1e3 --replay_capacity=1e4 --learning_rate=6.25e-5 --weights_entropy_lambda=0.0 --THER_replay_period=1000 --THER_nbr_training_iteration_per_update=100

#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_THER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=1e3 --replay_capacity=1e4 --learning_rate=6.25e-5 --weights_entropy_lambda=0.0 
#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_THER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=1e3 --replay_capacity=2e4 --learning_rate=3e-4 --weights_entropy_lambda=0.0 
