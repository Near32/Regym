WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_AgnosticFiLMedTHER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=5e3 --replay_capacity=10e3 --learning_rate=6.25e-5 --weights_entropy_lambda=0.0 --adam_weight_decay=0.0 --ther_adam_weight_decay=0.0 --THER_use_THER=True --THER_use_PER=True --THER_episode_length_reward_shaping=True --THER_replay_capacity=1e3 --THER_min_capacity=12 --THER_predictor_test_train_split_interval=5 --THER_test_replay_capacity=256 --THER_test_min_capacity=10 --THER_replay_period=2048 --THER_train_on_success=False --THER_nbr_training_iteration_per_update=128 --THER_predict_PADs=False --THER_predictor_accuracy_threshold=0.95 --THER_predictor_accuracy_safe_to_relabel_threshold=0.2 --THER_filter_predicate_fn=True --THER_relabel_terminal=False --THER_filter_out_timed_out_episode=True --THER_timing_out_episode_length_threshold=40

#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_AgnosticTHER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=5e3 --replay_capacity=1e4 --learning_rate=3e-4 --weights_entropy_lambda=0.0 --THER_replay_capacity=1e3 --THER_min_capacity=32 --THER_test_replay_capacity=100 --THER_test_min_capacity=10 --THER_replay_period=4096 --THER_train_on_success=False --THER_nbr_training_iteration_per_update=128 --THER_predict_PADs=False --THER_predictor_accuracy_safe_to_relabel_threshold=0.5 --THER_relabel_terminal=False 
#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_AgnosticTHER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=5e3 --replay_capacity=1e4 --learning_rate=3e-4 --weights_entropy_lambda=0.0 --THER_replay_capacity=1e3 --THER_min_capacity=32 --THER_test_replay_capacity=100 --THER_test_min_capacity=10 --THER_replay_period=16384 --THER_train_on_success=False --THER_nbr_training_iteration_per_update=128 --THER_predict_PADs=False
#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_AgnosticTHER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=5e3 --replay_capacity=1e4 --learning_rate=3e-4 --weights_entropy_lambda=0.0 --THER_replay_capacity=1e3 --THER_min_capacity=256 --THER_test_replay_capacity=100 --THER_replay_period=128 --THER_nbr_training_iteration_per_update=16
#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_THER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=1e3 --replay_capacity=1e4 --learning_rate=6.25e-5 --weights_entropy_lambda=0.0 --THER_replay_period=1000 --THER_nbr_training_iteration_per_update=100

#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_THER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=1e3 --replay_capacity=1e4 --learning_rate=6.25e-5 --weights_entropy_lambda=0.0 
#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_THER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=1e3 --replay_capacity=2e4 --learning_rate=3e-4 --weights_entropy_lambda=0.0 
