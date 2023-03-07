WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c benchmark_wandb_ether.py --seed=10 \
--config=babyAI_wandb_benchmark_ETHER_config.yaml \
--n_step=3 --nbr_actor=32 --nbr_minibatches=1 --batch_size=64 \
--min_capacity=2e3 --replay_capacity=5e3 --learning_rate=6.25e-5 \
--sequence_replay_burn_in_ratio=0.0 --weights_entropy_lambda=0.0 \
--sequence_replay_unroll_length=20 --sequence_replay_overlap_length=10 \
--sequence_replay_use_online_states=True --sequence_replay_use_zero_initial_states=False \
--sequence_replay_store_on_terminal=False --HER_target_clamping=False \
--adam_weight_decay=0.0 --ther_adam_weight_decay=0.0 \
--nbr_training_iteration_per_cycle=40 --nbr_episode_per_cycle=16 \
--ETHER_use_ETHER=True --THER_use_THER=True \
--ETHER_rg_learning_rate=3.0e-4 --ETHER_rg_weight_decay=1.0e-3 \
--ETHER_rg_vocab_size=64 --ETHER_rg_training_period=4096 \
--ETHER_rg_descriptive=False --ETHER_rg_use_curriculum_nbr_distractors=False \
--ETHER_rg_nbr_epoch_per_update=64 --ETHER_rg_accuracy_threshold=75 \
--ETHER_rg_nbr_train_distractors=63 --ETHER_rg_nbr_test_distractors=3 \
--ETHER_train_dataset_length=1024 --ETHER_test_dataset_length=512 \
--THER_use_PER=True --THER_observe_achieved_goal=False \
--THER_feedbacks_failure_reward=0 --THER_feedbacks_success_reward=1 \
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
--single_pick_episode=False --THER_timing_out_episode_length_threshold=40 \
--train_observation_budget=2.0e7

#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_AgnosticTHER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=5e3 --replay_capacity=1e4 --learning_rate=3e-4 --weights_entropy_lambda=0.0 --THER_replay_capacity=1e3 --THER_min_capacity=32 --THER_test_replay_capacity=100 --THER_test_min_capacity=10 --THER_replay_period=4096 --THER_train_on_success=False --THER_nbr_training_iteration_per_update=128 --THER_predict_PADs=False --THER_predictor_accuracy_safe_to_relabel_threshold=0.5 --THER_relabel_terminal=False 
#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_AgnosticTHER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=5e3 --replay_capacity=1e4 --learning_rate=3e-4 --weights_entropy_lambda=0.0 --THER_replay_capacity=1e3 --THER_min_capacity=32 --THER_test_replay_capacity=100 --THER_test_min_capacity=10 --THER_replay_period=16384 --THER_train_on_success=False --THER_nbr_training_iteration_per_update=128 --THER_predict_PADs=False
#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_AgnosticTHER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=5e3 --replay_capacity=1e4 --learning_rate=3e-4 --weights_entropy_lambda=0.0 --THER_replay_capacity=1e3 --THER_min_capacity=256 --THER_test_replay_capacity=100 --THER_replay_period=128 --THER_nbr_training_iteration_per_update=16
#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_THER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=1e3 --replay_capacity=1e4 --learning_rate=6.25e-5 --weights_entropy_lambda=0.0 --THER_replay_period=1000 --THER_nbr_training_iteration_per_update=100

#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_THER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=1e3 --replay_capacity=1e4 --learning_rate=6.25e-5 --weights_entropy_lambda=0.0 
#python -m ipdb -c c benchmark_wandb_ther.py --config=babyAI_wandb_benchmark_THER_config.yaml --n_step=1 --nbr_actor=32 --batch_size=32 --min_capacity=1e3 --replay_capacity=2e4 --learning_rate=3e-4 --weights_entropy_lambda=0.0 
