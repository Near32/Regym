#/bin/bash
#python benchmark_selfplay_s2b.py \
CUDA_LAUNCH_BLOCKING=1 \
WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c benchmark_diphyr.py \
--DIPhyR_average_window_length=128 \
--success_threshold=0.5 \
--use_cuda=True \
--seed=40 \
--saving_interval=1e20 \
--yaml_config=diphyr_benchmark_minimal_cot_HFTGI_OpenAI-GPT-4o-mini_config.yaml \
--use_grammar=False \
--nbr_episode_per_cycle=32 \
--nbr_training_iteration_per_cycle=4 \
--min_handled_experiences=1e1 \
--nbr_minibatches=4 \
--batch_size=64 \
--learning_rate=6.25e-05 \
--tau=None \
--inverted_tau=2500 \
--n_step=0 \
--nbr_actor=1 \
--replay_capacity=1e1 \
--min_capacity=1e2 \
--sequence_replay_burn_in_ratio=0.0 \
--sequence_replay_unroll_length=2 \
--r2d2_use_value_function_rescaling=False \
--train_observation_budget=2.0e3 

