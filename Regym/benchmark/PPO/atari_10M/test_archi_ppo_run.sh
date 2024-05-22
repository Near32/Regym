WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c benchmark_atari.py --seed=10 \
--config=atari_10M_archi_ppo_benchmark_config.yaml \
--nbr_max_random_steps=30 --grayscale=True \
--adam_eps=1.0e-5 --optimization_epochs=4 \
--nbr_actor=8 --mini_batch_size=256 --batch_size=256 \
--min_capacity=2e3 --replay_capacity=5e3 --learning_rate=2.5e-4 \
--weights_entropy_lambda=0.0 \
--adam_weight_decay=0.0 \
--use_random_network_distillation=False \
--train_observation_budget=2.0e7

