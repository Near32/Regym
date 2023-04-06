WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c benchmark_atari.py --seed=10 \
--config=montezuma_ppo_rnd_benchmark_config.yaml \
--nbr_max_random_steps=0 --grayscale=True \
--adam_eps=1.0e-5 --optimization_epochs=4 \
--nbr_actor=32 --mini_batch_size=1024 \
--min_capacity=2e3 --replay_capacity=5e3 --learning_rate=1.0e-4 \
--weights_entropy_lambda=0.0 --value_weight=0.5 --entropy_weight=0.001 \
--ppo_ratio_clip=0.1 --time_limit=18000 \
--discount=0.998 --intrinsic_discount=0.99 \
--adam_weight_decay=0.0 \
--single_life_episode=False \
--use_random_network_distillation=True \
--train_observation_budget=2.0e7

#--nbr_actor=128 --mini_batch_size=4096 --batch_size=4096 \


