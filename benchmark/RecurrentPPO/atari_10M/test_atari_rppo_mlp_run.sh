WANDB_CACHE_DIR=./wandb_cache/ xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m ipdb -c c benchmark_wandb_rppo.py --seed=10 \
--config=atari_wandb_benchmark_RPPO_MLP_config.yaml \
--nbr_max_random_steps=30 --grayscale=True \
--adam_eps=1.0e-5 --optimization_epochs=10 \
--standardized_adv=True \
--discount=0.99 --ppo_ratio_clip=0.1 \
--nbr_actor=8 --mini_batch_size=256 --batch_size=256 \
--learning_rate=2.5e-4 --gradient_clip=0.5 \
--entropy_weight=0.01 \
--sequence_replay_burn_in_ratio=0.0 --sequence_replay_unroll_length=1 \
--sequence_replay_overlap_length=0 \
--sequence_replay_use_online_states=False \
--sequence_replay_use_zero_initial_states=False \
--adam_weight_decay=0.0 \
--time_limit=40000 \
--use_random_network_distillation=False \
--train_observation_budget=1.0e7

