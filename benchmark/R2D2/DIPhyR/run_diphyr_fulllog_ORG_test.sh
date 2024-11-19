#/bin/bash
#python benchmark_selfplay_s2b.py \
CUDA_LAUNCH_BLOCKING=1 \
WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c benchmark_diphyr.py \
--multi_gpu_strategy='none' \
--use_ORG=True \
--ORG_use_supervised_training=True \
--ORG_rg_agent_loss_type="Hinge" \
--ORG_rg_nbr_train_distractors=4 \
--ORG_rg_nbr_test_distractors=4 \
--ORG_rg_batch_size=1 \
--ORG_rg_learning_rate=6.25e-5 \
--ORG_rg_weight_decay=0.01 \
--ORG_replay_capacity=128 \
--ORG_test_replay_capacity=32 \
--ORG_rg_training_period=64 \
--ORG_rg_optimizer_type='adamW8bit' \
--ORG_rg_optimizer_gradient_accumulation_steps=16 \
--DIPhyR_average_window_length=128 \
--success_threshold=0.5 \
--use_cuda=True \
--seed=10 \
--saving_interval=1e20 \
--yaml_config=diphyr_benchmark_fulllog_tr_test_config.yaml \
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
--train_observation_budget=2.0e4 

#--ORG_rg_graphtype="straight-through-gumbel-softmax" \

