#/bin/bash
#CUDA_LAUNCH_BLOCKING=1
#python benchmark_selfplay_s2b.py \
WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c benchmark_selfplay_s2b.py \
--success_threshold=0.5 \
--use_cuda=True --seed=30 --yaml_config=s2b_descr+feedback_comp_foc_1shot_N3m2M5_r2d2_lstm_benchmark_config.yaml \
--descriptive=True --max_nbr_values_per_latent=5 --min_nbr_values_per_latent=2 --sampling_strategy=component-focused-2shots \
--nbr_distractors=0 --nbr_latents=3 --nbr_object_centric_samples=4 --provide_listener_feedback=True \
--nbr_episode_per_cycle=32 --nbr_training_iteration_per_cycle=4 \
--min_handled_experiences=2e3 \
--nbr_minibatches=4 --batch_size=64 --learning_rate=6.25e-05 --critic_arch_feature_dim=256 \
--tau=None --inverted_tau=2500 --n_step=7 --nbr_actor=32 --replay_capacity=1e4 --min_capacity=1e3 \
--sequence_replay_burn_in_ratio=0.5 --sequence_replay_unroll_length=40 \
--r2d2_use_value_function_rescaling=False \
--use_ORG=False \
--use_rule_based_agent=True --use_speaker_rule_based_agent=True \
--node_id_to_extract=hidden \
--listener_comm_rec_adaptive_period=True --listener_rec_adaptive_period=True \
--listener_comm_rec_biasing=True --listener_comm_rec=True --listener_rec_biasing=True --listener_rec=True \
--listener_rec_period=2 --listener_comm_rec_period=2 --listener_rec_max_adaptive_period=1000 \
--listener_comm_rec_lambda=1.0e0 --listener_rec_lambda=1.0e0 --rec_threshold=0.02 \
--train_observation_budget=5.0e6 
