#/bin/bash
#python -m ipdb -c c benchmark_selfplay_s2b.py --batch_size=32 --config=s2b_descr+feedback_comp_foc_1shot_r2d2_largelstm_sad_vdn_benchmark_config.yaml --critic_arch_feature_dim=256 --descriptive=True --learning_rate=6.25e-05 --listener_comm_rec=True --listener_multimodal_rec_biasing=True --listener_rec=True --listener_rec_period=10 --max_nbr_values_per_latent=5 --n_step=3 --nbr_actor=32 --nbr_distractors=0 --nbr_latents=3 --nbr_object_centric_samples=4 --provide_listener_feedback=True --r2d2_use_value_function_rescaling=False --rec_threshold=0.02 --sampling_strategy=component-focused-2shots --seed=10 --sequence_replay_burn_in_ratio=0.5 --sequence_replay_unroll_length=20 --tau=0.0004 --train_observation_budget=5000000 --use_rule_based_agent=True --use_speaker_rule_based_agent=True --replay_capacity=2e4
python -m ipdb -c c benchmark_selfplay_s2b.py --batch_size=32 --config=s2b_descr+feedback_comp_foc_1shot_r2d2_dcesbn_sad_vdn_benchmark_config.yaml --critic_arch_feature_dim=256 --descriptive=True --learning_rate=6.25e-05 --listener_comm_rec=True --listener_multimodal_rec_biasing=True --listener_rec=True --listener_rec_period=10 --max_nbr_values_per_latent=3 --n_step=3 --nbr_actor=32 --nbr_distractors=0 --nbr_latents=3 --nbr_object_centric_samples=4 --provide_listener_feedback=True --r2d2_use_value_function_rescaling=False --rec_threshold=0.02 --sampling_strategy=component-focused-2shots --seed=10 --sequence_replay_burn_in_ratio=0.5 --sequence_replay_unroll_length=20 --tau=0.0004 --train_observation_budget=5000000 --use_rule_based_agent=True --use_speaker_rule_based_agent=True --replay_capacity=2e4
