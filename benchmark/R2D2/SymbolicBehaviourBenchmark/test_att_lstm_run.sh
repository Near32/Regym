#CUDA_VISIBLE_DEVICES=3  python -m ipdb -c c benchmark_selfplay_s2b.py s2b_r2d2_sad_vdn_benchmark_config.yaml --pubsub --pipelined --seed 0 --test_only --reload_path ./r2d2_s2b_debug/SymbolicBehaviourBenchmark-ReceptiveConstructiveTestEnv-v0/Train-Reward1/venv64/V6-MSL1-NCR3-L3Min2Max5-Distr3-CommPerm/3step_SAD_VDN_aID_1m3Ent_r2d2_AdamLR6d25m5_EPS1m12_L2AModelUpdate1Steps_EPSgreedyAPEX1m0_4m1OVER3p4_A2m0_gamma997_LargeMLPLSTM2Res_GradClip5m0_r1p5Min3e4_a9m1_b6m1_ovrN_e9m1_tau4m4_RepP1_NOBURNIN_b128_L20_O10_B0_NOZeroInitSt_OnlineSt_StoreOnDone/TRAINING/WithPosDisSpeakerRBAgent/PUBSUB/ListenerReconstruction+Biasing-1p0-BigArch/SEED0/3step_SAD_VDN_aID_1m3Ent_r2d2_AdamLR6d25m5_EPS1m12_L2AModelUpdate1Steps_EPSgreedyAPEX1m0_4m1OVER3p4_A2m0_gamma997_LargeMLPLSTM2Res_GradClip5m0_r1p5Min3e4_a9m1_b6m1_ovrN_e9m1_tau4m4_RepP1_NOBURNIN_b128_L20_O10_B0_NOZeroInitSt_OnlineSt_StoreOnDone.agent --speaker_rule_based_agent --path_suffix testreloadFromSEED0 --obs_budget 100000
python -m ipdb -c c benchmark_selfplay_s2b.py \
--batch_size=32 \
--config=s2b_descr+feedback_comp_foc_1shot_r2d2_att_lstm_sad_vdn_benchmark_config.yaml \
--critic_arch_feature_dim=256 \
--descriptive=True \
--learning_rate=6.25e-05 \
--listener_comm_rec=True --listener_multimodal_rec_biasing=True --listener_rec=True --listener_rec_period=2 --node_id_to_extract=cell \
--rec_threshold=0.02 \
--max_nbr_values_per_latent=3 \
--nbr_distractors=3 --nbr_latents=3 --nbr_object_centric_samples=4 --provide_listener_feedback=True \
--sampling_strategy=component-focused-2shots \
--n_step=3 --nbr_actor=32 \
--r2d2_use_value_function_rescaling=False \
--replay_capacity=20000 \
--sequence_replay_burn_in_ratio=0.5 --sequence_replay_unroll_length=20 --tau=0.0004 \
--seed=10 --train_observation_budget=5000000 \
--use_rule_based_agent=True --use_speaker_rule_based_agent=True --saving_interval=5e5

