python -m ipdb -c c batch_optimize_main.py  \
--model_name distilbert/distilgpt2  \
--dataset_path data/distilgpt2_diverse_targets_k1-5-25 \
--output_dir results/distilgpt2_STGS+Soft+LearnTau+SoftBPTT+LearnBTau+BS=32+LR=1e-1_test_k1-5-25-run \
--learning_rate=0.1  \
--epochs 1000 \
--model_precision full \
--gradient_checkpointing=False \
--losses=crossentropy \
--gradient_estimator=stgs \
--stgs_grad_variance_samples=10 \
--stgs_grad_variance_period=20 \
--stgs_grad_bias_samples=10 \
--stgs_grad_bias_period=50 \
--stgs_grad_bias_reference_samples=10 \
--stgs_grad_bias_reference_batch_size=64 \
--stgs_grad_bias_reference_use_baseline=True \
--stgs_grad_bias_reference_reward_scale=1.0 \
--stgs_grad_bias_reference_baseline_beta=0.9 \
--batch_size=64 \
--seed=2 \
--num_workers=1 \
--seq_len=80 \
--stgs_hard=False \
--learnable_temperature=True \
--decouple_learnable_temperature=True \
--temperature=10.0 \
--bptt=True \
--bptt_stgs_hard=False \
--bptt_learnable_temperature=True \
--bptt_hidden_state_conditioning=False \
--bptt_temperature=100.0 
#--target_indices="0,1" 

#--losses=embxentropy \
