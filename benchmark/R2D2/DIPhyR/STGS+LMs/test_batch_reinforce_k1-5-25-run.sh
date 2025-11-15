python -m ipdb -c c batch_optimize_main.py  \
--model_name distilbert/distilgpt2  \
--dataset_path data/distilgpt2_diverse_targets_k1-5-25 \
--output_dir results/distilgpt2_reinforce+BS=32+LR=1e-1_test_k1-5-25-run \
--learning_rate=0.1  \
--epochs 2000 \
--model_precision full \
--gradient_checkpointing=False \
--losses=crossentropy \
--gradient_estimator=reinforce \
--reinforce_use_baseline=True \
--reinforce_baseline_beta=0.9 \
--reinforce_reward_scale=1.0 \
--reinforce_grad_variance_samples=10 \
--reinforce_grad_variance_period=1 \
--batch_size=32 \
--seed=10 \
--num_workers=1 \
--seq_len=80 \

#--losses=embxentropy \
