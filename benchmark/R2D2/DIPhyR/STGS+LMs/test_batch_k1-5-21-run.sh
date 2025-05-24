#!/bin/bash
python -m ipdb -c c batch_optimize_main.py \
--model_name distilbert/distilgpt2  \
--dataset_path data/distilgpt2_diverse_targets_k1-5-21.json \
--output_dir results/distilgpt2_test_k1-5-21-run  \
--metric_groups "basic,semantic,distribution" \
--sentencebert_model "all-MiniLM-L6-v2" \
--bertscore_model "distilbert-base-uncased" \
--learning_rate=0.1  \
--epochs 2000 \
--model_precision full \
--gradient_checkpointing=False \
--losses=embxentropy \
--batch_size=1 \
--seed=30 \
--num_workers=1 \
--seq_len=80 \
--learnable_temperature=True \
--temperature=1000

