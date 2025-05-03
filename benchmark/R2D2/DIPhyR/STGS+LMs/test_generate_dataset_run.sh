python generate_dataset_with_perplexity.py \
--model_name distilbert/distilgpt2 \
--output_path data/distilgpt2_diverse_targets_k1-5-25 \
--k_min 1 \
--k_max 25 \
--k_step 5  \
--num_samples 5 \
--max_length 20 \
--noise_factor 1.0 
