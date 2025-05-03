import torch
import argparse
import random
import json
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_dataset")


def setup_model_and_tokenizer(model_name, device='cpu'):
    """
    Load model and tokenizer
    """
    logger.info(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add appropriate start and end of sentence tokens if they don't exist
    special_tokens = {}
    if tokenizer.bos_token is None:
        special_tokens['bos_token'] = '<s>'
    if tokenizer.eos_token is None:
        special_tokens['eos_token'] = '</s>'
    
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        # Resize model embeddings if new tokens were added
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def sample_k_skewed(k, noise_factor, vocab_size):
    """
    Sample a value k' that is centered around k but with some noise.
    
    Args:
        k: The central k value
        noise_factor: Controls the amount of deviation from k
        vocab_size: Maximum possible k value (vocabulary size)
    
    Returns:
        k': A sampled integer that tends to be close to k
    """
    # Use a truncated normal distribution centered at k
    if noise_factor <= 0:
        return k  # No noise
    
    # Calculate standard deviation based on noise factor and k
    std_dev = noise_factor #* k
    
    # Sample from normal distribution centered at k
    k_prime = int(round(np.random.normal(k, std_dev)))
    
    # Ensure 1 <= k_prime <= vocab_size
    k_prime = max(1, min(k_prime, vocab_size))
    
    return k_prime


def compute_perplexity(model, input_ids, attention_mask=None, device="cpu"):
    """
    Compute the perplexity of a sequence.
    
    Args:
        model: The language model
        input_ids: Tensor of token IDs
        attention_mask: Optional attention mask
        device: Device to run on
    
    Returns:
        perplexity: The perplexity score
    """
    with torch.no_grad():
        # Make sure input_ids is on the correct device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Get model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        
        # Extract loss
        loss = outputs.loss
        
        # Convert loss to perplexity (e^loss)
        perplexity = torch.exp(loss).item()
        
    return perplexity


def generate_sentence(model, tokenizer, k, noise_factor, max_length, device):
    """
    Generate a sentence by repeatedly sampling the k-th (approximately) most likely token.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        k: The rank of token to sample (1 = most likely, 2 = second most likely, etc.)
        noise_factor: How much to vary k during sampling (0 = no variation)
        max_length: Maximum number of tokens to generate
        device: Device to run the model on
    
    Returns:
        generated_text: The generated sentence as a string
        token_ranks: List of actual ranks used for each token
        token_probabilities: List of probabilities for each sampled token
    """
    # Start with BOS token
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    generated_tokens = [tokenizer.bos_token_id]
    token_ranks = []
    token_probabilities = []
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Convert logits to probabilities
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sort tokens by probability (descending)
            sorted_probs, sorted_indices = torch.sort(next_token_probs, descending=True)
            
            # Get actual vocabulary size (removing padding/special tokens if needed)
            effective_vocab_size = (next_token_probs > 0).sum().item()
            
            # Sample k' from distribution centered at k
            k_prime = sample_k_skewed(k, noise_factor, effective_vocab_size)
            
            # Ensure k_prime is within bounds
            k_prime = min(k_prime, effective_vocab_size)
            
            # Select the k'-th most likely token
            next_token = sorted_indices[0, k_prime - 1].unsqueeze(0).unsqueeze(0)
            prob_value = sorted_probs[0, k_prime - 1].item()
            
            # Store the actual rank used
            token_ranks.append(k_prime)
            token_probabilities.append(prob_value)
            
            # Append to generated tokens
            generated_tokens.append(next_token.item())
            
            # Check if EOS token was generated
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # Update input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    # Convert tokens to text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Calculate perplexity
    full_input_ids = torch.tensor([generated_tokens], device=device)
    perplexity = compute_perplexity(model, full_input_ids, device=device)
    
    return generated_text, token_ranks, token_probabilities, perplexity


def generate_dataset(model_name, output_path, k_values, num_samples, max_length, noise_factor, seed, wandb_project, wandb_entity=None):
    """
    Generate a dataset of sentences by sampling from the model with different k values.
    
    Args:
        model_name: Name of the model to use
        output_path: Path to save the dataset
        k_values: List of k values to use
        num_samples: Number of samples to generate for each k
        max_length: Maximum length of each generated sentence
        noise_factor: How much to vary k during sampling
        seed: Random seed for reproducibility
        wandb_project: W&B project name
        wandb_entity: W&B entity name
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize W&B
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            "model_name": model_name,
            "k_values": k_values,
            "num_samples": num_samples,
            "max_length": max_length,
            "noise_factor": noise_factor,
            "seed": seed
        }
    )
    
    # Initialize model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, device)
    
    # Prepare the dataset
    dataset = {
        "metadata": {
            "model_name": model_name,
            "k_values": k_values,
            "num_samples": num_samples,
            "max_length": max_length,
            "noise_factor": noise_factor,
            "seed": seed,
            "generation_timestamp": wandb.run.start_time
        },
        "samples": [],
        "k_summaries": {}
    }
    
    # Create a table to track samples
    columns = ["k_target", "k_actual_avg", "text", "length", "perplexity", "token_ranks", "token_probabilities"]
    samples_table = wandb.Table(columns=columns)
    
    # Dictionary to store metrics by k value
    k_metrics = {k: {"perplexities": [], "lengths": []} for k in k_values}
    
    # Generate samples for each k value
    for k in tqdm(k_values, desc="Generating samples for different k values"):
        for i in tqdm(range(num_samples), desc=f"Generating {num_samples} samples for k={k}", leave=False):
            text, ranks, probs, perplexity = generate_sentence(
                model, tokenizer, k, noise_factor, max_length, device
            )
            
            # Add to dataset
            sample = {
                "id": f"k{k}_sample{i}",
                "k_target": k,
                "text": text,
                "token_ranks": ranks,
                "token_probabilities": probs,
                "avg_rank": sum(ranks) / len(ranks) if ranks else 0,
                "length": len(ranks),
                "perplexity": perplexity
            }
            dataset["samples"].append(sample)
            
            # Add to metrics by k value
            k_metrics[k]["perplexities"].append(perplexity)
            k_metrics[k]["lengths"].append(len(ranks))
            
            # Add to W&B table
            samples_table.add_data(
                k,
                sample["avg_rank"],
                text,
                len(ranks),
                perplexity,
                ranks,
                probs
            )
            
            # Log progress
            if (i + 1) % 10 == 0:
                wandb.log({
                    "samples_generated": (k_values.index(k) * num_samples) + i + 1,
                    "current_k": k
                })
    
    # Compute summary metrics for each k value
    for k, metrics in k_metrics.items():
        perplexities = metrics["perplexities"]
        lengths = metrics["lengths"]
        
        dataset["k_summaries"][str(k)] = {
            "avg_perplexity": sum(perplexities) / len(perplexities) if perplexities else 0,
            "min_perplexity": min(perplexities) if perplexities else 0,
            "max_perplexity": max(perplexities) if perplexities else 0,
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "num_samples": len(perplexities)
        }
        
        # Log summary metrics to W&B
        wandb.log({
            f"k{k}/avg_perplexity": dataset["k_summaries"][str(k)]["avg_perplexity"],
            f"k{k}/avg_length": dataset["k_summaries"][str(k)]["avg_length"]
        })
    
    # Create summary table by k value
    k_summary_table = wandb.Table(columns=["k_value", "avg_perplexity", "min_perplexity", "max_perplexity", "avg_length"])
    for k, summary in dataset["k_summaries"].items():
        k_summary_table.add_data(
            int(k),
            summary["avg_perplexity"],
            summary["min_perplexity"],
            summary["max_perplexity"],
            summary["avg_length"]
        )
    
    # Log summary table
    wandb.log({"k_summaries": k_summary_table})
    
    # Save dataset to local file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"Dataset saved to {output_file}")
    
    # Log dataset to W&B
    wandb.log({"samples": samples_table})
    
    # Create artifact
    artifact = wandb.Artifact(
        name=f"target-sentences-dataset-{wandb.run.id}",
        type="dataset",
        description=f"Dataset of {len(k_values) * num_samples} target sentences generated with different k values"
    )
    artifact.add_file(output_file)
    wandb.log_artifact(artifact)
    
    # Finish W&B run
    wandb.finish()
    
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a dataset of target sentences by sampling from a language model")
    
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M",
                        help="Name of the language model to use")
    parser.add_argument("--output_path", type=str, default="target_sentences_dataset.json",
                        help="Path to save the generated dataset")
    parser.add_argument("--k_min", type=int, default=1,
                        help="Minimum k value (rank of token to sample)")
    parser.add_argument("--k_max", type=int, default=10,
                        help="Maximum k value (rank of token to sample)")
    parser.add_argument("--k_step", type=int, default=1,
                        help="Step size between k values")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate for each k value")
    parser.add_argument("--max_length", type=int, default=25,
                        help="Maximum length of each generated sentence")
    parser.add_argument("--noise_factor", type=float, default=0.3,
                        help="How much to vary k during sampling (0 = no variation)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--wandb_project", type=str, default="prompt-optimization-dataset",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity name")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    
    generate_dataset(
        model_name=args.model_name,
        output_path=args.output_path,
        k_values=k_values,
        num_samples=args.num_samples,
        max_length=args.max_length,
        noise_factor=args.noise_factor,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )
