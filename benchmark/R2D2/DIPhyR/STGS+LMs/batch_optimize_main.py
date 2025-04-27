import torch
import argparse
import numpy as np
import wandb
import logging
from pathlib import Path

# Import modularized utilities
from main import setup_model_and_tokenizer, TokenOverlapMetric
from data_loader import load_dataset, prepare_targets
from metrics_utils import aggregate_metrics_by_k, compute_auc_metrics, compute_overall_metrics, log_metrics_summary
from logging_utils import (
    create_summary_table, create_k_summary_table, log_k_metrics_to_wandb,
    update_k_summary_table, log_tables_to_wandb, log_auc_results_to_wandb,
    log_overall_metrics_to_wandb, save_results_to_file, create_and_log_artifact
)
from batch_processing import process_targets_sequential, process_targets_parallel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("batch_optimize")


def batch_optimize(dataset_path, model_name, output_dir, config, num_workers=1, target_indices=None):
    """
    Optimize prompts for multiple target sentences in parallel.
    
    Args:
        dataset_path: Path to the dataset file
        model_name: Name of the language model to use
        output_dir: Directory to save results
        config: Dictionary of optimization parameters
        num_workers: Number of parallel workers (if 1, runs sequentially)
        target_indices: Optional list of indices to select specific targets
        
    Returns:
        results: Dictionary mapping target IDs to optimization results
    """
    # Create run ID for grouping
    run_id = wandb.util.generate_id()
    logger.info(f"Starting batch optimization with run ID: {run_id}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset and prepare targets
    dataset = load_dataset(dataset_path)
    targets = dataset["samples"]
    if target_indices is not None:
        targets = [targets[i] for i in target_indices if i < len(targets)]
        logger.info(f"Selected {len(targets)} targets based on provided indices")
    
    # Initialize model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, device, model_precision=config.get("model_precision", "full"))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    

    # Create summary tables
    summary_table = create_summary_table()
    k_summary_table = create_k_summary_table()
    
    # Initialize dictionary to store metrics by k value
    k_values = sorted(set(target["k_target"] for target in targets))
    k_metrics = {k: {
        "perplexities": [],
        "exact_matches": [],
        "token_accuracies": [],
        "final_losses": [],
        "token_overlap_ratios": [],
        "target_hit_ratios": [],
        "lcs_ratios": [],
        "unigram_overlaps": [],
        "bigram_overlaps": []
    } for k in k_values}
    
    # Process targets based on number of workers
    if num_workers == 1:
        results = process_targets_sequential(
            targets, model, tokenizer, device, config, run_id, output_dir, 
            summary_table, k_metrics
        )
    else:
        results = process_targets_parallel(
            targets, model, tokenizer, config, run_id, output_dir, 
            summary_table, k_metrics, num_workers
        )
    
    # Initialize W&B run for the batch
    batch_run = wandb.init(
        project=config["wandb_project"],
        entity=config.get("wandb_entity"),
        name=f"batch_optimization_{run_id}",
        group=run_id,
        job_type="batch_coordination",
        config={
            **config,
            "dataset_path": dataset_path,
            "model_name": model_name,
            "num_targets": len(targets),
            "metadata": dataset.get("metadata", {})
        }
    ) 
    # Calculate aggregate metrics by k value
    k_aggregated = {}
    for k, metrics in k_metrics.items():
        num_samples = len(metrics["exact_matches"])
        if num_samples == 0:
            continue
            
        k_aggregated[k] = {
            "num_samples": num_samples,
            "success_rate": sum(metrics["exact_matches"]) / num_samples,
            "avg_token_accuracy": sum(metrics["token_accuracies"]) / num_samples,
            "avg_final_loss": sum(metrics["final_losses"]) / num_samples,
            "avg_perplexity": sum(metrics["perplexities"]) / num_samples if metrics["perplexities"] else 0,
            "avg_token_overlap_ratio": sum(metrics["token_overlap_ratios"]) / num_samples,
            "avg_target_hit_ratio": sum(metrics["target_hit_ratios"]) / num_samples,
            "avg_lcs_ratio": sum(metrics["lcs_ratios"]) / num_samples,
            "avg_unigram_overlap": sum(metrics["unigram_overlaps"]) / num_samples,
            "avg_bigram_overlap": sum(metrics["bigram_overlaps"]) / num_samples
        }
        
        # Add to k summary table
        update_k_summary_table(k_summary_table, k, k_aggregated[k])
        
        # Log to W&B
        log_k_metrics_to_wandb(k, k_aggregated[k], batch_run)
    
    # Log summary tables to W&B
    log_tables_to_wandb(summary_table, k_summary_table, batch_run)
    
    # Compute AUC metrics
    auc_results = compute_auc_metrics(k_metrics)
    
    # Log AUC results to W&B
    log_auc_results_to_wandb(auc_results, batch_run)
    
    # Calculate overall metrics
    all_exact_matches = [result["evaluation"]["exact_match"] for result in results.values()]
    all_token_accuracies = [result["evaluation"]["token_accuracy"] for result in results.values()]
    all_lcs_ratios = [result["evaluation"]["lcs_ratio"] for result in results.values()]
    all_unigram_overlaps = [result["evaluation"]["unigram_overlap"] for result in results.values()]
    all_bigram_overlaps = [result["evaluation"]["bigram_overlap"] for result in results.values()]
    
    # Calculate overall averages
    overall_metrics = {
        "success_rate": sum(all_exact_matches) / len(all_exact_matches) if all_exact_matches else 0,
        "avg_token_accuracy": sum(all_token_accuracies) / len(all_token_accuracies) if all_token_accuracies else 0,
        "avg_lcs_ratio": sum(all_lcs_ratios) / len(all_lcs_ratios) if all_lcs_ratios else 0,
        "avg_unigram_overlap": sum(all_unigram_overlaps) / len(all_unigram_overlaps) if all_unigram_overlaps else 0,
        "avg_bigram_overlap": sum(all_bigram_overlaps) / len(all_bigram_overlaps) if all_bigram_overlaps else 0,
        "num_samples": len(results)
    }
    
    # Add token metrics if available
    token_overlap_ratios = []
    target_hit_ratios = []
    for k, m in k_metrics.items():
        if m["token_overlap_ratios"]:
            token_overlap_ratios.extend(m["token_overlap_ratios"])
        if m["target_hit_ratios"]:
            target_hit_ratios.extend(m["target_hit_ratios"])
    
    if token_overlap_ratios:
        overall_metrics["avg_token_overlap_ratio"] = sum(token_overlap_ratios) / len(token_overlap_ratios)
    if target_hit_ratios:
        overall_metrics["avg_target_hit_ratio"] = sum(target_hit_ratios) / len(target_hit_ratios)
    
    # Log overall metrics to W&B
    log_overall_metrics_to_wandb(overall_metrics, batch_run)
    
    # Save results to file
    save_results_to_file(
        output_path=output_path,
        run_id=run_id,
        results=results,
        config=config,
        k_summaries={str(k): metrics for k, metrics in k_aggregated.items()},
        overall_metrics=overall_metrics,
        auc_results=auc_results,
        dataset_metadata=dataset.get("metadata", {})
    )
    
    # Create and log artifact
    create_and_log_artifact(batch_run, output_path, run_id)
    
    # Print summary to console
    log_metrics_summary(overall_metrics, logger)

    batch_run.finish() 
    # Return results
    return results


def str2bool(instr):
    """Convert string to boolean."""
    if isinstance(instr, bool):
        return instr
    if isinstance(instr, str):
        instr = instr.lower()
        if 'true' in instr:
            return True
        elif 'false' in instr:
            return False
        else:
            raise ValueError(f"Cannot convert '{instr}' to boolean")
    else:
        raise TypeError(f"Expected str or bool, got {type(instr)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch optimize prompts for multiple target sentences")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset file containing target sentences")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M",
                        help="Name of the language model to use")
    parser.add_argument("--model_precision", type=str, default="full",
                        help="Precision of the language model to use", choices=["full", "half"])
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False,
                        help="Whether to use gradient checkpointing")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="batch_optimization_results",
                        help="Directory to save optimization results")
    
    # Optimization parameters
    parser.add_argument("--losses", type=str, default="crossentropy",
                        help="Loss function(s) to use for optimization")
    parser.add_argument("--seq_len", type=int, default=40,
                        help="Length of the prompt sequence to optimize")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Number of optimization epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                        help="Learning rate for optimization")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for optimization")
    
    # ST-GS parameters
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for Gumbel-Softmax")
    parser.add_argument("--learnable_temperature", type=str2bool, default="False",
                        help="Whether to learn the temperature parameter")
    parser.add_argument("--stgs_hard", type=str2bool, default="False",
                        help="Whether to use hard ST-GS")
    parser.add_argument("--eps", type=float, default=1e-10,
                        help="Epsilon value for numerical stability")
    
    # BPTT parameters
    parser.add_argument("--bptt", type=str2bool, default="False",
                        help="Whether to use backpropagation through time")
    parser.add_argument("--bptt_temperature", type=float, default=1.0,
                        help="Temperature for BPTT Gumbel-Softmax")
    parser.add_argument("--bptt_learnable_temperature", type=str2bool, default="False",
                        help="Whether to learn the BPTT temperature parameter")
    parser.add_argument("--bptt_stgs_hard", type=str2bool, default="False",
                        help="Whether to use hard ST-GS for BPTT")
    parser.add_argument("--bptt_hidden_state_conditioning", type=str2bool, default="False",
                        help="Whether to condition BPTT on hidden states")
    parser.add_argument("--bptt_eps", type=float, default=1e-10,
                        help="Epsilon value for BPTT numerical stability")
    
    # Vocabulary parameters
    parser.add_argument("--filter_vocab", type=str2bool, default="False",
                        help="Whether to filter the vocabulary")
    parser.add_argument("--vocab_threshold", type=float, default=0.5,
                        help="Threshold for vocabulary filtering")
    
    # Other parameters
    parser.add_argument("--max_gradient_norm", type=float, default=0.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--plot_every", type=int, default=10000,
                        help="Frequency of plotting loss curves")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Parallelization parameters
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers (1 = sequential)")
    
    # Target selection parameters
    parser.add_argument("--target_indices", type=str, default=None,
                        help="Comma-separated list of target indices to optimize (e.g., '0,1,5')")
    
    # W&B parameters
    parser.add_argument("--wandb_project", type=str, default="prompt-optimization-batch",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity name")
    
    # Perplexity parameters
    parser.add_argument("--promptLambda", type=float, default=0.0,
                        help="Weight for prompt perplexity loss")
    parser.add_argument("--complLambda", type=float, default=0.0,
                        help="Weight for completion perplexity loss")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse target indices if provided
    target_indices = None
    if args.target_indices:
        target_indices = [int(idx) for idx in args.target_indices.split(",")]
    
    # Update losses with perplexity components if needed
    if args.promptLambda > 0.0:
        args.losses += '+promptPerplexity'
    if args.complLambda > 0.0:
        args.losses += '+completionPerplexity'
    
    # Prepare configuration dictionary
    config = vars(args)
    
    # Run batch optimization
    batch_optimize(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        config=config,
        num_workers=args.num_workers,
        target_indices=target_indices
    )
