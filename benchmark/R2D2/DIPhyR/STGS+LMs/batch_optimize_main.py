"""
Main entry point for batch optimization of prompts for target sentences.
"""
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from typing import Dict, List, Any, Optional
import wandb 

from transformers import AutoModelForCausalLM, AutoTokenizer

# Import custom modules
from metrics_registry import compute_all_metrics
from metrics_aggregator import MetricsAggregator
from metrics_logging import MetricsLogger
from evaluation_utils import evaluate_generated_output
from main import optimize_inputs

logger = logging.getLogger("batch_optimize")
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def setup_model_and_tokenizer(model_name: str, device: str = 'cpu', model_precision: str = 'full'):
    """
    Load model and tokenizer
    
    Args:
        model_name: HuggingFace model name
        device: Device to run on ('cpu' or 'cuda')
        model_precision: Precision for model ('full' or 'half')
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load model and move to device
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode (frozen)

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    
    # Convert to half precision if requested
    if model_precision == "half":
        model.half()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Load dataset from a JSON file.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        Dataset dictionary
    """
    import json
    
    logger.info(f"Loading dataset from {dataset_path}")
    
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded {len(dataset.get('samples', []))} samples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def prepare_targets(dataset: Dict[str, Any], target_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Prepare the list of targets, optionally filtering by indices.
    
    Args:
        dataset: The loaded dataset
        target_indices: Optional list of indices to select specific targets
        
    Returns:
        List of prepared target dictionaries
    """
    targets = dataset.get("samples", [])
    
    # Filter targets if indices are provided
    if target_indices is not None:
        targets = [targets[i] for i in target_indices if i < len(targets)]
        logger.info(f"Selected {len(targets)} targets based on provided indices")
    
    return targets

def optimize_for_target(target_info: Dict[str, Any], model, tokenizer, device: str, 
                       config: Dict[str, Any], run_id: str, output_dir: str, metric_groups: List[str] = None) -> Dict[str, Any]:
    """
    Optimize a prompt for a specific target sentence.
    
    Args:
        target_info: Dictionary containing target sentence information
        model: The language model
        tokenizer: The tokenizer
        device: The device to run optimization on
        config: Dictionary of optimization parameters
        run_id: Unique identifier for the W&B run
        output_dir: Directory to save results locally
        metric_groups: List of metric groups to compute
        
    Returns:
        result: Dictionary containing optimization results
    """
    
    target_id = target_info["id"]
    target_text = target_info["text"]
    k_target = target_info["k_target"]
    
    # Get pre-computed perplexity from dataset if available
    target_perplexity = target_info.get("perplexity", None)
    
    logger.info(f"Optimizing prompt for target {target_id}: '{target_text}'")
    
    # Create a run name that includes the target information
    run_name = f"{run_id}_target{target_id}_k{k_target}"
    
    # Initialize W&B for this target
    target_config = config.copy()
    target_config.update({
        "target_id": target_id,
        "target_text": target_text,
        "target_k": k_target,
        "target_avg_rank": target_info.get("avg_rank", 0),
        "target_length": target_info.get("length", len(target_text.split())),
        "target_perplexity": target_perplexity
    })
    
    # Setup metrics logger
    metrics_logger = MetricsLogger(
        run_id=f"{run_id}_{target_id}",
        output_dir=f"{output_dir}/target_{target_id}",
        wandb_project=config.get("wandb_project"),
        wandb_entity=config.get("wandb_entity")
    )
    
    metrics_logger.init_wandb(
        config=target_config,
        name=run_name,
        group=run_id,
        job_type="single_target_optimization"
    )
    
    # Tokenize target for later comparison
    target_tokens = tokenizer(target_text, return_tensors="pt").input_ids[0].cpu().tolist()
    
    # Optimize inputs for this target
    generated_tokens, optimized_inputs, losses = optimize_inputs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_text=target_text,
        losses=config["losses"],
        bptt=config.get("bptt", False),
        seq_len=config["seq_len"],
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        temperature=config.get("temperature", 1.0),
        bptt_temperature=config.get("bptt_temperature", 1.0),
        learnable_temperature=config.get("learnable_temperature", False),
        decouple_learnable_temperature=config.get("decouple_learnable_temperature", False),
        bptt_learnable_temperature=config.get("bptt_learnable_temperature", False),
        stgs_hard=config.get("stgs_hard", True),
        bptt_stgs_hard=config.get("bptt_stgs_hard", True),
        bptt_hidden_state_conditioning=config.get("bptt_hidden_state_conditioning", False),
        plot_every=config.get("plot_every", 100),
        stgs_grad_variance_samples=config.get("stgs_grad_variance_samples", 0),
        stgs_grad_variance_period=config.get("stgs_grad_variance_period", 1),
        stgs_grad_bias_samples=config.get("stgs_grad_bias_samples", 0),
        stgs_grad_bias_period=config.get("stgs_grad_bias_period", 1),
        stgs_grad_bias_reference_samples=config.get("stgs_grad_bias_reference_samples", 0),
        stgs_grad_bias_reference_batch_size=config.get("stgs_grad_bias_reference_batch_size", 0),
        stgs_grad_bias_reference_use_baseline=config.get("stgs_grad_bias_reference_use_baseline", True),
        stgs_grad_bias_reference_reward_scale=config.get("stgs_grad_bias_reference_reward_scale", 1.0),
        stgs_grad_bias_reference_baseline_beta=config.get("stgs_grad_bias_reference_baseline_beta", 0.9),
        reinforce_grad_variance_samples=config.get("reinforce_grad_variance_samples", 0),
        reinforce_grad_variance_period=config.get("reinforce_grad_variance_period", 1),
        gradient_estimator=config.get("gradient_estimator", "stgs"),
        reinforce_reward_scale=config.get("reinforce_reward_scale", 1.0),
        reinforce_use_baseline=config.get("reinforce_use_baseline", True),
        reinforce_baseline_beta=config.get("reinforce_baseline_beta", 0.9),
        eps=config.get("eps", 1e-10),
        bptt_eps=config.get("bptt_eps", 1e-10),
        vocab_threshold=config.get("vocab_threshold", 0.5),
        filter_vocab=config.get("filter_vocab", True),
        max_gradient_norm=config.get("max_gradient_norm", 0.0),
        batch_size=config.get("batch_size", 1),
        kwargs=config,
    )
    
    # Extract the optimized prompt tokens
    optimized_tokens = torch.argmax(optimized_inputs[0], dim=-1).cpu().tolist()
    optimized_text = tokenizer.decode(optimized_tokens)
    
    # Evaluate the generated output using our centralized metrics system
    evaluation_metrics = evaluate_generated_output(
        generated_tokens=generated_tokens, 
        target_tokens=target_tokens, 
        tokenizer=tokenizer,
        metric_groups=metric_groups,
        device=device
    )
    
    # Log evaluation metrics to W&B
    metrics_logger.log_metrics(evaluation_metrics)
    
    # Create result dictionary
    result = {
        "target_id": target_id,
        "target_text": target_text,
        "target_k": k_target,
        "target_perplexity": target_perplexity,
        "optimized_tokens": optimized_tokens,
        "optimized_text": optimized_text,
        "generated_tokens": generated_tokens,
        "generated_text": evaluation_metrics["generated_text"],
        "evaluation": evaluation_metrics,
        "final_loss": float(losses[-1]),
        "loss_history": [float(loss) for loss in losses]
    }
    
    # Save results to file
    metrics_logger.save_to_file(result, "result.json")
    
    # Save the tensor for future use
    target_output_dir = Path(f"{output_dir}/target_{target_id}")
    target_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(optimized_inputs, target_output_dir / "optimized_inputs.pt")
    
    # Finish logging
    metrics_logger.finish()
    
    logger.info(f"Optimization completed for target {target_id}")
    logger.info(f"Optimized prompt: '{optimized_text}'")
    logger.info(f"Final loss: {result['final_loss']}")
    logger.info(f"Exact match: {evaluation_metrics.get('exact_match', 0)}")
    logger.info(f"Token accuracy: {evaluation_metrics.get('token_accuracy', 0):.4f}")
    
    return result

def process_targets_sequential(targets: List[Dict[str, Any]], model, tokenizer, device: str, 
                              config: Dict[str, Any], run_id: str, output_dir: str, 
                              metrics_aggregator: MetricsAggregator,
                              metric_groups: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Process targets sequentially.
    
    Args:
        targets: List of target info dictionaries
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        config: Configuration dictionary
        run_id: Unique run identifier
        output_dir: Directory to save results
        metrics_aggregator: Metrics aggregator
        metric_groups: List of metric groups to compute
        
    Returns:
        results: Dictionary mapping target IDs to optimization results
    """
    results = {}
    
    for target in tqdm(targets, desc="Optimizing for targets"):
        result = optimize_for_target(
            target_info=target,
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            run_id=run_id,
            output_dir=output_dir,
            metric_groups=metric_groups
        )
        results[target["id"]] = result
        
        # Add metrics to aggregator
        metrics_aggregator.add_sample(result["evaluation"], k_value=target["k_target"])
    
    return results

def process_targets_parallel(targets: List[Dict[str, Any]], model, tokenizer, config: Dict[str, Any], 
                           run_id: str, output_dir: str, metrics_aggregator: MetricsAggregator,
                           num_workers: int, metric_groups: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Process targets in parallel using a ProcessPoolExecutor.
    
    Args:
        targets: List of target info dictionaries
        model: Language model
        tokenizer: Tokenizer
        config: Configuration dictionary
        run_id: Unique run identifier
        output_dir: Directory to save results
        metrics_aggregator: Metrics aggregator
        num_workers: Number of parallel workers
        metric_groups: List of metric groups to compute
        
    Returns:
        results: Dictionary mapping target IDs to optimization results
    """
    results = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_target = {
            executor.submit(
                optimize_for_target,
                target_info=target,
                model=model,
                tokenizer=tokenizer,
                device="cuda" if torch.cuda.is_available() else "cpu",
                config=config,
                run_id=run_id,
                output_dir=output_dir,
                metric_groups=metric_groups
            ): target for target in targets
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_target), 
                         total=len(targets), 
                         desc="Optimizing for targets"):
            target = future_to_target[future]
            try:
                result = future.result()
                results[target["id"]] = result
                
                # Add metrics to aggregator
                metrics_aggregator.add_sample(result["evaluation"], k_value=target["k_target"])

            except Exception as exc:
                logger.error(f"Target {target['id']} generated an exception: {exc}")
    
    return results

def batch_optimize(dataset_path: str, model_name: str, output_dir: str, 
                  config: Dict[str, Any], num_workers: int = 1, 
                  target_indices: Optional[List[int]] = None,
                  metric_groups: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Optimize prompts for multiple target sentences.
    
    Args:
        dataset_path: Path to the dataset file
        model_name: Name of the language model to use
        output_dir: Directory to save results
        config: Dictionary of optimization parameters
        num_workers: Number of parallel workers (if 1, runs sequentially)
        target_indices: Optional list of indices to select specific targets
        metric_groups: List of metric groups to compute
        
    Returns:
        summary: Dictionary with optimization summary
    """
    # Create run ID for grouping
    run_id = wandb.util.generate_id()
    logger.info(f"Starting batch optimization with run ID: {run_id}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset and prepare targets
    dataset = load_dataset(dataset_path)
    targets = prepare_targets(dataset, target_indices)
    
    # Initialize model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name, 
        device, 
        model_precision=config.get("model_precision", "full")
    )
    
    # Create output directory
    full_output_dir = f"{output_dir}/{run_id}"
    output_path = Path(full_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize the metrics aggregator
    metrics_aggregator = MetricsAggregator()
    
    # Process targets based on number of workers
    if num_workers == 1:
        results = process_targets_sequential(
            targets=targets, 
            model=model, 
            tokenizer=tokenizer, 
            device=device, 
            config=config, 
            run_id=run_id, 
            output_dir=full_output_dir, 
            metrics_aggregator=metrics_aggregator,
            metric_groups=metric_groups
        )
    else:
        results = process_targets_parallel(
            targets=targets, 
            model=model, 
            tokenizer=tokenizer, 
            config=config, 
            run_id=run_id, 
            output_dir=full_output_dir, 
            metrics_aggregator=metrics_aggregator,
            num_workers=num_workers,
            metric_groups=metric_groups
        )
    
    # Initialize the metrics logger
    metrics_logger = MetricsLogger(
        run_id=run_id,
        output_dir=full_output_dir,
        wandb_project=config.get("wandb_project"),
        wandb_entity=config.get("wandb_entity")
    )
    
    # Initialize W&B for batch coordination
    metrics_logger.init_wandb(
        group=run_id,
        config={
            **config,
            "dataset_path": dataset_path,
            "model_name": model_name,
            "num_targets": len(targets),
            "metadata": dataset.get("metadata", {})
        },
        name=f"batch_optimization_{run_id}",
        job_type="batch_coordination"
    )
    
    # Get the complete summary
    summary = metrics_aggregator.get_summary()
    
    # Log the summary
    metrics_logger.log_summary(summary)
    
    # Add results to summary
    summary["results"] = results
    
    # Save complete results
    metrics_logger.save_to_file(
        data={
            "run_id": run_id,
            "summary": summary,
            "config": config,
            "dataset_metadata": dataset.get("metadata", {})
        },
        filename="batch_results.json"
    )
    
    # Finish logging
    metrics_logger.finish()
    
    return summary

def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(v):
    """Convert comma-separated string to list."""
    if v is None:
        return None
    return [item.strip() for item in v.split(',')]

def parse_args():
    """Parse command-line arguments."""
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
    parser.add_argument("--learnable_temperature", type=str2bool, default=False,
                        help="Whether to learn the temperature parameter")
    parser.add_argument("--decouple_learnable_temperature", type=str2bool, default=False,
                        help="Whether to learn multiple decouple temperature parameters, one for each learnable input.")
    parser.add_argument("--stgs_hard", type=str2bool, default=True,
                        help="Whether to use hard ST-GS")
    parser.add_argument("--eps", type=float, default=1e-10,
                        help="Epsilon value for numerical stability")
    
    # Gradient estimator parameters
    parser.add_argument("--gradient_estimator", type=str, default="stgs", choices=["stgs", "reinforce"],
                        help="Gradient estimator to use for optimizing the discrete inputs")
    parser.add_argument("--reinforce_reward_scale", type=float, default=1.0,
                        help="Scaling factor applied to the REINFORCE policy loss")
    parser.add_argument("--reinforce_use_baseline", type=str2bool, default=True,
                        help="Enable running baseline to reduce REINFORCE variance")
    parser.add_argument("--reinforce_baseline_beta", type=float, default=0.9,
                        help="Exponential moving average coefficient for the REINFORCE baseline")
    
    # BPTT parameters
    parser.add_argument("--bptt", type=str2bool, default=False,
                        help="Whether to use backpropagation through time")
    parser.add_argument("--bptt_temperature", type=float, default=1.0,
                        help="Temperature for BPTT Gumbel-Softmax")
    parser.add_argument("--bptt_learnable_temperature", type=str2bool, default=False,
                        help="Whether to learn the BPTT temperature parameter")
    parser.add_argument("--bptt_stgs_hard", type=str2bool, default=False,
                        help="Whether to use hard ST-GS for BPTT")
    parser.add_argument("--bptt_hidden_state_conditioning", type=str2bool, default=False,
                        help="Whether to condition BPTT on hidden states")
    parser.add_argument("--bptt_eps", type=float, default=1e-10,
                        help="Epsilon value for BPTT numerical stability")
    
    # Vocabulary parameters
    parser.add_argument("--filter_vocab", type=str2bool, default=False,
                        help="Whether to filter the vocabulary")
    parser.add_argument("--vocab_threshold", type=float, default=0.5,
                        help="Threshold for vocabulary filtering")
    
    # Other parameters
    parser.add_argument("--max_gradient_norm", type=float, default=0.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--plot_every", type=int, default=100,
                        help="Frequency of plotting loss curves")
    parser.add_argument("--stgs_grad_variance_samples", type=int, default=0,
                        help="Number of STGS gradient samples to use when estimating variance (>=2 enables metric)")
    parser.add_argument("--stgs_grad_variance_period", type=int, default=1,
                        help="Epoch interval between two gradient variance measurements")
    parser.add_argument("--stgs_grad_bias_samples", type=int, default=0,
                        help="Number of STGS gradient samples to average when estimating bias (>=1 enables metric)")
    parser.add_argument("--stgs_grad_bias_period", type=int, default=1,
                        help="Epoch interval between two STGS bias measurements")
    parser.add_argument("--stgs_grad_bias_reference_samples", type=int, default=0,
                        help="Number of REINFORCE reference samples per bias estimate (>=1 enables metric)")
    parser.add_argument("--stgs_grad_bias_reference_batch_size", type=int, default=0,
                        help="Batch size for REINFORCE reference bias runs (0 reuses training batch size)")
    parser.add_argument("--stgs_grad_bias_reference_use_baseline", type=str2bool, default=True,
                        help="Enable REINFORCE baseline when computing STGS bias")
    parser.add_argument("--stgs_grad_bias_reference_reward_scale", type=float, default=1.0,
                        help="Reward scale to apply in the reference REINFORCE bias estimator")
    parser.add_argument("--stgs_grad_bias_reference_baseline_beta", type=float, default=0.9,
                        help="EMA beta for the reference REINFORCE baseline during bias estimation")
    parser.add_argument("--reinforce_grad_variance_samples", type=int, default=0,
                        help="Number of REINFORCE gradient samples to use when estimating variance (>=2 enables metric)")
    parser.add_argument("--reinforce_grad_variance_period", type=int, default=1,
                        help="Epoch interval between two REINFORCE gradient variance measurements")
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
    
    # Metrics parameters
    parser.add_argument("--metric_groups", type=str, default=None,
                        help="Comma-separated list of metric groups to compute (None = all)")
    parser.add_argument("--skip_metric_groups", type=str, default=None,
                        help="Comma-separated list of metric groups to skip")
    
    # SentenceBERT parameters
    parser.add_argument("--sentencebert_model", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceBERT model to use for semantic similarity")
    
    # BERTScore parameters
    parser.add_argument("--bertscore_model", type=str, default="distilbert-base-uncased",
                        help="Model to use for BERTScore computation")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse target indices if provided
    target_indices = None
    if args.target_indices:
        target_indices = [int(idx) for idx in args.target_indices.split(",")]
    
    # Parse metric groups if provided
    metric_groups = str2list(args.metric_groups)
    skip_metric_groups = str2list(args.skip_metric_groups)
    
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
        target_indices=target_indices,
        metric_groups=metric_groups
    )

if __name__ == "__main__":
    main()
