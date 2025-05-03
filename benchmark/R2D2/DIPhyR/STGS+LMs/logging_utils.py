"""
Utilities for logging metrics and results to wandb and local files.
"""
import wandb
import json
import logging
from pathlib import Path

logger = logging.getLogger("logging_utils")


def setup_wandb_logging(config, run_id, dataset_info):
    """
    Initialize and set up a W&B run for batch optimization.
    
    Args:
        config: Configuration dictionary
        run_id: Unique run identifier
        dataset_info: Information about the dataset
        
    Returns:
        wandb run object
    """
    return wandb.init(
        project=config["wandb_project"],
        entity=config.get("wandb_entity"),
        name=f"batch_optimization_{run_id}",
        job_type="batch_coordination",
        config={
            **config,
            "dataset_path": dataset_info.get("dataset_path"),
            "model_name": config.get("model_name"),
            "num_targets": dataset_info.get("num_targets", 0),
            "metadata": dataset_info.get("metadata", {})
        }
    )


def create_summary_table():
    """
    Create a W&B table for summarizing optimization results.
    
    Returns:
        wandb Table object
    """
    return wandb.Table(
        columns=[
            "target_id", "target_text", "target_k", "target_perplexity", 
            "optimized_text", "final_loss", "exact_match", "token_accuracy",
            "token_overlap_ratio", "target_hit_ratio", "lcs_ratio", 
            "unigram_overlap", "bigram_overlap", "bertscore_f1",
            "bertscore_precision", "bertscore_recall", "mauve_score"
        ]
    )


def create_k_summary_table():
    """
    Create a W&B table for summarizing results by k value.
    
    Returns:
        wandb Table object
    """
    return wandb.Table(columns=[
        "k_value", 
        "avg_perplexity", 
        "success_rate", 
        "avg_token_accuracy", 
        "avg_final_loss",
        "num_samples",
        "avg_token_overlap_ratio",
        "avg_target_hit_ratio",
        "avg_lcs_ratio",
        "avg_unigram_overlap",
        "avg_bigram_overlap",
        "avg_bertscore_f1",
        "avg_bertscore_precision",
        "avg_bertscore_recall",
        "avg_mauve_score"
    ])


def log_k_metrics_to_wandb(k, metrics, wandb_run):
    """
    Log metrics for a specific k value to W&B.
    
    Args:
        k: The k value
        metrics: Dictionary of metrics for this k value
        wandb_run: W&B run object
    """
    log_dict = {}
    
    # Add metrics with k prefix
    for metric_name, value in metrics.items():
        log_dict[f"k{k}/{metric_name}"] = value
    
    # Add metrics without prefix (for plotting)
    log_dict["k"] = k
    for metric_name, value in metrics.items():
        log_dict[metric_name] = value
    
    wandb_run.log(log_dict)


def update_summary_table(table, target_info, result, token_metrics):
    """
    Add a row to the summary table for a specific target.
    
    Args:
        table: W&B Table object
        target_info: Information about the target
        result: Optimization result for the target
        token_metrics: Token-based metrics
    """
    # Get BERTScore and MAUVE metrics if available
    bertscore_f1 = result["evaluation"].get("bertscore_f1", 0)
    bertscore_precision = result["evaluation"].get("bertscore_precision", 0)
    bertscore_recall = result["evaluation"].get("bertscore_recall", 0)
    mauve_score = result["evaluation"].get("mauve_score", 0)
    
    table.add_data(
        target_info["id"],
        target_info["text"],
        target_info["k_target"],
        target_info.get("perplexity", "N/A"),
        result["optimized_text"],
        result["final_loss"],
        result["evaluation"]["exact_match"],
        result["evaluation"]["token_accuracy"],
        token_metrics.get('token_overlap_ratio', 0),
        token_metrics.get('target_hit_ratio', 0),
        result["evaluation"]["lcs_ratio"],
        result["evaluation"]["unigram_overlap"],
        result["evaluation"]["bigram_overlap"],
        bertscore_f1,
        bertscore_precision,
        bertscore_recall,
        mauve_score
    )


def update_k_summary_table(table, k, metrics):
    """
    Add a row to the k summary table.
    
    Args:
        table: W&B Table object
        k: The k value
        metrics: Aggregated metrics for this k value
    """
    table.add_data(
        k,
        metrics.get("avg_perplexity", 0),
        metrics.get("success_rate", 0),
        metrics.get("avg_token_accuracy", 0),
        metrics.get("avg_final_loss", 0),
        metrics.get("num_samples", 0),
        metrics.get("avg_token_overlap_ratio", 0),
        metrics.get("avg_target_hit_ratio", 0),
        metrics.get("avg_lcs_ratio", 0),
        metrics.get("avg_unigram_overlap", 0),
        metrics.get("avg_bigram_overlap", 0),
        metrics.get("avg_bertscore_f1", 0),
        metrics.get("avg_bertscore_precision", 0),
        metrics.get("avg_bertscore_recall", 0),
        metrics.get("avg_mauve_score", 0)
    )


def log_tables_to_wandb(summary_table, k_summary_table, wandb_run):
    """
    Log summary tables to W&B.
    
    Args:
        summary_table: Table with individual target results
        k_summary_table: Table with aggregated k-value results
        wandb_run: W&B run object
    """
    wandb_run.log({
        "optimization_summary": summary_table,
        "k_value_summary": k_summary_table
    })


def log_auc_results_to_wandb(auc_results, wandb_run):
    """
    Log AUC results to W&B.
    
    Args:
        auc_results: Dictionary of AUC results
        wandb_run: W&B run object
    """
    wandb_run.log(auc_results)


def log_overall_metrics_to_wandb(metrics, wandb_run):
    """
    Log overall metrics to W&B.
    
    Args:
        metrics: Dictionary of overall metrics
        wandb_run: W&B run object
    """
    log_dict = {}
    for metric_name, value in metrics.items():
        log_dict[f"overall/{metric_name}"] = value
    
    wandb_run.log(log_dict)


def save_results_to_file(output_path, run_id, results, config, k_summaries, overall_metrics, auc_results=None, dataset_metadata=None):
    """
    Save all results to a JSON file.
    
    Args:
        output_path: Path to save the file
        run_id: Unique run identifier
        results: Dictionary of optimization results
        config: Configuration dictionary
        k_summaries: Dictionary of k-value summaries
        overall_metrics: Dictionary of overall metrics
        auc_results: Dictionary of AUC results
        dataset_metadata: Metadata about the dataset
    """
    data = {
        "run_id": run_id,
        "results": results,
        "config": config,
        "k_summaries": k_summaries,
        "overall": overall_metrics,
        "dataset_metadata": dataset_metadata or {}
    }
    
    if auc_results:
        data["k_value_auc"] = auc_results
    
    with open(output_path / "batch_results.json", "w") as f:
        json.dump(data, f, indent=2)


def create_and_log_artifact(batch_run, output_path, run_id):
    """
    Create and log a W&B artifact containing the results.
    
    Args:
        batch_run: W&B run object
        output_path: Path to the results file
        run_id: Unique run identifier
    """
    batch_artifact = wandb.Artifact(
        name=f"batch-optimization-results-{run_id}",
        type="results",
        description=f"Results from batch optimization"
    )
    batch_artifact.add_file(output_path / "batch_results.json")
    batch_run.log_artifact(batch_artifact)
