"""
Utilities for computing and processing metrics from batch optimization.
"""
import numpy as np
import logging
import torch
from collections import defaultdict
import bert_score
from mauve import compute_mauve

logger = logging.getLogger("metrics_utils")

# Custom normalization ranges for each metric
METRIC_NORMALIZATION = {
    "success_rate": {"min_y": 0, "max_y": 1},
    "avg_token_accuracy": {"min_y": 0, "max_y": 1},
    "avg_final_loss": {"min_y": 0, "max_y": 10},  # Adjust based on typical loss values
    "avg_perplexity": {"min_y": 1, "max_y": 100},  # Typical range for perplexity
    "avg_token_overlap_ratio": {"min_y": 0, "max_y": 1},
    "avg_target_hit_ratio": {"min_y": 0, "max_y": 1},
    "avg_unigram_overlap": {"min_y": 0, "max_y": 1},
    "avg_bigram_overlap": {"min_y": 0, "max_y": 1},
    "avg_lcs_ratio": {"min_y": 0, "max_y": 1},
    "avg_bertscore_f1": {"min_y": 0, "max_y": 1},
    "avg_bertscore_precision": {"min_y": 0, "max_y": 1},
    "avg_bertscore_recall": {"min_y": 0, "max_y": 1},
    "avg_mauve_score": {"min_y": 0, "max_y": 1},
}


def compute_auc(k_values, metrics, min_y=0, max_y=1):
    """
    Compute Area Under the Curve using trapezoidal integration with custom normalization.
    
    Args:
    - k_values: List of k values (x-axis)
    - metrics: List of corresponding metric values (y-axis)
    - min_y: Minimum expected value for the metric
    - max_y: Maximum expected value for the metric
    
    Returns:
    - Tuple of (Area under the curve, Normalized AuC)
    """
    if not k_values or not metrics:
        return 0.0, 0.0
        
    # Ensure k_values and metrics are sorted by k_values
    sorted_pairs = sorted(zip(k_values, metrics))
    k_values, metrics = zip(*sorted_pairs)
    
    # Compute AuC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(k_values)):
        x1, x2 = k_values[i-1], k_values[i]
        y1, y2 = metrics[i-1], metrics[i]
        auc += 0.5 * (x2 - x1) * (y1 + y2)
    
    # Compute max possible AuC (bounding rectangle)
    max_k = max(k_values)
    min_k = min(k_values)
    
    # Compute max possible AuC with given y range
    max_possible_auc = abs(max_y - min_y) * abs(max_k - min_k)
    
    # Normalize AuC
    normalized_auc = auc / max_possible_auc if max_possible_auc != 0 else 0
    
    return auc, normalized_auc


def compute_average_metric(values):
    """
    Safely compute the average of a list of values.
    
    Args:
    - values: List of numeric values
    
    Returns:
    - Average value or 0 if the list is empty
    """
    return sum(values) / len(values) if values else 0


def compute_bertscore(
    candidates, 
    references, 
    lang="en", 
    #model_type="microsoft/deberta-xlarge-mnli", 
    model_type="distilbert-base-uncased",
    batch_size=32, 
    device=None,
):
    """
    Compute BERTScore for a batch of candidates against references.
    
    Args:
    - candidates: List of candidate texts
    - references: List of reference texts
    - lang: Language code (default: "en")
    - model_type: Model to use for embeddings (default: "microsoft/deberta-xlarge-mnli")
    - batch_size: Batch size for processing (default: 32)
    - device: Device to run on (default: None, will use GPU if available)
    
    Returns:
    - Dictionary with precision, recall, and F1 scores
    """
    if not candidates or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    try:
        # Use the bert_score package
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        P, R, F1 = bert_score.score(
            candidates, references, 
            lang=lang, 
            model_type=model_type,
            batch_size=batch_size,
            device=device,
            verbose=False,
            use_fast_tokenizer=True,
        )
        
        # Convert tensors to float values
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }
    except Exception as e:
        logger.error(f"Error computing BERTScore: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def compute_mauve_score(candidates, references, model_name="gpt2-large", device=None, max_samples=None, verbose=False):
    """
    Compute MAUVE score for comparing distributions of generated and reference texts.
    
    Args:
    - candidates: List of candidate (generated) texts
    - references: List of reference texts
    - model_name: Model to use for text featurization (default: "gpt2-large")
    - device: Device to run on (default: None, will use GPU if available)
    - max_samples: Maximum number of samples to use (default: None, use all)
    - verbose: Whether to print progress information (default: False)
    
    Returns:
    - MAUVE score
    """
    if not candidates or not references:
        return 0.0
    
    # Make sure we have enough samples
    if len(candidates) < 2 or len(references) < 2:
        logger.warning("Not enough samples for MAUVE computation. Need at least 2 samples in each set.")
        return 0.0
    
    try:
        # Limit the number of samples if specified
        if max_samples is not None:
            candidates = candidates[:max_samples]
            references = references[:max_samples]
        
        # Use the mauve package
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Compute MAUVE score
        out = compute_mauve(
            p_text=references, 
            q_text=candidates,
            device_id=0 if device == "cuda" else -1,
            max_text_length=512,
            verbose=verbose,
            featurize_model_name=model_name
        )
        
        # Return MAUVE score
        return out.mauve
    except Exception as e:
        logger.error(f"Error computing MAUVE score: {e}")
        return 0.0


def aggregate_metrics_by_k(k_metrics):
    """
    Calculate aggregate metrics for each k value.
    
    Args:
    - k_metrics: Dictionary mapping k values to metric dictionaries
    
    Returns:
    - Dictionary mapping k values to aggregated metrics
    """
    aggregated = {}
    
    for k, metrics in k_metrics.items():
        #num_samples = len(metrics.get("exact_matches", []))
        # Estimate num_samples without exact_matches:
        num_samples = len(metrics.get("token_accuracies", []))
        if num_samples == 0:
            continue
            
        aggregated[k] = {
            "num_samples": num_samples,
            "success_rate": compute_average_metric(metrics.get("exact_matches", [])),
            "avg_token_accuracy": compute_average_metric(metrics.get("token_accuracies", [])),
            "avg_final_loss": compute_average_metric(metrics.get("final_losses", [])),
            "avg_perplexity": compute_average_metric(metrics.get("perplexities", [])),
            "avg_token_overlap_ratio": compute_average_metric(metrics.get("token_overlap_ratios", [])),
            "avg_target_hit_ratio": compute_average_metric(metrics.get("target_hit_ratios", [])),
            "avg_lcs_ratio": compute_average_metric(metrics.get("lcs_ratios", [])),
            "avg_unigram_overlap": compute_average_metric(metrics.get("unigram_overlaps", [])),
            "avg_bigram_overlap": compute_average_metric(metrics.get("bigram_overlaps", [])),
            "avg_bertscore_precision": compute_average_metric(metrics.get("bertscore_precisions", [])),
            "avg_bertscore_recall": compute_average_metric(metrics.get("bertscore_recalls", [])),
            "avg_bertscore_f1": compute_average_metric(metrics.get("bertscore_f1s", [])),
            "avg_mauve_score": compute_average_metric(metrics.get("mauve_scores", []))
        }
    
    return aggregated


def compute_auc_metrics(k_metrics):
    """
    Compute AUC metrics for various evaluation metrics.
    
    Args:
    - k_metrics: Dictionary mapping k values to metric dictionaries
    
    Returns:
    - Dictionary of AUC results
    """
    # Define all metrics to compute AUC for
    auc_metrics = {
        "success_rate": [],
        "avg_token_accuracy": [],
        "avg_final_loss": [],
        "avg_perplexity": [],
        "avg_token_overlap_ratio": [],
        "avg_target_hit_ratio": [],
        "avg_lcs_ratio": [],
        "avg_unigram_overlap": [],
        "avg_bigram_overlap": [],
        "avg_bertscore_precision": [],
        "avg_bertscore_recall": [],
        "avg_bertscore_f1": [],
        "avg_mauve_score": [],
    }
    
    # Prepare k values and collect metrics for each k
    k_values = sorted(k_metrics.keys())
    
    # Collect metrics for each k
    for k in k_values:
        metrics = k_metrics[k]
        #num_samples = len(metrics.get("exact_matches", []))
        # Estimate num_samples without exact_matches:
        num_samples = len(metrics.get("token_accuracies", []))
        if num_samples == 0:
            continue
            
        for metric_name in auc_metrics:
            # Extract metric values based on the metric name
            if metric_name == "success_rate":
                value = compute_average_metric(metrics.get("exact_matches", []))
            elif metric_name == "avg_token_accuracy":
                value = compute_average_metric(metrics.get("token_accuracies", []))
            elif metric_name == "avg_final_loss":
                value = compute_average_metric(metrics.get("final_losses", []))
            elif metric_name == "avg_perplexity":
                value = compute_average_metric(metrics.get("perplexities", []))
            elif metric_name == "avg_token_overlap_ratio":
                value = compute_average_metric(metrics.get("token_overlap_ratios", []))
            elif metric_name == "avg_target_hit_ratio":
                value = compute_average_metric(metrics.get("target_hit_ratios", []))
            elif metric_name == "avg_lcs_ratio":
                value = compute_average_metric(metrics.get("lcs_ratios", []))
            elif metric_name == "avg_unigram_overlap":
                value = compute_average_metric(metrics.get("unigram_overlaps", []))
            elif metric_name == "avg_bigram_overlap":
                value = compute_average_metric(metrics.get("bigram_overlaps", []))
            elif metric_name == "avg_bertscore_precision":
                value = compute_average_metric(metrics.get("bertscore_precisions", []))
            elif metric_name == "avg_bertscore_recall":
                value = compute_average_metric(metrics.get("bertscore_recalls", []))
            elif metric_name == "avg_bertscore_f1":
                value = compute_average_metric(metrics.get("bertscore_f1s", []))
            elif metric_name == "avg_mauve_score":
                value = compute_average_metric(metrics.get("mauve_scores", []))
            else:
                value = 0
                
            auc_metrics[metric_name].append(value)
    
    # Compute AUC for each metric
    auc_results = {}
    for metric_name, metric_values in auc_metrics.items():
        if not metric_values:
            continue
            
        # Get normalization parameters, default to 0-1 if not specified
        norm_params = METRIC_NORMALIZATION.get(metric_name, {"min_y": 0, "max_y": 1})
        
        # Compute raw and normalized AuC
        raw_auc, normalized_auc = compute_auc(
            k_values, 
            metric_values, 
            min_y=norm_params["min_y"], 
            max_y=norm_params["max_y"]
        )
        
        auc_results[f"AuC/Raw/{metric_name}"] = raw_auc
        auc_results[f"AuC/Normalized/{metric_name}"] = normalized_auc
    
    return auc_results


def compute_overall_metrics(results):
    """
    Compute overall metrics across all optimization results.
    
    Args:
    - results: Dictionary of optimization results
    
    Returns:
    - Dictionary of overall metrics
    """
    # Extract all evaluation metrics
    all_exact_matches = [result["evaluation"]["exact_match"] for result in results.values()]
    all_token_accuracies = [result["evaluation"]["token_accuracy"] for result in results.values()]
    all_lcs_ratios = [result["evaluation"]["lcs_ratio"] for result in results.values()]
    all_unigram_overlaps = [result["evaluation"]["unigram_overlap"] for result in results.values()]
    all_bigram_overlaps = [result["evaluation"]["bigram_overlap"] for result in results.values()]
    
    # Extract BERTScore metrics if available
    all_bertscore_precisions = []
    all_bertscore_recalls = []
    all_bertscore_f1s = []
    all_mauve_scores = []
    
    # Extract token metrics
    all_token_overlap_ratios = []
    all_target_hit_ratios = []
    
    for result in results.values():
        # Extract token metrics
        if "token_overlap_ratio" in result:
            all_token_overlap_ratios.append(result["token_overlap_ratio"])
        if "target_hit_ratio" in result:
            all_target_hit_ratios.append(result["target_hit_ratio"])
            
        # Extract BERTScore and MAUVE metrics if available
        if "evaluation" in result:
            eval_metrics = result["evaluation"]
            if "bertscore_precision" in eval_metrics:
                all_bertscore_precisions.append(eval_metrics["bertscore_precision"])
            if "bertscore_recall" in eval_metrics:
                all_bertscore_recalls.append(eval_metrics["bertscore_recall"])
            if "bertscore_f1" in eval_metrics:
                all_bertscore_f1s.append(eval_metrics["bertscore_f1"])
            if "mauve_score" in eval_metrics:
                all_mauve_scores.append(eval_metrics["mauve_score"])
    
    # Compute overall metrics
    overall_metrics = {
        "success_rate": compute_average_metric(all_exact_matches),
        "avg_token_accuracy": compute_average_metric(all_token_accuracies),
        "avg_lcs_ratio": compute_average_metric(all_lcs_ratios),
        "avg_unigram_overlap": compute_average_metric(all_unigram_overlaps),
        "avg_bigram_overlap": compute_average_metric(all_bigram_overlaps),
        "num_samples": len(results),
    }
    
    # Add token metrics if available
    if all_token_overlap_ratios:
        overall_metrics["avg_token_overlap_ratio"] = compute_average_metric(all_token_overlap_ratios)
    if all_target_hit_ratios:
        overall_metrics["avg_target_hit_ratio"] = compute_average_metric(all_target_hit_ratios)
        
    # Add BERTScore and MAUVE metrics if available
    if all_bertscore_precisions:
        overall_metrics["avg_bertscore_precision"] = compute_average_metric(all_bertscore_precisions)
    if all_bertscore_recalls:
        overall_metrics["avg_bertscore_recall"] = compute_average_metric(all_bertscore_recalls)
    if all_bertscore_f1s:
        overall_metrics["avg_bertscore_f1"] = compute_average_metric(all_bertscore_f1s)
    if all_mauve_scores:
        overall_metrics["avg_mauve_score"] = compute_average_metric(all_mauve_scores)
    
    return overall_metrics


def log_metrics_summary(metrics, logger):
    """
    Log a summary of metrics to the given logger.
    
    Args:
    - metrics: Dictionary of metrics
    - logger: Logger object to log to
    """
    logger.info(f"Overall success rate: {metrics.get('success_rate', 0):.4f}")
    logger.info(f"Overall token accuracy: {metrics.get('avg_token_accuracy', 0):.4f}")
    logger.info(f"Overall LCS ratio: {metrics.get('avg_lcs_ratio', 0):.4f}")
    logger.info(f"Overall unigram overlap: {metrics.get('avg_unigram_overlap', 0):.4f}")
    logger.info(f"Overall bigram overlap: {metrics.get('avg_bigram_overlap', 0):.4f}")
    
    if "avg_token_overlap_ratio" in metrics:
        logger.info(f"Overall token overlap ratio: {metrics['avg_token_overlap_ratio']:.4f}")
    if "avg_target_hit_ratio" in metrics:
        logger.info(f"Overall target hit ratio: {metrics['avg_target_hit_ratio']:.4f}")
        
    # Log BERTScore and MAUVE metrics if available
    if "avg_bertscore_f1" in metrics:
        logger.info(f"Overall BERTScore F1: {metrics['avg_bertscore_f1']:.4f}")
    if "avg_mauve_score" in metrics:
        logger.info(f"Overall MAUVE score: {metrics['avg_mauve_score']:.4f}")
