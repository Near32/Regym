"""
Utilities for computing and processing metrics from batch optimization.
"""
import numpy as np
import logging

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
            "avg_bigram_overlap": compute_average_metric(metrics.get("bigram_overlaps", []))
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
    
    # Extract token metrics
    all_token_overlap_ratios = []
    all_target_hit_ratios = []
    
    for result in results.values():
        if "token_overlap_ratio" in result:
            all_token_overlap_ratios.append(result["token_overlap_ratio"])
        if "target_hit_ratio" in result:
            all_target_hit_ratios.append(result["target_hit_ratio"])
    
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
