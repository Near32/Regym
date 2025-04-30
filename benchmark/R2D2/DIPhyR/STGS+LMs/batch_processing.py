"""
Functions for processing multiple targets in batch mode.
"""
import concurrent.futures
import logging
from tqdm import tqdm
from pathlib import Path
import torch

logger = logging.getLogger("batch_processing")


def process_targets_sequential(targets, model, tokenizer, device, config, run_id, output_dir, 
                              summary_table, k_metrics):
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
        summary_table: W&B summary table
        k_metrics: Dictionary to store metrics by k value
        
    Returns:
        results: Dictionary mapping target IDs to optimization results
    """
    from target_optimization import optimize_for_target
    from main import TokenOverlapMetric
    from logging_utils import update_summary_table
    
    results = {}
    
    for target in tqdm(targets, desc="Optimizing for targets"):
        result = optimize_for_target(
            target_info=target,
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            run_id=run_id,
            output_dir=output_dir
        )
        results[target["id"]] = result
        
        # Add to k-value metrics
        k = target["k_target"]
        if k not in k_metrics:
            k_metrics[k] = {
                "perplexities": [],
                "exact_matches": [],
                "token_accuracies": [],
                "final_losses": [],
                "token_overlap_ratios": [],
                "target_hit_ratios": [],
                "lcs_ratios": [],
                "unigram_overlaps": [],
                "bigram_overlaps": [],
                "bertscore_precisions": [],
                "bertscore_recalls": [],
                "bertscore_f1s": [],
                "mauve_scores": []
            }
            
        k_metrics[k]["perplexities"].append(target.get("perplexity", 0))
        k_metrics[k]["exact_matches"].append(result["evaluation"]["exact_match"])
        k_metrics[k]["token_accuracies"].append(result["evaluation"]["token_accuracy"])
        k_metrics[k]["final_losses"].append(result["final_loss"])
        k_metrics[k]["lcs_ratios"].append(result["evaluation"]["lcs_ratio"])
        k_metrics[k]["unigram_overlaps"].append(result["evaluation"]["unigram_overlap"])
        k_metrics[k]["bigram_overlaps"].append(result["evaluation"]["bigram_overlap"])
        
        # Add BERTScore metrics if available
        if "bertscore_precision" in result["evaluation"]:
            k_metrics[k]["bertscore_precisions"].append(result["evaluation"]["bertscore_precision"])
            k_metrics[k]["bertscore_recalls"].append(result["evaluation"]["bertscore_recall"])
            k_metrics[k]["bertscore_f1s"].append(result["evaluation"]["bertscore_f1"])
        
        # Add MAUVE score if available
        if "mauve_score" in result["evaluation"]:
            k_metrics[k]["mauve_scores"].append(result["evaluation"]["mauve_score"])
        
        # Add token overlap and target hit metrics
        token_overlap_metric = TokenOverlapMetric(
            target_text=target["text"],
            tokenizer=tokenizer
        )
        token_metrics = token_overlap_metric.measure(
            prompt_tokens=result["optimized_tokens"]
        )
        k_metrics[k]["token_overlap_ratios"].append(token_metrics.get('token_overlap_ratio', 0))
        k_metrics[k]["target_hit_ratios"].append(token_metrics.get('target_hit_ratio', 0))
        
        # Add to summary table
        update_summary_table(summary_table, target, result, token_metrics)
    
    return results


def process_targets_parallel(targets, model, tokenizer, config, run_id, output_dir, 
                           summary_table, k_metrics, num_workers):
    """
    Process targets in parallel using a ProcessPoolExecutor.
    
    Args:
        targets: List of target info dictionaries
        model: Language model
        tokenizer: Tokenizer
        config: Configuration dictionary
        run_id: Unique run identifier
        output_dir: Directory to save results
        summary_table: W&B summary table
        k_metrics: Dictionary to store metrics by k value
        num_workers: Number of parallel workers
        
    Returns:
        results: Dictionary mapping target IDs to optimization results
    """
    from target_optimization import optimize_for_target
    from main import TokenOverlapMetric
    from logging_utils import update_summary_table
    import torch
    
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
                output_dir=output_dir
            ): target for target in targets
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_target), 
                         total=len(targets), 
                         desc="Optimizing for targets"):
            target = future_to_target[future]
            try:
                result = future.result()
                results[target["id"]] = result
                
                # Add to k-value metrics
                k = target["k_target"]
                if k not in k_metrics:
                    k_metrics[k] = {
                        "perplexities": [],
                        "exact_matches": [],
                        "token_accuracies": [],
                        "final_losses": [],
                        "token_overlap_ratios": [],
                        "target_hit_ratios": [],
                        "lcs_ratios": [],
                        "unigram_overlaps": [],
                        "bigram_overlaps": [],
                        "bertscore_precisions": [],
                        "bertscore_recalls": [],
                        "bertscore_f1s": [],
                        "mauve_scores": []
                    }
                
                k_metrics[k]["perplexities"].append(target.get("perplexity", 0))
                k_metrics[k]["exact_matches"].append(result["evaluation"]["exact_match"])
                k_metrics[k]["token_accuracies"].append(result["evaluation"]["token_accuracy"])
                k_metrics[k]["final_losses"].append(result["final_loss"])
                k_metrics[k]["lcs_ratios"].append(result["evaluation"]["lcs_ratio"])
                k_metrics[k]["unigram_overlaps"].append(result["evaluation"]["unigram_overlap"])
                k_metrics[k]["bigram_overlaps"].append(result["evaluation"]["bigram_overlap"])
                
                # Add BERTScore metrics if available
                if "bertscore_precision" in result["evaluation"]:
                    k_metrics[k]["bertscore_precisions"].append(result["evaluation"]["bertscore_precision"])
                    k_metrics[k]["bertscore_recalls"].append(result["evaluation"]["bertscore_recall"])
                    k_metrics[k]["bertscore_f1s"].append(result["evaluation"]["bertscore_f1"])
                
                # Add MAUVE score if available
                if "mauve_score" in result["evaluation"]:
                    k_metrics[k]["mauve_scores"].append(result["evaluation"]["mauve_score"])
                
                # Add token overlap and target hit metrics
                token_overlap_metric = TokenOverlapMetric(
                    target_text=target["text"],
                    tokenizer=tokenizer
                )
                token_metrics = token_overlap_metric.measure(
                    prompt_tokens=torch.tensor(result["optimized_tokens"])
                )
                k_metrics[k]["token_overlap_ratios"].append(token_metrics.get('token_overlap_ratio', 0))
                k_metrics[k]["target_hit_ratios"].append(token_metrics.get('target_hit_ratio', 0))
                
                # Add to summary table
                update_summary_table(summary_table, target, result, token_metrics)

            except Exception as exc:
                logger.error(f"Target {target['id']} generated an exception: {exc}")
    
    return results


def process_targets(targets, model, tokenizer, device, config, run_id, output_dir, 
                  summary_table, k_metrics, num_workers=1):
    """
    Process targets either sequentially or in parallel based on num_workers.
    
    Args:
        targets: List of target info dictionaries
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        config: Configuration dictionary
        run_id: Unique run identifier
        output_dir: Directory to save results
        summary_table: W&B summary table
        k_metrics: Dictionary to store metrics by k value
        num_workers: Number of parallel workers (1 = sequential)
        
    Returns:
        results: Dictionary mapping target IDs to optimization results
    """
    if num_workers == 1:
        return process_targets_sequential(
            targets, model, tokenizer, device, config, run_id, output_dir, 
            summary_table, k_metrics
        )
    else:
        return process_targets_parallel(
            targets, model, tokenizer, config, run_id, output_dir, 
            summary_table, k_metrics, num_workers
        )
