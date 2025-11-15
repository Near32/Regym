"""
Utilities for logging metrics to various destinations (WandB, files, etc.)
"""
import logging
import json
import wandb
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger("metrics_logging")

class MetricsLogger:
    """
    Class for logging metrics to various destinations.
    """
    
    def __init__(self, run_id: str = None, output_dir: Optional[str] = None, 
                wandb_project: Optional[str] = None, wandb_entity: Optional[str] = None):
        """
        Initialize the metrics logger.
        
        Args:
            run_id: Unique identifier for this run
            output_dir: Directory to save logs and files
            wandb_project: W&B project name (if None, W&B logging is disabled)
            wandb_entity: W&B entity name
        """
        self.run_id = run_id or wandb.util.generate_id()
        self.output_dir = Path(output_dir) if output_dir else None
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run = None
        
        # Create output directory if specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tables
        self.summary_table = None
        self.k_summary_table = None
        
    def init_wandb(
        self, 
        config: Dict[str, Any] = None, 
        name: Optional[str] = None, 
        group: Optional[str] = None, 
        job_type: Optional[str] = None,
        resume: Optional[str] = 'allow',
    ):
        """
        Initialize W&B logging.
        
        Args:
            config: Configuration dictionary
            name: Run name
            group: Run group
            job_type: Job type
        """
        if not self.wandb_project:
            logger.warning("W&B project not specified, skipping W&B initialization")
            return
            
        name = name or f"run_{self.run_id}"
        
        self.wandb_run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=name,
            id=self.run_id,
            group=group,
            job_type=job_type,
            config=config,
            reinit='finish_previous', #True,
            #reinit=False,
            #reinit=True,
            resume=resume,
        )
        
        # Initialize tables
        self.summary_table = self._create_summary_table()
        self.k_summary_table = self._create_k_summary_table()
        
        return self.wandb_run
        
    def _create_summary_table(self):
        """Create a W&B table for individual sample metrics."""
        if not self.wandb_run:
            return None
            
        return wandb.Table(
            columns=[
                "sample_id", "target_text", "k_value", "target_perplexity", 
                "generated_text", "final_loss", "exact_match", "token_accuracy",
                "token_overlap_ratio", "target_hit_ratio", "lcs_ratio", 
                "unigram_overlap", "bigram_overlap", "bertscore_f1",
                "bertscore_precision", "bertscore_recall", "mauve_score",
                "sentencebert_similarity"
            ]
        )
        
    def _create_k_summary_table(self):
        """Create a W&B table for summarizing results by k value."""
        if not self.wandb_run:
            return None
            
        return wandb.Table(columns=[
            "k_value", 
            "sample_count",
            "avg_perplexity", 
            "success_rate", 
            "avg_token_accuracy", 
            "avg_final_loss",
            "avg_token_overlap_ratio",
            "avg_target_hit_ratio",
            "avg_lcs_ratio",
            "avg_unigram_overlap",
            "avg_bigram_overlap",
            "avg_bertscore_f1",
            "avg_bertscore_precision",
            "avg_bertscore_recall",
            "avg_mauve_score",
            "avg_sentencebert_similarity"
        ])
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: Optional[str] = None):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step for W&B logging
            prefix: Optional prefix for metric names
        """
        if not self.wandb_run:
            return
            
        # Prepare metrics for logging
        log_metrics = {}
        
        for name, value in metrics.items():
            # Skip non-scalar values
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
                
            # Add prefix if specified
            if prefix:
                log_name = f"{prefix}/{name}"
            else:
                log_name = name
                
            log_metrics[log_name] = value
            
        # Log to W&B
        try:
            self.wandb_run.log(log_metrics, step=step)
        except Exception as e:
            print(f"Exception caught in metrics logger: {e}")
            print(f"Resuming W&B Run:")
            self.wandb_run = wandb.init(
                entity=self.wandb_run.entity, 
                project=self.wandb_run.project, 
                id=self.wandb_run.id, 
                group=self.wandb_run.group,
                resume='allow',
            )
            self.wandb_run.log(log_metrics, step=step)

        
    def log_k_metrics(self, k: int, metrics: Dict[str, Any]):
        """
        Log metrics for a specific k value, following the original format for plotting.
        
        Args:
            k: The k value
            metrics: Dictionary of metrics for this k value
        """
        if not self.wandb_run:
            return
            
        log_dict = {}
        
        # First, add metrics with k prefix (matches original format)
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Format: "k{k}/{metric}" - e.g., "k1/avg_token_accuracy"
                log_dict[f"k{k}/{metric_name}"] = value
        
        # Add k value and raw metrics for scatter plots (matches original format)
        log_dict["k"] = k
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Add without prefix for scatter plots
                log_dict[metric_name] = value
                
        # Log to W&B
        self.wandb_run.log(log_dict)
        
    def update_summary_table(self, sample_id: str, target_info: Dict[str, Any], 
                            result: Dict[str, Any], metrics: Dict[str, Any]):
        """
        Add a row to the summary table.
        
        Args:
            sample_id: Unique identifier for the sample
            target_info: Information about the target
            result: Optimization result for the target
            metrics: Evaluation metrics
        """
        if not self.summary_table:
            return
            
        # Extract values with safe defaults
        target_text = target_info.get("text", "")
        k_value = target_info.get("k_target", 0)
        target_perplexity = target_info.get("perplexity", 0.0)
        generated_text = result.get("optimized_text", "")
        final_loss = result.get("final_loss", 0.0)
        
        # Add row to table
        self.summary_table.add_data(
            sample_id,
            target_text,
            k_value,
            target_perplexity,
            generated_text,
            final_loss,
            metrics.get("exact_match", 0),
            metrics.get("token_accuracy", 0.0),
            metrics.get("token_overlap_ratio", 0.0),
            metrics.get("target_hit_ratio", 0.0),
            metrics.get("lcs_ratio", 0.0),
            metrics.get("unigram_overlap", 0.0),
            metrics.get("bigram_overlap", 0.0),
            metrics.get("bertscore_f1", 0.0),
            metrics.get("bertscore_precision", 0.0),
            metrics.get("bertscore_recall", 0.0),
            metrics.get("mauve_score", 0.0),
            metrics.get("sentencebert_similarity", 0.0)
        )
        
    def update_k_summary_table(self, k_value: int, metrics: Dict[str, Any]):
        """
        Add a row to the k summary table.
        
        Args:
            k_value: The k value
            metrics: Aggregated metrics for this k value
        """
        if not self.k_summary_table:
            return
            
        # Extract values with safe defaults
        sample_count = metrics.get("sample_count", 0)
        
        # For backward compatibility, handle both "success_rate" and "avg_exact_match"
        success_rate = metrics.get("success_rate", metrics.get("avg_exact_match", 0.0))
        
        # Add row to table
        self.k_summary_table.add_data(
            k_value,
            sample_count,
            metrics.get("avg_perplexity", 0.0),
            success_rate,
            metrics.get("avg_token_accuracy", 0.0),
            metrics.get("avg_final_loss", 0.0),
            metrics.get("avg_token_overlap_ratio", 0.0),
            metrics.get("avg_target_hit_ratio", 0.0),
            metrics.get("avg_lcs_ratio", 0.0),
            metrics.get("avg_unigram_overlap", 0.0),
            metrics.get("avg_bigram_overlap", 0.0),
            metrics.get("avg_bertscore_f1", 0.0),
            metrics.get("avg_bertscore_precision", 0.0),
            metrics.get("avg_bertscore_recall", 0.0),
            metrics.get("avg_mauve_score", 0.0),
            metrics.get("avg_sentencebert_similarity", 0.0)
        )
        
    def log_tables(self):
        """Log summary tables to W&B."""
        if not self.wandb_run:
            return
            
        tables = {}
        
        if self.summary_table:
            tables["optimization_summary"] = self.summary_table
            
        if self.k_summary_table:
            tables["k_value_summary"] = self.k_summary_table
            
        if tables:
            self.wandb_run.log(tables)
            
    def save_to_file(self, data: Dict[str, Any], filename: str):
        """
        Save data to a JSON file.
        
        Args:
            data: Data to save
            filename: Name of the file (without directory)
        """
        if not self.output_dir:
            return
            
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved data to {filepath}")
        
        # Create and log W&B artifact if enabled
        if self.wandb_run:
            artifact = wandb.Artifact(
                name=f"{filename.split('.')[0]}-{self.run_id}",
                type="results",
                description=f"Results saved to {filename}"
            )
            artifact.add_file(filepath)
            self.wandb_run.log_artifact(artifact)
            
    def log_summary(self, summary: Dict[str, Any]):
        """
        Log a complete summary of results.
        
        Args:
            summary: Summary dictionary from MetricsAggregator.get_summary()
        """
        # Log overall metrics
        if "overall" in summary:
            self.log_metrics(summary["overall"], prefix="overall")
            
        # Log metrics by k value with proper formatting for plotting
        if "by_k" in summary:
            for k, metrics in summary["by_k"].items():
                # Use the k_metrics format that preserves the original plotting format
                self.log_k_metrics(k, metrics)
                self.update_k_summary_table(k, metrics)
                
        # Log AUC results
        if "auc" in summary:
            self.log_metrics(summary["auc"], prefix="auc")
            
        # Log tables
        self.log_tables()
        
        # Save complete summary to file
        self.save_to_file(summary, "evaluation_summary.json")
        
    def finish(self):
        """Finish logging and clean up."""
        if self.wandb_run:
            self.wandb_run.finish()
