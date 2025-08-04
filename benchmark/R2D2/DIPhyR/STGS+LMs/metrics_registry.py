"""
Centralized registry for all text evaluation metrics.
This module defines all available metrics and their properties in one place.
"""
import logging
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Any, Callable, Tuple, Optional, Union

logger = logging.getLogger("metrics_registry")

# Define normalization ranges for each metric
# These are used for normalizing values when computing AUC
METRIC_RANGES = {
    "exact_match": {"min": 0, "max": 1, "higher_is_better": True},
    "token_accuracy": {"min": 0, "max": 1, "higher_is_better": True},
    "final_loss": {"min": 0, "max": 10, "higher_is_better": False},
    "perplexity": {"min": 1, "max": 100, "higher_is_better": False},
    "token_overlap_ratio": {"min": 0, "max": 1, "higher_is_better": True},
    "target_hit_ratio": {"min": 0, "max": 1, "higher_is_better": True},
    "lcs_ratio": {"min": 0, "max": 1, "higher_is_better": True},
    "unigram_overlap": {"min": 0, "max": 1, "higher_is_better": True},
    "bigram_overlap": {"min": 0, "max": 1, "higher_is_better": True},
    "bertscore_precision": {"min": 0, "max": 1, "higher_is_better": True},
    "bertscore_recall": {"min": 0, "max": 1, "higher_is_better": True},
    "bertscore_f1": {"min": 0, "max": 1, "higher_is_better": True},
    "mauve_score": {"min": 0, "max": 1, "higher_is_better": True},
    "sentencebert_similarity": {"min": 0, "max": 1, "higher_is_better": True},
}

# Define the available metrics and their computation methods
class MetricRegistry:
    """
    Registry for all available metrics and their computation methods.
    """
    def __init__(self):
        self.metrics = {}  # {metric_name: {'compute_fn': fn, 'requires': [], 'group': ''}}
        self.groups = defaultdict(list)  # Group metrics by category
        
    def register(self, name: str, compute_fn: Callable, requires: List[str] = None, 
                group: str = "basic", higher_is_better: bool = True, 
                min_val: float = 0.0, max_val: float = 1.0):
        """
        Register a new metric.
        
        Args:
            name: Name of the metric
            compute_fn: Function to compute the metric
            requires: List of required inputs
            group: Group the metric belongs to
            higher_is_better: Whether higher values are better
            min_val: Minimum expected value
            max_val: Maximum expected value
        """
        if requires is None:
            requires = []
            
        self.metrics[name] = {
            'compute_fn': compute_fn,
            'requires': requires,
            'group': group,
            'higher_is_better': higher_is_better,
            'min_val': min_val,
            'max_val': max_val
        }
        
        self.groups[group].append(name)
        
        # Update METRIC_RANGES if not already defined
        if name not in METRIC_RANGES:
            METRIC_RANGES[name] = {
                "min": min_val,
                "max": max_val,
                "higher_is_better": higher_is_better
            }
        
    def get_metric_fn(self, name: str) -> Callable:
        """Get the computation function for a metric."""
        return self.metrics[name]['compute_fn']
        
    def get_metrics_by_group(self, group: str) -> List[str]:
        """Get all metrics in a specific group."""
        return self.groups.get(group, [])
        
    def get_all_metrics(self) -> List[str]:
        """Get all registered metric names."""
        return list(self.metrics.keys())
        
    def get_metric_info(self, name: str) -> Dict:
        """Get information about a metric."""
        return self.metrics.get(name, {})
        
    def compute_metric(self, name: str, **kwargs) -> Any:
        """Compute a specific metric with the provided inputs."""
        if name not in self.metrics:
            logger.warning(f"Metric '{name}' not found in registry")
            return None
            
        try:
            metric_info = self.metrics[name]
            # Check required inputs
            for req in metric_info['requires']:
                if req not in kwargs:
                    logger.warning(f"Required input '{req}' for metric '{name}' not provided")
                    return None
                    
            # Compute the metric
            return metric_info['compute_fn'](**kwargs)
        except Exception as e:
            logger.error(f"Error computing metric '{name}': {e}")
            return None

# Create a global registry instance
registry = MetricRegistry()

# Define utility functions for computing basic metrics
def calculate_ngram_overlap(generated_tokens, reference_tokens, n=1):
    """Calculate n-gram overlap between two token sequences."""
    if len(generated_tokens) < n or len(reference_tokens) < n:
        return 0.0
        
    gen_ngrams = [tuple(generated_tokens[i:i+n]) for i in range(len(generated_tokens)-n+1)]
    ref_ngrams = [tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens)-n+1)]
    
    gen_counter = defaultdict(int)
    ref_counter = defaultdict(int)
    
    for ngram in gen_ngrams:
        gen_counter[ngram] += 1
    
    for ngram in ref_ngrams:
        ref_counter[ngram] += 1
    
    matches = 0
    for ngram, count in gen_counter.items():
        matches += min(count, ref_counter[ngram])
        
    return matches / len(ref_ngrams) if ref_ngrams else 0.0

def lcs_length(a, b):
    """Calculate length of longest common subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    return dp[m][n]

# Register basic metrics
def compute_exact_match(generated_tokens, reference_tokens, **kwargs):
    """Compute exact match (boolean)."""
    return int(generated_tokens == reference_tokens)

registry.register(
    name="exact_match",
    compute_fn=compute_exact_match,
    requires=["generated_tokens", "reference_tokens"],
    group="basic"
)

def compute_token_accuracy(generated_tokens, reference_tokens, **kwargs):
    """Compute token-level accuracy."""
    min_len = min(len(generated_tokens), len(reference_tokens))
    if len(reference_tokens) == 0:
        return 0.0
    matching_tokens = sum(generated_tokens[i] == reference_tokens[i] for i in range(min_len))
    return matching_tokens / len(reference_tokens)

registry.register(
    name="token_accuracy",
    compute_fn=compute_token_accuracy,
    requires=["generated_tokens", "reference_tokens"],
    group="basic"
)

def compute_unigram_overlap(generated_tokens, reference_tokens, **kwargs):
    """Compute unigram overlap."""
    return calculate_ngram_overlap(generated_tokens, reference_tokens, n=1)

registry.register(
    name="unigram_overlap",
    compute_fn=compute_unigram_overlap,
    requires=["generated_tokens", "reference_tokens"],
    group="basic"
)

def compute_bigram_overlap(generated_tokens, reference_tokens, **kwargs):
    """Compute bigram overlap."""
    return calculate_ngram_overlap(generated_tokens, reference_tokens, n=2)

registry.register(
    name="bigram_overlap",
    compute_fn=compute_bigram_overlap,
    requires=["generated_tokens", "reference_tokens"],
    group="basic"
)

def compute_lcs_ratio(generated_tokens, reference_tokens, **kwargs):
    """Compute longest common subsequence ratio."""
    if len(reference_tokens) == 0:
        return 0.0
    lcs = lcs_length(generated_tokens, reference_tokens)
    return lcs / len(reference_tokens)

registry.register(
    name="lcs_ratio",
    compute_fn=compute_lcs_ratio,
    requires=["generated_tokens", "reference_tokens"],
    group="basic"
)

# Register advanced metrics
def compute_bertscore(generated_text, reference_text, model_type="distilbert-base-uncased", batch_size=32, device=None, **kwargs):
    """Compute BERTScore."""
    try:
        import bert_score
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure inputs are lists
        if isinstance(generated_text, str):
            generated_text = [generated_text]
        if isinstance(reference_text, str):
            reference_text = [reference_text]
        
        P, R, F1 = bert_score.score(
            generated_text, reference_text, 
            model_type=model_type,
            batch_size=batch_size,
            device=device,
            verbose=False
        )
        
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item()
        }
    except Exception as e:
        logger.error(f"Error computing BERTScore: {e}")
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0
        }

registry.register(
    name="bertscore",
    compute_fn=compute_bertscore,
    requires=["generated_text", "reference_text"],
    group="semantic"
)

def compute_mauve_score(generated_text, reference_text, model_name="gpt2-medium", device=None, max_samples=None, **kwargs):
    """Compute MAUVE score."""
    try:
        from mauve import compute_mauve
        
        # Ensure inputs are lists
        if isinstance(generated_text, str):
            generated_text = [generated_text]
        if isinstance(reference_text, str):
            reference_text = [reference_text]
        
        # Check if we have enough samples
        if len(generated_text) < 2 or len(reference_text) < 2:
            # Duplicate samples if needed to have at least 2
            generated_text = generated_text * max(1, 2 // len(generated_text))
            reference_text = reference_text * max(1, 2 // len(reference_text))
        
        # Limit samples if specified
        if max_samples is not None:
            generated_text = generated_text[:max_samples]
            reference_text = reference_text[:max_samples]
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Compute MAUVE
        out = compute_mauve(
            p_text=reference_text, 
            q_text=generated_text,
            device_id=0 if device == "cuda" else -1,
            max_text_length=512,
            verbose=False,
            featurize_model_name=model_name
        )
        
        return {"mauve_score": out.mauve}
    except Exception as e:
        logger.error(f"Error computing MAUVE score: {e}")
        return {"mauve_score": 0.0}

registry.register(
    name="mauve",
    compute_fn=compute_mauve_score,
    requires=["generated_text", "reference_text"],
    group="distribution"
)

def compute_sentencebert_similarity(generated_text, reference_text, model_name="all-MiniLM-L6-v2", batch_size=32, device=None, **kwargs):
    """Compute SentenceBERT similarity score."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Ensure inputs are lists
        if isinstance(generated_text, str):
            generated_text = [generated_text]
        if isinstance(reference_text, str):
            reference_text = [reference_text]
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        model = SentenceTransformer(model_name, device=device)
        
        # Encode sentences
        generated_embeddings = model.encode(generated_text, batch_size=batch_size, show_progress_bar=False)
        reference_embeddings = model.encode(reference_text, batch_size=batch_size, show_progress_bar=False)
        
        # Compute similarities
        similarities = []
        for gen_emb, ref_emb in zip(generated_embeddings, reference_embeddings):
            # Normalize vectors
            gen_norm = gen_emb / np.linalg.norm(gen_emb)
            ref_norm = ref_emb / np.linalg.norm(ref_emb)
            
            # Compute cosine similarity
            similarity = np.dot(gen_norm, ref_norm)
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        return {"sentencebert_similarity": avg_similarity}
    except Exception as e:
        logger.error(f"Error computing SentenceBERT similarity: {e}")
        return {"sentencebert_similarity": 0.0}

registry.register(
    name="sentencebert",
    compute_fn=compute_sentencebert_similarity,
    requires=["generated_text", "reference_text"],
    group="semantic"
)

# Function to compute all metrics
def compute_all_metrics(generated_tokens, reference_tokens, tokenizer=None, 
                      compute_groups=None, skip_groups=None, **kwargs):
    """
    Compute all registered metrics or those in specific groups.
    
    Args:
        generated_tokens: Generated token IDs
        reference_tokens: Reference token IDs
        tokenizer: Tokenizer for decoding texts
        compute_groups: List of groups to compute (None = all)
        skip_groups: List of groups to skip
        **kwargs: Additional inputs for specific metrics
        
    Returns:
        Dictionary of computed metrics
    """
    results = {}
    
    # Determine which groups to compute
    if compute_groups is None:
        metric_names = registry.get_all_metrics()
    else:
        metric_names = []
        for group in compute_groups:
            metric_names.extend(registry.get_metrics_by_group(group))
    
    # Filter out skipped groups
    if skip_groups:
        excluded = []
        for group in skip_groups:
            excluded.extend(registry.get_metrics_by_group(group))
        metric_names = [name for name in metric_names if name not in excluded]
    
    # Prepare inputs for text-based metrics if tokenizer is provided
    inputs = {
        "generated_tokens": generated_tokens,
        "reference_tokens": reference_tokens,
        **kwargs
    }
    
    if tokenizer:
        inputs["generated_text"] = tokenizer.decode(generated_tokens)
        inputs["reference_text"] = tokenizer.decode(reference_tokens)
    
    # Compute each metric
    for metric_name in metric_names:
        # Skip metrics that require text if tokenizer is not provided
        metric_info = registry.get_metric_info(metric_name)
        requires_text = any(req.endswith('_text') for req in metric_info.get('requires', []))
        
        if requires_text and not tokenizer:
            continue
            
        # Compute the metric
        result = registry.compute_metric(metric_name, **inputs)
        
        # Add to results
        if isinstance(result, dict):
            results.update(result)
        elif result is not None:
            results[metric_name] = result
    
    return results

# Export functions
__all__ = [
    'registry', 
    'compute_all_metrics',
    'METRIC_RANGES',
    'calculate_ngram_overlap',
    'lcs_length'
]
