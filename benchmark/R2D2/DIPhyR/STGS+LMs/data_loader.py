"""
Functions for loading datasets and preparing targets.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger("data_loader")


def load_dataset(dataset_path):
    """
    Load the dataset of target sentences from a JSON file.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        dataset: The loaded dataset as a dictionary
    """
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        logger.info(f"Loaded {len(dataset['samples'])} samples")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def prepare_targets(dataset, target_indices=None):
    """
    Prepare the list of targets, optionally filtering by indices.
    
    Args:
        dataset: The loaded dataset
        target_indices: Optional list of indices to select specific targets
        
    Returns:
        targets: List of prepared target dictionaries
    """
    targets = dataset.get("samples", [])
    
    # Filter targets if indices are provided
    if target_indices is not None:
        targets = [targets[i] for i in target_indices if i < len(targets)]
        logger.info(f"Selected {len(targets)} targets based on provided indices")
    
    return targets


def get_k_values(targets):
    """
    Extract the set of unique k values from the targets.
    
    Args:
        targets: List of target dictionaries
        
    Returns:
        k_values: Sorted list of unique k values
    """
    return sorted(set(target.get("k_target", 0) for target in targets))
