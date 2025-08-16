"""Utility functions for wandb logging organization."""


def reorganize_metrics_for_wandb(metrics):
    """Reorganize metrics for wandb logging to group std metrics separately.
    
    This function moves all metrics ending with _std that are under eval/ to eval_std/
    to avoid cluttering the main eval section in wandb.
    
    Args:
        metrics: Dictionary of metrics with original keys
        
    Returns:
        Dictionary with reorganized keys for wandb logging
    """
    reorganized_metrics = {}
    
    for key, value in metrics.items():
        if key.endswith("_std") and key.startswith("eval/"):
            # Move eval std metrics to eval_std/ section
            new_key = key.replace("eval/", "eval_std/")
            reorganized_metrics[new_key] = value
        else:
            # Keep non-std metrics as they are
            reorganized_metrics[key] = value
    
    return reorganized_metrics


def log_metrics_to_wandb(metrics, step, wandb_instance):
    """Log metrics to wandb with reorganized std metrics.
    
    Args:
        metrics: Dictionary of metrics with original keys
        step: Current training step
        wandb_instance: wandb instance to log to
    """
    reorganized_metrics = reorganize_metrics_for_wandb(metrics)
    wandb_instance.log(reorganized_metrics, step=step) 