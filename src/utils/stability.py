"""Numerical stability utilities for PVF-U."""

import numpy as np
from typing import List, Tuple


def safe_softmax(logits: np.ndarray, clip_value: float = 20.0) -> np.ndarray:
    """Compute softmax with numerical stability.
    
    Args:
        logits: Input logits
        clip_value: Clipping value for logits
        
    Returns:
        Softmax probabilities
    """
    clipped = np.clip(logits, -clip_value, clip_value)
    max_logit = np.max(clipped)
    exp_logits = np.exp(clipped - max_logit)
    return exp_logits / (np.sum(exp_logits) + 1e-12)


def safe_entropy(probs: np.ndarray, eps_log: float = 1e-12) -> float:
    """Compute entropy with numerical stability.
    
    Args:
        probs: Probability distribution
        eps_log: Small constant to prevent log(0)
        
    Returns:
        Entropy value
    """
    return -np.sum(probs * np.log(probs + eps_log))


def safe_variance(values: np.ndarray, eps_var: float = 1e-8) -> float:
    """Compute variance with numerical stability.
    
    Args:
        values: Input array
        eps_var: Small constant to prevent zero variance
        
    Returns:
        Variance value
    """
    return float(np.var(values) + eps_var)


def safe_exponential(values: np.ndarray, clip_value: float = 100.0) -> np.ndarray:
    """Compute exponential with numerical stability.
    
    Args:
        values: Input array
        clip_value: Clipping value to prevent overflow
        
    Returns:
        Exponential values
    """
    clipped = np.clip(values, -clip_value, clip_value)
    return np.exp(clipped)


def check_numerical_stability(
    tensor: np.ndarray,
    raise_on_error: bool = True
) -> Tuple[bool, str]:
    """Check for numerical stability issues.
    
    Args:
        tensor: Input tensor to check
        raise_on_error: Whether to raise exception on error
        
    Returns:
        Tuple of (is_stable, error_message)
    """
    if np.any(np.isnan(tensor)):
        msg = f"NaN detected in tensor: {tensor}"
        if raise_on_error:
            raise RuntimeError(msg)
        return False, msg
    
    if np.any(np.isinf(tensor)):
        msg = f"Inf detected in tensor: {tensor}"
        if raise_on_error:
            raise RuntimeError(msg)
        return False, msg
    
    return True, "OK"


def clip_gradients(gradients: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """Clip gradients to prevent exploding gradients.
    
    Args:
        gradients: Input gradients
        max_norm: Maximum L2 norm
        
    Returns:
        Clipped gradients
    """
    norm = np.linalg.norm(gradients)
    if norm > max_norm:
        return gradients * (max_norm / norm)
    return gradients
