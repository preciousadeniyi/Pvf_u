"""Differential privacy utilities for sensitive deployments."""

import numpy as np
from typing import Optional


def add_differential_privacy(
    tensor: np.ndarray,
    epsilon: float = 0.1,
    delta: float = 1e-5,
    sensitivity: Optional[float] = None
) -> np.ndarray:
    """Add differential privacy noise to a tensor.
    
    Implements Gaussian mechanism for differential privacy.
    
    Args:
        tensor: Input tensor
        epsilon: Privacy budget (ε)
        delta: Failure probability (δ)
        sensitivity: L2 sensitivity (auto-computed if None)
        
    Returns:
        Tensor with added noise
    """
    if sensitivity is None:
        sensitivity = np.linalg.norm(tensor) + 1e-8
    
    # Compute noise scale
    noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    # Add Gaussian noise
    noise = np.random.randn(*tensor.shape) * noise_scale
    return tensor + noise


def compute_privacy_budget(
    num_queries: int,
    total_epsilon: float = 1.0,
    total_delta: float = 1e-5
) -> Tuple[float, float]:
    """Compute per-query privacy budget under composition.
    
    Args:
        num_queries: Number of queries
        total_epsilon: Total epsilon budget
        total_delta: Total delta budget
        
    Returns:
        Tuple of (per_query_epsilon, per_query_delta)
    """
    per_query_epsilon = total_epsilon / (2 * np.sqrt(num_queries))
    per_query_delta = total_delta / num_queries
    return per_query_epsilon, per_query_delta
