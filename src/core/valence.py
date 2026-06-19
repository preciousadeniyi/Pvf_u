"""Valence computation and BIBO stability guarantees."""

import numpy as np
from typing import Tuple


def compute_total_valence(v_pri: float, alpha: float, v_soc: float) -> float:
    """Compute total integrated valence.
    
    v_tot = v_pri + α·v_soc
    
    Args:
        v_pri: Intrinsic valence from Layer 0
        alpha: Relational gate from Layer 1
        v_soc: Social valence from Layer 1
        
    Returns:
        Total valence (bounded by Theorem 6.2: |v_tot| ≤ 2)
    """
    v_tot = v_pri + alpha * v_soc
    
    # BIBO guarantee: |v_tot| ≤ 2
    # Clip to ensure stability (should not be necessary if layers are correct)
    v_tot = np.clip(v_tot, -2.0, 2.0)
    
    return float(v_tot)


def compute_panic_signal(
    delta: np.ndarray,
    panic_threshold: np.ndarray
) -> Tuple[float, bool]:
    """Compute panic signal for Biological Veto.
    
    || |δ| ⊙ θ_panic^{-1} ||_2
    
    Args:
        delta: Prediction error vector
        panic_threshold: Modality-specific panic thresholds
        
    Returns:
        Tuple of (panic_magnitude, is_panic)
    """
    normalized = np.abs(delta) / (panic_threshold + 1e-8)
    magnitude = np.linalg.norm(normalized)
    is_panic = magnitude > 1.0
    return magnitude, is_panic


def compute_temporal_derivative(
    current_delta: np.ndarray,
    prev_delta: np.ndarray,
    tau: float
) -> float:
    """Compute temporal derivative |\dot{δ}|.
    
    Args:
        current_delta: Current prediction error
        prev_delta: Previous prediction error
        tau: Time constant
        
    Returns:
        Temporal derivative magnitude
    """
    derivative = (np.abs(current_delta) - np.abs(prev_delta)) / tau
    return float(derivative)
