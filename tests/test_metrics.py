"""Unit tests for metrics computation."""

import pytest
import numpy as np

from pvf_u.utils.metrics import (
    compute_gini_coefficient,
    compute_pei,
    compute_creative_tension,
    compute_narrative_coherence,
)


def test_gini_coefficient():
    """Test Gini coefficient computation."""
    # Perfect equality
    assert compute_gini_coefficient([1, 1, 1, 1]) == 0.0
    
    # Perfect inequality
    assert compute_gini_coefficient([0, 0, 0, 4]) > 0.7
    
    # Known values
    vals = [1, 2, 3, 4, 5]
    gini = compute_gini_coefficient(vals)
    # Gini should be between 0 and 1
    assert 0.0 <= gini <= 1.0


def test_pei():
    """Test Power Equity Index computation."""
    # Perfect equity
    weights = [{1: 0.25, 2: 0.25}, {0: 0.25, 2: 0.25}, {0: 0.25, 1: 0.25}, {0: 0.25, 1: 0.25}]
    pei = compute_pei(weights, N=4)
    assert pei > 0.9
    
    # Perfect inequity
    weights = [{1: 1.0, 2: 0.0}, {0: 0.0, 2: 0.0}, {0: 0.0, 1: 0.0}, {0: 0.0, 1: 0.0}]
    pei = compute_pei(weights, N=4)
    assert pei < 0.1


def test_creative_tension():
    """Test CreativeTension computation."""
    alpha = [0.5, 0.6, 0.4]
    narratives = [
        np.array([1.0, 0.0]),
        np.array([0.8, 0.6]),
        np.array([0.0, 1.0])
    ]
    costs = [0.1, 0.2, 0.3]
    i_history = [0.1, 0.15, 0.2, 0.25, 0.3]
    
    psi = compute_creative_tension(alpha, narratives, costs, i_history)
    assert 0.0 <= psi < 1.0


def test_narrative_coherence():
    """Test Narrative Coherence computation."""
    narratives = [
        np.array([1.0, 0.0]),
        np.array([0.8, 0.6]),
        np.array([0.0, 1.0])
    ]
    
    nc = compute_narrative_coherence(narratives)
    assert 0.0 <= nc <= 1.0
