
"""Metrics for PVF-U evaluation: PEI, CreativeTension, Narrative Coherence."""

import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient for a list of values.
    
    Args:
        values: List of values (e.g., incoming powers)
        
    Returns:
        Gini coefficient in [0, 1]
    """
    n = len(values)
    if n == 0:
        return 0.0
    
    sorted_values = np.sort(values)
    numerator = 2 * np.sum((np.arange(1, n + 1) * sorted_values))
    denominator = n * np.sum(sorted_values)
    if denominator == 0:
        return 0.0
    return (numerator / denominator) - (n + 1) / n


def compute_pei(
    influence_weights: List[Dict[int, float]],
    N: int,
    epsilon: float = 1e-8
) -> float:
    """Compute Power Equity Index (PEI).
    
    PEI = (1 - G_in) × (1 - C_cent) × MarginProt × ProcJustice
    
    Args:
        influence_weights: List of dicts mapping neighbor_id -> weight for each agent
        N: Number of agents
        epsilon: Numerical stability
        
    Returns:
        PEI in [0, 1]
    """
    # Compute incoming powers
    powers = []
    for weights in influence_weights:
        powers.append(sum(weights.values()))
    
    # Gini coefficient
    gini = compute_gini_coefficient(powers)
    
    # Centralization
    total_power = sum(powers)
    if total_power == 0:
        centralization = 0.0
    else:
        centralization = max(powers) / (total_power + epsilon)
    
    # Margin protection
    mean_power = np.mean(powers) if powers else 0.0
    min_power = min(powers) if powers else 0.0
    margin_protection = min_power / (mean_power + epsilon)
    margin_protection = min(margin_protection, 1.0)
    
    # Procedural justice (simplified: inverse of std/mean)
    if mean_power == 0:
        procedural_justice = 1.0
    else:
        std_power = np.std(powers)
        procedural_justice = 1.0 - std_power / (mean_power + epsilon)
        procedural_justice = max(min(procedural_justice, 1.0), 0.0)
    
    pei = (1 - gini) * (1 - centralization) * margin_protection * procedural_justice
    return max(min(pei, 1.0), 0.0)


def compute_creative_tension(
    alpha: List[float],
    narratives: List[np.ndarray],
    relational_costs: List[float],
    multi_information_history: List[float],
    gamma: float = 0.5,
    tau: float = 10.0,
    epsilon: float = 1e-8
) -> float:
    """Compute CreativeTension Ψ(t).
    
    Ψ = tanh(α_mean · D_div · (1 + [\dot{I}]^+) · exp(-γ·R_mean))
    
    Args:
        alpha: List of gate values for each agent
        narratives: List of narrative embeddings for each agent
        relational_costs: List of relational costs for each agent
        multi_information_history: History of collective multi-information I(t)
        gamma: Relational cost weight
        tau: Time constant for derivative
        epsilon: Numerical stability
        
    Returns:
        CreativeTension in [0, 1)
    """
    N = len(alpha)
    if N == 0:
        return 0.0
    
    # Narrative divergence
    mean_narrative = np.mean(narratives, axis=0)
    divergences = [np.linalg.norm(n - mean_narrative) for n in narratives]
    max_div = max(divergences) if divergences else 0.0
    if max_div > epsilon:
        D_div = np.mean([d / (max_div + epsilon) for d in divergences])
    else:
        D_div = 0.0
    
    # Derivative of multi-information
    if len(multi_information_history) < 2:
        I_dot = 0.0
    else:
        I_dot = (multi_information_history[-1] - multi_information_history[-2]) / tau
        I_dot = max(0.0, I_dot)
    
    # Means
    alpha_mean = np.mean(alpha)
    R_mean = np.mean(relational_costs) if relational_costs else 0.0
    
    # CreativeTension
    psi = alpha_mean * D_div * (1.0 + I_dot) * np.exp(-gamma * R_mean)
    return float(np.tanh(psi))


def compute_narrative_coherence(
    narratives: List[np.ndarray],
    epsilon: float = 1e-8
) -> float:
    """Compute Narrative Coherence (NC).
    
    NC = mean pairwise cosine similarity
    
    Args:
        narratives: List of narrative embeddings
        epsilon: Numerical stability
        
    Returns:
        Narrative Coherence in [0, 1]
    """
    N = len(narratives)
    if N < 2:
        return 1.0
    
    similarities = []
    for i in range(N):
        for j in range(i + 1, N):
            min_K = min(len(narratives[i]), len(narratives[j]))
            emb1 = narratives[i][:min_K]
            emb2 = narratives[j][:min_K]
            if np.linalg.norm(emb1) < epsilon or np.linalg.norm(emb2) < epsilon:
                sim = 0.0
            else:
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + epsilon)
            similarities.append(sim)
    
    return float(np.mean(similarities))
