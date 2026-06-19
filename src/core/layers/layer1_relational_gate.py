


"""Layer 1: Relational Gate - Dynamic social permeability."""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class RelationalGateConfig:
    """Configuration for Relational Gate."""
    gamma: float = 0.5          # Relational cost weight
    window_size: int = 100     # Window for I(t) estimation
    epsilon: float = 1e-8      # Numerical stability
    gate_sensitivity: float = 1.0  # Sensitivity of sigmoid


class RelationalGate:
    """Layer 1: Relational Gate.
    
    Implements:
    - Social valence: v_soc = tanh(Σ w_ab·s_ab·(π_b·δ_b) - γ·R_a)
    - Relational cost: R_a = Σ divergence + communication loss
    - Epistemic gate: α = σ((E[I] - IG) / sqrt(Var[IG] + Var[I]))
    """
    
    def __init__(self, config: dict):
        """Initialize Relational Gate.
        
        Args:
            config: Configuration dictionary with gamma, window_size, etc.
        """
        self.gamma = config.get('gamma', 0.5)
        self.window_size = config.get('window_size', 100)
        self.epsilon = config.get('epsilon', 1e-8)
        self.gate_sensitivity = config.get('gate_sensitivity', 1.0)
        
        self.ig_history = []
        self.i_history = []
        self.agent = None
    
    def compute_social_valence(
        self,
        agent: 'PVFAgent',
        neighbors: List['PVFAgent'],
        influence_weights: Dict[int, float]
    ) -> Tuple[float, float]:
        """Compute social valence v_soc.
        
        v_soc = tanh(Σ w_ab·s_ab·(π_b·δ_b) - γ·R_a)
        
        Args:
            agent: Current agent
            neighbors: List of neighboring agents
            influence_weights: Dictionary mapping neighbor_id -> weight
            
        Returns:
            Tuple of (social_valence, relational_cost)
        """
        self.agent = agent
        total_influence = 0.0
        total_cost = 0.0
        
        for neighbor in neighbors:
            neighbor_id = neighbor.agent_id
            
            # Get influence weight
            w_ab = influence_weights.get(neighbor_id, 0.0)
            
            # Compute similarity between narratives
            s_ab = self.compute_similarity(
                agent.narrative.embedding,
                neighbor.narrative.embedding
            )
            
            # Compute neighbor's precision
            variance = np.var(neighbor.history['delta']) if len(neighbor.history['delta']) > 1 else 1.0
            pi_b = 1.0 / (variance + self.epsilon)
            
            # Neighbor's prediction error
            delta_b = np.mean(np.abs(neighbor.state.delta))
            
            # Weighted influence
            total_influence += w_ab * s_ab * pi_b * delta_b
            
            # Relational cost
            cost = self.compute_relational_cost(agent, neighbor)
            total_cost += cost
        
        # Compute social valence
        v_soc = np.tanh(total_influence - self.gamma * total_cost)
        return float(v_soc), float(total_cost)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between narrative embeddings."""
        if np.linalg.norm(emb1) < self.epsilon or np.linalg.norm(emb2) < self.epsilon:
            return 0.0
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + self.epsilon))
    
    def compute_relational_cost(self, agent: 'PVFAgent', neighbor: 'PVFAgent') -> float:
        """Compute relational cost R_a.
        
        R_a = Σ divergence + communication_loss
        
        Args:
            agent: Current agent
            neighbor: Neighboring agent
            
        Returns:
            Relational cost value
        """
        # Compute divergence between prediction errors
        delta_a = np.mean(np.abs(agent.state.delta))
        delta_b = np.mean(np.abs(neighbor.state.delta))
        max_delta = max(delta_a, delta_b, self.epsilon)
        divergence = abs(delta_a - delta_b) / max_delta
        
        # Communication loss (simulated)
        communication_loss = 0.0  # In practice, this would be a learned signal
        
        return divergence + communication_loss
    
    def compute_gate(self, ig: float, collective_i: float) -> float:
        """Compute epistemic gate α.
        
        α = σ((E[I] - IG) / sqrt(Var[IG] + Var[I]))
        
        Args:
            ig: Agent's Information Gain
            collective_i: Collective multi-information
            
        Returns:
            Gate value α in [0, 1]
        """
        self.ig_history.append(ig)
        self.i_history.append(collective_i)
        
        # Need enough history for variance estimation
        if len(self.ig_history) < self.window_size:
            return 0.5  # Neutral gate during warm-up
        
        # Sliding window estimates
        recent_ig = self.ig_history[-self.window_size:]
        recent_i = self.i_history[-self.window_size:]
        
        mean_i = np.mean(recent_i)
        var_ig = np.var(recent_ig)
        var_i = np.var(recent_i)
        
        # Gate input
        z = self.gate_sensitivity * (mean_i - ig) / (np.sqrt(var_ig + var_i) + self.epsilon)
        
        # Sigmoid
        alpha = 1.0 / (1.0 + np.exp(-z))
        return float(alpha)
    
    def reset(self):
        """Reset the Relational Gate."""
        self.ig_history = []
        self.i_history = []
