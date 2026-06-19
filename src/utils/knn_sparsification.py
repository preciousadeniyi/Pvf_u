"""Biased k-NN sparsification for efficient social coupling."""

import numpy as np
from typing import List, Dict, Tuple, Optional
import heapq


class BiasedKNN:
    """k-Nearest Neighbor sparsification with oversampling of marginalized agents.
    
    Features:
    - Efficient k-NN selection with O(N log N)
    - Oversampling of bottom quartile by 1.5x
    - Supports dynamic neighbor selection
    """
    
    def __init__(self, k: int = 50, oversample_factor: float = 1.5):
        """Initialize Biased KNN.
        
        Args:
            k: Number of neighbors to select
            oversample_factor: Oversampling factor for marginalized agents
        """
        self.k = k
        self.oversample_factor = oversample_factor
        
    def select_neighbors(
        self,
        agent_id: int,
        all_agents: List[any],
        powers: List[float],
        positions: Optional[np.ndarray] = None
    ) -> List[int]:
        """Select neighbors with biased oversampling.
        
        Args:
            agent_id: ID of the current agent
            all_agents: List of all agents (with positions/narratives)
            powers: List of powers for each agent
            positions: Optional positions for spatial k-NN
            
        Returns:
            List of selected neighbor IDs
        """
        N = len(all_agents)
        if N <= 1:
            return []
        
        # Compute distances (spatial or narrative)
        if positions is not None:
            distances = self._compute_spatial_distances(agent_id, positions)
        else:
            distances = self._compute_narrative_distances(agent_id, all_agents)
        
        # Sort by distance
        sorted_neighbors = sorted([(distances[i], i) for i in range(N) if i != agent_id], 
                                 key=lambda x: x[0])
        
        # Identify bottom quartile for oversampling
        sorted_powers = np.sort(powers)
        if len(sorted_powers) > 0:
            quartile_idx = max(0, int(len(sorted_powers) * 0.25))
            quartile_power = sorted_powers[quartile_idx]
        else:
            quartile_power = 0.0
        
        marginalized = [i for i in range(N) if i != agent_id and powers[i] <= quartile_power]
        
        # Select neighbors with oversampling
        selected = []
        selected_set = set()
        
        # First pass: prioritize marginalized agents
        for idx, (_, neighbor_id) in enumerate(sorted_neighbors):
            if len(selected) >= self.k:
                break
            if neighbor_id in marginalized and np.random.rand() < self.oversample_factor / self.k:
                selected.append(neighbor_id)
                selected_set.add(neighbor_id)
        
        # Second pass: fill remaining slots with nearest neighbors
        for _, neighbor_id in sorted_neighbors:
            if len(selected) >= self.k:
                break
            if neighbor_id not in selected_set:
                selected.append(neighbor_id)
                selected_set.add(neighbor_id)
        
        return selected
    
    def _compute_spatial_distances(self, agent_id: int, positions: np.ndarray) -> Dict[int, float]:
        """Compute spatial distances between agents."""
        distances = {}
        for i, pos in enumerate(positions):
            if i != agent_id:
                distances[i] = np.linalg.norm(positions[agent_id] - pos)
        return distances
    
    def _compute_narrative_distances(self, agent_id: int, all_agents: List) -> Dict[int, float]:
        """Compute narrative distances between agents."""
        distances = {}
        agent_narrative = getattr(all_agents[agent_id], 'narrative', None)
        if agent_narrative is None:
            return {i: np.random.rand() for i in range(len(all_agents)) if i != agent_id}
        
        for i, other in enumerate(all_agents):
            if i != agent_id:
                other_narrative = getattr(other, 'narrative', None)
                if other_narrative is None:
                    distances[i] = 1.0
                else:
                    # Cosine distance (1 - similarity)
                    sim = self._compute_similarity(agent_narrative, other_narrative)
                    distances[i] = 1.0 - sim
        return distances
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        min_K = min(len(emb1), len(emb2))
        e1 = emb1[:min_K]
        e2 = emb2[:min_K]
        if np.linalg.norm(e1) == 0 or np.linalg.norm(e2) == 0:
            return 0.0
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
