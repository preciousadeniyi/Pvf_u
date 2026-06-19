"""Dark Room environment for testing solitary survival."""

import numpy as np
from typing import List, Dict, Tuple


class DarkRoom:
    """Dark Room environment.
    
    A perfectly predictable environment with δ = 0 and no social partners.
    Used to test Theorem 6.1 (Solitary Survival).
    """
    
    def __init__(self, dimension: int = 1):
        """Initialize Dark Room environment.
        
        Args:
            dimension: Sensory dimension
        """
        self.dimension = dimension
        self.steps = 0
        self.max_steps = 5000
        self.agents = {}
        
        # Constant observation (perfectly predictable)
        self.observation = np.zeros(dimension)
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Process actions (all actions lead to same observation).
        
        Args:
            actions: Dictionary mapping agent_id -> action
            
        Returns:
            Tuple of (observation, prediction_errors, episode_complete)
        """
        self.steps += 1
        
        # δ = 0 for all agents (perfectly predictable)
        deltas = np.zeros((len(actions), self.dimension))
        
        # Episode never completes in Dark Room
        complete = False
        
        return self.observation.copy(), deltas, complete
    
    def get_neighbors(self, agent_id: int) -> List:
        """No neighbors in Dark Room."""
        return []
    
    def observe(self, agent_id: int) -> np.ndarray:
        """Get observation."""
        return self.observation.copy()
    
    def is_complete(self) -> bool:
        """Check if episode is complete."""
        return self.steps >= self.max_steps
    
    def reset(self):
        """Reset environment."""
        self.steps = 0
        return self.observation.copy()
