"""Shape Game: Collaborative Tangram arrangement benchmark."""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any


class ShapeGame:
    """Shape Game environment.
    
    Two agents collaboratively arrange 7 Tangram pieces into a meaningful
    configuration through mutual negotiation.
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        pieces: int = 7,
        grid_size: int = 10
    ):
        """Initialize Shape Game environment.
        
        Args:
            num_agents: Number of agents
            pieces: Number of Tangram pieces
            grid_size: Size of the grid
        """
        self.num_agents = num_agents
        self.pieces = pieces
        self.grid_size = grid_size
        
        # Current configuration (positions of pieces)
        self.configuration = np.random.rand(pieces, 2) * grid_size
        
        # Target configuration (hidden, for evaluation)
        self.target_configuration = self._generate_target()
        
        # Episode state
        self.episode_complete = False
        self.steps = 0
        self.max_steps = 5000
        self.commitments = [False] * num_agents
        
        # Agent registry
        self.agents = {}
        self.agent_positions = {}
    
    def _generate_target(self) -> np.ndarray:
        """Generate target configuration."""
        # Simple pattern: arranged in a circle
        angles = np.linspace(0, 2*np.pi, self.pieces + 1)[:-1]
        radius = self.grid_size * 0.3
        center = self.grid_size / 2
        target = np.array([
            [center + radius * np.cos(a), center + radius * np.sin(a)]
            for a in angles
        ])
        return target
    
    def step(
        self,
        actions: Dict[int, np.ndarray],
        narratives: Optional[Dict[int, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Process agent actions and update state.
        
        Args:
            actions: Dictionary mapping agent_id -> action vector
            narratives: Dictionary mapping agent_id -> narrative embedding
            
        Returns:
            Tuple of (configuration, prediction_errors, episode_complete)
        """
        self.steps += 1
        
        # Apply actions to configurations
        for agent_id, action in actions.items():
            if agent_id < self.pieces:  # Each agent controls pieces
                piece_idx = agent_id % self.pieces
                self.configuration[piece_idx] += action[:2]
                # Clip to grid
                self.configuration[piece_idx] = np.clip(
                    self.configuration[piece_idx],
                    0, self.grid_size
                )
        
        # Check for mutual commitment
        if all(self.commitments):
            if self._check_configurational_convergence():
                if narratives and self._check_semantic_agreement(narratives):
                    self.episode_complete = True
        
        # Compute prediction errors (relative to target)
        deltas = self._compute_prediction_errors()
        
        return self.configuration.copy(), deltas, self.episode_complete
    
    def _compute_prediction_errors(self) -> np.ndarray:
        """Compute prediction errors for each agent."""
        # Each agent sees the configuration and predicts target
        errors = []
        for agent_id in range(self.num_agents):
            # Prediction is current configuration
            predicted = self.configuration.flatten()
            # Target is hidden, but agents have internal models
            # Simplified: use distance to target
            error = np.linalg.norm(self.configuration - self.target_configuration, axis=1)
            errors.append(error)
        return np.array(errors)
    
    def _check_configurational_convergence(self) -> bool:
        """Check if configuration matches target."""
        distances = np.linalg.norm(self.configuration - self.target_configuration, axis=1)
        return np.all(distances < 0.1)
    
    def _check_semantic_agreement(self, narratives: Dict[int, np.ndarray]) -> bool:
        """Check if agents agree on narrative meaning."""
        if len(narratives) < 2:
            return True
        
        # Compute pairwise cosine similarities
        embeddings = list(narratives.values())
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                )
                similarities.append(sim)
        
        return np.mean(similarities) > 0.85
    
    def commit(self, agent_id: int):
        """Agent commits to current configuration."""
        self.commitments[agent_id] = True
    
    def get_neighbors(self, agent_id: int) -> List[Any]:
        """Get neighboring agents."""
        # In Shape Game, all agents are neighbors
        return [self.agents[i] for i in range(self.num_agents) if i != agent_id]
    
    def observe(self, agent_id: int) -> np.ndarray:
        """Get observation for an agent."""
        # Agent sees the current configuration
        return self.configuration.flatten()
    
    def is_complete(self) -> bool:
        """Check if episode is complete."""
        return self.episode_complete or self.steps >= self.max_steps
    
    def reset(self):
        """Reset environment."""
        self.configuration = np.random.rand(self.pieces, 2) * self.grid_size
        self.target_configuration = self._generate_target()
        self.episode_complete = False
        self.steps = 0
        self.commitments = [False] * self.num_agents
        return self.configuration.copy()
