"""Cooperative Navigation for generalization experiments (SI S7)."""

import numpy as np
from typing import List, Dict, Tuple, Optional


class CooperativeNavigation:
    """Cooperative Navigation environment.
    
    Agents cooperate to reach goals in a 2D grid.
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        grid_size: int = 10,
        obstacle_probability: float = 0.1
    ):
        """Initialize Cooperative Navigation environment.
        
        Args:
            num_agents: Number of agents
            grid_size: Size of the grid
            obstacle_probability: Probability of obstacles
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        
        # Agent positions
        self.positions = np.random.randint(0, grid_size, (num_agents, 2))
        
        # Goals (different for each agent)
        self.goals = self._generate_goals()
        
        # Obstacles
        self.obstacles = self._generate_obstacles(obstacle_probability)
        
        # Episode state
        self.episode_complete = False
        self.steps = 0
        self.max_steps = 1000
        self.completion_time = None
    
    def _generate_goals(self) -> np.ndarray:
        """Generate goals for each agent."""
        goals = np.random.randint(0, self.grid_size, (self.num_agents, 2))
        # Ensure goals are not on obstacles (if possible)
        return goals
    
    def _generate_obstacles(self, prob: float) -> np.ndarray:
        """Generate obstacles."""
        obstacles = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if np.random.rand() < prob:
                    obstacles.append((i, j))
        return np.array(obstacles) if obstacles else np.zeros((0, 2))
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Process actions and update positions.
        
        Args:
            actions: Dictionary mapping agent_id -> action vector
            
        Returns:
            Tuple of (positions, prediction_errors, episode_complete)
        """
        self.steps += 1
        
        # Apply actions
        for agent_id, action in actions.items():
            # Action is delta in x,y
            self.positions[agent_id] += action[:2]
            # Clip to grid
            self.positions[agent_id] = np.clip(
                self.positions[agent_id],
                0, self.grid_size - 1
            )
            # Check obstacles
            pos_tuple = tuple(self.positions[agent_id].astype(int))
            if len(self.obstacles) > 0 and any(np.all(self.obstacles == pos_tuple, axis=1)):
                # Collision with obstacle: revert
                self.positions[agent_id] -= action[:2]
                self.positions[agent_id] = np.clip(
                    self.positions[agent_id],
                    0, self.grid_size - 1
                )
        
        # Check completion
        distances = np.linalg.norm(self.positions - self.goals, axis=1)
        if np.all(distances < 0.5):
            self.episode_complete = True
            if self.completion_time is None:
                self.completion_time = self.steps
        
        # Prediction errors: distance to goal
        deltas = distances
        
        return self.positions.copy(), deltas, self.episode_complete
    
    def get_neighbors(self, agent_id: int) -> List:
        """Get neighboring agents within distance threshold."""
        threshold = 3.0
        neighbors = []
        for i in range(self.num_agents):
            if i != agent_id:
                dist = np.linalg.norm(self.positions[agent_id] - self.positions[i])
                if dist < threshold:
                    neighbors.append(i)
        return neighbors
    
    def observe(self, agent_id: int) -> np.ndarray:
        """Get observation for an agent."""
        # Agent sees its position and goal
        return np.concatenate([
            self.positions[agent_id],
            self.goals[agent_id]
        ])
    
    def is_complete(self) -> bool:
        """Check if episode is complete."""
        return self.episode_complete or self.steps >= self.max_steps
    
    def reset(self):
        """Reset environment."""
        self.positions = np.random.randint(0, self.grid_size, (self.num_agents, 2))
        self.goals = self._generate_goals()
        self.episode_complete = False
        self.steps = 0
        self.completion_time = None
        return self.positions.copy()
