



"""Generative model for prediction generation."""

import numpy as np
from typing import Dict, Any, Optional


class GenerativeModel:
    """Generative model p(s | ψ) for prediction generation.
    
    Generates predictions based on hidden states ψ.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize generative model.
        
        Args:
            config: Configuration dictionary
        """
        self.hidden_dim = config.get('hidden_dim', 64)
        self.sensor_dim = config.get('sensor_dim', 1)
        self.learning_rate = config.get('learning_rate', 0.01)
        
        # Simple linear generative model
        self.weights = np.random.randn(self.sensor_dim, self.hidden_dim) * 0.1
        self.bias = np.zeros(self.sensor_dim)
        
        self.hidden_state = np.zeros(self.hidden_dim)
    
    def generate_prediction(self, hidden_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate prediction from hidden state.
        
        Args:
            hidden_state: Optional hidden state (uses internal if None)
            
        Returns:
            Predicted sensory input
        """
        if hidden_state is None:
            hidden_state = self.hidden_state
        return self.weights @ hidden_state + self.bias
    
    def update(self, prediction_error: np.ndarray):
        """Update model parameters (simplified learning).
        
        Args:
            prediction_error: δ = predicted - actual
        """
        # Simple Hebbian-like update
        self.weights += self.learning_rate * np.outer(prediction_error, self.hidden_state)
        self.bias += self.learning_rate * prediction_error
