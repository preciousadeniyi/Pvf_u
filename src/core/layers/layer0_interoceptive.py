"""Layer 0: Interoceptive Core - Guarantees solitary survival."""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class InteroceptiveConfig:
    """Configuration for Interoceptive Core."""
    beta: float = 1.0          # Weight for Information Gain
    tau: float = 10.0          # Time constant for temporal derivative
    epsilon: float = 1e-8      # Numerical stability
    precision_window: int = 100  # Window for variance estimation


class InteroceptiveCore:
    """Layer 0: Interoceptive Core.
    
    Implements:
    - Prediction error computation: δ = predicted - actual
    - Temporal derivative: |\dot{δ}|
    - Information Gain: IG = H[posterior] - E[H[posterior|action]]
    - Intrinsic valence: v_pri = tanh(-π·δ·|\dot{δ}| + β·IG)
    """
    
    def __init__(self, config: dict):
        """Initialize Interoceptive Core.
        
        Args:
            config: Configuration dictionary with beta, tau, epsilon, etc.
        """
        self.beta = config.get('beta', 1.0)
        self.tau = config.get('tau', 10.0)
        self.epsilon = config.get('epsilon', 1e-8)
        self.precision_window = config.get('precision_window', 100)
        
        self.prev_delta = None
        self.delta_history = []
        self.ig_history = []
        
        # Internal generative model (simplified)
        self.posterior = None
        self.posterior_given_action = {}
    
    def compute_prediction_error(self, observation: np.ndarray) -> np.ndarray:
        """Compute prediction error δ = predicted - actual.
        
        Args:
            observation: Actual sensory input
            
        Returns:
            Prediction error vector
        """
        predicted = self._generate_prediction()
        delta = predicted - observation
        self.delta_history.append(delta)
        
        # Keep history bounded
        if len(self.delta_history) > self.precision_window * 2:
            self.delta_history = self.delta_history[-self.precision_window:]
            
        return delta
    
    def _generate_prediction(self) -> np.ndarray:
        """Generate endogenous prediction (simplified)."""
        # In practice, this would use a learned generative model
        # For simulation, add small noise to previous prediction
        if self.prev_delta is None:
            return np.zeros(1)  # Default prediction
        
        # Simple predictive model: assume no change
        return self.prev_delta * 0.0
    
    def compute_temporal_derivative(self, delta: np.ndarray) -> float:
        """Compute temporal derivative |\dot{δ}|.
        
        Args:
            delta: Current prediction error
            
        Returns:
            Temporal derivative magnitude
        """
        if self.prev_delta is None:
            self.prev_delta = delta
            return 0.0
        
        derivative = (np.abs(delta) - np.abs(self.prev_delta)) / self.tau
        self.prev_delta = delta
        return float(derivative)
    
    def compute_information_gain(self, actions: List[np.ndarray]) -> float:
        """Compute Information Gain IG_i^a(t).
        
        IG = H[posterior] - E_a[H[posterior|action]]
        
        Args:
            actions: List of candidate actions
            
        Returns:
            Information Gain value
        """
        # Compute current entropy
        if self.posterior is None:
            self.posterior = self._compute_posterior()
        current_entropy = self._compute_entropy(self.posterior)
        
        # Compute expected entropy after each action
        expected_entropy = 0.0
        for action in actions:
            if action.tobytes() not in self.posterior_given_action:
                post = self._compute_posterior_given_action(action)
                self.posterior_given_action[action.tobytes()] = post
            else:
                post = self.posterior_given_action[action.tobytes()]
            expected_entropy += self._compute_entropy(post)
        
        expected_entropy /= len(actions) if actions else 1.0
        
        ig = current_entropy - expected_entropy
        self.ig_history.append(ig)
        return max(0.0, ig)
    
    def _compute_posterior(self) -> np.ndarray:
        """Compute current posterior distribution."""
        # Simplified: normal distribution with variance from history
        if len(self.delta_history) < 2:
            return np.array([1.0])
        variance = np.var(self.delta_history) + self.epsilon
        return np.random.normal(0, np.sqrt(variance), 10)
    
    def _compute_posterior_given_action(self, action: np.ndarray) -> np.ndarray:
        """Compute posterior distribution conditioned on action."""
        # Simplified: perturb variance based on action
        if len(self.delta_history) < 2:
            return np.array([1.0])
        variance = np.var(self.delta_history) + self.epsilon
        # Action reduces uncertainty by some amount
        variance *= 0.9
        return np.random.normal(0, np.sqrt(variance), 10)
    
    def _compute_entropy(self, distribution: np.ndarray) -> float:
        """Compute entropy of a distribution."""
        # Normalize
        probs = np.abs(distribution) / (np.sum(np.abs(distribution)) + self.epsilon)
        # Entropy
        return -np.sum(probs * np.log(probs + self.epsilon))
    
    def compute_intrinsic_valence(
        self,
        delta: np.ndarray,
        delta_dot: float,
        ig: float
    ) -> float:
        """Compute intrinsic valence v_pri.
        
        v_pri = tanh(-π·δ·|\dot{δ}| + β·IG)
        
        Args:
            delta: Prediction error
            delta_dot: Temporal derivative
            ig: Information Gain
            
        Returns:
            Intrinsic valence value
        """
        # Compute precision (inverse variance)
        variance = np.var(self.delta_history) if len(self.delta_history) > 1 else 1.0
        precision = 1.0 / (variance + self.epsilon)
        
        # Compute valence
        delta_magnitude = np.mean(np.abs(delta))
        raw = -precision * delta_magnitude * np.abs(delta_dot) + self.beta * ig
        return np.tanh(raw)
    
    def reset(self):
        """Reset the Interoceptive Core."""
        self.prev_delta = None
        self.delta_history = []
        self.ig_history = []
        self.posterior = None
        self.posterior_given_action = {}
