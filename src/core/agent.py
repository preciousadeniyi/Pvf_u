
"""Main PVFAgent class implementing the three-tiered architecture."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from pvf_u.core.layers.layer0_interoceptive import InteroceptiveCore
from pvf_u.core.layers.layer1_relational_gate import RelationalGate
from pvf_u.core.layers.layer2_ecosystemic import EcosystemicSuperstructure
from pvf_u.core.narrative import NarrativeEmbedding
from pvf_u.core.valence import compute_total_valence
from pvf_u.utils.stability import safe_softmax


@dataclass
class AgentState:
    """Container for agent's current state."""
    agent_id: int
    delta: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    delta_dot: float = 0.0
    information_gain: float = 0.0
    intrinsic_valence: float = 0.0
    social_valence: float = 0.0
    total_valence: float = 0.0
    alpha: float = 0.0
    narrative_embedding: np.ndarray = field(default_factory=lambda: np.zeros(8))
    panic_active: bool = False
    steps_since_panic: int = 0


class PVFAgent:
    """Main PVF-U agent implementing the three-tiered heterarchical architecture.
    
    The agent integrates:
    - Layer 0: Interoceptive Core for solitary survival
    - Layer 1: Relational Gate for social permeability
    - Layer 2: Ecosystemic Superstructure for collective negotiation
    - Biological Veto for safety
    """
    
    def __init__(
        self,
        agent_id: int,
        config: Dict[str, Any],
        initial_delta: Optional[np.ndarray] = None
    ):
        """Initialize PVF-U agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration dictionary with hyperparameters
            initial_delta: Initial prediction error (default: zeros)
        """
        self.agent_id = agent_id
        self.config = config
        
        # Initialize layers
        self.layer0 = InteroceptiveCore(config)
        self.layer1 = RelationalGate(config)
        self.layer2 = EcosystemicSuperstructure(config)
        
        # Initialize narrative
        dim = config.get('dimension', 512)
        K_max = config.get('K_max', 128)
        self.narrative = NarrativeEmbedding(dim, K_max)
        
        # Initialize state
        self.state = AgentState(
            agent_id=agent_id,
            delta=initial_delta if initial_delta is not None else np.array([0.0])
        )
        
        # History for tracking
        self.history = {
            'delta': [],
            'alpha': [],
            'valence': [],
            'narrative': [],
            'panic': [],
        }
        
        # Influence weights (dict: neighbor_id -> weight)
        self.influence_weights: Dict[int, float] = {}
    
    def step(
        self,
        observation: np.ndarray,
        neighbors: List[Any],
        actions: Optional[List[np.ndarray]] = None
    ) -> Tuple[float, float, np.ndarray]:
        """Execute one step of the PVF-U loop.
        
        Args:
            observation: Current sensory input
            neighbors: List of neighboring agents
            actions: Candidate actions for IG computation (optional)
            
        Returns:
            Tuple of (total_valence, alpha, narrative_embedding)
        """
        # --- Layer 0: Interoceptive Core ---
        delta = self.layer0.compute_prediction_error(observation)
        delta_dot = self.layer0.compute_temporal_derivative(delta)
        
        # Compute Information Gain
        if actions is None:
            actions = self._generate_candidate_actions()
        ig = self.layer0.compute_information_gain(actions)
        
        # Compute intrinsic valence
        v_pri = self.layer0.compute_intrinsic_valence(delta, delta_dot, ig)
        
        # --- Layer 1: Relational Gate ---
        if neighbors:
            # Compute social valence
            v_soc, relational_cost = self.layer1.compute_social_valence(
                self, neighbors, self.influence_weights
            )
            
            # Compute collective multi-information
            collective_i = self.layer2.collective_multi_information
            
            # Compute gate
            alpha = self.layer1.compute_gate(ig, collective_i)
        else:
            v_soc = 0.0
            relational_cost = 0.0
            alpha = 0.0
        
        # --- Layer 2: Ecosystemic Superstructure ---
        # Check Biological Veto
        if self.layer2.check_panic(delta):
            self._trigger_veto()
        
        # If veto active, override alpha and social valence
        if self.state.panic_active:
            alpha = 0.0
            v_soc = 0.0
            self.state.steps_since_panic += 1
            
            # Check if recovery time has elapsed
            recovery_steps = self.config.get('recovery_steps', 50)
            if self.state.steps_since_panic >= recovery_steps:
                if not self.layer2.check_panic(delta):
                    self.state.panic_active = False
                    self.state.steps_since_panic = 0
        
        # Compute total valence
        v_tot = compute_total_valence(v_pri, alpha, v_soc)
        
        # Update narrative
        self.narrative.update(v_tot, alpha, neighbors)
        
        # Update influence weights (periodically)
        if len(self.history['delta']) % self.config.get('social_update_interval', 10) == 0:
            if neighbors:
                self.influence_weights = self.layer2.update_influence_weights(
                    self.agent_id, neighbors
                )
        
        # Update state
        self.state.delta = delta
        self.state.delta_dot = delta_dot
        self.state.information_gain = ig
        self.state.intrinsic_valence = v_pri
        self.state.social_valence = v_soc
        self.state.total_valence = v_tot
        self.state.alpha = alpha
        self.state.narrative_embedding = self.narrative.embedding
        
        # Update history
        self.history['delta'].append(float(np.mean(np.abs(delta))))
        self.history['alpha'].append(alpha)
        self.history['valence'].append(v_tot)
        self.history['narrative'].append(self.narrative.embedding.copy())
        self.history['panic'].append(self.state.panic_active)
        
        return v_tot, alpha, self.narrative.embedding
    
    def _generate_candidate_actions(self) -> List[np.ndarray]:
        """Generate candidate actions for Information Gain computation."""
        # In practice, this would sample from the policy
        # For simplicity, generate random actions
        num_actions = self.config.get('num_candidate_actions', 10)
        dim = self.config.get('action_dim', 2)
        return [np.random.randn(dim) for _ in range(num_actions)]
    
    def _trigger_veto(self):
        """Biological Veto: Freeze-Isolate Protocol."""
        self.state.panic_active = True
        self.state.steps_since_panic = 0
        
        # Freeze influence weights (do NOT reset)
        # The layer2 keeps track of this
        self.layer2.trigger_veto()
    
    def select_action(self, valence: float) -> np.ndarray:
        """Select action based on valence.
        
        Policy: π(a|s) ∝ exp(β·valence)
        """
        beta = self.config.get('action_beta', 1.0)
        candidates = self._generate_candidate_actions()
        probabilities = safe_softmax(np.array([valence * beta for _ in candidates]))
        idx = np.random.choice(len(candidates), p=probabilities)
        return candidates[idx]
    
    def get_state(self) -> AgentState:
        """Return current agent state."""
        return self.state
    
    def reset(self):
        """Reset agent to initial state."""
        self.state = AgentState(agent_id=self.agent_id)
        self.narrative.reset()
        self.influence_weights = {}
        self.history = {
            'delta': [],
            'alpha': [],
            'valence': [],
            'narrative': [],
            'panic': [],
        }
