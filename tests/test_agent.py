"""Unit tests for PVFAgent class."""

import pytest
import numpy as np
import yaml
from pathlib import Path

from pvf_u.core.agent import PVFAgent
from pvf_u.environment.dark_room import DarkRoom
from pvf_u.environment.shape_game import ShapeGame


def load_config():
    """Load default configuration."""
    config_path = Path(__file__).parent.parent / "src" / "config" / "default_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_agent_initialization():
    """Test that agent initializes correctly."""
    config = load_config()
    agent = PVFAgent(agent_id=0, config=config)
    
    assert agent.agent_id == 0
    assert agent.state.alpha == 0.0
    assert agent.narrative.K == 8
    assert agent.layer0.beta == 1.0
    assert agent.layer1.gamma == 0.5
    assert agent.layer2.lambda_reg == 0.29


def test_solitary_survival():
    """Test Theorem 6.1: Solitary survival in Dark Room."""
    config = load_config()
    agent = PVFAgent(agent_id=0, config=config)
    env = DarkRoom()
    
    movements = []
    for _ in range(100):
        obs = env.observe(agent_id=0)
        val, alpha, _ = agent.step(obs, neighbors=[])
        action = agent.select_action(val)
        movements.append(1 if np.any(action != 0) else 0)
    
    # Agent should move in Dark Room
    assert np.mean(movements) > 0.5


def test_bibo_stability():
    """Test Theorem 6.2: BIBO stability |v_tot| ≤ 2."""
    config = load_config()
    agent = PVFAgent(agent_id=0, config=config)
    env = ShapeGame()
    
    valences = []
    for _ in range(100):
        obs = env.observe(agent_id=0)
        neighbors = env.get_neighbors(agent_id=0)
        # Create dummy neighbors
        dummy_neighbors = [PVFAgent(i, config) for i in range(1, env.num_agents)]
        for n in dummy_neighbors:
            n.state.delta = np.array([0.1])
        val, _, _ = agent.step(obs, neighbors=dummy_neighbors)
        valences.append(val)
    
    # |v_tot| ≤ 2 (Theorem 6.2)
    assert np.max(np.abs(valences)) <= 2.0 + 1e-6


def test_biological_veto():
    """Test Biological Veto mechanism."""
    config = load_config()
    # Set low panic threshold to trigger veto
    config['layers']['layer2']['panic_threshold'] = 0.1
    agent = PVFAgent(agent_id=0, config=config)
    env = DarkRoom()
    
    # Create large prediction error
    obs = np.array([10.0])  # Large error
    
    for _ in range(10):
        val, alpha, _ = agent.step(obs, neighbors=[])
    
    # Veto should be active
    assert agent.state.panic_active or agent.layer2.veto_active


def test_information_gain():
    """Test Information Gain computation."""
    config = load_config()
    agent = PVFAgent(agent_id=0, config=config)
    
    # Generate some actions
    actions = [np.random.randn(2) for _ in range(10)]
    
    # Compute IG
    ig = agent.layer0.compute_information_gain(actions)
    
    # IG should be non-negative
    assert ig >= 0.0


def test_narrative_update():
    """Test narrative embedding update."""
    config = load_config()
    agent = PVFAgent(agent_id=0, config=config)
    neighbor = PVFAgent(agent_id=1, config=config)
    
    initial_embedding = agent.narrative.embedding.copy()
    
    # Update with social influence
    val = 0.5
    alpha = 0.7
    agent.narrative.update(val, alpha, [neighbor], {1: 0.5})
    
    # Embedding should change
    assert not np.allclose(initial_embedding, agent.narrative.embedding)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
