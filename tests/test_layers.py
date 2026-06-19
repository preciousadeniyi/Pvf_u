

"""Unit tests for PVF-U layers."""

import pytest
import numpy as np
import yaml
from pathlib import Path

from pvf_u.core.layers.layer0_interoceptive import InteroceptiveCore
from pvf_u.core.layers.layer1_relational_gate import RelationalGate
from pvf_u.core.layers.layer2_ecosystemic import EcosystemicSuperstructure


def load_config():
    """Load default configuration."""
    config_path = Path(__file__).parent.parent / "src" / "config" / "default_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_layer0_prediction_error():
    """Test prediction error computation."""
    config = load_config()
    layer0 = InteroceptiveCore(config['layers']['layer0'])
    
    observation = np.array([1.0, 2.0])
    delta = layer0.compute_prediction_error(observation)
    
    assert delta.shape == observation.shape
    assert not np.any(np.isnan(delta))


def test_layer0_temporal_derivative():
    """Test temporal derivative computation."""
    config = load_config()
    layer0 = InteroceptiveCore(config['layers']['layer0'])
    
    delta1 = np.array([1.0])
    delta2 = np.array([2.0])
    
    # First call should return 0
    d1 = layer0.compute_temporal_derivative(delta1)
    assert d1 == 0.0
    
    # Second call should compute derivative
    d2 = layer0.compute_temporal_derivative(delta2)
    # |2 - 1| / tau = 1/10 = 0.1
    assert abs(d2 - 0.1) < 0.01


def test_layer0_information_gain():
    """Test Information Gain computation."""
    config = load_config()
    layer0 = InteroceptiveCore(config['layers']['layer0'])
    
    # Seed with some history
    for _ in range(10):
        layer0.compute_prediction_error(np.array([np.random.randn()]))
    
    actions = [np.random.randn(2) for _ in range(5)]
    ig = layer0.compute_information_gain(actions)
    
    assert ig >= 0.0


def test_layer1_gate():
    """Test Relational Gate computation."""
    config = load_config()
    layer1 = RelationalGate(config['layers']['layer1'])
    
    # Simulate history
    for i in range(150):
        layer1.ig_history.append(np.random.randn())
        layer1.i_history.append(np.random.randn())
    
    alpha = layer1.compute_gate(ig=0.5, collective_i=0.3)
    assert 0.0 <= alpha <= 1.0


def test_layer2_relief():
    """Test relief provision."""
    config = load_config()
    layer2 = EcosystemicSuperstructure(config['layers']['layer2'])
    
    # Create dummy agents
    class DummyAgent:
        def __init__(self, delta, narrative):
            self.state = type('obj', (object,), {'delta': delta, 'delta_dot': 0.0})
            self.narrative = type('obj', (object,), {'embedding': narrative})
            self.history = {'delta': [delta]}
    
    agent_a = DummyAgent(np.array([0.5]), np.array([1.0, 0.0]))
    agent_b = DummyAgent(np.array([0.3]), np.array([0.8, 0.6]))
    
    relief = layer2.compute_relief(agent_a, agent_b)
    assert 0.0 <= relief <= 1.0


def test_layer2_panic():
    """Test panic detection."""
    config = load_config()
    config['layers']['layer2']['panic_threshold'] = 1.0
    layer2 = EcosystemicSuperstructure(config['layers']['layer2'])
    
    # Should not panic
    delta = np.array([0.5])
    assert not layer2.check_panic(delta)
    
    # Should panic
    delta = np.array([2.0])
    assert layer2.check_panic(delta)
