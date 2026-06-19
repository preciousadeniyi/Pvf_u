#!/usr/bin/env python
"""Run Shape Game experiments from Section 8."""

import numpy as np
import pandas as pd
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any

from pvf_u.core.agent import PVFAgent
from pvf_u.environment.shape_game import ShapeGame
from pvf_u.environment.dark_room import DarkRoom
from pvf_u.utils.metrics import compute_pei, compute_creative_tension, compute_narrative_coherence


def run_experiment(
    experiment_type: str,
    num_runs: int = 50,
    max_steps: int = 5000,
    config_path: str = "src/config/default_config.yaml",
    output_dir: str = "results/"
) -> pd.DataFrame:
    """Run Shape Game experiment.
    
    Args:
        experiment_type: 'E-Dark', 'E-Novel', 'E-Social', 'E-Social+Bully'
        num_runs: Number of runs
        max_steps: Maximum steps per episode
        config_path: Path to configuration file
        output_dir: Directory to save results
        
    Returns:
        DataFrame with results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    results = []
    
    for run in tqdm(range(num_runs), desc=f"Running {experiment_type}"):
        # Setup environment
        if experiment_type == 'E-Dark':
            env = DarkRoom()
            num_agents = 1
        elif experiment_type == 'E-Novel':
            env = DarkRoom()  # Simplified
            num_agents = 1
        elif experiment_type == 'E-Social':
            env = ShapeGame(num_agents=2, pieces=7)
            num_agents = 2
        elif experiment_type == 'E-Social+Bully':
            env = ShapeGame(num_agents=10, pieces=7)
            num_agents = 10
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        # Create agents
        agents = [PVFAgent(i, config) for i in range(num_agents)]
        
        # Run episode
        episode_data = run_episode(env, agents, max_steps)
        episode_data['run'] = run
        episode_data['experiment'] = experiment_type
        results.append(episode_data)
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path(output_dir) / f"shape_game_{experiment_type}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df


def run_episode(env, agents, max_steps: int) -> Dict[str, Any]:
    """Run a single episode."""
    env.reset()
    episode_data = {
        'movement_entropy': [],
        'completion_time': None,
        'pei': [],
        'creative_tension': [],
        'alpha': [],
        'delta': [],
        'narrative_coherence': []
    }
    
    for step in range(max_steps):
        actions = {}
        neighbors_by_id = {}
        
        for agent in agents:
            obs = env.observe(agent.agent_id)
            neighbors = env.get_neighbors(agent.agent_id) if hasattr(env, 'get_neighbors') else []
            neighbors_by_id[agent.agent_id] = neighbors
            
            val, alpha, narrative = agent.step(obs, neighbors)
            action = agent.select_action(val)
            actions[agent.agent_id] = action
            
            # Collect data
            episode_data['alpha'].append(alpha)
            episode_data['delta'].append(np.mean(np.abs(agent.state.delta)))
        
        # Step environment
        config, deltas, complete = env.step(actions)
        
        # Compute metrics
        if env.num_agents > 1:
            # PEI
            weights = [agent.influence_weights for agent in agents]
            pei = compute_pei(weights, len(agents))
            episode_data['pei'].append(pei)
            
            # CreativeTension
            narratives = [agent.narrative.embedding for agent in agents]
            alphas = [agent.state.alpha for agent in agents]
            costs = [0.1 for _ in agents]  # Placeholder
            i_history = []  # Placeholder
            ct = compute_creative_tension(alphas, narratives, costs, i_history)
            episode_data['creative_tension'].append(ct)
            
            # Narrative Coherence
            nc = compute_narrative_coherence(narratives)
            episode_data['narrative_coherence'].append(nc)
        
        if complete:
            episode_data['completion_time'] = step
            break
    
    return episode_data


def main():
    parser = argparse.ArgumentParser(description='Run Shape Game experiments')
    parser.add_argument('--experiment', type=str, default='E-Social',
                       choices=['E-Dark', 'E-Novel', 'E-Social', 'E-Social+Bully'])
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--config', type=str, default='src/config/default_config.yaml')
    parser.add_argument('--output', type=str, default='results/')
    
    args = parser.parse_args()
    
    df = run_experiment(
        experiment_type=args.experiment,
        num_runs=args.runs,
        max_steps=args.max_steps,
        config_path=args.config,
        output_dir=args.output
    )
    
    print(f"Results saved to {args.output}")
    print(f"Mean completion time: {df['completion_time'].mean():.2f} steps")
    print(f"Mean PEI: {df['pei'].mean():.3f}")


if __name__ == "__main__":
    main()
