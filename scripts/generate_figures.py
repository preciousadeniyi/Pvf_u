#!/usr/bin/env python
"""Generate figures from paper results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse


def generate_figures(results_dir: str, output_dir: str):
    """Generate all paper figures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Distribution of key metrics
    generate_distribution_figure(results_dir, output_path)
    
    # Figure 2: Ablation study results
    generate_ablation_figure(results_dir, output_path)
    
    # Figure 3: Sensitivity analysis
    generate_sensitivity_figure(results_dir, output_path)
    
    # Figure 4: Convergence diagnostics
    generate_convergence_figure(results_dir, output_path)


def generate_distribution_figure(results_dir: str, output_path: Path):
    """Generate distribution plots for key metrics."""
    # Load results
    df_list = []
    for exp in ['E-Dark', 'E-Novel', 'E-Social', 'E-Social+Bully']:
        path = Path(results_dir) / f"shape_game_{exp}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df['experiment'] = exp
            df_list.append(df)
    
    if not df_list:
        print("No results found for distribution figure")
        return
    
    df = pd.concat(df_list, ignore_index=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = {
        'movement_entropy': 'Movement Entropy (bits)',
        'completion_time': 'Completion Time (steps)',
        'pei': 'Power Equity Index (PEI)',
        'creative_tension': 'CreativeTension'
    }
    
    for i, (metric, label) in enumerate(metrics.items()):
        ax = axes[i // 2, i % 2]
        if metric in df.columns:
            sns.boxplot(data=df, x='experiment', y=metric, ax=ax)
            ax.set_title(label)
            ax.set_xlabel('')
            ax.set_ylabel(label)
            
            # Add horizontal line for target
            if metric == 'pei':
                ax.axhline(y=0.7, color='red', linestyle='--', label='Target (0.7)')
            elif metric == 'creative_tension':
                ax.axhline(y=0.3, color='red', linestyle='--', label='Target (0.3)')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure1_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_ablation_figure(results_dir: str, output_path: Path):
    """Generate ablation study figure."""
    # Expected results from Table 8.3
    configurations = [
        'Full PVF-U',
        'w/o Layer 0',
        'w/o Layer 1',
        'w/o Layer 2'
    ]
    
    pei_values = [0.82, 0.78, 0.61, 0.53]
    pei_std = [0.04, 0.05, 0.07, 0.09]
    
    ct_values = [0.45, 0.38, 0.22, 0.09]
    ct_std = [0.06, 0.07, 0.05, 0.04]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(configurations))
    width = 0.6
    
    # PEI
    bars1 = ax1.bar(x, pei_values, width, yerr=pei_std, capsize=5)
    ax1.axhline(y=0.7, color='red', linestyle='--', label='Target (0.7)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configurations, rotation=15, ha='right')
    ax1.set_ylabel('Power Equity Index (PEI)')
    ax1.set_title('Ablation: PEI')
    ax1.legend()
    
    # CreativeTension
    bars2 = ax2.bar(x, ct_values, width, yerr=ct_std, capsize=5)
    ax2.axhline(y=0.3, color='red', linestyle='--', label='Target (0.3)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configurations, rotation=15, ha='right')
    ax2.set_ylabel('CreativeTension')
    ax2.set_title('Ablation: CreativeTension')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure2_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_sensitivity_figure(results_dir: str, output_path: Path):
    """Generate sensitivity analysis figure."""
    # Data from Table 8.4
    parameters = ['β', 'γ', 'λ', 'θ_panic', 'θ_collective', 'τ', 'T_recovery']
    pei_sensitivity = [0.08, -0.12, 0.72, -0.56, -0.04, 0.03, 0.02]
    ct_sensitivity = [0.21, -0.34, -0.06, -0.18, 0.41, 0.07, -0.03]
    edark_sensitivity = [0.92, -0.04, -0.02, -0.01, -0.02, 0.06, -0.01]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(parameters))
    width = 0.25
    
    bars1 = ax.bar(x - width, pei_sensitivity, width, label='PEI')
    bars2 = ax.bar(x, ct_sensitivity, width, label='CreativeTension')
    bars3 = ax.bar(x + width, edark_sensitivity, width, label='E-Dark Entropy')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.3, color='red', linestyle='--', label='Sensitivity threshold')
    ax.axhline(y=-0.3, color='red', linestyle='--')
    
    ax.set_xticks(x)
    ax.set_xticklabels(parameters)
    ax.set_ylabel('Sensitivity Elasticity')
    ax.set_title('Sensitivity Analysis')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure3_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_convergence_figure(results_dir: str, output_path: Path):
    """Generate convergence diagnostics figure."""
    # Data from Table 8.5
    variables = ['Gate α(t)', 'K(t)', 'PEI(t)', 'CT Ψ(t)']
    rhat_values = [1.004, 1.006, 1.008, 1.007]
    ess_values = [842, 756, 714, 738]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # R-hat
    bars1 = ax1.bar(variables, rhat_values)
    ax1.axhline(y=1.1, color='red', linestyle='--', label='Threshold (1.1)')
    ax1.set_ylabel('R-hat')
    ax1.set_title('Convergence: R-hat')
    ax1.legend()
    
    # ESS
    bars2 = ax2.bar(variables, ess_values)
    ax2.axhline(y=700, color='red', linestyle='--', label='Threshold (700)')
    ax2.set_ylabel('Effective Sample Size')
    ax2.set_title('Convergence: ESS')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'figure4_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--results', type=str, default='results/')
    parser.add_argument('--output', type=str, default='figures/')
    
    args = parser.parse_args()
    
    generate_figures(args.results, args.output)
    print(f"Figures saved to {args.output}")


if __name__ == "__main__":
    main()
