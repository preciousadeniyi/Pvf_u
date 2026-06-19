#!/usr/bin/env python
"""Run full benchmark suite for PVF-U."""

import subprocess
import argparse
from pathlib import Path


def run_benchmark_suite(config_path: str, output_dir: str):
    """Run all benchmarks."""
    experiments = ['E-Dark', 'E-Novel', 'E-Social', 'E-Social+Bully']
    
    for experiment in experiments:
        print(f"\n=== Running {experiment} ===")
        cmd = [
            'python',
            'benchmarks/run_shape_game.py',
            '--experiment', experiment,
            '--runs', '50',
            '--config', config_path,
            '--output', output_dir
        ]
        subprocess.run(cmd, check=True)
    
    # Run ablation
    print("\n=== Running Ablation Studies ===")
    cmd = [
        'python',
        'benchmarks/run_ablation.py',
        '--config', config_path,
        '--output', output_dir
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='Run full benchmark suite')
    parser.add_argument('--config', type=str, default='src/config/default_config.yaml')
    parser.add_argument('--output', type=str, default='results/')
    
    args = parser.parse_args()
    
    run_benchmark_suite(args.config, args.output)


if __name__ == "__main__":
    main()
