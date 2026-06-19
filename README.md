# README.md

```markdown
# PVF-U: Unified Primitive Valence Foundation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.20+-green.svg)](https://github.com/google/jax)
[![CI Tests](https://github.com/precious-adeniyi/pvf-u/actions/workflows/tests.yml/badge.svg)](https://github.com/precious-adeniyi/pvf-u/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/gh/precious-adeniyi/pvf-u/branch/main/graph/badge.svg)](https://codecov.io/gh/precious-adeniyi/pvf-u)

Implementation of **"The Unified Primitive Valence Foundation: From Solitary Survival to Ecosystemic Negotiation in Affective AI"** (Adeniyi, 2026).

---

## 📖 Overview

PVF-U is a three-tiered architecture for affective AI that resolves the fundamental "dark room" paradox where AI systems collapse in either social contexts (individualist architectures) or solitary contexts (purely social architectures).

### Key Features

1. **🧠 Layer 0: Interoceptive Core** — Guarantees solitary survival via epistemic foraging (Information Gain) and temporal affect tracking
2. **🚪 Layer 1: Relational Gate** — Dynamically modulates social permeability based on comparative epistemic utility
3. **🌐 Layer 2: Ecosystemic Superstructure** — Implements equitable influence, relief-based trust, and a Biological Veto
4. **🛡️ Biological Veto** — Non-negotiable safety mechanism that prevents collective negotiation from overriding individual homeostasis
5. **⚖️ Power Equity** — Built-in equity mechanisms with tunable PEI > 0.7, NC > 0.8, CreativeTension > 0.3

### Theoretical Guarantees

- **Solitary Survival** (Theorem 6.1): Agents explore even in perfectly predictable environments
- **BIBO Stability** (Theorem 6.2): Total valence is uniformly bounded: |v_tot| ≤ 2
- **Gate Convergence** (Theorem 6.3): Relational gate converges with emergency fast-trigger
- **Bounded Influence** (Theorem 6.4): Gini coefficient of influence distribution is bounded
- **Veto Stabilization** (Theorem 6.5): Biological Veto ensures bounded recovery

---

## 🚀 Quick Start

### Installation

**Using Conda (Recommended):**

```bash
# Clone the repository
git clone https://github.com/precious-adeniyi/pvf-u.git
cd pvf-u

# Create conda environment
conda env create -f environment.yml
conda activate pvf-u

# Install in development mode
pip install -e .
```

**Using pip:**

```bash
# Clone the repository
git clone https://github.com/precious-adeniyi/pvf-u.git
cd pvf-u

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Minimal Example

```python
from pvf_u.core.agent import PVFAgent
from pvf_u.environment.shape_game import ShapeGame
from pvf_u.utils.metrics import compute_pei, compute_creative_tension

# Load configuration
import yaml
with open('src/config/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create agent and environment
agent = PVFAgent(agent_id=0, config=config)
env = ShapeGame(num_agents=2)

# Run episode
for step in range(5000):
    observation = env.observe(agent.agent_id)
    neighbors = env.get_neighbors(agent.agent_id)
    
    # PVF-U step: compute valence, gate, and narrative
    valence, alpha, narrative = agent.step(observation, neighbors)
    
    # Select action based on valence
    action = agent.select_action(valence)
    
    # Step environment
    env.step({agent.agent_id: action})
    
    if env.is_complete():
        print(f"Episode completed in {step} steps!")
        break

# Compute metrics
influence_weights = [agent.influence_weights]
pei = compute_pei(influence_weights, N=2)
print(f"Power Equity Index: {pei:.3f}")
```

---

## 📊 Reproducing Paper Results

### Run All Benchmarks

```bash
# Run full benchmark suite (Shape Game experiments + ablations + sensitivity)
python scripts/run_benchmark_suite.py --config src/config/default_config.yaml --output results/
```

### Run Specific Experiments

```bash
# Shape Game: Solitary Survival (E-Dark)
python benchmarks/run_shape_game.py --experiment E-Dark --runs 50

# Shape Game: Social (E-Social)
python benchmarks/run_shape_game.py --experiment E-Social --runs 50

# Shape Game: With Bully Agent (E-Social+Bully)
python benchmarks/run_shape_game.py --experiment E-Social+Bully --runs 50

# Ablation Studies
python benchmarks/run_ablation.py --config src/config/default_config.yaml

# Sensitivity Analysis
python benchmarks/run_sensitivity.py --config src/config/default_config.yaml
```

### Generate Figures

```bash
# Generate all paper figures (distributions, ablation, sensitivity, convergence)
python scripts/generate_figures.py --results results/ --output figures/
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_agent.py -v

# Run with coverage
pytest --cov=src tests/
```

---

## 📁 Project Structure

```
pvf-u/
├── README.md                     # This file
├── LICENSE                       # MIT License
├── pyproject.toml                # Project metadata
├── requirements.txt              # Python dependencies (pip)
├── environment.yml               # Conda environment
│
├── .github/
│   └── workflows/
│       └── tests.yml             # CI/CD tests
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── core/                     # Core PVF-U implementation
│   │   ├── __init__.py
│   │   ├── agent.py              # Main PVFAgent class
│   │   ├── valence.py            # Valence computation
│   │   ├── narrative.py          # Narrative embeddings
│   │   └── layers/
│   │       ├── __init__.py
│   │       ├── layer0_interoceptive.py   # Interoceptive Core
│   │       ├── layer1_relational_gate.py # Relational Gate
│   │       └── layer2_ecosystemic.py     # Ecosystemic Superstructure
│   │
│   ├── models/                   # Generative models
│   │   ├── __init__.py
│   │   ├── generative_model.py   # p(s|ψ) for predictions
│   │   └── variational.py        # q(ψ) for posterior inference
│   │
│   ├── environment/              # Environments
│   │   ├── __init__.py
│   │   ├── shape_game.py         # Shape Game benchmark
│   │   ├── cooperative_navigation.py  # Generalization benchmark
│   │   └── dark_room.py          # Dark Room for solitary survival
│   │
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── metrics.py            # PEI, CT, NC computation
│   │   ├── stability.py          # Numerical stability helpers
│   │   ├── knn_sparsification.py # Biased k-NN sparsification
│   │   └── differential_privacy.py  # DP utilities
│   │
│   └── config/                   # Configuration files
│       ├── __init__.py
│       ├── default_config.yaml   # Default hyperparameters
│       └── experiment_configs/
│           ├── phase1_sovereign.yaml   # Layer 0 only
│           ├── phase2_porous.yaml      # Layers 0-1
│           ├── phase3_dyadic.yaml      # Layers 0-2 (λ=0)
│           └── phase4_ecosystem.yaml   # Full PVF-U
│
├── benchmarks/                   # Benchmark scripts
│   ├── __init__.py
│   ├── run_shape_game.py         # Shape Game experiments
│   ├── run_navigation.py         # Navigation experiments
│   ├── run_ablation.py           # Ablation studies
│   └── run_sensitivity.py        # Sensitivity analysis
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_introduction.ipynb
│   ├── 02_layer0_demo.ipynb
│   ├── 03_gate_dynamics.ipynb
│   ├── 04_shape_game_analysis.ipynb
│   ├── 05_normative_metrics.ipynb
│   └── 06_reproduce_figures.ipynb
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_layers.py
│   ├── test_metrics.py
│   ├── test_stability.py
│   └── test_numerical.py
│
├── scripts/                      # Utility scripts
│   ├── run_benchmark_suite.py    # Full benchmark runner
│   ├── run_grid_search.py        # Hyperparameter search
│   └── generate_figures.py       # Figure generation
│
├── data/                         # Data storage (empty)
├── results/                      # Experiment results (generated)
└── docs/                         # Documentation
    ├── api_reference.md
    ├── contributing.md
    └── installation.md
```

---

## 🧪 Experimental Results

### Shape Game Benchmark

The Shape Game benchmark (Section 8 of the paper) demonstrates PVF-U's superiority over PVF-AB across four key metrics:

| Metric | PVF-U | PVF-AB | Improvement |
|--------|-------|--------|-------------|
| Solitary Survival (Movement Entropy) | 1.24 ± 0.15 bits | 0.12 ± 0.03 bits | **+933%** |
| Completion Time | 84 ± 9 s | 126 ± 14 s | **-33%** |
| Power Equity Index (PEI) | 0.82 ± 0.04 | 0.52 ± 0.08 | **+58%** |
| CreativeTension (CT) | 0.45 ± 0.06 | 0.08 ± 0.05 | **+463%** |

### Ablation Studies

| Configuration | Completion Time (s) | PEI | CT | E-Dark Movement |
|---------------|:---:|:---:|:---:|:---:|
| Full PVF-U | 84 ± 9 | 0.82 ± 0.04 | 0.45 ± 0.06 | ✅ |
| w/o Layer 0 (IG removed) | 162 ± 21 | 0.78 ± 0.05 | 0.38 ± 0.07 | ❌ |
| w/o Layer 1 (Gate forced = 1) | 118 ± 15 | 0.61 ± 0.07 | 0.22 ± 0.05 | ✅ |
| w/o Layer 2 (Veto disabled) | 93 ± 12 | 0.53 ± 0.09 | 0.09 ± 0.04 | ✅ |

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{adeniyi2026pvf,
  title={The Unified Primitive Valence Foundation: From Solitary Survival to Ecosystemic Negotiation in Affective AI},
  author={Adeniyi, Precious},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/precious-adeniyi/pvf-u},
  doi={10.5281/zenodo.XXXXX}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Ethical Use Statement

PVF-U is designed to promote equitable, biologically-grounded relational responsibility. It includes explicit ethical safeguards:

- **Biological Veto**: Non-negotiable safety mechanism
- **Power Equity Index**: Hardcoded equity monitoring
- **Relief-based Influence**: Social trust earned through reducing others' prediction errors

**PVF-U should NOT be used for:**
- Autonomous weapon systems
- Unconsented surveillance
- Systems designed to manipulate or coerce human participants
- Any application that violates fundamental human rights

---

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guidelines](docs/contributing.md) for details on our code of conduct and the process for submitting pull requests.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ benchmarks/ tests/

# Lint code
flake8 src/ benchmarks/ tests/
```

### Reporting Issues

Please report bugs and feature requests through the [GitHub Issues](https://github.com/precious-adeniyi/pvf-u/issues) page.

---

## 🙏 Acknowledgments

- **Active Inference Institute** community for discussions on collective intelligence
- **Valerii Zaiats** for collaboration on related implementations
- **Anonymous reviewers** for constructive feedback
- All researchers and practitioners working on equitable AI and participatory sense-making

---

## 📧 Contact

**Precious Adeniyi**  
Independent Researcher, Oyo, Nigeria  
Email: preciousadeniyi165@gmail.com  
GitHub: [@precious-adeniyi](https://github.com/precious-adeniyi)

---

## 🔗 Links

- **Paper**: [arXiv] (Coming soon)
- **Code**: [GitHub](https://github.com/precious-adeniyi/pvf-u)
- **Zenodo**: [DOI: 10.5281/zenodo.XXXXX](https://doi.org/10.5281/zenodo.XXXXX)
- **Issues**: [GitHub Issues](https://github.com/precious-adeniyi/pvf-u/issues)

---

## 📊 Badges

| Badge | Purpose |
|-------|---------|
| [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX) | Permanent citation link |
| [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | Open-source license |
| [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) | Python version support |
| [![JAX](https://img.shields.io/badge/JAX-0.4.20+-green.svg)](https://github.com/google/jax) | Core dependency |
| [![CI Tests](https://github.com/precious-adeniyi/pvf-u/actions/workflows/tests.yml/badge.svg)](https://github.com/precious-adeniyi/pvf-u/actions/workflows/tests.yml) | CI test status |
| [![Codecov](https://codecov.io/gh/precious-adeniyi/pvf-u/branch/main/graph/badge.svg)](https://codecov.io/gh/precious-adeniyi/pvf-u) | Test coverage |

---

## 📖 Further Reading

### Theoretical Foundations
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
- De Jaegher, H., & Di Paolo, E. (2007). Participatory sense-making. *Phenomenology and the Cognitive Sciences*, 6(4), 485-507.
- Ramstead, M. J. D., et al. (2023). Variational ecology and the free energy principle. *Physics of Life Reviews*.

### Related Implementations
- [Active Inference Institute](https://activeinference.org/)
- [Nomura & Horii Social Reality AIF](https://github.com/nomura-t/social-reality-aif)
- [Albarracin et al. Empathy AIF](https://github.com/albarracin/empathy-aif)

---

**PVF-U: From Solitary Survival to Ecosystemic Negotiation** — Building equitable, biologically-grounded AI together. 🌍🤖

---

*Last updated: June 2026*
```

