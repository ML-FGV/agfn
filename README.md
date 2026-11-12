# Ancestral GFlowNets 

A JAX and Flax (NNX) implementation of Generative Flow Networks (GFlowNets) for causal discovery over **Ancestral Graphs** (and DAGs) under latent confounding.

## What It Does

- **Ancestral Graph Environment (`agfx.gym`)**: An environment that constructs valid ancestral graphs (handling directed edges, bidirected edges for confounding, and avoiding invalid cyclic structures).
- **GFlowNet Training (`agfx.policies`, `agfx.training`)**: Trajectory Balance (TB) and Conditional Balance (CB) training to sample posterior distributions over graph structures.
- **Scoring & Likelihoods (`agfx.rewards`, `agfx.special.ricf`)**: Scoring based on uniform, sparsity, and linear Gaussian likelihoods evaluated with Residual Iterative Conditional Fitting (RICF).
- **Active Learning (`agfx.eitl`)**: Experiment-in-the-loop active learning strategies for causal graph discovery.

---

## Installation

Requires **Python >= 3.13**.

### Using [uv](https://docs.astral.sh/uv/) (recommended)

```bash
# Clone the repository and sync dependencies
uv sync
```
---

## Quickstart

Run training via the CLI entrypoint:

```bash
# Inspect available CLI options
uv run python -m agfx.main --help

# Run GFlowNet training on 3-node graphs
uv run python -m agfx.main --num-nodes 3 --iterations 512
```

To run tests:

```bash
uv run pytest
```
