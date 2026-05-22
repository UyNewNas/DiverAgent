<p align="center">
  <h1 align="center">CBDP: Convergent Backbone with Divergent Probe</h1>
  <p align="center">
    <em>Directional probes for zero-shot attribute-object composition</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/framework-PyTorch-red.svg" alt="PyTorch">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
    <img src="https://img.shields.io/badge/status-research-yellow.svg" alt="Research">
  </p>
</p>

---

## Overview

**CBDP (Convergent Backbone with Divergent Probe)** is a research paradigm for zero-shot attribute-object compositional generation. The core hypothesis: *can a directional probe learn decoupled attribute-direction perturbations that shift object features along attribute axes to generate unseen combinations?*

The approach uses a **two-stage training pipeline**:

1. **Stage 1 — Convergent Backbone**: Train a disentangled autoencoder that separates object identity from attributes in latent space.
2. **Stage 2 — Divergent Probe**: Freeze the backbone and train a directional probe that learns to generate K diverse perturbations guided by attribute embeddings, producing novel attribute-object compositions.

---

## Architecture

```
                      Stage 1 (frozen)              Stage 2 (trained)
                 ┌────────────────────────┐    ┌────────────────────────┐
                 │    SharedEncoder       │    │   DirectionalProbe     │
                 │    ObjectProjector     │    │   ├─ transport MLP     │
                 │    AttributeProjector  │    │   ├─ inj_mlp 0–3       │
                 │    ConditionalDecoder  │    │   └─ DivergenceGate     │
                 │    Embedding anchors   │    │   NoveltyLoss (memory)  │
                 └────────────────────────┘    └────────────────────────┘
                            │                            │
   Input x ───▶ SharedEncoder ───▶ obj_emb ───▶ DirectionalProbe ───▶ K × z_k
                                        │                        │
                                  attr_emb ───▶ directional guide  │
                                                                   │
                                                        ┌─────────▼─────────┐
                                                        │ConditionalDecoder  │
                                                        │(+ per-layer inj)   │
                                                        └─────────┬─────────┘
                                                                 ▼
                                                        K diverse outputs
```

### Key Components

| Component | Description |
|---|---|
| **SharedEncoder/CNNEncoder** | Multi-layer Conv-BN-ReLU + FC, outputs shared feature vector |
| **ObjectProjector** | Shared feature → object latent space (64/128 dim) |
| **AttributeProjector** | Shared feature → attribute latent space (64/128 dim) |
| **DirectionalProbe** | `z_k = obj_emb + transport_k(obj⊕attr) + attr_emb` — learns to shift object features along attribute directions, producing K perturbations with per-layer decoder injections |
| **DivergenceGate** | Lightweight classifier that learns when to apply divergence |
| **NoveltyLoss** | Memory bank (ring buffer) + cosine similarity threshold; penalizes probe outputs that too closely resemble training samples |

### Three-Anchor Loss

```python
total_loss = 0.15 × (diversity - 0.7)² + 0.80 × plausibility + 0.50 × novelty
```

| Term | Weight | Goal |
|---|---|---|
| Diversity | 0.15 | Push K outputs toward target pairwise cosine similarity τ = 0.3 |
| Plausibility | 0.80 | At least one output should closely resemble a valid reconstruction |
| Novelty | 0.50 | Probe outputs should differ from all training samples in memory |

### DCI Metric

Geometric mean of three dimensions — **not** the traditional disentanglement metric:

```python
DCI = (plausibility × divergence_quality × novelty)^(1/3)
```

where `divergence_quality = max(0, 1 - |cos_sim - τ|)` with τ = 0.3.

---

## Experiments

### Phase 2a: Colored dSprites (Synthetic)

**Setup**: 3 shapes × 6 colors = 18 combinations. Training on 4 colors per shape, zero-shot evaluation on 2 held-out colors.

| Metric | Backbone (baseline) | CBDP (full) |
|---|---|---|
| **DCI (train)** | 0.101 | **0.652** |
| **DCI (test, zero-shot)** | 0.122 | **0.621** |
| Cosine similarity (test) | 1.000 | **0.386** |
| Plausibility (test) | 0.925 | **0.948** |
| Novelty (test) | 0.007 | **0.276** |
| Shape retention (test) | — | **0.801** |
| Color retention (test) | — | **0.514** |
| **CBDP vs Random baseline** | — | **+198%** |

**Ablation (test set)**:

| Config | DCI | cos_sim | novelty |
|---|---|---|---|
| **Full** | **0.677** | 0.307 | 0.328 |
| λ_div = 0 | 0.586 | 0.439 | 0.244 |
| λ_pla = 0 | 0.671 | 0.296 | 0.318 |
| λ_nov = 0 | 0.674 | 0.303 | 0.328 |

### Phase 2b: MIT-States (Real Images)

**Setup**: 245 objects × 115 attributes, ~53K images. 196 objects for training, 49 held-out for zero-shot evaluation.

| Metric | Backbone (baseline) | CBDP (full) |
|---|---|---|
| **DCI (train)** | 0.083 | **0.580** |
| **DCI (test, unseen combos)** | 0.083 | **0.580** |
| Cosine similarity (test) | 1.000 | **0.421** |
| Plausibility (test) | 0.869 | **0.948** |
| Novelty (test) | 0.002 | **0.234** |
| **Train-test DCI gap** | — | **~0.0001** |
| Object retention (test) | — | **0.403** |
| Attribute retention (test) | — | **~1.000** |
| **CBDP vs Random baseline** | — | **+275%** |

**Ablation (test set, fresh probe init)**:

| Config | DCI | cos_sim | novelty |
|---|---|---|---|
| **Full** | **0.575** | 0.381 | 0.215 |
| λ_div = 0 | 0.241 | 0.928 | 0.041 |
| λ_pla = 0 | 0.501 | 0.429 | 0.167 |
| λ_nov = 0 | 0.520 | 0.368 | 0.158 |

**Key findings**:
- Near-zero train-test DCI gap demonstrates strong generalization to unseen attribute-object compositions.
- Corrected ablation (fresh random probe init, not deepcopy from trained) confirms all three loss terms are critical — λ_div ablation drops DCI to 0.241, the largest single impact.
- Diversity (cos_sim=0.421) still deviates from target τ=0.3, and novelty (0.234) remains constrained, indicating the probe struggles to escape the well-converged backbone latent space.

---

## Repository Structure

```
DiverAgent/
├── src/diver_agent/          # Core CBDP library
│   ├── backbone.py           # Shared backbone components
│   ├── cbdp.py               # CBDP model wrapper
│   ├── gate.py               # Divergence gate
│   ├── losses.py             # Three-anchor loss functions
│   └── probe.py              # Directional probe
├── experiments/
│   ├── dsprites/             # Phase 2a: colored dSprites
│   │   ├── backbone.py       # DSpritesBackbone + DirectionalProbe
│   │   ├── config.py         # Hyperparameters
│   │   ├── dataset.py        # Train/test combinatorial split
│   │   ├── gen_dataset.py    # Dataset generation
│   │   ├── stage1_train_backbone.py
│   │   ├── stage2_train_probe.py
│   │   ├── evaluate.py       # DCI, ablation, random baseline
│   │   └── results/          # Logs and result JSONs
│   ├── mitstates/            # Phase 2b: MIT-States
│   │   ├── backbone.py       # CNNEncoder + MITStatesBackbone
│   │   ├── config.py         # Hyperparameters
│   │   ├── dataset.py        # Zip-direct read, object-level split
│   │   ├── stage1_train_backbone.py
│   │   ├── stage2_train_probe.py
│   │   ├── evaluate.py       # DCI + CLIP metrics
│   │   └── results/          # Logs, result JSONs, visualizations
│   └── omniglot/             # Phase 1: Omniglot (archived)
├── .gitignore
├── LICENSE                   # MIT
├── pyproject.toml
└── README.md
```

---

## Reproduce

### Requirements

```bash
pip install torch torchvision numpy tqdm matplotlib lpips
```

### Run Pipeline

**dSprites (Phase 2a):**
```bash
cd experiments/dsprites
python gen_dataset.py        # Generate colored dSprites dataset
python stage1_train_backbone.py  # Stage 1: convergent backbone
python stage2_train_probe.py     # Stage 2: divergent probe
python evaluate.py               # DCI evaluation + ablations
```

**MIT-States (Phase 2b):**
```bash
cd experiments/mitstates
python stage1_train_backbone.py  # Stage 1: CNN backbone (60 epochs)
python stage2_train_probe.py     # Stage 2: directional probe (20 epochs)
python evaluate.py               # DCI + CLIP evaluation
```

The dataset for MIT-States will be automatically downloaded (~808MB zip) on first run.

---

## License

[MIT](LICENSE) © 2026 Slava Chan
