# Physics-Informed Neural Operators for Aerodynamic Flow-Field Prediction
*Physics-aware Fourier Neural Operator surrogate modelling for reducing non-physical flow artifacts during geometric extrapolation.*

## Project Overview
Aerodynamic flow-field prediction is traditionally expensive due to the high computational cost of conventional Computational Fluid Dynamics (CFD). While neural surrogates like Fourier Neural Operators (FNO) can drastically accelerate prediction speeds, they often generate physically inconsistent vector fields—particularly when extrapolating to unseen geometries.

This project investigates whether adding a continuity-based physical regularization loss to an FNO (yielding a PI-FNO) can force the network to respect fluid dynamics equations better, minimizing non-physical flow artifacts during geometric extrapolation.

---

## Key Result
> **98.5% reduction in aggregate held-out continuity residual across 50 test geometries**, while the PI-FNO incurred a **+0.0036 absolute Relative L2 trade-off**.

- **50/50** test geometries exhibited strictly improved physical continuity.
- Hyperparameter selection (loss weighting $\lambda = 1.0$) was executed **strictly on the validation set**.
- The held-out test set remained **completely unseen** during model and hyperparameter selection.

---

## What I Built
The technical pipeline was designed and executed from scratch:
**Geometry $\rightarrow$ Flow-field data $\rightarrow$ Preprocessing $\rightarrow$ Baseline FNO $\rightarrow$ Physics-aware FNO (PI-FNO) $\rightarrow$ Validation-based lambda selection $\rightarrow$ Blind test evaluation $\rightarrow$ Field-level analysis**

---

## Why Physics Regularization?
Pure data-driven models (like a baseline FNO) minimize global structural error (Relative $L_2$ against ground-truth fields) but remain unaware of the underlying physics, often producing velocity fields that violate local continuity (conservation of mass).

By explicitly encoding a discrete local continuity residual into the training objective, the PI-FNO is penalized for non-physical divergence. 

**The Central Trade-off**:
The PI-FNO dedicates network capacity to satisfying the physics residual, leading to substantially lower physical violation at the expense of a very small reduction in global field accuracy.

---

## Experimental Protocol

### Data Split
- **100** train geometries
- **25** validation geometries
- **50** held-out test geometries

### Hyperparameter Selection
- The physics-loss weighting factor ($\lambda$) candidates were evaluated exclusively on the **25 validation geometries**.
- $\lambda = 1.0$ was selected using a predefined validation rule (maximizing continuity improvement while bounding $L_2$ degradation).
- The 50 test geometries were absolutely blind to this selection.

### Final Evaluation
- Final selected models were evaluated on the held-out test set exactly once.

---

## Results

| Model | Relative L2 | Continuity Residual |
|-------|------------|---------------------|
| FNO | 0.1558 | 1.5468 |
| PI-FNO | 0.1594 | 0.0225 |

**Conclusion**: The PI-FNO achieved a **98.5%** local continuity reduction across the blind test set. **100% (50/50)** of the evaluated geometries improved physically, confirming that the physics regularization acts effectively as a domain-wide structural corrector, albeit requiring a small (+0.0036) accuracy trade-off.

---

## Representative Flow-Field Comparisons

### 1. Representative Median Case (Geometry 000110)
![CASE_A_000110](figures/results/CASE_A_000110.png)
*Observation: PI-FNO largely retains the macro flow features while suppressing local unphysical high-frequency artifacts (visible in the continuity residual field).*

### 2. Strongest Continuity Improvement (Geometry 000177)
![CASE_B_000177](figures/results/CASE_B_000177.png)
*Observation: FNO struggles profoundly with physical consistency in this boundary configuration. PI-FNO massively regularizes the field, eliminating the chaotic residual violations.*

### 3. Optimal Trade-off Profile (Geometry 000068)
![CASE_K_000068](figures/results/CASE_K_000068.png)
*Observation: In optimal cases, PI-FNO suppresses the continuity residual without inducing any perceptible loss in visual structural accuracy.*

---

## Local Hardware
The complete training, validation, and testing lifecycle was successfully executed locally on standard workstation hardware:
- **CPU**: AMD Ryzen 7 6800H (~16 GB RAM)
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM)
- **Environment**: PyTorch 2.5.1 + CUDA 12.1, FP32 Precision

---

## Reproducibility

### 1. Quick Smoke Test
To verify the environment and code execution loop (1 epoch, dummy data logic):
```bash
python scripts/train_and_evaluate.py --smoke-test
```

### 2. Data Preparation
To acquire the geometry fields:
```bash
python scripts/download_data.py
```

### 3. Full Local GPU Reproduction
To reproduce the full pipeline (Train $\rightarrow$ Validation Selection $\rightarrow$ Final Test):
```bash
python scripts/train_and_evaluate.py --config configs/pifno.yaml
```

---

## Repository Structure
```
physics-informed-fno-bladenet/
├── configs/
│   └── pifno.yaml
├── data/
├── docs/
├── figures/
│   └── results/
├── results/
│   ├── claim_summary.md
│   └── final_metrics.csv
├── scripts/
│   ├── download_data.py
│   └── train_and_evaluate.py
├── src/
│   └── models/
│       ├── fno3d.py
│       └── spectral_conv3d.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Limitations
- **Local Continuity vs. Global Mass Conservation**: The implemented continuity residual acts as a local physics regularizer; it does not theoretically guarantee strict domain-wide mass conservation.
- **Not a Strict direct CFD substitute**: Due to the intrinsic (+0.0036) Relative $L_2$ accuracy trade-off, this surrogate model accelerates inference but does not match exact CFD structural fidelity.
- **Regime Specificity**: Findings are strictly bound to the evaluated BladeNet geometric dataset regime (128x32x16 grids).
- **Hyperparameter Dependency**: The selected $\lambda = 1.0$ is the specific optimal factor derived for this dataset under this protocol, not a universal optimum for all PDE constraints.

---

## Dataset Attribution
The dataset utilized in this repository originates from the `lrwei/bladenet` collection hosted on HuggingFace. All dataset usage strictly defers to the original repository's licensing and attribution guidelines. 

