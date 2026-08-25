# Limitations

While the Physics-Informed Fourier Neural Operator (PI-FNO) improves the physical realism of predicted flow fields, it is important to acknowledge its engineering boundaries.

- **Local vs. Global Conservation:** The continuity residual acts as a local spatial regularizer. It penalizes divergence pointwise during training, but it does not guarantee exact global mass conservation across the entire domain boundaries in the way a finite volume CFD solver would.
- **Accuracy Trade-off:** Enforcing physical constraints uses network capacity. The PI-FNO incurs a measurable (+0.0036) Relative $L_2$ accuracy penalty compared to the baseline FNO. For applications where absolute pointwise matching is more critical than physical consistency, the baseline model may be preferable.
- **Dataset Dependency:** The optimal physics weighting parameter ($\lambda = 1.0$) is specific to the evaluated BladeNet geometric dataset regime (128x32x16 grids). Different flow regimes or grid resolutions would require re-evaluating the $\lambda$ hyperparameter.
- **Not a CFD Replacement:** This surrogate accelerates inference significantly, but it does not achieve the exact structural fidelity of a converged Computational Fluid Dynamics simulation. It is a complementary predictive tool, not a replacement for high-fidelity verification.
