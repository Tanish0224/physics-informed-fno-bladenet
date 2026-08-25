# Results

The final evaluation compares the baseline Fourier Neural Operator (FNO) against the selected Physics-Informed Fourier Neural Operator (PI-FNO) across the 50 held-out test geometries.

## Primary Metrics

| Metric | FNO | PI-FNO | Difference |
|--------|-----|--------|------------|
| **Relative L2** | 0.1558 | 0.1594 | +0.0036 |
| **Continuity Residual** | 1.5468 | 0.0225 | -98.5% |

The inclusion of the continuity regularization term successfully suppressed non-physical flow artifacts. 100% of the tested geometries (50 out of 50) showed a strictly improved continuity residual.

The trade-off for this physical consistency is a minor degradation in global structural accuracy, represented by the +0.0036 shift in the Relative $L_2$ error.

## Spatial Interpretation
Visually, the baseline FNO often predicts high-frequency, noisy continuity violations, particularly around complex geometry boundaries where extrapolation is difficult. The PI-FNO actively smooths out these unphysical artifacts. While the macroscopic flow field (velocity and pressure contours) looks structurally similar to the baseline, the underlying vector field obeys mass conservation principles far more closely.

## Low-Data Performance
To understand the model's behavior under severe data scarcity, the training set was restricted from 100 to 25 geometries. Across three random seeds, the PI-FNO maintained strong physical consistency, and the accuracy trade-off shrank to approximately +0.00175 Relative $L_2$. This indicates that physics regularization provides an outsized benefit when large datasets are unavailable to naturally smooth out unphysical predictions.