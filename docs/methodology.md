# Methodology

This project implements a controlled comparison between a baseline Fourier Neural Operator (FNO) and a Physics-Informed Fourier Neural Operator (PI-FNO) for aerodynamic flow-field prediction. 

## Network Architecture
Both models use the identical underlying 3D FNO architecture, initialized with:
- 8 Fourier modes across all three spatial dimensions.
- A channel width of 16.
- A standard MSE loss formulation for the data-driven component.

## Physics Regularization
The baseline FNO is trained purely on the data-driven MSE loss, optimizing for global structural accuracy (Relative $L_2$). 

For the PI-FNO, I introduced a custom PDE residual loss based on the continuity equation (conservation of mass). The continuity residual is defined as the discrete spatial divergence of the density-velocity flux:

$$ \nabla \cdot (\rho \mathbf{u}) = 0 $$

During training, this discrete residual is calculated across the internal domain points. The PI-FNO objective function then becomes:

$$ \mathcal{L} = \mathcal{L}_{MSE} + \lambda \mathcal{L}_{Continuity} $$

By incorporating this penalty, the network is forced to dedicate capacity to generating physically consistent velocity and density fields, rather than solely minimizing pointwise error.