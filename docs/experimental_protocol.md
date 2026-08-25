# Experimental Protocol

To ensure the integrity of the comparison and avoid overfitting the physics regularization term to the test data, the dataset was strictly partitioned.

## Dataset Partitioning
- **Training Set:** 100 geometries. Used to fit the network weights via Adam optimization.
- **Validation Set:** 25 geometries. Used exclusively to evaluate different candidate values for the physics loss weighting parameter ($\lambda$).
- **Test Set:** 50 geometries. Held out completely until the final models were locked.

## Model Selection
The baseline FNO ($\lambda = 0$) and several PI-FNO candidates were trained. The validation set was used to select the optimal physics weight that maximized continuity improvement without causing an unacceptable degradation in global $L_2$ accuracy. 

This process identified **$\lambda = 1.0$** as the optimal weight for this dataset.

## Held-Out Evaluation
The final baseline FNO and the selected PI-FNO ($\lambda = 1.0$) were evaluated exactly once on the 50 held-out test geometries. The resulting metrics form the definitive performance claims of this project.

## Low-Data Analysis
To understand how physics constraints impact performance when data is scarce, an additional experiment was run using only 25 training geometries. This experiment was repeated across three random seeds to ensure the resulting accuracy penalty (~0.00175) was stable and not a statistical anomaly.