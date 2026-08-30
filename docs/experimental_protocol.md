# Experimental Protocol

To maintain a fair comparison and prevent the physics regularization term from being tuned against the test data, the dataset was strictly divided into separate subsets.

## Dataset Partitioning

* **Training Set:** 100 geometries were used to train the network weights using Adam optimization.
* **Validation Set:** 25 geometries were reserved solely for testing different candidate values of the physics loss weighting parameter ($\lambda$).
* **Test Set:** 50 geometries were kept completely separate and were not accessed until the final models had been finalized.

## Model Selection

A baseline FNO ($\lambda = 0$) along with several PI-FNO variants were trained using different physics loss weights. The validation set was then used to identify the $\lambda$ value that provided the greatest improvement in continuity while maintaining an acceptable level of global $L_2$ accuracy.

This evaluation resulted in **$\lambda = 1.0$** being selected as the optimal physics weight for the dataset.

## Held-Out Evaluation

Once the model configurations were finalized, the baseline FNO and the selected PI-FNO ($\lambda = 1.0$) were evaluated once on the 50 held-out test geometries. The metrics obtained from this evaluation were used as the final performance results reported in the project.

## Low-Data Analysis

An additional experiment was conducted to examine the effect of physics constraints under limited training data. Only 25 geometries were used for training, and the experiment was repeated with three different random seeds. This was done to confirm that the observed accuracy penalty of approximately **0.00175** was consistent rather than the result of random statistical variation.
