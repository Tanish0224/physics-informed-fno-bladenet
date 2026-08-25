# Reproducibility

This project is structured to be completely reproducible on local hardware. The pipeline handles dataset acquisition, normalization, training, validation, and final test evaluation.

## Environment Setup
Ensure you have a standard Python environment with PyTorch installed. The code was tested using PyTorch 2.5.1 with CUDA 12.1.
```bash
pip install -r requirements.txt
```

## Quick Verification
You can execute a minimal synthetic execution test (instantiation, forward pass, and backward pass) to confirm that your environment and the underlying model imports are correctly configured:
```bash
python scripts/train_and_evaluate.py --smoke-test
```

## Data Preparation
The dataset consists of 3D flow fields and geometric SDFs. Run the download script to fetch the `.npz` files from HuggingFace and unpack them into the appropriate `data/pilot/` structure:
```bash
python scripts/download_data.py  # Prints instructions
```

## Full Execution
Once the data is downloaded, you can trigger the full end-to-end pipeline. The script will automatically train the baseline and PI-FNO models, evaluate the models on the held-out test set, and save the final predictions to `results/predictions/`.

```bash
python scripts/train_and_evaluate.py --config configs/pifno.yaml
```

## Low-Data Study (Optional)
To reproduce the N=25 experiment, append the flag:
```bash
python scripts/train_and_evaluate.py --config configs/pifno.yaml --run-low-data
```