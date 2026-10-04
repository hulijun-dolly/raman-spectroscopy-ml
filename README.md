# Raman Spectroscopy ML Pipeline

This repository provides a practical workflow for Raman spectroscopy classification using:

- Raman-specific preprocessing
- PCA + SVM
- PLS-DA
- Random Forest

The project includes:

- a reusable Python pipeline in `src/raman_pipeline.py`
- a notebook in `notebooks/raman_model_comparison.ipynb`
- setup instructions and model comparison guidance

## Contents

- `src/raman_pipeline.py` — synthetic Raman dataset generation, preprocessing, model training, and evaluation
- `notebooks/raman_model_comparison.ipynb` — interactive notebook walkthrough
- `requirements.txt` — dependencies

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
python src/raman_pipeline.py
```

## Raman preprocessing workflow

The workflow is tailored for Raman spectra:

1. Cosmic ray spike removal
2. Baseline correction
3. Smoothing with Savitzky–Golay filter
4. Intensity normalization
5. Train/test split with stratification
6. Model comparison and report generation

## Example models

- PCA + SVM
- PLS-DA implemented with `PLSRegression` + argmax class assignment
- Random Forest classifier

## Notes

- For real Raman data, replace the synthetic dataset with your experimental spectra.
- Ensure the training/test split is done on a per-sample basis, not per-spectrum segment.
- For small sample sizes, use repeated cross-validation or nested CV for more reliable performance estimates.

## License

MIT
