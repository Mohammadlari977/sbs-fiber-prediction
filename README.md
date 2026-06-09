# Physics-Informed Machine Learning for Nanofiber Diameter Prediction in Solution Blow Spinning

This repository contains the dataset, the complete modelling pipeline, and the trained
models for predicting the diameter of solution-blow-spun (SBS) polymer nanofibers from
physics-informed descriptors. Twelve machine-learning and deep-learning models are
benchmarked under a three-level validation protocol, with post-hoc calibration for
transfer to new laboratories.

## Overview

- **Dataset.** 408 pure-polymer samples curated from 57 published studies spanning 29
  polymers, with eight physics-informed input features and measured fiber diameter as
  the target.
- **Features.** Polymer concentration (wt %), intrinsic viscosity, reduced concentration
  c/c\*, solvent surface tension, solvent boiling point, air pressure, feed rate, and
  working distance.
- **Models.** Eight classical models — Extra Trees, Random Forest, XGBoost, LightGBM,
  CatBoost, Support Vector Regression, ElasticNet, and k-Nearest Neighbours — and four
  deep models — MLP, 1D-CNN, FT-Transformer, and TabNet.
- **Validation.** Three levels: (1) cross-validation with random and study-grouped
  folds; (2) a held-out set of unseen studies for zero-shot prediction and per-study
  calibration; (3) an independent external laboratory set (poly(ethylene oxide)
  nanofibers) for zero-shot prediction and affine calibration with bootstrap confidence
  estimates.
- **Reliability analysis.** Intraclass correlation, a linear mixed-effects model
  (conditional ICC), and sequential variance decomposition characterise between-study
  variance; SHAP and feature-ablation analyses assess feature importance.

## Repository structure

```
.
├── notebooks/
│   └── SBS_pipeline_core.ipynb   End-to-end pipeline: data cleaning, physics-informed
│                                 feature engineering, model training, all three
│                                 validation levels, calibration, and analysis.
├── src/
│   └── sbs_dl_models.py          Deep-learning model classes (MLP, 1D-CNN,
│                                 FT-Transformer) and a portable loader that rebuilds
│                                 the trained networks from their saved weights.
├── data/                         Dataset and splits.
│   ├── dataset_full.csv          Full curated dataset with all recorded columns.
│   ├── dataset_ml_ready.csv      Cleaned modelling dataset (eight features + target).
│   ├── cv_training_set.csv       Cross-validation training split.
│   ├── holdout_set.csv           Held-out studies used for Level-2 evaluation.
│   ├── external_validation.csv   Independent external laboratory measurements (Level 3).
│   ├── split_info.json           Split definition, random seed, and scaler statistics.
│   ├── mark_houwink_parameters.csv  Mark-Houwink constants used to compute intrinsic
│   │                                viscosity for each polymer.
│   └── solvent_properties.csv    Solvent surface tension, boiling point, and density.
├── models/
│   ├── scalers/                  Fitted StandardScaler objects.
│   │   ├── standard_scaler.pkl
│   │   └── standard_scaler_full.pkl
│   └── dl/                        Portable export of the trained deep-learning models.
│       ├── MLP_state.pt
│       ├── CNN1D_state.pt
│       ├── FT_Transformer_state.pt
│       ├── TabNet.zip
│       ├── dl_configs.json        Architecture configuration for each network.
│       └── sbs_dl_models.py       Loader and model classes used to rebuild the networks.
├── README.md
├── LICENSE
├── CITATION.cff
└── requirements.txt
```

## Installation

The pipeline targets Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Usage

Open `notebooks/SBS_pipeline_core.ipynb`, set `PROJECT_ROOT` at the top of the notebook
to the folder that holds the `data` and `models` files, then run the cells in order. The
pipeline is organised into sequential stages — data cleaning, physics-informed feature
engineering, feature scaling and the study-aware split, hyperparameter optimisation,
cross-validation, Level-2 hold-out and calibration, Level-3 external validation, and
reliability and feature-importance analysis — and checkpoints intermediate results, so
stages can be re-run independently. A GPU is recommended for the deep-learning and
bootstrap stages.

To load the trained deep-learning models from the portable export:

```python
from sbs_dl_models import load_dl_models
models = load_dl_models('models/dl')
```

## License

The code is released under the MIT License (see `LICENSE`). The dataset is released under
the Creative Commons Attribution 4.0 International (CC-BY-4.0) license.

## Citation

Citation metadata is provided in `CITATION.cff`.
