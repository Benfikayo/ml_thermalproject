# mlThermalproject -- End-to-End ML + Explainable AI for Thermoelectric zT Prediction

Adapted from and benchmarked against:
> Zeng, Y., Cao, W., Peng, T., Hou, Y., Miao, L., Wang, Z., & Shi, J. (2025).
> *A machine learning-based framework for predicting the power factor of thermoelectric materials.*
> Applied Materials Today, 43, 102627. https://doi.org/10.1016/j.apmt.2025.102627

## What's different from the paper (and why)

| Paper (PF) | This project (zT) | Reason |
|---|---|---|
| Target: power factor (PF = S²σ), split n-type / p-type | Target: zT (figure of merit), single unified model | Project scope -- zT is the more decision-relevant metric for material selection, and doping-type labels aren't part of this dataset |
| Feature set: Magpie descriptors + basic MP features (band gap, crystal system, energy above hull) | Composition-derived descriptors (av_x, av_atomic_radius, av_valence_electrons, oxidation states, group/period) + lattice parameters + band gap + T | zT = S²σT/κ, so S, σ, κ, ρ, PF are **excluded as inputs** -- including them would let the model "predict" zT by recomputing its own definition. `data_transformation.py` hard-blocks these columns even if present in the raw CSV. |
| Random train/test split (implied) | **Group-aware split by composition** (`GroupShuffleSplit`) | The same compound appears at multiple temperatures. A random split lets the model see a compound in training (at one T) and get evaluated on it in test (at another T), inflating reported R²/MAE. This wasn't controlled for in the paper. |
| Interpretability: CatBoost feature importance + SISSO | Feature importance (tree models) + **SHAP** + SISSO | SHAP gives per-sample, signed contributions rather than only a global ranking -- needed to support directional design-rule claims ("higher X tends to raise zT"), not just "X matters". |
| 5 models: RR, FCNN, XGBoost, LightGBM, CatBoost, + SISSO | Same 6, FCNN implemented with `MLPRegressor` by default (swap for PyTorch/Keras for closer replication of the paper's [128,64,32] architecture + Optuna-tuned hyperparameters) | Keeps the base pipeline dependency-light; the paper's exact FCNN + Bayesian hyperparameter search (Optuna) is documented as a drop-in upgrade in `model_trainer.py`. |

## Project structure

```
mlThermalproject/
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── app.py                      # optional Flask prediction UI
├── templates/index.html
├── notebook/
│   ├── data/Zt_data.csv        # your composition-enriched dataset
│   └── ZT_performance.ipynb    # EDA + interactive pipeline run + SHAP
├── src/mlThermalproject/
│   ├── __init__.py
│   ├── logger.py
│   ├── exception.py
│   ├── utils.py                 # save/load, group-aware split, evaluate_models
│   ├── components/
│   │   ├── data_ingestion.py    # reads Zt_data.csv, group-aware train/test split
│   │   ├── data_transformation.py  # preprocessing, leakage-column blocklist
│   │   ├── model_trainer.py     # RR, FCNN, LightGBM, CatBoost, XGBoost, SISSO stub
│   │   └── visualization.py     # full figure suite (see below)
│   └── pipeline/
│       ├── train_pipeline.py    # orchestrates ingestion -> transform -> train -> figures
│       └── predict_pipeline.py  # load model.pkl + preprocessor.pkl, predict new zT
├── artifacts/                   # generated: data.csv, train.csv, test.csv, model.pkl,
│                                 #   preprocessor.pkl, model_comparison.csv, figures/
└── logs/                        # generated: timestamped run logs
```

## Figures produced (matches the paper's suite, adapted to a single zT target)

1. Test R² and MAE comparison across RR / FCNN / LightGBM / CatBoost / XGBoost / SISSO
2. Train + test scatter plot (predicted vs true zT) per model, full dataset
3. log(zT)_ML vs log(zT)_true per model
4. Frequency vs log(zT) distribution, before/after sampling
5. Prediction error vs zT value, across models
6. Feature importance ranking per model
7. Descriptor-pair mapping to zT with iso-density contour lines
8. Univariate analysis (single descriptor vs zT with trend line)
9. SISSO symbolic-formula fitting result (run externally -- see `run_sisso_stub()`)

## Setup

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python -m src.mlThermalproject.pipeline.train_pipeline
```

Check `artifacts/model_comparison.csv` for the metrics table and `artifacts/figures/` for all plots.

## Predict on a new composition

Edit the example in `src/mlThermalproject/pipeline/predict_pipeline.py`, or run the Flask app:

```bash
python app.py
```

## Known data gaps to resolve before submission

- ~26 compositions are missing lattice parameters/band gap (no Materials Project match) -- see `mp_structural_lookup.py`.
- Oxidation states, group/period were computed from a hardcoded reference table for portability; regenerate
  via `pymatgen`/`matminer` locally for a citable, reproducible methodology section.
- FCNN uses `MLPRegressor` as a placeholder for the paper's custom architecture + Optuna Bayesian hyperparameter
  search -- swap in before claiming a like-for-like model comparison.