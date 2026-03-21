# Mapping Urban Change in Nuremberg with Machine Learning

This repository contains the final assignment project for **Machine Learning WT 25/26** at **UTN**. It is a **strictly tabular machine learning** project for modeling land-cover composition and urban change in **Nuremberg, Germany** using aggregated Sentinel-2 features and ESA WorldCover labels.

The repository now contains two closely related layers of work:

- the original course scaffold and tabular geospatial ML pipeline in `src/data`, `src/models`, and `src/app`
- the current **Earth Engine-driven temporal pipeline** that builds annual tabular features and trains multi-output regression models for `2019 -> 2020` and `2020 -> 2021`

The current focus of the repository is:

- **multi-output regression** from T1 satellite-derived tabular features to T2 land-cover proportions
- evaluation on both an in-period held-out split and a harder forward temporal split
- a **file-based Streamlit dashboard** that visualizes predictions, labels, uncertainty, error summaries, and model comparison

## Updated Overview

This project does **not** use CNNs, segmentation models, or spatial deep learning. Each row in the modeling dataset represents one stable **250 m x 250 m grid cell** with a `cell_id`. The model input is a fixed-length vector of spectral and index-based features derived from **Sentinel-2**. The target is a 4-part land-cover composition vector derived from **ESA WorldCover**:

- `built_up`
- `vegetation`
- `water`
- `other`

The current experimental setup is:

- `2019 -> 2020` for **train / validation / test**
- `2020 -> 2021` for **forward external evaluation only**

Predictions are post-processed to remain valid proportions:

- clipped to `[0, 1]`
- row-normalized so the four outputs sum to `1`

## Assignment Goal

The assignment asks for a model-based system that:

- uses **satellite imagery** for at least two time periods
- uses **ESA WorldCover** as the main source of land-cover labels
- works with **tabular features**, not CNNs or end-to-end vision models
- predicts **land-cover composition** and supports change analysis
- evaluates models beyond plain accuracy
- communicates **uncertainty**, **limitations**, and **trust** to non-experts
- provides a working **interactive product**

This repository was built specifically around those requirements.

## Study Area and Grid

### Study area

- City: **Nuremberg, Germany**
- Geometry source:
  - preferred: Nuremberg boundary file
  - fallback in the original repo code: a tighter Nuremberg bounding box if the exact boundary is missing

### Spatial unit

- fixed analysis unit: **250 m x 250 m grid cells**
- stable identifier: `cell_id`

This aggregation level was chosen because it:

- reduces pixel-level noise
- keeps the problem in a tabular ML setting
- makes outputs easier to summarize and explain
- is practical for course-scale experimentation and dashboard visualization

### Intended users

- non-expert city stakeholders
- students and instructors reviewing the workflow
- public users exploring broad spatial patterns

### Non-intended use

The system should **not** be used for:

- parcel-level planning
- legal or regulatory decisions
- property valuation
- causal claims about why one specific location changed

## Current End-to-End Workflow

### Pipeline at a glance

```text
Sentinel-2 + ESA WorldCover + Nuremberg boundary
    -> annual preprocessing and compositing
    -> 250 m grid aggregation
    -> T1 tabular feature creation
    -> T2 proportion label creation
    -> temporal dataset assembly
    -> multi-output regression training
    -> in-period and forward evaluation
    -> exported CSV / JSON artifacts
    -> Streamlit dashboard
```

### Updated project pipeline

1. Raw data are collected for Nuremberg.
2. Sentinel-2 imagery is filtered by year and cloud conditions.
3. Annual median composites are created.
4. Spectral and index features are aggregated to stable 250 m cells.
5. ESA WorldCover is grouped into four classes and converted to per-cell proportions.
6. Temporal datasets are built for `2019 -> 2020` and `2020 -> 2021`.
7. Multi-output regression models are trained on `2019 -> 2020` only.
8. Performance is checked on a held-out in-period test split and a forward external split.
9. Predictions, metrics, and summaries are exported to file-based artifacts.
10. The Streamlit dashboard reads those artifacts and visualizes model behavior.

## Data Sources

### Sentinel-2

- source: **`COPERNICUS/S2_SR_HARMONIZED`**
- use in this project:
  - annual feature generation for `2019` and `2020` inputs
  - spectral and index-based tabular predictors

### Sentinel-2 preprocessing

The Earth Engine workflow is based on:

- full-year filtering
- cloud filtering
- annual **median compositing**

The current feature set is intentionally simple and tabular.

### ESA WorldCover

- **2020**: WorldCover **v100**
- **2021**: WorldCover **v200**

These layers are used to construct the target land-cover composition per grid cell.

### Optional context data

The original repository also contains optional OSM context-feature code for:

- `road_density`
- `building_density`

That path is separate from the current Earth Engine temporal dataset workflow.

## Feature Engineering

The current T1 feature vector contains the following tabular predictors:

- `B2`
- `B3`
- `B4`
- `B8`
- `B11`
- `NDVI`
- `NDWI`
- `NDBI`

These features are aggregated per 250 m grid cell, giving one fixed-length row per `cell_id`.

The project deliberately avoids:

- raw image patches
- CNN feature extraction
- spatial deep learning
- any T2-derived predictors in the input feature set

## Label Generation

ESA WorldCover classes are grouped into four land-cover categories:

- `vegetation`
- `built_up`
- `water`
- `other`

For each 250 m cell, the label is stored as a **proportion vector** whose components sum to approximately 1:

- `vegetation_prop`
- `built_up_prop`
- `water_prop`
- `other_prop`

This turns the labeling problem into a mixed-land-cover composition task instead of a single dominant-class classification task.

## Dataset Construction

### Temporal datasets

Two temporal datasets are used in the current pipeline:

1. `2019 -> 2020`
   - used for **train / validation / test**
2. `2020 -> 2021`
   - used only for **forward external evaluation**

### Current dataset files

The current Earth Engine-derived files in the repository include:

- `data/processed_esa_ee/nuremberg_2019_features_250m.csv`
- `data/processed_esa_ee/nuremberg_2020_composition_250m.csv`
- `data/processed_esa_ee/nuremberg_2021_composition_250m.csv`
- `data/train_esa_ee/nuremberg_2019_to_2020_model.csv`
- `data/train_esa_ee/nuremberg_2020_to_2021_model.csv`

### Current split convention

For the `_ee` modeling pipeline, the final split is:

- `80%` train
- `10%` validation
- `10%` test

on the `2019 -> 2020` dataset.

## Machine Learning Formulation

### Problem type

The current ML task is:

- **multi-output regression**

### Inputs and outputs

- `X`: T1 tabular satellite-derived features only
- `y`: T2 land-cover proportions

For each row:

- input year = earlier year (`2019` or `2020`)
- output year = later year (`2020` or `2021`)

### Target columns

The target vector contains four outputs:

- `vegetation_t2`
- `built_up_t2`
- `water_t2`
- `other_t2`

### Output constraints

Because the outputs represent proportions, model predictions are post-processed by:

1. clipping to `[0, 1]`
2. row-normalizing to force the four outputs to sum to `1`

This keeps prediction vectors interpretable as compositions.

## Models Used

The current `_ee` experiments include four tabular models:

- **Elastic Net**
- **Random Forest**
- **XGBoost**
- **MLP**

### Role of each model

- **Elastic Net**
  - interpretable linear baseline
  - coefficient export available
- **Random Forest**
  - nonlinear ensemble baseline
  - feature importance export available
  - uncertainty estimate from tree disagreement
- **XGBoost**
  - boosted tree model with stronger nonlinear capacity
- **MLP**
  - small tabular neural baseline using dense layers only
  - still tabular, not image-based deep learning

## Evaluation Strategy

### In-period evaluation

The first evaluation is performed on the held-out test split from:

- `2019 -> 2020`

This measures generalization to unseen rows from the same temporal transition.

### Forward temporal evaluation

The second evaluation is performed on:

- `2020 -> 2021`

This is the more realistic and harder evaluation because it tests **forward temporal generalization**.

### Metrics

The current `_ee` pipelines compute:

- `MAE`
- `RMSE`
- `R²`

for:

- each class separately
- overall macro-style summary across outputs

### Why this matters

A model can look strong on an in-period test split but still degrade badly on the forward split. This repository is designed to make that gap visible rather than hiding it.

## Current Findings

The current experiments show a consistent pattern:

- nonlinear models perform better on the held-out in-period test split
- forward temporal generalization is weaker across the board
- **Elastic Net** is more interpretable and is relatively steadier forward than the stronger in-period nonlinear models
- the project highlights the risk of **temporal overfitting** when training on a single-year transition only

In short:

- stronger in-sample performance does **not** automatically mean better forward-time robustness
- temporal evaluation is necessary for a realistic interpretation of model quality

## Streamlit Dashboard

The dashboard lives in:

- `src/app/app.py`

### Current dashboard design

The app is **file-based**, not a live inference service. It reads exported artifacts and visualizes:

- predicted T2 composition
- actual T2 composition labels
- optional change views when the required T1 composition exists
- selected model metrics
- uncertainty summaries
- error summaries
- model comparison across splits

### Current top layout

The top visible section contains:

- the left sidebar with controls and metrics
- the predicted map panel
- the actual labels map panel
- the explanation panel

### Analytical sections below the top view

Below the top map-based section, the dashboard is being extended with additional analytical views such as:

- **Land-Cover Composition Summary**
- **Confidence / Uncertainty Analysis**
- **Land-Cover Change Summary**
- **Prediction Error Analysis**
- **Model Comparison Across Splits**
- **Dominant Land-Cover Summary**
- **Download / Export**

### Model comparison design

The model comparison section is organized in this order:

1. **category-wise comparison first**
   - built_up
   - vegetation
   - water
   - other
2. **overall comparison second**
   - overall R²
   - overall MAE
   - overall RMSE

### Dashboard artifact files

The dashboard currently expects file-based inputs such as:

- `data/processed/app_predictions.csv`
- `models/metrics.json`
- `data/interim/grid.geojson`

and, when needed for geometry compatibility with the Earth Engine grid:

- `data/processed_esa_ee/nuremberg_2020_composition_250m.csv`
- `data/processed_esa_ee/nuremberg_2021_composition_250m.csv`
- `data/processed_esa_ee/nuremberg_2019_features_250m.csv`

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── processed_esa_ee/
│   └── train_esa_ee/
├── models/
│   ├── elastic_net_ee/
│   ├── random_forest_ee/
│   ├── xgboost_ee/
│   └── mlp_ee/
├── reports/
├── src/
│   ├── app/
│   ├── data/
│   ├── models/
│   └── utils/
├── README.md
└── requirements.txt
```

### Important folders

- `src/app`
  - Streamlit dashboard, map utilities, explanation text, and analytical visualizations
- `src/data`
  - original grid-based feature engineering and label-generation pipeline
- `src/models`
  - original change-model code plus the current `_ee` training and export scripts
- `src/utils`
  - config, I/O helpers, and current `_ee` model configuration files
- `data/processed_esa_ee`
  - annual Earth Engine-derived grid-level features and label tables
- `data/train_esa_ee`
  - temporal training tables for `2019 -> 2020` and `2020 -> 2021`
- `models/*_ee`
  - trained model artifacts, prediction exports, and metrics for each model
- `reports`
  - report scaffold, reflection notes, checklist, and demo script

## How to Run the Project

Run everything from the project root.

### 1. Create and activate the environment

#### Ubuntu / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train the current `_ee` models

Examples:

```bash
python -m src.models.train_elastic_net_ee
python -m src.models.train_random_forest_ee
python -m src.models.train_xgboost_ee
python -m src.models.train_mlp_ee
```

These scripts write artifacts under:

- `models/elastic_net_ee/`
- `models/random_forest_ee/`
- `models/xgboost_ee/`
- `models/mlp_ee/`

### 3. Build the dashboard exports

```bash
python -m src.models.export_app_artifacts
```

This writes:

- `data/processed/app_predictions.csv`
- `models/metrics.json`

### 4. Launch the Streamlit dashboard

```bash
streamlit run src/app/app.py
```

## Original Repo Pipeline

The original repository pipeline is still present and useful for reference.

### Original pipeline scripts

- `src.data.prepare_dataset`
- `src.models.train_all`

### Original outputs

- `data/processed/dataset_t1.csv`
- `data/processed/dataset_t2.csv`
- `data/processed/change_dataset.csv`
- `models/elastic_net.joblib`
- `models/random_forest.joblib`
- `models/metrics.json` in the original scaffold setup

That path is centered on **change prediction** and the earlier course scaffold, while the current `_ee` workflow is centered on **T1 -> T2 composition regression** and forward temporal evaluation.

## Main Output Artifacts

### Current Earth Engine model artifacts

- `models/elastic_net_ee/elastic_net_model_ee.joblib`
- `models/elastic_net_ee/elastic_net_test_predictions_ee.csv`
- `models/elastic_net_ee/elastic_net_external_predictions_ee.csv`
- `models/elastic_net_ee/elastic_net_metrics_ee.json`
- `models/elastic_net_ee/elastic_net_coefficients_ee.csv`
- `models/random_forest_ee/random_forest_model_ee.joblib`
- `models/random_forest_ee/random_forest_test_predictions_ee.csv`
- `models/random_forest_ee/random_forest_external_predictions_ee.csv`
- `models/random_forest_ee/random_forest_metrics_ee.json`
- `models/random_forest_ee/random_forest_feature_importances_ee.csv`
- `models/xgboost_ee/xgboost_model_ee.joblib`
- `models/xgboost_ee/xgboost_test_predictions_ee.csv`
- `models/xgboost_ee/xgboost_external_predictions_ee.csv`
- `models/xgboost_ee/xgboost_metrics_ee.json`
- `models/xgboost_ee/xgboost_feature_importances_ee.csv`
- `models/mlp_ee/mlp_model_ee.joblib`
- `models/mlp_ee/mlp_test_predictions_ee.csv`
- `models/mlp_ee/mlp_external_predictions_ee.csv`
- `models/mlp_ee/mlp_metrics_ee.json`

### Dashboard-ready exports

- `data/processed/app_predictions.csv`
- `models/metrics.json`
- `data/interim/grid.geojson`

## Current Limitations

- The project still depends on a **single-year transition** for training, which makes forward generalization difficult.
- The `2019 -> 2020` test split is only a held-out subset, so those dashboard views are not full-city inference maps.
- A full `2019` composition table is not currently present in the repository, which limits some change-mode views for the `test_2019_2020` split.
- ESA WorldCover remains an imperfect proxy for ground truth.
- Mixed cells and boundary cells remain noisy.
- Temporal mismatch can still exist between annual labels and image compositing choices.
- The dashboard is exploratory and educational, not an operational decision-support system.

## Future Improvements / Next Steps

- add a full-city `2019 -> 2020` inference export for easier visual comparison
- strengthen temporal robustness using more years or multi-transition training
- improve uncertainty summaries and calibration analysis
- add more careful treatment of mixed-cell uncertainty
- continue dashboard polishing and reporting for the final course submission

## Files for Submission Support

- `reports/final_report_outline.md`
- `reports/assignment_checklist.md`
- `reports/chatgpt_reflection.md`
- `reports/chatgpt_usage_log.md`
- `reports/demo_video_script.md`
