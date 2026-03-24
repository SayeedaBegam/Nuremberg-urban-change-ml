# Mapping Urban Change in Nuremberg with Machine Learning

This repository contains the full final assignment submission for **Machine Learning WT 25/26 (UTN)**. The current `master` branch is the submission branch and includes the merged work from the Earth Engine composition-prediction workflow.

## Start Here

If you are reviewing the project for grading, use this order:

1. Read this `README.md` for the project scope and run instructions.
2. Check the deliverable notes in `reports/`.
3. Review the main training code in `src/models/`.
4. Launch the dashboard with `streamlit run src/app/app.py`.

Main submission entry points:

- `src/app/app.py`: Streamlit dashboard for predictions, maps, errors, uncertainty, interpretability, and stress tests
- `src/models/`: model training, evaluation, export, and comparison scripts
- `data/train_esa_ee/`: temporal training tables for the Sentinel-2 workflow
- `data/train_esa_osm/`: temporal training tables for the Sentinel-2 + OSM workflow
- `reports/`: supporting course-deliverable documents

## Short Overview

The project is a **tabular machine learning** system for land-cover modeling in **Nuremberg, Germany**. It is built around:

- Sentinel-2-derived fixed-length features
- ESA WorldCover-based land-cover labels
- multi-output regression for land-cover composition
- derived land-cover change analysis
- a file-based Streamlit dashboard for visualization and model diagnostics

The repository contains two related tracks:

- an older scaffold pipeline for change-target modeling from processed geospatial tables
- a newer temporal composition workflow with:
  - `ee`: Sentinel-2 only
  - `ee_osm`: Sentinel-2 + OSM context features

## Problem Statement

Urban areas change over time, but those changes are not always well captured by single-label maps. This project models land-cover composition per grid cell and uses those predictions to support change analysis.

Instead of predicting one dominant class, the main temporal workflow predicts four land-cover proportions:

- `built_up`
- `vegetation`
- `water`
- `other`

This keeps the project aligned with the assignment requirement to use **tabular / fixed-length features only**, without CNNs, Transformers, or end-to-end image models.

## Project Objectives

- Build a tabular ML pipeline for Nuremberg using satellite-derived predictors and ESA WorldCover labels
- Predict **T2 land-cover composition** from **T1 features**
- Support derived land-cover change analysis from those composition predictions
- Compare multiple classical ML models
- Evaluate both in-period performance and forward temporal generalization
- Provide an interactive dashboard with maps, actual-vs-predicted views, error summaries, uncertainty, interpretability, and stress testing

## Repository Structure

```text
.
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   |-- processed_esa_ee/
|   |-- processed_esa_ee_osm/
|   |-- train_esa_ee/
|   `-- train_esa_osm/
|-- models/
|   |-- elastic_net_ee/
|   |-- random_forest_ee/
|   |-- xgboost_ee/
|   |-- mlp_ee/
|   |-- elastic_net_ee_osm/
|   |-- random_forest_ee_osm/
|   |-- xgboost_ee_osm/
|   |-- mlp_ee_osm/
|   `-- model_comparison_ee_osm/
|-- reports/
|-- src/
|   |-- app/
|   |-- data/
|   |-- models/
|   `-- utils/
|-- README.md
`-- requirements.txt
```

Key folders:

- `src/app/`: Streamlit app and dashboard helper utilities
- `src/data/`: original dataset-preparation pipeline
- `src/models/`: model training, evaluation, and export scripts
- `src/utils/`: configuration and data-preparation helpers
- `models/`: trained model artifacts, metrics, prediction exports, and comparison outputs
- `reports/`: assignment support documents and submission notes

## Data Sources

Main data sources:

- Sentinel-2-derived tabular predictors for multiple time periods
- ESA WorldCover labels for land-cover composition
- optional OSM-derived context features for the `ee_osm` workflow

Examples of processed files already included:

- `data/train_esa_ee/nuremberg_2019_to_2020_model.csv`
- `data/train_esa_ee/nuremberg_2020_to_2021_model.csv`
- `data/train_esa_osm/nuremberg_2019_to_2020_model_osm.csv`
- `data/train_esa_osm/nuremberg_2020_to_2021_model_osm.csv`
- `data/processed_esa_ee/nuremberg_2019_features_250m.csv`
- `data/processed_esa_ee/nuremberg_2020_composition_250m.csv`
- `data/processed_esa_ee/nuremberg_2021_composition_250m.csv`
- `data/processed_esa_ee_osm/nuremberg_grid_250m_wgs84.geojson`

The repository also includes raster/vector inputs from the earlier scaffold workflow.

## Method Overview

### Study area and spatial unit

- **Study area**: Nuremberg, Germany
- **Spatial unit**: stable **250 m x 250 m grid cells**
- **Identifier**: `cell_id`

### Main temporal composition workflow

```text
Processed Sentinel-2 / ESA WorldCover tables
    -> temporal train table construction
    -> T1 feature selection
    -> T2 composition labels
    -> train / validation / test split on 2019 -> 2020
    -> forward evaluation on 2020 -> 2021
    -> prediction exports + metrics
    -> Streamlit dashboard
```

### Learning setup

- **Task type**: multi-output regression
- **Input**: T1 tabular features only
- **Output**: T2 land-cover composition
- **Targets**:
  - `built_up`
  - `vegetation`
  - `water`
  - `other`

### Temporal setup

- `2019 -> 2020` is used for train / validation / test
- `2020 -> 2021` is used for forward external evaluation

### Output post-processing

Predictions are converted to valid proportions:

- clipped to `[0, 1]`
- row-normalized so outputs sum to `1`

In the OSM-based workflow, three targets are predicted directly and `other` is reconstructed during post-processing, but the final outputs still form a four-part composition.

## Implemented Workflows and Models

### Original scaffold change pipeline

Files:

- `src/data/prepare_dataset.py`
- `src/models/train_all.py`

This older path trains:

- Elastic Net
- Random Forest

for change targets such as `delta_built_up`, `delta_vegetation`, `delta_water`, and `delta_other`.

### Sentinel-2 temporal composition pipeline (`ee`)

Training scripts:

- `src/models/train_elastic_net_ee.py`
- `src/models/train_random_forest_ee.py`
- `src/models/train_xgboost_ee.py`
- `src/models/train_mlp_ee.py`

### Sentinel-2 + OSM temporal composition pipeline (`ee_osm`)

Training scripts:

- `src/models/train_elastic_net_ee_osm.py`
- `src/models/train_random_forest_ee_osm.py`
- `src/models/train_xgboost_ee_osm.py`
- `src/models/train_mlp_ee_osm.py`

### Model families

- Elastic Net
- Random Forest
- XGBoost
- MLP

## Dashboard Overview

The dashboard is implemented in `src/app/app.py`. It is a **file-based dashboard**, not a live inference service.

Current capabilities:

- data-source switching between `Sentinel-2` and `Sentinel-2 + OSM`
- model selection
- split selection (`test_2019_2020`, `forward_2020_2021`)
- composition vs change views
- predicted and actual map panels
- uncertainty summaries
- error analysis
- model comparison
- Elastic Net coefficient view
- feature-noise stress testing

The app reads from saved artifacts such as:

- `data/processed/app_predictions.csv`
- `models/metrics.json`
- `models/*_ee/*.json`
- `models/*_ee_osm/*.json`
- `models/*_ee/*.csv`
- `models/*_ee_osm/*.csv`
- `data/interim/grid.geojson`
- `data/processed_esa_ee_osm/nuremberg_grid_250m_wgs84.geojson`

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional dependency:

```bash
pip install xgboost
```

`xgboost` is used by the XGBoost training scripts but is not listed in `requirements.txt`.

## How to Run

### Quick review path

If the goal is only to inspect the final project output:

```bash
streamlit run src/app/app.py
```

### Full workflow

1. Run the original scaffold pipeline

```bash
python -m src.data.prepare_dataset
python -m src.models.train_all
```

2. Train the Sentinel-2 temporal models

```bash
python -m src.models.train_elastic_net_ee
python -m src.models.train_random_forest_ee
python -m src.models.train_xgboost_ee
python -m src.models.train_mlp_ee
```

3. Build app-ready exports for the Sentinel-2 baseline view

```bash
python -m src.models.export_app_artifacts
```

4. Train the Sentinel-2 + OSM temporal models

```bash
python -m src.models.train_elastic_net_ee_osm
python -m src.models.train_random_forest_ee_osm
python -m src.models.train_xgboost_ee_osm
python -m src.models.train_mlp_ee_osm
```

5. Build the optional OSM comparison summary

```bash
python -m src.models.compare_models_ee_osm
```

6. Launch the dashboard

```bash
streamlit run src/app/app.py
```

## Submission Notes

The `master` branch is intended to contain the complete assignment in one place. Supporting course-deliverable files include:

- `reports/assignment_checklist.md`
- `reports/final_report_outline.md`
- `reports/chatgpt_reflection.md`
- `reports/chatgpt_usage_log.md`
- `reports/demo_video_script.md`

## Limitations

- The project is restricted to **tabular fixed-length features**; it does not use dense image models
- Forward generalization is harder than in-period test performance and should be interpreted carefully
- Some dashboard change views are derived from composition outputs rather than directly learned change labels
- Some dashboard sections depend on the availability of specific saved artifacts
- The repository contains both an older scaffold and a newer temporal workflow, so not every script serves the same prediction objective

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- geopandas
- rasterio / rioxarray
- Streamlit
- Folium / streamlit-folium
- joblib
- matplotlib / seaborn
- optional: xgboost

## Authors / Course Context

- Project: **Mapping Urban Change in Nuremberg with Machine Learning**
- Context: **Machine Learning WT 25/26**
- Institution: **UTN**
