# Assignment Checklist

This file maps the assignment requirements to concrete project outputs so nothing gets missed during the final rush.

## Core task

- Predict land-cover composition and land-cover change from tabular satellite features
- Use ESA WorldCover as the main label source
- Use Sentinel-2 imagery for at least two time periods

## Problem framing and scope

- Focus classes: `built_up`, `vegetation`, `water`, `other`
- Spatial unit: `250m x 250m` grid cells by default
- Temporal setup: 2020 to 2021
- Intended user: non-expert city stakeholders
- Non-intended use: parcel-level or legal decision making

## Data exploration and reality check

- Feature distributions
- Label distributions
- Change-label distributions
- Three non-trivial issues to discuss:
  - cloud and missing-data effects
  - WorldCover label noise
  - spatial autocorrelation
- One issue explicitly not fixed:
  - residual label noise at mixed-cell boundaries

## Modeling and change prediction

- Model 1: Elastic Net
- Model 2: Random Forest
- Targets:
  - `delta_built_up`
  - `delta_vegetation`
  - `delta_water`
  - `delta_other`

## Evaluation beyond accuracy

- Spatial hold-out strategy in `src/models/train_all.py`
- Metrics:
  - MAE
  - RMSE
  - false change rate
  - stability score
- Stress test:
  - add Gaussian noise to features and compare performance

## Explainability and trust

- Helpful explanation text in `src/app/explain_utils.py`
- Misleading explanation text in `src/app/explain_utils.py`
- Uncertainty estimate from Random Forest disagreement

## Mandatory final product

- Streamlit app with:
  - map of Nuremberg study area
  - target selection
  - predicted change visualization
  - uncertainty display
  - limitation notes

## Mandatory reflection

- `reports/chatgpt_reflection.md`
- `reports/chatgpt_usage_log.md`

## Still dependent on real data

- Optional exact Nuremberg boundary file
- Optional OSM roads/buildings vector file for bonus features
- Real model outputs and screenshots for the report/demo
