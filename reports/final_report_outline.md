# Final Report Outline

## 1. Problem Framing

- Why urban change in Nuremberg matters
- Intended user
- Non-intended use
- Spatial and temporal setup

## 2. Data

- Sentinel-2 imagery
- ESA WorldCover labels
- Grid-cell aggregation at 250m by default
- CRS and alignment choices

## 3. Exploratory Analysis and Data Issues

- Label balance
- Change distribution
- Spatial clustering
- Three key issues:
  - cloud contamination
  - label noise
  - spatial autocorrelation
- One issue not fixed and why

## 4. Features and Models

- Spectral summaries
- NDVI, NDBI, NDWI
- Elastic Net
- Random Forest

## 5. Evaluation

- Spatial hold-out strategy
- MAE / RMSE
- False change rate
- Stability
- Stress test under noisy features

## 6. Explainability and Trust

- Helpful explanation example
- Misleading explanation example
- Uncertainty communication

## 7. Limitations and Responsible Use

- Resolution mismatch
- Mixed cells
- Non-causal interpretation warning

## 8. Conclusion

- Main findings
- What the system is useful for
- What it should not be used for
