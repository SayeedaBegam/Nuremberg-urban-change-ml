# Demo Video Script

## 1. Introduction

This project analyzes urban change in Nuremberg using satellite-derived tabular features and ESA WorldCover labels. The goal is to highlight where built-up land and vegetation have changed over time, while also communicating uncertainty and model limitations.

## 2. Data and Method

The system aggregates Sentinel-2 imagery and WorldCover labels into 100-meter grid cells. For each cell, it computes spectral summaries and indices such as NDVI, NDBI, and NDWI, then trains two tabular machine learning models: Elastic Net and Random Forest.

## 3. Evaluation

The evaluation uses a spatial hold-out strategy rather than a random split. This matters because neighboring cells are similar, and a random split would overestimate performance. The project also includes a stress test by adding noise to features.

## 4. Product Walkthrough

The Streamlit dashboard lets the user inspect predicted built-up and vegetation change across the city. It also surfaces uncertainty and states clearly that the map is intended for broad pattern exploration, not parcel-level decisions.

## 5. Limitations

The main limitations are label noise, mixed cells, seasonal variation, and possible mismatch between imagery dates and annual label products. These limitations are shown directly in the dashboard and discussed in the report.
