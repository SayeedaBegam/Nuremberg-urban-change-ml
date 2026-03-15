# ChatGPT Usage Log

## Example prompts used

1. Help me scope a one-day ML assignment on urban change in Nuremberg using tabular models.
2. Propose a defensible feature set from Sentinel-2 imagery for land-cover change prediction.
3. Suggest evaluation methods beyond accuracy for spatial land-cover change modeling.
4. Draft a Streamlit app structure for communicating predicted change and uncertainty to non-experts.

## Misleading suggestion example

A generic suggestion was to use a standard random train/test split. That would likely inflate performance because neighboring cells are spatially correlated, so I rejected it in favor of a spatial hold-out strategy.

## Disagreement summary

I rejected suggestions that pushed toward unnecessary model complexity or evaluation shortcuts. The final project favors a smaller but more defensible system that matches the assignment constraints.
