"""Visualization utilities for predictions vs truth analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_predictions_vs_truth(
    predictions: pd.DataFrame,
    predicted_col: str,
    actual_col: str,
    class_name: str,
) -> plt.Figure:
    """Scatter plot of predictions vs actual values with perfect prediction line."""
    fig, ax = plt.subplots(figsize=(8, 8))

    valid_mask = predictions[predicted_col].notna() & predictions[actual_col].notna()
    pred = predictions.loc[valid_mask, predicted_col].values
    actual = predictions.loc[valid_mask, actual_col].values

    ax.scatter(actual, pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction', zorder=5)

    ax.set_xlabel(f'Actual {class_name}', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Predicted {class_name}', fontsize=11, fontweight='bold')
    ax.set_title(f'Predictions vs Truth\n{class_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')

    return fig


def plot_residuals(
    predictions: pd.DataFrame,
    predicted_col: str,
    actual_col: str,
    class_name: str,
) -> plt.Figure:
    """Residual plot showing errors across prediction range."""
    fig, ax = plt.subplots(figsize=(8, 6))

    valid_mask = predictions[predicted_col].notna() & predictions[actual_col].notna()
    pred = predictions.loc[valid_mask, predicted_col].values
    actual = predictions.loc[valid_mask, actual_col].values
    residuals = actual - pred

    ax.scatter(pred, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2.5)

    ax.set_xlabel(f'Predicted {class_name}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
    ax.set_title(f'Residual Plot\n{class_name}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')

    return fig


def plot_distribution_comparison(
    predictions: pd.DataFrame,
    predicted_col: str,
    actual_col: str,
    class_name: str,
) -> plt.Figure:
    """Overlaid histograms comparing predicted and actual distributions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    valid_mask = predictions[predicted_col].notna() & predictions[actual_col].notna()
    pred = predictions.loc[valid_mask, predicted_col].values
    actual = predictions.loc[valid_mask, actual_col].values

    ax.hist(actual, bins=40, alpha=0.6, label='Actual (Truth)', color='#2E86AB', edgecolor='black')
    ax.hist(pred, bins=40, alpha=0.6, label='Predicted', color='#A23B72', edgecolor='black')

    ax.set_xlabel(f'{class_name} Value', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'Distribution: Predicted vs Actual\n{class_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--', axis='y')

    return fig


def plot_error_distribution(
    predictions: pd.DataFrame,
    predicted_col: str,
    actual_col: str,
    class_name: str,
) -> plt.Figure:
    """Distribution of prediction errors (residuals)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    valid_mask = predictions[predicted_col].notna() & predictions[actual_col].notna()
    residuals = (predictions.loc[valid_mask, actual_col] - predictions.loc[valid_mask, predicted_col]).values

    ax.hist(residuals, bins=40, color='#F18F01', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', lw=2.5, label='Zero Error')
    ax.axvline(residuals.mean(), color='green', linestyle='--', lw=2.5, label=f'Mean Error: {residuals.mean():.4f}')

    ax.set_xlabel('Error (Actual - Predicted)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'Error Distribution\n{class_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--', axis='y')

    return fig


def plot_quantile_quantile(
    predictions: pd.DataFrame,
    predicted_col: str,
    actual_col: str,
    class_name: str,
) -> plt.Figure:
    """Q-Q plot to check if prediction errors are normally distributed."""
    from scipy import stats

    fig, ax = plt.subplots(figsize=(8, 8))

    valid_mask = predictions[predicted_col].notna() & predictions[actual_col].notna()
    residuals = (predictions.loc[valid_mask, actual_col] - predictions.loc[valid_mask, predicted_col]).values

    stats.probplot(residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_marker('o')
    ax.get_lines()[0].set_markersize(6)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linewidth(2.5)

    ax.set_title(f'Q-Q Plot (Normal Distribution Check)\n{class_name}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')

    return fig


def compute_error_metrics(
    predictions: pd.DataFrame,
    predicted_col: str,
    actual_col: str,
) -> dict:
    """Compute comprehensive error metrics."""
    valid_mask = predictions[predicted_col].notna() & predictions[actual_col].notna()
    pred = predictions.loc[valid_mask, predicted_col].values
    actual = predictions.loc[valid_mask, actual_col].values
    residuals = actual - pred

    mae = np.abs(residuals).mean()
    rmse = np.sqrt((residuals ** 2).mean())
    mape = np.abs(residuals / (np.abs(actual) + 1e-8)).mean() * 100
    r2 = 1 - (residuals.var() / actual.var()) if actual.var() > 0 else 0

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2,
        'Mean Error': residuals.mean(),
        'Std Error': residuals.std(),
        'Min Error': residuals.min(),
        'Max Error': residuals.max(),
    }
