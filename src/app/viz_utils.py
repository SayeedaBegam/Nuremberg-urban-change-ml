from __future__ import annotations

import math
from typing import Iterable

import altair as alt
import numpy as np
import pandas as pd


CLASS_ORDER = ["built_up", "vegetation", "water", "other"]
CLASS_COLORS = {
    "built_up": "#de2d26",
    "vegetation": "#31a354",
    "water": "#2171b5",
    "other": "#7f7f7f",
}
SPLIT_ORDER = ["test_2019_2020", "forward_2020_2021"]
MODEL_ORDER = ["elastic_net", "random_forest", "xgboost", "mlp"]


alt.data_transformers.disable_max_rows()


def _available_class_columns(frame: pd.DataFrame, template: str) -> dict[str, str]:
    return {
        class_name: template.format(class_name=class_name)
        for class_name in CLASS_ORDER
        if template.format(class_name=class_name) in frame.columns
    }


def prepare_dashboard_frame(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()

    if "uncertainty_mean" not in enriched.columns:
        if "uncertainty_mean_row" in enriched.columns:
            enriched["uncertainty_mean"] = enriched["uncertainty_mean_row"]
        else:
            uncertainty_columns = [
                column for column in [f"uncertainty_{class_name}" for class_name in CLASS_ORDER] if column in enriched.columns
            ]
            if uncertainty_columns:
                enriched["uncertainty_mean"] = enriched[uncertainty_columns].mean(axis=1)

    error_columns = []
    for class_name in CLASS_ORDER:
        pred_column = f"pred_{class_name}_prop_t2"
        actual_column = f"actual_{class_name}_prop_t2"
        error_column = f"abs_error_{class_name}"
        if error_column not in enriched.columns and pred_column in enriched.columns and actual_column in enriched.columns:
            enriched[error_column] = (enriched[pred_column] - enriched[actual_column]).abs()
        if error_column in enriched.columns:
            error_columns.append(error_column)

    if error_columns and "abs_error_mean" not in enriched.columns:
        enriched["abs_error_mean"] = enriched[error_columns].mean(axis=1)

    pred_columns = [f"pred_{class_name}_prop_t2" for class_name in CLASS_ORDER if f"pred_{class_name}_prop_t2" in enriched.columns]
    if len(pred_columns) == len(CLASS_ORDER):
        pred_frame = enriched[pred_columns].copy()
        if "dominant_class_pred" not in enriched.columns:
            enriched["dominant_class_pred"] = pred_frame.idxmax(axis=1).str.replace("pred_", "", regex=False).str.replace(
                "_prop_t2", "", regex=False
            )
        if "dominant_margin_pred" not in enriched.columns:
            values = np.sort(pred_frame.to_numpy(dtype=float), axis=1)
            enriched["dominant_margin_pred"] = values[:, -1] - values[:, -2]

    return enriched


def composition_summary_long(frame: pd.DataFrame) -> pd.DataFrame:
    stage_specs = [
        ("T1 Actual", _available_class_columns(frame, "{class_name}_prop_t1")),
        ("T2 Predicted", _available_class_columns(frame, "pred_{class_name}_prop_t2")),
        ("T2 Actual", _available_class_columns(frame, "actual_{class_name}_prop_t2")),
    ]
    rows = []
    for stage_name, columns in stage_specs:
        for class_name, column in columns.items():
            rows.append({"stage": stage_name, "class": class_name, "value": float(frame[column].mean())})
    return pd.DataFrame(rows)


def average_composition(frame: pd.DataFrame, template: str) -> pd.DataFrame:
    rows = []
    for class_name, column in _available_class_columns(frame, template).items():
        rows.append({"class": class_name, "value": float(frame[column].mean())})
    return pd.DataFrame(rows)


def change_summary_long(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_name, template in [("Predicted", "pred_delta_{class_name}"), ("Actual", "actual_delta_{class_name}")]:
        for class_name, column in _available_class_columns(frame, template).items():
            rows.append({"source": source_name, "class": class_name, "value": float(frame[column].mean())})
    return pd.DataFrame(rows)


def positive_change_share(frame: pd.DataFrame, template: str) -> pd.DataFrame:
    rows = []
    total = 0.0
    for class_name, column in _available_class_columns(frame, template).items():
        positive_mean = float(frame[column].clip(lower=0.0).mean())
        rows.append({"class": class_name, "value": positive_mean})
        total += positive_mean
    if total > 0:
        for row in rows:
            row["value"] = row["value"] / total
    return pd.DataFrame(rows)


def uncertainty_long(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {class_name: f"uncertainty_{class_name}" for class_name in CLASS_ORDER if f"uncertainty_{class_name}" in frame.columns}
    if not columns:
        return pd.DataFrame()
    melted = frame[list(columns.values())].rename(columns={value: key for key, value in columns.items()}).melt(
        var_name="class", value_name="uncertainty"
    )
    return melted.dropna()


def top_rows(frame: pd.DataFrame, sort_column: str, columns: Iterable[str], n_rows: int = 10) -> pd.DataFrame:
    available = []
    seen = set()
    for column in columns:
        if column in frame.columns and column not in seen:
            available.append(column)
            seen.add(column)
    if sort_column not in frame.columns or not available:
        return pd.DataFrame()
    return frame.loc[:, available].sort_values(sort_column, ascending=False).head(n_rows).reset_index(drop=True)


def error_scatter_frame(frame: pd.DataFrame, class_name: str) -> pd.DataFrame:
    pred_column = f"pred_{class_name}_prop_t2"
    actual_column = f"actual_{class_name}_prop_t2"
    if pred_column not in frame.columns or actual_column not in frame.columns:
        return pd.DataFrame()
    scatter = frame[[pred_column, actual_column]].dropna().rename(columns={pred_column: "predicted", actual_column: "actual"})
    return scatter


def dominant_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "dominant_class_pred" not in frame.columns or "dominant_margin_pred" not in frame.columns:
        return pd.DataFrame(), pd.DataFrame()

    counts = (
        frame.groupby("dominant_class_pred", dropna=False).size().rename("count").reset_index().rename(columns={"dominant_class_pred": "class"})
    )
    bin_edges = np.linspace(0.0, 1.0, 6)
    bin_labels = [f"{bin_edges[idx]:.1f}-{bin_edges[idx + 1]:.1f}" for idx in range(len(bin_edges) - 1)]
    margin_binned = pd.cut(
        frame["dominant_margin_pred"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=True,
    )
    margins = margin_binned.value_counts().sort_index().reset_index()
    margins.columns = ["margin_bin", "count"]
    margins["margin_bin"] = margins["margin_bin"].astype(str)
    return counts, margins


def _metric_value(split_metrics: dict, candidates: list[str]) -> float | None:
    for key in candidates:
        if key in split_metrics and split_metrics[key] is not None:
            return float(split_metrics[key])
    return None


def category_metrics_frame(metrics_payload: dict, class_name: str) -> pd.DataFrame:
    rows = []
    for model_name, model_payload in metrics_payload.items():
        for split_name, split_metrics in model_payload.get("metrics_by_split", {}).items():
            row = {"model": model_name, "split": split_name}
            row["r2"] = _metric_value(split_metrics, [f"{class_name}_t2_r2", f"{class_name}_r2"])
            row["mae"] = _metric_value(split_metrics, [f"{class_name}_t2_mae", f"{class_name}_mae"])
            row["rmse"] = _metric_value(split_metrics, [f"{class_name}_t2_rmse", f"{class_name}_rmse"])
            if any(row[metric] is not None for metric in ["r2", "mae", "rmse"]):
                rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["model"] = pd.Categorical(frame["model"], MODEL_ORDER, ordered=True)
        frame["split"] = pd.Categorical(frame["split"], SPLIT_ORDER, ordered=True)
        frame = frame.sort_values(["model", "split"]).reset_index(drop=True)
    return frame


def overall_metrics_frame(metrics_payload: dict) -> pd.DataFrame:
    rows = []
    for model_name, model_payload in metrics_payload.items():
        for split_name, split_metrics in model_payload.get("metrics_by_split", {}).items():
            row = {"model": model_name, "split": split_name}
            row["r2"] = _metric_value(split_metrics, ["overall_r2", "overall_R2"])
            row["mae"] = _metric_value(split_metrics, ["overall_mae", "overall_MAE"])
            row["rmse"] = _metric_value(split_metrics, ["overall_rmse", "overall_RMSE"])
            if any(row[metric] is not None for metric in ["r2", "mae", "rmse"]):
                rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["model"] = pd.Categorical(frame["model"], MODEL_ORDER, ordered=True)
        frame["split"] = pd.Categorical(frame["split"], SPLIT_ORDER, ordered=True)
        frame = frame.sort_values(["model", "split"]).reset_index(drop=True)
    return frame


def composition_stacked_bar(dataframe: pd.DataFrame) -> alt.Chart:
    chart_frame = dataframe.copy()
    chart_frame["class_order"] = chart_frame["class"].map({class_name: idx for idx, class_name in enumerate(CLASS_ORDER)})
    return (
        alt.Chart(chart_frame)
        .mark_bar()
        .encode(
            x=alt.X("stage:N", title=None),
            y=alt.Y("value:Q", title="Average proportion", stack="normalize"),
            color=alt.Color("class:N", scale=alt.Scale(domain=CLASS_ORDER, range=[CLASS_COLORS[c] for c in CLASS_ORDER])),
            order=alt.Order("class_order:Q"),
            tooltip=["stage", "class", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=320)
    )


def pie_chart(dataframe: pd.DataFrame, title: str) -> alt.Chart:
    return (
        alt.Chart(dataframe)
        .mark_arc(outerRadius=82)
        .encode(
            theta=alt.Theta("value:Q"),
            color=alt.Color(
                "class:N",
                scale=alt.Scale(domain=CLASS_ORDER, range=[CLASS_COLORS[c] for c in CLASS_ORDER]),
                legend=alt.Legend(orient="bottom", title="class", columns=2),
            ),
            tooltip=["class", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(title=alt.TitleParams(title, anchor="middle"), width=240, height=320)
        .configure_view(stroke=None)
    )


def grouped_bar_chart(dataframe: pd.DataFrame, x_field: str, y_field: str, color_field: str, title: str) -> alt.Chart:
    return (
        alt.Chart(dataframe)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_field}:N", title=None),
            xOffset=alt.XOffset(f"{color_field}:N"),
            y=alt.Y(f"{y_field}:Q", title=title),
            color=alt.Color(f"{color_field}:N", title=None),
            tooltip=[x_field, color_field, alt.Tooltip(f"{y_field}:Q", format=".4f")],
        )
        .properties(height=300)
    )


def histogram_chart(frame: pd.DataFrame, column: str, title: str) -> alt.Chart:
    return (
        alt.Chart(frame.dropna(subset=[column]))
        .mark_bar()
        .encode(
            x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=30), title=title),
            y=alt.Y("count():Q", title="Cells"),
            tooltip=[alt.Tooltip("count():Q", title="Cells")],
        )
        .properties(height=280)
    )


def boxplot_chart(dataframe: pd.DataFrame, value_column: str, category_column: str, title: str) -> alt.Chart:
    return (
        alt.Chart(dataframe.dropna(subset=[value_column]))
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X(f"{category_column}:N", title=None),
            y=alt.Y(f"{value_column}:Q", title=title),
            color=alt.Color(f"{category_column}:N", scale=alt.Scale(domain=CLASS_ORDER, range=[CLASS_COLORS[c] for c in CLASS_ORDER])),
            tooltip=[category_column, alt.Tooltip(value_column, format=".4f")],
        )
        .properties(height=280)
    )


def scatter_chart(dataframe: pd.DataFrame, title: str) -> alt.LayerChart:
    if dataframe.empty:
        return alt.Chart(pd.DataFrame({"predicted": [], "actual": []})).mark_point().properties(height=300)

    lower = float(min(dataframe["predicted"].min(), dataframe["actual"].min()))
    upper = float(max(dataframe["predicted"].max(), dataframe["actual"].max()))
    base = alt.Chart(dataframe).properties(height=320)
    points = base.mark_circle(size=45, opacity=0.45).encode(
        x=alt.X("actual:Q", title="Actual"),
        y=alt.Y("predicted:Q", title="Predicted"),
        tooltip=[alt.Tooltip("actual:Q", format=".3f"), alt.Tooltip("predicted:Q", format=".3f")],
    )
    line_df = pd.DataFrame({"actual": [lower, upper], "predicted": [lower, upper]})
    reference = alt.Chart(line_df).mark_line(strokeDash=[6, 4], color="#cccccc").encode(x="actual:Q", y="predicted:Q")
    return (points + reference).properties(title=title)


def confusion_matrix_chart(dataframe: pd.DataFrame) -> alt.Chart:
    label_order = ["Unchanged", "Changed"]
    return (
        alt.Chart(dataframe)
        .mark_rect()
        .encode(
            x=alt.X("predicted_label:N", sort=label_order, title="Predicted"),
            y=alt.Y("actual_label:N", sort=label_order, title="Actual"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), title="Cells"),
            tooltip=["actual_label", "predicted_label", "cell_type", alt.Tooltip("count:Q", format=",d")],
        )
        .properties(height=260)
    )


def confusion_text_overlay(dataframe: pd.DataFrame) -> alt.Chart:
    return alt.Chart(dataframe).mark_text(fontSize=14, fontWeight="bold").encode(
        x=alt.X("predicted_label:N", sort=["Unchanged", "Changed"]),
        y=alt.Y("actual_label:N", sort=["Unchanged", "Changed"]),
        text=alt.Text("count:Q", format=",d"),
        color=alt.value("white"),
    )


def coefficient_bar_chart(dataframe: pd.DataFrame, title: str) -> alt.Chart:
    chart_frame = dataframe.copy()
    chart_frame["direction"] = np.where(chart_frame["coefficient"] >= 0, "Positive", "Negative")
    return (
        alt.Chart(chart_frame)
        .mark_bar()
        .encode(
            x=alt.X("coefficient:Q", title="Coefficient"),
            y=alt.Y("feature:N", sort=alt.SortField(field="sort_order", order="ascending"), title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Positive", "Negative"], range=["#31a354", "#de2d26"]),
                title=None,
            ),
            tooltip=["feature", alt.Tooltip("coefficient:Q", format=".4f"), alt.Tooltip("abs_coefficient:Q", format=".4f")],
        )
        .properties(title=title, height=360)
    )
