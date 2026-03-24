from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.common_config_ee_osm import MODELS_DIR_EE_OSM


MODEL_DIRS_EE_OSM = {
    "elastic_net": MODELS_DIR_EE_OSM / "elastic_net_ee_osm" / "elastic_net_metrics_ee_osm.json",
    "random_forest": MODELS_DIR_EE_OSM / "random_forest_ee_osm" / "random_forest_metrics_ee_osm.json",
    "xgboost": MODELS_DIR_EE_OSM / "xgboost_ee_osm" / "xgboost_metrics_ee_osm.json",
    "mlp": MODELS_DIR_EE_OSM / "mlp_ee_osm" / "mlp_metrics_ee_osm.json",
}

OUTPUT_DIR_EE_OSM = MODELS_DIR_EE_OSM / "model_comparison_ee_osm"
OUTPUT_PATH_EE_OSM = OUTPUT_DIR_EE_OSM / "model_comparison_summary_ee_osm.csv"


def main() -> None:
    rows: list[dict[str, object]] = []
    missing: list[str] = []

    for model_name, metrics_path in MODEL_DIRS_EE_OSM.items():
        if not metrics_path.exists():
            missing.append(model_name)
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for split_key, split_label in [
            ("test_metrics_ee_osm", "test_2019_2020"),
            ("external_forward_metrics_ee_osm", "forward_2020_2021"),
        ]:
            metrics = payload.get(split_key, {})
            rows.append(
                {
                    "model": model_name,
                    "split": split_label,
                    "overall_mae": metrics.get("overall_mae"),
                    "overall_rmse": metrics.get("overall_rmse"),
                    "overall_r2": metrics.get("overall_r2"),
                    "vegetation_r2": metrics.get("vegetation_r2"),
                    "built_up_r2": metrics.get("built_up_r2"),
                    "water_r2": metrics.get("water_r2"),
                    "other_r2": metrics.get("other_r2"),
                }
            )

    if not rows:
        raise FileNotFoundError("No _ee_osm metrics JSON files were found. Train at least one model first.")

    summary = pd.DataFrame(rows).sort_values(["split", "overall_mae", "overall_rmse"], ascending=[True, True, True])
    OUTPUT_DIR_EE_OSM.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH_EE_OSM, index=False)

    print("Model comparison EE OSM summary")
    print(summary.to_string(index=False))
    if missing:
        print(f"Missing metrics for: {', '.join(missing)}")
    print(f"Summary written to: {OUTPUT_PATH_EE_OSM}")


if __name__ == "__main__":
    main()
