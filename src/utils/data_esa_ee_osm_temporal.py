import os
import pandas as pd

# ================================
# Paths
# ================================
base_dir = "/home/dk/Nuremberg-urban-change-ml/data"

features_2019_path = os.path.join(base_dir, "processed_esa_ee_osm", "nuremberg_2019_features_plus_osm_250m.csv")
features_2020_path = os.path.join(base_dir, "processed_esa_ee_osm", "nuremberg_2020_features_plus_osm_250m.csv")

labels_2020_path = os.path.join(base_dir, "processed_esa_ee_osm", "nuremberg_2020_composition_250m.csv")
labels_2021_path = os.path.join(base_dir, "processed_esa_ee_osm", "nuremberg_2021_composition_250m.csv")

output_dir = os.path.join(base_dir, "train_esa_osm")
os.makedirs(output_dir, exist_ok=True)

output_2019_2020 = os.path.join(output_dir, "nuremberg_2019_to_2020_model_osm.csv")
output_2020_2021 = os.path.join(output_dir, "nuremberg_2020_to_2021_model_osm.csv")

# ================================
# Helper
# ================================
label_cols = ["cell_id", "vegetation", "built_up", "water", "other"]
drop_cols = [".geo", "system:index"]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    df["cell_id"] = df["cell_id"].astype(str)
    return df

def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df[label_cols].copy()
    df["cell_id"] = df["cell_id"].astype(str)
    return df

def build_temporal_table(features_path: str, labels_path: str, output_path: str) -> None:
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    features = prepare_features(features)
    labels = prepare_labels(labels)

    assert features["cell_id"].is_unique, f"Duplicate cell_id in features: {features_path}"
    assert labels["cell_id"].is_unique, f"Duplicate cell_id in labels: {labels_path}"

    merged = features.merge(labels, on="cell_id", how="left")

    for col in ["vegetation", "built_up", "water", "other"]:
        merged[col] = merged[col].fillna(0)

    merged["label_sum"] = (
        merged["vegetation"] +
        merged["built_up"] +
        merged["water"] +
        merged["other"]
    )

    print(f"\nBuilt: {os.path.basename(output_path)}")
    print("Rows:", len(merged))
    print("Unique cell_id:", merged["cell_id"].nunique())
    print("Null label rows:", merged[["vegetation", "built_up", "water", "other"]].isnull().sum().sum())
    print("Label sum stats:")
    print(merged["label_sum"].describe())

    merged = merged.drop(columns=["label_sum"])
    merged.to_csv(output_path, index=False)

# ================================
# Build temporal datasets
# ================================
build_temporal_table(features_2019_path, labels_2020_path, output_2019_2020)
build_temporal_table(features_2020_path, labels_2021_path, output_2020_2021)

print("\nSaved files:")
print(output_2019_2020)
print(output_2020_2021)