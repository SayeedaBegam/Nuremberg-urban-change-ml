import pandas as pd

# ================================
# 2020 MERGE
# ================================
features_2020 = pd.read_csv("data/processed_esa_ee_osm/nuremberg_2020_features_plus_osm_250m.csv")
labels_2020 = pd.read_csv("data/processed_esa_ee_osm/nuremberg_2020_labels_grouped_250m.csv")

# Clean unnecessary columns if present
for col in [".geo", "system:index"]:
    if col in features_2020.columns:
        features_2020 = features_2020.drop(columns=[col])

# Ensure same dtype
features_2020["cell_id"] = features_2020["cell_id"].astype(str)
labels_2020["cell_id"] = labels_2020["cell_id"].astype(str)

# Merge
merged_2020 = features_2020.merge(labels_2020, on="cell_id", how="left")

# Fill label NaNs (safety)
for col in ["vegetation", "built_up", "water", "other"]:
    merged_2020[col] = merged_2020[col].fillna(0)

# Save
merged_2020.to_csv("data/processed_esa_ee_osm/nuremberg_2020_composition_250m.csv", index=False)


# ================================
# 2021 MERGE
# ================================
features_2021 = pd.read_csv("data/processed_esa_ee_osm/nuremberg_2021_features_plus_osm_250m.csv")
labels_2021 = pd.read_csv("data/processed_esa_ee_osm/nuremberg_2021_labels_grouped_250m.csv")

# Clean unnecessary columns if present
for col in [".geo", "system:index"]:
    if col in features_2021.columns:
        features_2021 = features_2021.drop(columns=[col])

# Ensure same dtype
features_2021["cell_id"] = features_2021["cell_id"].astype(str)
labels_2021["cell_id"] = labels_2021["cell_id"].astype(str)

# Merge
merged_2021 = features_2021.merge(labels_2021, on="cell_id", how="left")

# Fill label NaNs (safety)
for col in ["vegetation", "built_up", "water", "other"]:
    merged_2021[col] = merged_2021[col].fillna(0)

# Save
merged_2021.to_csv("data/processed_esa_ee_osm/nuremberg_2021_composition_250m.csv", index=False)


# ================================
# OPTIONAL VALIDATION
# ================================
for name, df in [("2020", merged_2020), ("2021", merged_2021)]:
    df["label_sum"] = df["vegetation"] + df["built_up"] + df["water"] + df["other"]
    print(f"\n{name} label sum stats:")
    print(df["label_sum"].describe())