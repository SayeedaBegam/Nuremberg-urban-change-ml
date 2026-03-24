import pandas as pd

spectral_2021 = pd.read_csv("data/processed_esa_ee_osm/nuremberg_2021_features_plus_location_250m.csv")
osm_250m = pd.read_csv("data/processed_esa_ee_osm/nuremberg_2021_osm_features_250m.csv")

# Drop only non-model metadata, NOT cell_id
spectral_2021 = spectral_2021.drop(columns=["system:index", ".geo"], errors="ignore")

updated_2021 = pd.merge(
    spectral_2021,
    osm_250m,
    on="cell_id",
    how="inner"
)

# Optional: fill unmatched OSM cells with 0
updated_2021["road_density"] = updated_2021["road_density"].fillna(0)
updated_2021["building_area_ratio"] = updated_2021["building_area_ratio"].fillna(0)

# Put cell_id first
cols = ["cell_id"] + [c for c in updated_2021.columns if c != "cell_id"]
updated_2021 = updated_2021[cols]

print(updated_2021.columns.tolist())
print(updated_2021.head())
print(updated_2021.shape)
print(updated_2021.isnull().sum())

updated_2021.to_csv(
    "data/processed_esa_ee_osm/nuremberg_2021_features_plus_osm_250m.csv",
    index=False
)