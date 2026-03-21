# import pandas as pd

# # -----------------------------
# # 1. LOAD FILES
# # -----------------------------
# df_2019 = pd.read_csv("data/processed_esa_ee/nuremberg_2019_features_250m.csv")
# df_2020 = pd.read_csv("data/processed_esa_ee/nuremberg_2020_composition_250m.csv")

# # -----------------------------
# # 2. KEEP ONLY 2019 FEATURES
# # -----------------------------
# features_2019 = df_2019[
#     ["cell_id", "B2", "B3", "B4", "B8", "B11", "NDVI", "NDWI", "NDBI"]
# ].copy()

# # Rename 2019 feature columns
# features_2019 = features_2019.rename(columns={
#     "B2": "B2_2019",
#     "B3": "B3_2019",
#     "B4": "B4_2019",
#     "B8": "B8_2019",
#     "B11": "B11_2019",
#     "NDVI": "NDVI_2019",
#     "NDWI": "NDWI_2019",
#     "NDBI": "NDBI_2019"
# })

# # -----------------------------
# # 3. KEEP ONLY 2020 LABELS
# # -----------------------------
# labels_2020 = df_2020[
#     ["cell_id", "vegetation", "built_up", "water", "other"]
# ].copy()

# # Rename 2020 label columns
# labels_2020 = labels_2020.rename(columns={
#     "vegetation": "vegetation_2020",
#     "built_up": "built_up_2020",
#     "water": "water_2020",
#     "other": "other_2020"
# })

# # -----------------------------
# # 4. JOIN ON cell_id
# # -----------------------------
# df_model = pd.merge(
#     features_2019,
#     labels_2020,
#     on="cell_id",
#     how="inner"
# )

# # -----------------------------
# # 5. CHECK RESULT
# # -----------------------------
# print("Shape:", df_model.shape)
# print(df_model.head())
# print("\nNull values:")
# print(df_model.isnull().sum())

# # -----------------------------
# # 6. SAVE
# # -----------------------------
# df_model.to_csv("data/train_esa_ee/nuremberg_2019_to_2020_model.csv", index=False)

import pandas as pd

# -----------------------------
# 1. LOAD FILES
# -----------------------------
df_2020 = pd.read_csv("data/processed_esa_ee/nuremberg_2020_composition_250m.csv")
df_2021 = pd.read_csv("data/processed_esa_ee/nuremberg_2021_composition_250m.csv")

# -----------------------------
# 2. KEEP ONLY 2020 FEATURES
# -----------------------------
features_2020 = df_2020[
    ["cell_id", "B2", "B3", "B4", "B8", "B11", "NDVI", "NDWI", "NDBI"]
].copy()

features_2020 = features_2020.rename(columns={
    "B2": "B2_2020",
    "B3": "B3_2020",
    "B4": "B4_2020",
    "B8": "B8_2020",
    "B11": "B11_2020",
    "NDVI": "NDVI_2020",
    "NDWI": "NDWI_2020",
    "NDBI": "NDBI_2020"
})

# -----------------------------
# 3. KEEP ONLY 2021 LABELS
# -----------------------------
labels_2021 = df_2021[
    ["cell_id", "vegetation", "built_up", "water", "other"]
].copy()

labels_2021 = labels_2021.rename(columns={
    "vegetation": "vegetation_2021",
    "built_up": "built_up_2021",
    "water": "water_2021",
    "other": "other_2021"
})

# -----------------------------
# 4. JOIN ON cell_id
# -----------------------------
df_model = pd.merge(
    features_2020,
    labels_2021,
    on="cell_id",
    how="inner"
)

# -----------------------------
# 5. CHECK RESULT
# -----------------------------
print("Shape:", df_model.shape)
print(df_model.head())
print("\nNull values:")
print(df_model.isnull().sum())

# -----------------------------
# 6. SAVE
# -----------------------------
df_model.to_csv("data/train_esa_ee/nuremberg_2020_to_2021_model.csv", index=False)