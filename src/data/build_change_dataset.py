import pandas as pd


def build_change_dataset(dataset_t1: pd.DataFrame, dataset_t2: pd.DataFrame) -> pd.DataFrame:
    """Join two yearly datasets and derive change targets."""
    merged = dataset_t1.merge(
        dataset_t2,
        on="cell_id",
        suffixes=("_t1", "_t2"),
        how="inner",
    )
    merged["delta_built_up"] = merged["built_up_prop_t2"] - merged["built_up_prop_t1"]
    merged["delta_vegetation"] = merged["vegetation_prop_t2"] - merged["vegetation_prop_t1"]
    merged["delta_water"] = merged["water_prop_t2"] - merged["water_prop_t1"]
    merged["delta_other"] = merged["other_prop_t2"] - merged["other_prop_t1"]
    merged["change_binary"] = (
        (merged["delta_built_up"].abs() >= 0.1) | (merged["delta_vegetation"].abs() >= 0.1)
    ).astype(int)
    return merged
