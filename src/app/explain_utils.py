from __future__ import annotations


def helpful_explanation(layer_mode: str, target_class: str, pipeline_key: str = "ee") -> str:
    class_label = target_class.replace("_", " ")
    source_phrase = "Sentinel-2 tabular features"
    if pipeline_key == "ee_osm":
        source_phrase = (
            "Sentinel-2 tabular features plus year-specific OSM auxiliary context such as road density "
            "and building area ratio"
        )

    if layer_mode == "change":
        return (
            f"This map shows model-estimated change in {class_label} relative to the earlier time step for each "
            f"250 m grid cell using {source_phrase}. Use it to compare broad spatial patterns across models and "
            "evaluation splits, not to explain why one individual cell changed."
        )

    return (
        f"This map shows predicted {class_label} proportion at T2 for each 250 m grid cell using {source_phrase}. "
        "It is useful for screening city-scale land-cover patterns and comparing model behavior, but it is not a "
        "parcel-level decision tool."
    )


def misleading_explanation(pipeline_key: str = "ee") -> str:
    if pipeline_key == "ee_osm":
        return (
            "A stronger in-period score after adding OSM features does not prove that the model will generalize "
            "forward in time. Auxiliary variables such as road density or building area ratio can improve fit on "
            "2019 to 2020 and still fail on the forward 2020 to 2021 evaluation. Feature importance or coefficient "
            "magnitude also does not prove causation."
        )
    return (
        "A strong score on the in-year test split does not prove that the model will generalize forward in time. "
        "Nonlinear models can fit 2019 to 2020 well and still fail on the forward 2020 to 2021 evaluation. "
        "Feature importance or weight magnitude also does not prove causation."
    )


def limitations_text(pipeline_key: str = "ee") -> str:
    if pipeline_key == "ee_osm":
        return (
            "Predictions are derived from tabular satellite features aggregated to 250 m cells plus auxiliary OSM "
            "context features. They still inherit label noise, mixed land-cover cells, incomplete contextual data, "
            "and temporal dataset limitations. Treat uncertainty and forward evaluation results as part of the "
            "interpretation, especially when model rankings change across splits."
        )
    return (
        "Predictions are derived from tabular satellite features aggregated to 250 m cells and inherit label noise, "
        "mixed land-cover cells, and temporal dataset limitations. Treat uncertainty and forward evaluation results "
        "as part of the interpretation, especially when model rankings change across splits."
    )
