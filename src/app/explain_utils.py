def helpful_explanation() -> str:
    return (
        "High predicted built-up change usually appears where the model sees lower vegetation signals "
        "and stronger built-up spectral patterns. This is useful for screening broad urban change, not "
        "for proving why a specific cell changed."
    )


def misleading_explanation() -> str:
    return (
        "A high feature importance score does not prove causation. For example, if a spectral feature is "
        "important, that does not mean it caused development; it only helped the model separate patterns "
        "in the training data."
    )
