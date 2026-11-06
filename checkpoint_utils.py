"""Validate experiment architecture metadata before strict checkpoint loading."""

FEATURE_FLAGS = (
    "use_category_supervision", "use_confidence_fusion", "use_category_pan",
    "use_dsm_geometry", "use_dsm_geometry_branch",
)


def _canonical_options(options):
    result = dict(options)
    # Old category-guidance checkpoints always applied the N2 gate.
    result.setdefault("use_n2_category_gate", True)
    # Existing R1/R2 checkpoints predate the new geometry branch.
    result.setdefault("use_dsm_geometry_branch", False)
    windows = tuple(result.pop("dsm_geometry_windows", (9, 25)))
    if result["use_dsm_geometry_branch"]:
        result["dsm_geometry_windows"] = windows
    return result


def checkpoint_state_dict(checkpoint, model_options, num_classes):
    """Return weights only if the saved model matches the requested architecture.

    This does not warm-start a new architecture from an older checkpoint.
    The caller must still use model.load_state_dict(..., strict=True).
    """
    if "model_state_dict" in checkpoint:
        saved = _canonical_options(checkpoint["model_options"])
        current = _canonical_options(model_options)
        if saved != current or checkpoint["num_classes"] != num_classes:
            raise ValueError("MODEL_OPTIONS / N_CLASSES must match the saved checkpoint, "
                             "including the geometry branch and its active window sizes.")
        return checkpoint["model_state_dict"]
    if any(model_options.get(key, False) for key in FEATURE_FLAGS):
        raise ValueError("Legacy checkpoints require all added feature modules disabled. "
                         "Train the new modules before testing them.")
    return checkpoint
