"""Real-only FlowGMM detector roles shared by named and CLIMB comparisons."""
import json

import numpy as np
from sklearn.model_selection import train_test_split


FLOW_PARAMETERS = {
    "flow_epochs": "epochs", "flow_layers": "layers", "flow_hidden": "hidden",
    "flow_batch_size": "batch_size", "flow_learning_rate": "learning_rate",
}


def flow_for_partition(dataset, split, partitions, role, seed, parameters, device="auto"):
    """Reserve calibration inside real inner fit; never fit on validation or test.

    The detector keeps these roles for outer replay too. The downstream classifier
    still receives all real inner-fit/outer-training rows, exactly as canonical U2.
    Only the inverse classifier-coordinate bridge changes at outer replay.
    """
    from samplers.flow_gmm import FlowGMMConfig, FlowGate, fit_flow_gmm

    if role not in ("inner", "outer"):
        raise ValueError("FlowGMM requires an inner/outer role")
    real = np.asarray(split["fit_indices"])
    fit, calibration = train_test_split(real, test_size=.2,
        stratify=np.asarray(dataset["y"])[real], random_state=int(seed) + 7919)
    validation = np.asarray(split["validation_indices"])
    identities = np.asarray(dataset["row_ids"])
    options = {target: parameters[source] for source, target in FLOW_PARAMETERS.items() if source in parameters}
    config = FlowGMMConfig(**options, device=device)
    key = (int(seed), tuple(fit.tolist()), json.dumps(options, sort_keys=True), device)
    cache = partitions.setdefault("flow_cores", {})
    if key not in cache:
        cache[key] = fit_flow_gmm(dataset["X"].iloc[fit], np.asarray(dataset["y"])[fit],
                                  identities[fit], seed, config)
    gate = FlowGate(cache[key], partitions[role],
                    calibration_row_ids=identities[calibration],
                    validation_row_ids=identities[validation])
    gate.role = role
    gate.calibration_class_counts = {int(label): int((np.asarray(dataset["y"])[calibration] == label).sum())
                                     for label in np.unique(dataset["y"])}
    return gate
