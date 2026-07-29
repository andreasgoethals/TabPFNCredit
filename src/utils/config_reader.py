# src/utils/config_reader.py
"""
Unified YAML configuration loader for TabPFNCredit experiments.

Each experiment under ``scripts/ExperimentN/config/`` ships three YAML files:

* ``CONFIG_DATA.yaml``       -- split settings, dataset toggles, paths.
* ``CONFIG_METHOD.yaml``     -- per-task method enable/disable.
* ``CONFIG_EXPERIMENT.yaml`` -- training / HPO parameters.

Experiment 2 (learning curve) and Experiment 3 (class imbalance) additionally
define their own extra-section schema inside ``CONFIG_EXPERIMENT.yaml``. A
single factory driven by an ``ExperimentSpec`` dataclass loads all of them, so
adding Experiment 4 is a two-line change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# Experiment specification
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ExperimentSpec:
    """Declarative spec for an experiment's extra config section(s).

    Attributes:
        name: Experiment identifier (directory name, e.g. ``"Experiment2"``).
        extra_section: Optional key under which extra parameters are stored in
            the final config dict (``"learning_curve"``, ``"imbalance"``, ...).
        extra_params: List of keys in ``CONFIG_EXPERIMENT.yaml`` that must
            exist under the extra section. All required; missing keys raise
            ``ValueError``.
        validator: Optional function called with the extra-section dict that
            raises ``ValueError`` on invalid parameter relationships.
        datasets_from_row_max: When True, the dataset selection is derived
            from ``<extra_section>.<task>.row_max`` (datasets with at least
            that many rows) instead of from ``CONFIG_DATA``'s dataset blocks.
            Used by Experiment 2 so the learning-curve ceiling (row_max) and
            the dataset inclusion threshold are a single knob that can't
            drift apart.
        filter_by_minority_max: When True, the PD dataset list is additionally
            filtered to datasets whose natural minority-class proportion
            EXCEEDS ``<extra_section>.minority_proportion_max``. Used by
            Experiment 3: you can only subsample the minority class down, so
            a dataset that's already less imbalanced than the top of the
            sweep can't take part.
    """

    name: str
    extra_section: Optional[str] = None
    extra_params: List[str] = field(default_factory=list)
    validator: Optional[Callable[[Dict[str, Any]], None]] = None
    datasets_from_row_max: bool = False
    filter_by_minority_max: bool = False


# --------------------------------------------------------------------------- #
# Per-experiment validators
# --------------------------------------------------------------------------- #

def _validate_learning_curve(params: Dict[str, Any]) -> None:
    """Validate the per-task learning_curve block.

    Shape::

        learning_curve:
          pd:  {row_max: ..., row_min: ..., row_step: ...}
          lgd: {row_max: ..., row_min: ..., row_step: ...}

    ``row_max`` does double duty: it's both the top of the training-size
    sweep AND the dataset-inclusion threshold (a dataset is used iff it has
    at least ``row_max`` rows -- see ``ExperimentSpec.datasets_from_row_max``).
    So there is exactly one knob per task.

    The legacy flat shape (``row_max`` / ``row_min`` / ``row_step`` at the
    top level) is still accepted for backwards compatibility; it gets
    duplicated into both ``pd`` and ``lgd`` so the per-task helpers can
    read a unified shape downstream.
    """
    # Backwards-compat: legacy flat keys
    if "row_max" in params and "pd" not in params:
        legacy = {
            "row_max": params["row_max"],
            "row_min": params["row_min"],
            "row_step": params["row_step"],
        }
        params["pd"] = dict(legacy)
        params["lgd"] = dict(legacy)
        for k in ("row_max", "row_min", "row_step", "min_dataset_size"):
            params.pop(k, None)

    for task in ("pd", "lgd"):
        if task not in params:
            raise ValueError(
                f"learning_curve.{task} block missing; "
                f"required keys: row_max, row_min, row_step"
            )
        block = params[task]
        for key in ("row_max", "row_min", "row_step"):
            if key not in block:
                raise ValueError(f"learning_curve.{task}.{key} is required")
        if block["row_min"] > block["row_max"]:
            raise ValueError(
                f"learning_curve.{task}: row_min ({block['row_min']}) "
                f"cannot exceed row_max ({block['row_max']})"
            )
        if block["row_step"] <= 0:
            raise ValueError(
                f"learning_curve.{task}: row_step must be positive, got {block['row_step']}"
            )


def _validate_imbalance(params: Dict[str, Any]) -> None:
    p_min = params["minority_proportion_min"]
    p_max = params["minority_proportion_max"]
    p_step = params["minority_proportion_step"]
    if p_min > p_max:
        raise ValueError(
            f"minority_proportion_min ({p_min}) cannot exceed "
            f"minority_proportion_max ({p_max})"
        )
    if p_step <= 0:
        raise ValueError(f"minority_proportion_step must be positive, got {p_step}")
    if p_min < 0 or p_max > 0.5:
        raise ValueError(
            f"minority_proportion must be in [0, 0.5]; got min={p_min}, max={p_max}"
        )


# --------------------------------------------------------------------------- #
# Registry of known experiments
# --------------------------------------------------------------------------- #

_EXPERIMENT_REGISTRY: Dict[str, ExperimentSpec] = {
    "Experiment0": ExperimentSpec(name="Experiment0"),
    "Experiment1": ExperimentSpec(name="Experiment1"),
    "Experiment2": ExperimentSpec(
        name="Experiment2",
        extra_section="learning_curve",
        # Per-task block now lives at learning_curve.{pd,lgd}; the validator
        # also accepts legacy flat keys for backwards-compat.
        extra_params=[],
        validator=_validate_learning_curve,
        # Datasets are chosen by learning_curve.<task>.row_max (a dataset is
        # included iff it has >= row_max rows). CONFIG_DATA's dataset blocks
        # are intentionally empty for Experiment 2.
        datasets_from_row_max=True,
    ),
    "Experiment3": ExperimentSpec(
        name="Experiment3",
        extra_section="imbalance",
        extra_params=[
            "minority_proportion_max",
            "minority_proportion_min",
            "minority_proportion_step",
        ],
        validator=_validate_imbalance,
        # Beyond the min_rows filter in CONFIG_DATA, keep only PD datasets
        # whose natural minority proportion exceeds minority_proportion_max
        # (otherwise the sweep can't start at the top).
        filter_by_minority_max=True,
    ),
}


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #

def get_config_dir(experiment_name: str) -> Path:
    """Return the config directory for an experiment, or raise FileNotFoundError."""
    project_root = Path(__file__).resolve().parent.parent.parent
    config_dir = project_root / "scripts" / experiment_name / "config"
    if not config_dir.exists():
        raise FileNotFoundError(
            f"Config directory not found for experiment '{experiment_name}'.\n"
            f"Expected location: {config_dir}"
        )
    return config_dir


# --------------------------------------------------------------------------- #
# Core loader
# --------------------------------------------------------------------------- #

def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_dataset_block(task: str, block: Any) -> Dict[str, bool]:
    """Turn a ``dataset_pd`` / ``dataset_lgd`` YAML block into ``{name: True}``.

    Two shapes are accepted:

    1. **Per-dataset toggle** (Experiments 0 and 1)::

           dataset_pd:
             0001.gmsc: true
             0002.taiwan_creditcard: false

    2. **Minimum-row filter** (Experiments 2 and 3)::

           dataset_pd:
             min_rows: 30000

       The selection is "every dataset whose row count is >= min_rows".
       Row counts come from ``src.data.dataset_inventory.row_counts``.

    Returns a ``{dataset_name: True}`` dict for the selected datasets.
    """
    if not block:
        return {}
    if isinstance(block, dict) and "min_rows" in block:
        from src.data.dataset_inventory import datasets_with_min_rows
        try:
            min_rows = int(block["min_rows"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"dataset_{task}.min_rows must be an integer, got {block['min_rows']!r}"
            ) from exc
        return {name: True for name in datasets_with_min_rows(task, min_rows)}
    # Per-dataset toggle dict.
    return {k: True for k, v in block.items() if v}


def _base_config(
    data_config: Dict[str, Any],
    method_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Shared skeleton for every experiment."""
    return {
        "split": data_config["split"],
        "paths": data_config["paths"],
        "datasets": {
            "pd": _resolve_dataset_block("pd", data_config.get("dataset_pd")),
            "lgd": _resolve_dataset_block("lgd", data_config.get("dataset_lgd")),
        },
        "methods": {
            "pd": {k: v for k, v in method_config["methods"]["pd"].items() if v},
            "lgd": {k: v for k, v in method_config["methods"]["lgd"].items() if v},
        },
        "training": {
            "max_epochs": experiment_config["max_epochs"],
            "batch_size": experiment_config["batch_size"],
            "early_stopping": experiment_config["early_stopping"],
            "early_stopping_patience": experiment_config["early_stopping_patience"],
            # Optional: TALENT's packaged defaults would otherwise refit every
            # model 15x per fold and report only the last fit. Absent from a
            # config file => 1 (fit once), which is what every downstream
            # consumer in this repo assumes.
            "seed_num": int(experiment_config.get("seed_num", 1)),
        },
        "tuning": {
            "n_trials": experiment_config["n_trials"],
        },
    }


def load_config(experiment_name: str) -> Dict[str, Any]:
    """
    Load and validate the config for a given experiment.

    Dispatches to a single generic implementation driven by
    :data:`_EXPERIMENT_REGISTRY`. Each experiment can declare additional
    required parameters via an :class:`ExperimentSpec`.
    """
    if not experiment_name:
        raise ValueError(
            "experiment_name must be provided (e.g. 'Experiment0', 'Experiment1')"
        )
    if experiment_name not in _EXPERIMENT_REGISTRY:
        raise NotImplementedError(
            f"Unknown experiment '{experiment_name}'. "
            f"Known experiments: {sorted(_EXPERIMENT_REGISTRY)}"
        )

    spec = _EXPERIMENT_REGISTRY[experiment_name]
    config_dir = get_config_dir(experiment_name)

    try:
        data_config = _read_yaml(config_dir / "CONFIG_DATA.yaml")
        method_config = _read_yaml(config_dir / "CONFIG_METHOD.yaml")
        experiment_config = _read_yaml(config_dir / "CONFIG_EXPERIMENT.yaml")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing required YAML config file in {config_dir}.\nError: {exc}"
        ) from exc

    config = _base_config(data_config, method_config, experiment_config)

    # Append extra section if this experiment declares one.
    #
    # Two shapes are supported per experiment:
    #   1. A nested block keyed by ``spec.extra_section`` (current Experiment 2):
    #        learning_curve:
    #          pd:  {row_max: ..., row_min: ..., row_step: ..., ...}
    #          lgd: {row_max: ..., row_min: ..., row_step: ..., ...}
    #   2. Legacy flat keys (Experiment 3 still uses this):
    #        minority_proportion_min: ...
    #        minority_proportion_max: ...
    #        minority_proportion_step: ...
    if spec.extra_section is not None:
        if spec.extra_section in experiment_config:
            # Nested shape -- copy the block verbatim
            extras = dict(experiment_config[spec.extra_section])
        else:
            # Legacy flat shape -- collect ``extra_params`` keys
            for param in spec.extra_params:
                if param not in experiment_config:
                    raise ValueError(
                        f"Missing required parameter '{param}' in CONFIG_EXPERIMENT.yaml "
                        f"for {spec.name}"
                    )
            extras = {p: experiment_config[p] for p in spec.extra_params}
        if spec.validator is not None:
            spec.validator(extras)
        config[spec.extra_section] = extras

    # Experiment 2: derive the dataset selection from the per-task
    # learning-curve ceiling (row_max). A dataset is included iff it has at
    # least row_max rows -- otherwise the curve could never reach the top of
    # the sweep for that dataset. This keeps the sweep ceiling and the
    # dataset filter as ONE knob (CONFIG_DATA's dataset blocks are empty).
    if spec.datasets_from_row_max and spec.extra_section:
        from src.data.dataset_inventory import datasets_with_min_rows
        sweep = config.get(spec.extra_section, {})
        for task in ("pd", "lgd"):
            block = sweep.get(task) or {}
            row_max = block.get("row_max")
            if row_max:
                config["datasets"][task] = {
                    name: True for name in datasets_with_min_rows(task, int(row_max))
                }
            else:
                config["datasets"][task] = {}

    # Experiment 3: on top of the min_rows filter, keep only PD datasets
    # whose natural minority proportion EXCEEDS minority_proportion_max --
    # the sweep subsamples the minority class DOWN from that ceiling, so a
    # dataset that's already less imbalanced can't participate.
    if spec.filter_by_minority_max and spec.extra_section:
        from src.data.dataset_inventory import datasets_with_min_minority
        imb = config.get(spec.extra_section, {})
        p_max = imb.get("minority_proportion_max")
        if p_max is not None:
            kept = datasets_with_min_minority(
                "pd", list(config["datasets"]["pd"]), float(p_max)
            )
            config["datasets"]["pd"] = {name: True for name in kept}

    return config


if __name__ == "__main__":
    for exp in _EXPERIMENT_REGISTRY:
        try:
            cfg = load_config(exp)
            print(f"[OK] {exp} config loaded successfully.")
            spec = _EXPERIMENT_REGISTRY[exp]
            if spec.extra_section:
                print(f"  {spec.extra_section}: {cfg[spec.extra_section]}")
        except FileNotFoundError:
            print(f"[--] {exp} config not found (directory doesn't exist yet)")
        except Exception as exc:
            print(f"[ERR] {exp} error: {exc}")
