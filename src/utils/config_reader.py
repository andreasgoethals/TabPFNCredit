# src/utils/config_reader.py
"""
Unified YAML configuration loader for TabPFNCredit experiments.

Each experiment under ``scripts/ExperimentN/config/`` ships three YAML files:

* ``CONFIG_DATA.yaml``       -- split settings, dataset toggles, paths.
* ``CONFIG_METHOD.yaml``     -- per-task method enable/disable.
* ``CONFIG_EXPERIMENT.yaml`` -- training / HPO parameters.

Experiment 2 (learning curve) and Experiment 3 (class imbalance) additionally
define their own extra-section schema inside ``CONFIG_EXPERIMENT.yaml``.
Previously each experiment had a near-duplicate ``_load_exp*_config`` helper;
this module collapses them into a single factory driven by an ``ExperimentSpec``
dataclass so that adding Experiment 4 is a two-line change.
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
    """

    name: str
    extra_section: Optional[str] = None
    extra_params: List[str] = field(default_factory=list)
    validator: Optional[Callable[[Dict[str, Any]], None]] = None


# --------------------------------------------------------------------------- #
# Per-experiment validators
# --------------------------------------------------------------------------- #

def _validate_learning_curve(params: Dict[str, Any]) -> None:
    """Validate the per-task learning_curve block.

    Shape::

        learning_curve:
          pd:  {row_max: ..., row_min: ..., row_step: ..., min_dataset_size: ...}
          lgd: {row_max: ..., row_min: ..., row_step: ..., min_dataset_size: ...}

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
            "min_dataset_size": params.get("min_dataset_size", 0),
        }
        params["pd"] = dict(legacy)
        params["lgd"] = dict(legacy)
        for k in ("row_max", "row_min", "row_step", "min_dataset_size"):
            params.pop(k, None)

    for task in ("pd", "lgd"):
        if task not in params:
            raise ValueError(
                f"learning_curve.{task} block missing; "
                f"required keys: row_max, row_min, row_step, min_dataset_size"
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
        block.setdefault("min_dataset_size", 0)


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
