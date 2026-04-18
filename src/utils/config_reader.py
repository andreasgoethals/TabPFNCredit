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
    if params["row_min"] > params["row_max"]:
        raise ValueError(
            f"row_min ({params['row_min']}) cannot exceed row_max ({params['row_max']})"
        )
    if params["row_step"] <= 0:
        raise ValueError(f"row_step must be positive, got {params['row_step']}")


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
        extra_params=["row_max", "row_min", "row_step"],
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
            "pd": {k: v for k, v in data_config.get("dataset_pd", {}).items() if v},
            "lgd": {k: v for k, v in data_config.get("dataset_lgd", {}).items() if v},
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
    if spec.extra_section is not None:
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
