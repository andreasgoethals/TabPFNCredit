"""Regression guard: exactly ONE model fit per fold.

TALENT's ``build_args`` layers its packaged ``deep_configs.json`` /
``classical_configs.json`` on top of its own baked-in defaults, and those
packaged files carry ``seed_num: 15``. Any method built without an explicit
``seed_num`` is therefore refit **15x per fold** -- while
``RunResult.predictions`` / ``.predict_proba`` / ``.metrics`` (the only fields
this repo reads) carry just the LAST repeat. On the cluster that cost 15x the
GPU hours for a number that was thrown away, and it is what pushed TabFM's
largest dataset past a 25 h wall-clock limit.

These tests pin the behaviour at both ends of the chain: the YAML configs and
``_build_talent_args``. If the ``overrides.setdefault("seed_num", ...)`` line
in ``_build_talent_args`` ever disappears, ``test_talent_default_is_still_15``
documents what comes back and the other tests fail loudly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.methods.method_runner import _build_talent_args
from src.utils.config_reader import load_config

EXPERIMENTS = ["Experiment0", "Experiment1", "Experiment2", "Experiment3"]

#: One per architecture branch of ``build_args`` (deep vs classical) plus the
#: foundation models this actually bit.
SAMPLE_METHODS = ["tabfm", "tabpfn_v3", "mlp", "catboost", "LogReg"]


def _args(method: str, **kwargs):
    defaults = dict(
        method=method, seed=42, is_regression=False,
        save_path=Path(tempfile.mkdtemp()), tune=False, n_trials=1,
        max_epoch=50, batch_size=255, early_stopping=True,
        early_stopping_patience=10, evaluate_option="best-val",
        user_overrides={},
    )
    defaults.update(kwargs)
    return _build_talent_args(**defaults)


class TestConfigFiles:

    @pytest.mark.parametrize("experiment", EXPERIMENTS)
    def test_every_experiment_pins_seed_num(self, experiment):
        config = load_config(experiment)
        assert config["training"]["seed_num"] == 1, (
            f"{experiment} must fit each fold once; anything else multiplies "
            f"compute without changing what is reported"
        )

    def test_missing_key_falls_back_to_one(self):
        """An older config file without the key must not inherit TALENT's 15."""
        from src.utils.config_reader import _base_config

        base = _base_config(
            data_config={"split": {}, "paths": {}},
            method_config={"methods": {"pd": {}, "lgd": {}}},
            experiment_config={
                "max_epochs": 1, "batch_size": 1, "early_stopping": False,
                "early_stopping_patience": 1, "n_trials": 1,
            },
        )
        assert base["training"]["seed_num"] == 1


class TestArgsBuilder:

    @pytest.mark.parametrize("method", SAMPLE_METHODS)
    def test_default_is_a_single_fit(self, method):
        assert _args(method).seed_num == 1

    @pytest.mark.parametrize("method", SAMPLE_METHODS)
    def test_explicit_value_is_honoured(self, method):
        assert _args(method, seed_num=15).seed_num == 15

    @pytest.mark.parametrize("bad", [0, -3])
    def test_nonsense_values_are_clamped_to_one(self, bad):
        """seed_num=0 would make TALENT's ``for s in range(seed_num)`` loop run
        zero times and then index ``per_seed[0]`` -- an IndexError deep in a
        cluster job. Clamp instead."""
        assert _args("catboost", seed_num=bad).seed_num == 1

    def test_talent_default_is_still_15(self):
        """Documents WHY the explicit pin exists. If TALENT ever ships
        ``seed_num: 1`` this test fails and the pin can be reconsidered."""
        import TALENT

        raw = TALENT.build_args("catboost", save_path=tempfile.mkdtemp())
        assert raw.seed_num == 15, (
            "TALENT's packaged default changed -- revisit the seed_num pin in "
            "_build_talent_args and the CONFIG_EXPERIMENT.yaml comments"
        )
