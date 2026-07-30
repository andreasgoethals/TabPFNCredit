"""The dataset count declared in CONFIG_DATA must be enforced, not assumed.

Selecting datasets by row count (``min_rows``) keeps proprietary slugs out of the
published config, but it has a failure mode: a dataset that is missing or
unreadable is not reported as *missing*, it is simply not *selected*. The
benchmark then shrinks silently.

That is not hypothetical. Two datasets' processed arrays were deleted while their
raw files were absent from the cluster; both dropped out of every experiment, and
``resubmit`` reported "nothing to do" immediately after 203 result files had been
removed. ``expect_datasets`` converts that into a hard error at config-load time.
"""

from __future__ import annotations

import pytest

from src.utils import config_reader


def _block(n_expected: int | None = None) -> dict:
    block = {"min_rows": 0}
    if n_expected is not None:
        block["expect_datasets"] = n_expected
    return block


def test_matching_count_passes(monkeypatch):
    monkeypatch.setattr(
        "src.data.dataset_inventory.datasets_with_min_rows",
        lambda task, min_rows: ["a", "b", "c"],
    )
    got = config_reader._resolve_dataset_block("pd", _block(3))
    assert set(got) == {"a", "b", "c"}


def test_short_count_raises_with_an_actionable_message(monkeypatch):
    """A dataset disappearing from disk must stop the run, not shrink it."""
    monkeypatch.setattr(
        "src.data.dataset_inventory.datasets_with_min_rows",
        lambda task, min_rows: ["a", "b"],          # one went missing
    )
    with pytest.raises(ValueError) as exc:
        config_reader._resolve_dataset_block("pd", _block(3))
    message = str(exc.value)
    assert "expected 3" in message and "found 2" in message
    assert "data/raw" in message, "the message must say where to look"


def test_extra_count_also_raises(monkeypatch):
    """More datasets than declared is equally worth stopping for."""
    monkeypatch.setattr(
        "src.data.dataset_inventory.datasets_with_min_rows",
        lambda task, min_rows: ["a", "b", "c", "d"],
    )
    with pytest.raises(ValueError):
        config_reader._resolve_dataset_block("pd", _block(3))


def test_absent_guard_keeps_old_behaviour(monkeypatch):
    """expect_datasets is optional -- Experiments 2/3 select a varying count."""
    monkeypatch.setattr(
        "src.data.dataset_inventory.datasets_with_min_rows",
        lambda task, min_rows: ["a"],
    )
    assert set(config_reader._resolve_dataset_block("pd", _block(None))) == {"a"}


def test_non_integer_guard_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "src.data.dataset_inventory.datasets_with_min_rows",
        lambda task, min_rows: ["a"],
    )
    with pytest.raises(ValueError, match="expect_datasets"):
        config_reader._resolve_dataset_block("pd", {"min_rows": 0,
                                                    "expect_datasets": "many"})


@pytest.mark.parametrize("experiment,task,count",
                         [("Experiment0", "pd", 14), ("Experiment0", "lgd", 7),
                          ("Experiment1", "pd", 14), ("Experiment1", "lgd", 7)])
def test_headline_experiments_declare_their_dataset_count(experiment, task, count):
    """The two full-coverage experiments must pin their counts."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (root / "scripts" / experiment / "config" / "CONFIG_DATA.yaml")
        .read_text(encoding="utf-8"))
    block = cfg[f"dataset_{task}"]
    assert block.get("expect_datasets") == count, (
        f"{experiment} dataset_{task} must declare expect_datasets: {count}")
