"""Tests for notebooks/CONFIG_NOTEBOOKS.yaml + its loader in experiment_plots.

Pins the MECHANISM of the notebook-level method filters, not the current
list contents (both lists are meant to be edited freely): the shipped YAML
parses with both sections, the loader helpers return string lists, the
champion lists are non-empty and never contain an excluded method, and the
control method hardcoded in the champion-stat notebooks stays included.
"""

from __future__ import annotations

from src.visualizations.experiment_plots import (
    _notebook_config_path,
    _notebook_method_filters,
    champion_methods,
    excluded_methods,
)


class TestNotebookConfig:

    def test_config_file_exists_and_parses(self):
        assert _notebook_config_path().exists()
        cfg = _notebook_method_filters()
        assert set(cfg) >= {"exclude", "champions"}

    def test_exclude_lists_are_string_lists(self):
        for task in ("pd", "lgd"):
            excl = excluded_methods(task)
            assert isinstance(excl, list)
            assert all(isinstance(m, str) for m in excl)

    def test_champion_lists(self):
        for task in ("pd", "lgd"):
            champs = champion_methods(task)
            # The champion-stat notebooks break on an empty inclusion list.
            assert champs, f"{task} champions must not be empty"
            # Experiment1.3/1.6 hardcode tabpfn_v3 as the Bonferroni-Dunn
            # control, so it must stay in the inclusion list.
            assert "tabpfn_v3" in champs
            # champion_methods applies the exclude list on top.
            assert not set(champs) & set(excluded_methods(task))

    def test_unknown_task_is_empty(self):
        assert excluded_methods("nope") == []
        assert champion_methods("nope") == []
