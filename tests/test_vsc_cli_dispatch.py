"""Submission previews must never execute a benchmark on a login node."""

import pytest
from typer.testing import CliRunner

from src.utils import cli


@pytest.mark.parametrize("on_vsc", [False, True])
@pytest.mark.parametrize("flag", [None, "--submit", "--no-submit"])
def test_execution_and_preview_dispatch(monkeypatch, on_vsc, flag):
    calls = []
    cells = [{"task": "pd", "dataset": "example", "method": "xgboost"}]
    config = {"sentinel": "unchanged scientific configuration"}
    monkeypatch.setattr(cli, "configure_quiet_runtime", lambda: None)
    monkeypatch.setattr(cli, "load_config", lambda name: config)
    monkeypatch.setattr(cli, "_build_task_list", lambda *a, **kw: cells)
    monkeypatch.setattr(cli, "_on_vsc", lambda: on_vsc)

    def preprocess(received):
        assert received is cells
        calls.append("preprocess")
        return set()

    def plan(name, received, received_config, *, submit, after_job_id):
        assert received is cells and received_config is config
        assert name == "Experiment0" and after_job_id is None
        calls.append(("slurm", submit))

    def local(name, received, received_config):
        assert received is cells and received_config is config
        calls.append("local")

    monkeypatch.setattr(cli, "_preprocess_if_needed", preprocess)
    monkeypatch.setattr(cli, "_run_experiment_vsc", plan)
    monkeypatch.setattr(cli, "_run_cells_locally", local)
    monkeypatch.setattr(cli, "_summarize_now", lambda name: calls.append("summarize"))
    args = ["experiment", "Experiment0"] + ([flag] if flag else [])
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    if flag == "--no-submit":
        assert calls == [("slurm", False)]
    elif flag == "--submit" or on_vsc:
        assert calls == ["preprocess", ("slurm", True)]
    else:
        assert calls == ["preprocess", "local", "summarize"]
