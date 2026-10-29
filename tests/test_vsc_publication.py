"""A successful process alone must not authorize scratch cleanup."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.utils import cli


@pytest.mark.parametrize("complete", [True, False])
def test_receipt_requires_durable_complete_results(tmp_path, monkeypatch, complete):
    monkeypatch.setattr(cli, "_PROJECT_ROOT", tmp_path / "repo")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "work/tmp"))
    base = tmp_path / "project/results"
    result = base / "experiment0/pd/example/m.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"folds": {"0": {"metrics": {"AUC": .8}}}}))
    np.savez(result.with_suffix(".npz"), y_true=np.array([0, 1]))
    plan = tmp_path / "plan.json"
    plan.write_text("immutable plan")
    slot = [{"task": "pd", "dataset": "example", "method": "m", "name": "m"}]
    config = {"split": {"cv_splits": 1 if complete else 2}}
    receipt = tmp_path / "repo/logs/manifests/publications/123_0.json"
    if complete:
        cli._verify_publication("Experiment0", slot, config, base, plan, 0)
        assert json.loads(receipt.read_text())["status"] == "published"
    else:
        with pytest.raises(RuntimeError, match="incomplete"):
            cli._verify_publication("Experiment0", slot, config, base, plan, 0)
        assert not receipt.exists()


def test_corrupt_predictions_are_not_published(tmp_path, monkeypatch):
    import zipfile

    monkeypatch.setattr(cli, "_PROJECT_ROOT", tmp_path / "repo")
    result = tmp_path / "experiment0/pd/example/m.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"folds":{"0":{}}}')
    result.with_suffix(".npz").write_bytes(b"corrupt")
    with pytest.raises(zipfile.BadZipFile):
        cli._verify_publication("Experiment0", [{"task": "pd", "dataset": "example", "method": "m"}],
                                {"split": {"cv_splits": 1}}, tmp_path, Path("unused"), 0)
    assert not (tmp_path / "repo/logs/manifests/publications").exists()
