"""Planning occurs in the image; scheduler RPCs and receipts stay on the host."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("preview", [True, False])
def test_host_submission_boundary(tmp_path, monkeypatch, preview):
    spec = importlib.util.spec_from_file_location("vsc_host", Path(__file__).parents[1] / "scripts/vsc.py")
    host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    data, project, scratch = (tmp_path / name for name in ("data", "project", "scratch"))
    repo = data / "repo"
    (repo / "scripts").mkdir(parents=True)
    (project / "runtimes").mkdir(parents=True)
    scratch.mkdir()
    image = project / "runtimes/tabpfncredit-runtime-20261029-cli-fixed.sif"
    image.touch()
    job = repo / "scripts/job.slurm"
    job.touch()
    monkeypatch.setattr(host, "__file__", str(repo / "scripts/vsc.py"))
    for key, value in {"VSC_DATA": data, "TABPFN_STAGING_ROOT": project,
                       "VSC_SCRATCH": scratch, "VSC_SCRATCH_LUSTRE1": scratch,
                       "VSC_PROJECT_LUSTRE1": project.parent}.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.delenv("TABPFN_CONTAINER_IMAGE", raising=False)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "apptainer":
            assert command[-1] == "--no-submit"
            assert "--submit" not in command
            manifest = Path(kwargs["env"]["TABPFN_HOST_JOB_MANIFEST"])
            manifest.write_text(json.dumps([{"path": str(job), "cluster": "wice",
                                             "dependency": "afterok:12"}]))
            return subprocess.CompletedProcess(command, 0)
        assert command == ["sbatch", "--parsable", "--dependency=afterok:12", str(job)]
        return subprocess.CompletedProcess(command, 0, "123;wice\n", "")

    monkeypatch.setattr(host.subprocess, "run", run)
    args = ["experiment", "Experiment0", "--no-submit" if preview else "--submit"]
    assert host.main(args) == 0
    assert len(calls) == (1 if preview else 2)
    assert not (scratch / "tabpfncredit").exists()
    receipts = list((repo / "logs/manifests/submissions").glob("*.receipt.json"))
    assert len(receipts) == (0 if preview else 1)
