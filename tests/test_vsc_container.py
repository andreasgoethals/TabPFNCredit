"""Framework checks use temporary storage and a fake container; no Slurm work."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from src.utils import slurm_generator as sg


@pytest.fixture(autouse=True)
def clean_profile(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(("TABPFN_", "VSC_")):
            monkeypatch.delenv(key)


def _bash_path(path: Path) -> str:
    text = path.as_posix()
    return "/" + text[0].lower() + text[2:] if path.drive else text


@pytest.fixture
def deployment(monkeypatch, tmp_path):
    data = tmp_path / "data"
    repo = data / "checkout"
    project = tmp_path / "project's space" / "TabPFNCredit"
    scratch = tmp_path / "scratch"
    for path in (repo / ".git", project / "runtimes", scratch):
        path.mkdir(parents=True)
    image = project / "runtimes" / "paper.sif"
    image.write_text("fake image", encoding="utf-8")
    monkeypatch.setattr(sg, "_REPO_ROOT", _bash_path(repo))
    for key, value in {
        "TABPFN_VSC_PROFILE": "container",
        "TABPFN_PROJECT_ROOT": _bash_path(project.parent),
        "TABPFN_CONTAINER_IMAGE": _bash_path(image),
        "VSC_DATA": _bash_path(data),
        "VSC_SCRATCH": _bash_path(scratch),
        "VSC_SCRATCH_LUSTRE1": _bash_path(scratch),
        "VSC_PROJECT_LUSTRE1": _bash_path(project.parent),
    }.items():
        monkeypatch.setenv(key, value)
    return repo, project, scratch


def _items(method="xgboost"):
    return [{"dataset": "0001.example", "task": "pd", "method": method,
             "name": f"{method}__row{n}", "tune": False, "row_limit": n,
             "sampling": None, "copy_from": None, "est_seconds": 30}
            for n in (100, 200)]


def test_legacy_is_default_and_new_profile_requires_explicit_settings(monkeypatch):
    assert sg._container_settings() is None
    monkeypatch.setenv("TABPFN_VSC_PROFILE", "container")
    with pytest.raises(ValueError, match="absolute TABPFN_STAGING_ROOT"):
        sg._container_settings()


def test_snapshot_preserves_legacy_full_directory_precedence(deployment, monkeypatch):
    repo, project, _ = deployment
    monkeypatch.setenv("TABPFN_STAGING_ROOT", _bash_path(project))
    monkeypatch.setenv("TABPFN_PROJECT_ROOT", "/ignored")
    settings = sg._container_settings()
    assert settings["TABPFN_STAGING_ROOT"] == _bash_path(project)
    monkeypatch.setenv("TABPFN_CONTAINER_IMAGE", "/changed.sif")
    assert settings["TABPFN_CONTAINER_IMAGE"].endswith("/paper.sif")


def test_empty_legacy_root_is_not_ignored_for_shared_alias(deployment, monkeypatch):
    monkeypatch.setenv("TABPFN_STAGING_ROOT", "  ")
    with pytest.raises(ValueError, match="absolute TABPFN_STAGING_ROOT"):
        sg._container_settings()


def test_legacy_generation_refuses_unpropagated_shared_root(deployment, monkeypatch):
    monkeypatch.delenv("TABPFN_VSC_PROFILE")
    with pytest.raises(ValueError, match="legacy jobs must use TABPFN_STAGING_ROOT"):
        sg._container_settings()


@pytest.mark.parametrize("experiment", ["Experiment0", "Experiment1", "Experiment2", "Experiment3"])
def test_container_changes_runtime_not_point_plans(deployment, monkeypatch, tmp_path, experiment):
    jobs = sg.generate_scripts_for_experiment(
        experiment=experiment, work_items=_items(), out_dir=tmp_path / "new")
    container_script = jobs[0].path.read_text()
    container_plan = json.loads(next((tmp_path / "new").glob("*_plan.json")).read_text())
    monkeypatch.delenv("TABPFN_VSC_PROFILE")
    monkeypatch.delenv("TABPFN_PROJECT_ROOT")
    old = sg.generate_scripts_for_experiment(
        experiment=experiment, work_items=_items(), out_dir=tmp_path / "old")
    legacy_plan = json.loads(next((tmp_path / "old").glob("*_plan.json")).read_text())
    assert container_plan == legacy_plan
    assert jobs[0].partition_key == old[0].partition_key
    assert jobs[0].n_array_slots == old[0].n_array_slots
    assert "#SBATCH --cpus-per-task=" in container_script
    assert "#SBATCH --mem=" not in container_script
    assert "#SBATCH --mem=" in old[0].path.read_text()
    assert "tabpfncreditvenv/bin/activate" in old[0].path.read_text()


def test_container_rejects_retired_routes_without_remapping(deployment, monkeypatch, tmp_path):
    monkeypatch.setattr(sg, "partition_for_method", lambda *a, **kw: "cpu_genius")
    with pytest.raises(ValueError, match="retired/unvalidated routes: cpu_genius"):
        sg.generate_scripts_for_experiment(
            experiment="Experiment1", work_items=_items(), out_dir=tmp_path / "generated")
    assert not list((tmp_path / "generated").glob("*.slurm"))


def _bash():
    executable = shutil.which("bash")
    if not executable and os.name == "nt":
        git = shutil.which("git")
        if git:
            candidate = Path(git).parents[1] / "bin" / "bash.exe"
            if candidate.exists():
                executable = str(candidate)
    if not executable:
        pytest.skip("Bash is needed for the mocked launcher check")
    return executable


def _run_fake_container(tmp_path, deployment, *, expected_status=23, settings=None, method="xgboost"):
    repo, project, scratch = deployment
    capture = tmp_path / "captured.txt"
    script = tmp_path / "job.sh"
    # Git Bash's readlink can use its /tmp alias for native Windows links and
    # /c/ for regular paths. Linux uses one form; normalize only this test host.
    normalize_links = (
        'readlink() { local value drive; value="$(command readlink "$@")" || return; '
        'value="$(cygpath -m "$value")"; drive="${value:0:1}"; '
        'printf "/%s%s\\n" "${drive,,}" "${value:2}"; }\n'
        if os.name == "nt" else ""
    )
    script.write_text(
        normalize_links + "git() { printf '%s\\n' " + shlex.quote(_bash_path(repo / ".git")) + "; }\n"
        "apptainer() { printf '%s\\n' \"$@\" > " + shlex.quote(_bash_path(capture)) + "; "
        'printf "STAGING=%s\\nCACHE=%s\\nTMP=%s\\nJOB=%s\\nTHREADS=%s\\nIMAGE=%s\\n" "$APPTAINERENV_TABPFN_STAGING_ROOT" '
        '"$APPTAINERENV_TABPFN_CACHE_ROOT" "$APPTAINERENV_TMPDIR" '
        '"$APPTAINERENV_SLURM_ARRAY_JOB_ID" "$APPTAINERENV_SLURM_CPUS_PER_TASK" "$APPTAINERENV_TABPFN_CONTAINER_IMAGE" >> '
        + shlex.quote(_bash_path(capture)) + f"; return {expected_status}; }}\n"
        + sg._container_prologue(settings or sg._container_settings(), "wice", 0)
        + sg._container_invocation("Experiment2", "cpu", PurePosixPath("/stable/plan.json"), _items(method)),
        encoding="utf-8", newline="\n")
    env = os.environ.copy()
    # Simulate Leuven discarding the user's exports. Only site/Slurm variables remain.
    for key in list(env):
        if key.startswith("TABPFN_"):
            env.pop(key)
    env.update(SLURM_JOB_ID="123", SLURM_ARRAY_TASK_ID="7", SLURM_ARRAY_JOB_ID="120", SLURM_CPUS_PER_TASK="18")
    proc = subprocess.run([_bash(), "--noprofile", "--norc", _bash_path(script)],
                          env=env, capture_output=True, text=True, timeout=20)
    return proc, capture


def test_launcher_uses_snapshot_propagates_exit_and_retains_owned_workspace(deployment, tmp_path):
    proc, capture = _run_fake_container(tmp_path, deployment)
    assert proc.returncode == 23, proc.stdout + proc.stderr
    recorded = capture.read_text()
    assert "-m\nsrc.utils.cli\nslurm-task\n--experiment\nExperiment2\n" in recorded
    assert "--array-id\n7\n--plan-path\n/stable/plan.json" in recorded
    assert f"STAGING={_bash_path(deployment[1])}" in recorded
    assert "--cleanenv" in recorded
    assert "JOB=120\nTHREADS=18" in recorded
    assert f"IMAGE={_bash_path(deployment[1] / 'runtimes/paper.sif')}" in recorded
    workspaces = list((deployment[2] / "tabpfncredit" / "workspaces").iterdir())
    assert len(workspaces) == 1
    assert (workspaces[0] / "OWNER.txt").is_file()
    assert (workspaces[0] / "cache").is_dir()
    assert (workspaces[0] / "tmp").is_dir()
    assert (workspaces[0] / "apptainer-cache").is_dir()
    assert (workspaces[0] / "apptainer-tmp").is_dir()


def test_launcher_rejects_results_outside_project_before_work(deployment, tmp_path):
    settings = sg._container_settings()
    settings["TABPFN_RESULTS_ROOT"] = _bash_path(tmp_path / "wrong")
    proc, capture = _run_fake_container(tmp_path, deployment, settings=settings)
    assert proc.returncode != 0
    assert "Results must reside" in proc.stderr
    assert not capture.exists()


def test_launcher_rejects_non_native_scratch(deployment, tmp_path, monkeypatch):
    other = tmp_path / "other-scratch"
    other.mkdir()
    monkeypatch.setenv("VSC_SCRATCH", _bash_path(other))
    proc, capture = _run_fake_container(tmp_path, deployment)
    assert proc.returncode != 0
    assert "Scratch must be writable and native" in proc.stderr
    assert not capture.exists()


def test_launcher_rejects_non_native_project(deployment, tmp_path, monkeypatch):
    other = tmp_path / "other-project"
    other.mkdir()
    monkeypatch.setenv("VSC_PROJECT_LUSTRE1", _bash_path(other))
    proc, capture = _run_fake_container(tmp_path, deployment)
    assert proc.returncode != 0
    assert "PROJECT must be native to wice" in proc.stderr
    assert not capture.exists()


def test_workspace_parent_link_cannot_create_files_outside_scratch(deployment, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (deployment[2] / "tabpfncredit").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks unavailable: {exc}")
    proc, capture = _run_fake_container(tmp_path, deployment)
    assert proc.returncode != 0
    assert "Workspace directory escapes native scratch" in proc.stderr
    assert not (outside / "workspaces").exists()
    assert not capture.exists()


def test_image_must_be_on_project(deployment, tmp_path):
    image = tmp_path / "outside.sif"
    image.write_text("fake", encoding="utf-8")
    settings = sg._container_settings()
    settings["TABPFN_CONTAINER_IMAGE"] = _bash_path(image)
    proc, capture = _run_fake_container(tmp_path, deployment, settings=settings)
    assert proc.returncode != 0
    assert "SIF image must reside" in proc.stderr
    assert not capture.exists()


def test_project_mount_alias_is_bound_for_existing_asset_links(deployment, tmp_path):
    alias = tmp_path / "project-alias"
    try:
        alias.symlink_to(deployment[1], target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks unavailable: {exc}")
    settings = sg._container_settings()
    settings["TABPFN_STAGING_ROOT"] = _bash_path(alias)
    proc, capture = _run_fake_container(tmp_path, deployment, settings=settings)
    assert proc.returncode == 23, proc.stderr
    assert f"{_bash_path(deployment[1])}:{_bash_path(alias)}" in capture.read_text()


def test_hyperfast_missing_asset_fails_before_container(deployment, tmp_path):
    proc, capture = _run_fake_container(tmp_path, deployment, method="hyperfast")
    assert proc.returncode != 0
    assert "HyperFast staged checkpoint is missing" in proc.stderr
    assert not capture.exists()


def test_mitra_checks_immutable_package_assets_without_provisioning():
    command = sg._container_invocation("Experiment1", "gpu_h100", Path("/plan.json"), _items("mitra"))
    assert "Mitra weights must be prepared in the immutable image" in command
    assert "model.safetensors" in command
    assert "cp " not in command and "pip " not in command
