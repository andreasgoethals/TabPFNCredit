"""Storage opt-in checks must not change published runs' legacy path choices."""

from pathlib import Path

import pytest

from src.utils import paths


@pytest.fixture
def storage(monkeypatch, tmp_path):
    for name in (
        "TABPFN_STAGING_ROOT", "TABPFN_PROJECT_ROOT", "VSC_PROJECT_ROOT",
        "TABPFN_RESULTS_ROOT", "TABPFN_CACHE_ROOT", "TABPFN_VSC_PROFILE", "VSC_SCRATCH",
    ):
        monkeypatch.delenv(name, raising=False)
    repo = tmp_path / "data-tier" / "checkout"
    project = tmp_path / "allocation" / "TabPFNCredit"
    scratch = tmp_path / "scratch"
    for directory in (repo, project, scratch):
        directory.mkdir(parents=True)
    monkeypatch.setattr(paths, "PROJECT_ROOT", repo)
    monkeypatch.setattr(paths, "_DEFAULT_STAGING_ROOT", str(tmp_path / "unmounted"))
    return repo, project, scratch


@pytest.fixture
def container(monkeypatch, storage):
    repo, project, scratch = storage
    monkeypatch.setenv("TABPFN_VSC_PROFILE", "container")
    monkeypatch.setenv("TABPFN_STAGING_ROOT", str(project))
    monkeypatch.setenv("VSC_SCRATCH", str(scratch))
    return repo, project, scratch


def test_local_defaults_are_unchanged(storage):
    repo, _, _ = storage
    assert paths.staging_root() is None
    assert paths.data_roots() == [repo / "data"]
    assert paths.checkpoint_roots() == [repo / "checkpoints"]
    assert paths.results_root() == repo / "results"
    assert paths.logs_root() == repo / "logs"
    assert paths.cache_root() == repo / ".cache"
    assert paths.processed_write_dir("PD", "0001.demo") == (
        repo / "data" / "processed" / "pd" / "0001.demo"
    )


@pytest.mark.parametrize("setting", ["TABPFN_PROJECT_ROOT", "VSC_PROJECT_ROOT"])
def test_allocation_alias_appends_project_name(monkeypatch, storage, setting):
    repo, project, _ = storage
    monkeypatch.setenv(setting, str(project.parent))
    assert paths.staging_root() == project
    assert paths.results_root() == project / "results"
    assert paths.data_roots() == [repo / "data", project / "data"]
    assert paths.checkpoint_roots() == [repo / "checkpoints", project / "checkpoints"]


def test_legacy_exact_root_wins_over_aliases(monkeypatch, storage):
    _, project, _ = storage
    monkeypatch.setenv("TABPFN_STAGING_ROOT", str(project))
    monkeypatch.setenv("TABPFN_PROJECT_ROOT", "missing-new-allocation")
    monkeypatch.setenv("VSC_PROJECT_ROOT", "missing-other-allocation")
    assert paths.staging_root() == project
    assert paths.results_root() == project / "results"


def test_project_specific_alias_wins_over_general_alias(monkeypatch, storage):
    _, project, _ = storage
    monkeypatch.setenv("TABPFN_PROJECT_ROOT", str(project.parent))
    monkeypatch.setenv("VSC_PROJECT_ROOT", "missing-other-allocation")
    assert paths.staging_root() == project


@pytest.mark.parametrize("legacy", ["", "missing-old-root"])
def test_unavailable_legacy_root_keeps_local_fallback(monkeypatch, storage, legacy):
    repo, _, _ = storage
    monkeypatch.setenv("TABPFN_STAGING_ROOT", legacy)
    monkeypatch.setenv("VSC_PROJECT_ROOT", "ignored-alias")
    assert paths.staging_root() is None
    assert paths.results_root() == repo / "results"


def test_legacy_output_overrides_and_scratch_are_unchanged(monkeypatch, storage):
    _, _, scratch = storage
    monkeypatch.setenv("TABPFN_RESULTS_ROOT", "relative-results")
    assert paths.results_root() == Path("relative-results")
    monkeypatch.setenv("VSC_SCRATCH", str(scratch))
    assert paths.cache_root() == scratch / "tabpfncredit" / "cache"
    monkeypatch.setenv("TABPFN_CACHE_ROOT", "relative-cache")
    assert paths.cache_root() == Path("relative-cache")


@pytest.mark.parametrize("setting", ["TABPFN_PROJECT_ROOT", "VSC_PROJECT_ROOT"])
@pytest.mark.parametrize("kind", ["empty", "relative", "missing", "file"])
def test_new_alias_never_silently_falls_back(monkeypatch, storage, setting, kind):
    repo, project, _ = storage
    values = {"empty": "", "relative": "allocation", "missing": str(repo / "missing")}
    if kind == "file":
        invalid_allocation = project.parent / "invalid"
        invalid_allocation.mkdir()
        (invalid_allocation / "TabPFNCredit").write_text("not a directory")
        value = str(invalid_allocation)
    else:
        value = values[kind]
    monkeypatch.setenv(setting, value)
    # A legacy results override must not conceal an explicitly invalid new root.
    monkeypatch.setenv("TABPFN_RESULTS_ROOT", str(repo / "results"))
    for resolver in (paths.staging_root, paths.data_roots, paths.results_root):
        with pytest.raises((ValueError, OSError)):
            resolver()


def test_existing_data_selection_is_unchanged_with_new_alias(monkeypatch, storage):
    repo, project, _ = storage
    monkeypatch.setenv("TABPFN_PROJECT_ROOT", str(project.parent))
    for root in (repo, project):
        raw = root / "data" / "raw" / "pd"
        processed = root / "data" / "processed" / "pd" / "0001.demo"
        raw.mkdir(parents=True)
        processed.mkdir(parents=True)
        (raw / "0001.demo.csv").write_text("source\n")
        (processed / "y.npy").write_bytes(b"marker only; never loaded")
    assert paths.find_raw_path("PD", "0001.demo") == repo / "data/raw/pd/0001.demo"
    assert paths.find_processed_dir("PD", "0001.demo") == repo / "data/processed/pd/0001.demo"
    assert paths.processed_write_dir("PD", "0001.demo") == repo / "data/processed/pd/0001.demo"


def test_container_needs_explicit_storage_even_if_legacy_default_exists(monkeypatch, storage):
    _, project, _ = storage
    monkeypatch.setenv("TABPFN_VSC_PROFILE", "container")
    monkeypatch.setattr(paths, "_DEFAULT_STAGING_ROOT", str(project))
    with pytest.raises(ValueError, match="explicit PROJECT"):
        paths.staging_root()


def test_container_rejects_unavailable_legacy_root(monkeypatch, container):
    _, project, _ = container
    monkeypatch.setenv("TABPFN_STAGING_ROOT", str(project / "unmounted"))
    with pytest.raises(FileNotFoundError):
        paths.results_root()


def test_container_results_validate_without_creating_files(monkeypatch, container):
    _, project, _ = container
    assert paths.results_root() == project / "results"
    assert not (project / "results").exists()
    custom = project / "results" / "future-campaign"
    monkeypatch.setenv("TABPFN_RESULTS_ROOT", str(custom))
    assert paths.results_root() == custom
    assert not custom.exists()


@pytest.mark.parametrize("destination", ["data", "scratch", "root", "relative"])
def test_container_results_cannot_escape_project(monkeypatch, container, destination):
    repo, project, scratch = container
    value = {"data": repo / "results", "scratch": scratch / "results",
             "root": project, "relative": Path("results")}[destination]
    monkeypatch.setenv("TABPFN_RESULTS_ROOT", str(value))
    with pytest.raises(ValueError):
        paths.results_root()


def test_container_checks_existing_write_ancestor(monkeypatch, container):
    _, project, _ = container
    denied = project / "protected"
    denied.mkdir()
    monkeypatch.setenv("TABPFN_RESULTS_ROOT", str(denied / "new" / "results"))
    monkeypatch.setattr(paths.os, "access", lambda path, mode: path != denied)
    with pytest.raises(PermissionError, match="not writable"):
        paths.results_root()


def test_container_refuses_file_as_write_ancestor(monkeypatch, container):
    _, project, _ = container
    blocked = project / "file"
    blocked.write_text("preserve me")
    monkeypatch.setenv("TABPFN_RESULTS_ROOT", str(blocked / "results"))
    with pytest.raises((NotADirectoryError, OSError)):
        paths.results_root()


@pytest.mark.parametrize("missing", ["VSC_SCRATCH", "TABPFN_CACHE_ROOT"])
def test_container_cache_has_no_fallback(monkeypatch, container, missing):
    _, _, scratch = container
    monkeypatch.setenv("TABPFN_CACHE_ROOT", str(scratch / "job-123" / "cache"))
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="explicit TABPFN_CACHE_ROOT"):
        paths.cache_root()


def test_container_cache_must_be_beneath_existing_scratch(monkeypatch, container):
    _, project, scratch = container
    cache = scratch / "job-123" / "cache"
    monkeypatch.setenv("TABPFN_CACHE_ROOT", str(cache))
    assert paths.cache_root() == cache
    assert not cache.exists()
    monkeypatch.setenv("TABPFN_CACHE_ROOT", str(project / "cache"))
    with pytest.raises(ValueError, match="must be beneath"):
        paths.cache_root()
    monkeypatch.setenv("VSC_SCRATCH", str(scratch / "missing"))
    with pytest.raises(FileNotFoundError):
        paths.cache_root()


def _directory_link(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks unavailable on this host: {exc}")


@pytest.mark.parametrize("kind", ["results", "cache"])
def test_container_rejects_symlink_escape(monkeypatch, container, kind):
    repo, project, scratch = container
    parent = project if kind == "results" else scratch
    link = parent / "escape"
    _directory_link(link, repo)
    monkeypatch.setenv(f"TABPFN_{kind.upper()}_ROOT", str(link / "output"))
    resolver = paths.results_root if kind == "results" else paths.cache_root
    with pytest.raises(ValueError, match="must be beneath"):
        resolver()


def test_container_refuses_data_reads_and_writes(container):
    repo, _, _ = container
    raw = repo / "data" / "raw" / "pd"
    raw.mkdir(parents=True)
    (raw / "0001.demo.csv").write_text("source\n")
    with pytest.raises(ValueError, match="Raw dataset must be beneath PROJECT"):
        paths.find_raw_path("pd", "0001.demo")
    with pytest.raises(ValueError, match="Raw dataset must be beneath PROJECT"):
        paths.processed_write_dir("pd", "0001.demo")
    with pytest.raises(ValueError, match="Processed data must be beneath"):
        paths.processed_write_dir("pd", "0002.missing")


def test_container_processed_write_stays_with_project_source(container):
    _, project, _ = container
    raw = project / "data" / "raw" / "pd"
    raw.mkdir(parents=True)
    (raw / "0001.demo.csv").write_text("source\n")
    destination = project / "data" / "processed" / "pd" / "0001.demo"
    assert paths.processed_write_dir("PD", "0001.demo") == destination
    assert not destination.exists()


def _file_link(link, target):
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"File symlinks unavailable on this host: {exc}")


@pytest.mark.parametrize("kind", ["raw", "processed"])
def test_container_rejects_selected_data_copy_without_falling_through(monkeypatch, container, kind):
    repo, project, _ = container
    for root in (repo, project):
        directory = root / "data" / kind / "pd"
        if kind == "processed":
            directory /= "0001.demo"
        directory.mkdir(parents=True)
        name = "0001.demo.csv" if kind == "raw" else "y.npy"
        (directory / name).write_bytes(b"synthetic input marker")
    resolver = paths.find_raw_path if kind == "raw" else paths.find_processed_dir
    with pytest.raises(ValueError, match="must be beneath PROJECT"):
        resolver("pd", "0001.demo")
    # The same fixtures still choose the historical repo-local copy in legacy mode.
    monkeypatch.delenv("TABPFN_VSC_PROFILE")
    assert resolver("pd", "0001.demo") == repo / "data" / kind / "pd" / "0001.demo"


def test_container_allows_repo_data_link_into_project(container):
    repo, project, _ = container
    raw = project / "data" / "raw" / "pd"
    processed = project / "data" / "processed" / "pd" / "0001.demo"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)
    (raw / "0001.demo.csv").write_text("source\n")
    for name in ("y.npy", "N.npy", "C.npy", "info.json"):
        (processed / name).write_bytes(b"synthetic input marker")
    _directory_link(repo / "data", project / "data")
    assert paths.find_raw_path("pd", "0001.demo") == repo / "data/raw/pd/0001.demo"
    assert paths.find_processed_dir("pd", "0001.demo") == repo / "data/processed/pd/0001.demo"
    assert paths.processed_write_dir("pd", "0001.demo") == repo / "data/processed/pd/0001.demo"


def test_container_ignores_empty_repo_data_scaffolding(container):
    repo, project, _ = container
    for kind in ("raw", "processed"):
        scaffolding = repo / "data" / kind / "pd"
        scaffolding.mkdir(parents=True)
        (scaffolding / ".gitkeep").write_text("")
    processed = project / "data" / "processed" / "pd" / "0001.demo"
    processed.mkdir(parents=True)
    (processed / "y.npy").write_bytes(b"synthetic input marker")
    assert paths.find_raw_path("pd", "0001.missing") is None
    assert paths.find_processed_dir("pd", "0001.demo") == processed


@pytest.mark.parametrize("extension", [".csv", ".parquet"])
def test_container_checks_raw_file_link_not_nonexistent_stem(container, extension):
    repo, project, _ = container
    raw = repo / "data" / "raw" / "pd"
    raw.mkdir(parents=True)
    original = project / ("source" + extension)
    original.write_bytes(b"synthetic input marker")
    _file_link(raw / ("0001.demo" + extension), original)
    assert paths.find_raw_path("pd", "0001.demo") == raw / "0001.demo"


def test_container_rejects_csv_escape_without_switching_to_safe_parquet(container):
    repo, project, _ = container
    raw = project / "data" / "raw" / "pd"
    raw.mkdir(parents=True)
    outside = repo / "outside.csv"
    outside.write_text("source\n")
    _file_link(raw / "0001.demo.csv", outside)
    (raw / "0001.demo.parquet").write_bytes(b"unused synthetic marker")
    with pytest.raises(ValueError, match="Raw dataset must be beneath PROJECT"):
        paths.find_raw_path("pd", "0001.demo")


def test_container_keeps_csv_precedence_over_unused_parquet(container):
    repo, project, _ = container
    raw = project / "data" / "raw" / "pd"
    raw.mkdir(parents=True)
    (raw / "0001.demo.csv").write_text("source\n")
    outside = repo / "unused.parquet"
    outside.write_bytes(b"unused synthetic marker")
    _file_link(raw / "0001.demo.parquet", outside)
    assert paths.find_raw_path("pd", "0001.demo") == raw / "0001.demo"


@pytest.mark.parametrize("name", ["y.npy", "N.npy", "C.npy", "info.json"])
def test_container_rejects_processed_file_symlink_escape(monkeypatch, container, name):
    repo, project, _ = container
    processed = project / "data" / "processed" / "pd" / "0001.demo"
    processed.mkdir(parents=True)
    if name != "y.npy":
        (processed / "y.npy").write_bytes(b"synthetic input marker")
    outside = repo / name
    outside.write_bytes(b"synthetic input marker")
    _file_link(processed / name, outside)
    with pytest.raises(ValueError, match="must be beneath PROJECT"):
        paths.find_processed_dir("pd", "0001.demo")
    monkeypatch.delenv("TABPFN_VSC_PROFILE")
    assert paths.find_processed_dir("pd", "0001.demo") == processed
