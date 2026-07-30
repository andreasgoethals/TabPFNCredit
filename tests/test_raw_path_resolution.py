"""Raw dataset files must resolve, including on project storage.

Every dataset slug has the form ``NNNN.name`` ("0001.gmsc"), so pathlib reads
``.gmsc`` as a suffix. ``find_raw_path`` used ``stem.with_suffix(".csv")``, which
REPLACES that suffix and looked for "0001.csv" -- a file that never exists. The
consequences were entirely silent and cost a long cluster debugging session:

* preprocessing always fell back to the repo-local path, so raw data on the
  cluster's project storage was never found;
* ``row_count``'s raw fallback never fired, so a dataset whose processed cache was
  missing reported no row count, dropped out of ``min_rows`` selection, and
  vanished from every experiment while ``resubmit`` said "nothing to do".

These tests use a temporary directory rather than the real data, so they hold in a
clone with no datasets at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import paths


def test_raw_file_for_appends_rather_than_replaces():
    """The core mistake, pinned directly."""
    stem = Path("/somewhere/data/raw/pd/0001.gmsc")
    assert paths.raw_file_for(stem, ".csv").name == "0001.gmsc.csv"
    assert stem.with_suffix(".csv").name == "0001.csv", (
        "pathlib's behaviour changed; the comment in raw_file_for needs revisiting")


@pytest.mark.parametrize("dataset", ["0001.gmsc", "0012.home_credit", "0008.german"])
@pytest.mark.parametrize("ext", [".csv", ".parquet"])
def test_find_raw_path_resolves_dotted_slugs(monkeypatch, tmp_path, dataset, ext):
    raw = tmp_path / "data" / "raw" / "pd"
    raw.mkdir(parents=True)
    (raw / f"{dataset}{ext}").write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(paths, "data_roots", lambda: [tmp_path / "data"])

    found = paths.find_raw_path("pd", dataset)
    assert found is not None, f"{dataset}{ext} exists but was not found"
    assert paths.raw_file_for(found, ext).exists()


def test_find_raw_path_prefers_the_first_root(monkeypatch, tmp_path):
    """Repo-local must win over project storage, as documented."""
    first, second = tmp_path / "repo" / "data", tmp_path / "staging" / "data"
    for root in (first, second):
        (root / "raw" / "lgd").mkdir(parents=True)
        (root / "raw" / "lgd" / "0001.heloc.csv").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(paths, "data_roots", lambda: [first, second])
    assert paths.find_raw_path("lgd", "0001.heloc") == first / "raw" / "lgd" / "0001.heloc"


def test_find_raw_path_falls_through_to_the_second_root(monkeypatch, tmp_path):
    """The cluster case: nothing in the repo, the file on project storage."""
    first, second = tmp_path / "repo" / "data", tmp_path / "staging" / "data"
    (first / "raw" / "lgd").mkdir(parents=True)          # exists but empty
    (second / "raw" / "lgd").mkdir(parents=True)
    (second / "raw" / "lgd" / "0001.heloc.csv").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(paths, "data_roots", lambda: [first, second])
    assert paths.find_raw_path("lgd", "0001.heloc") == second / "raw" / "lgd" / "0001.heloc"


def test_missing_dataset_still_returns_none(monkeypatch, tmp_path):
    (tmp_path / "data" / "raw" / "pd").mkdir(parents=True)
    monkeypatch.setattr(paths, "data_roots", lambda: [tmp_path / "data"])
    assert paths.find_raw_path("pd", "0099.absent") is None


def test_row_count_uses_the_raw_file_when_there_is_no_processed_cache(monkeypatch,
                                                                     tmp_path):
    """The fallback that silently never ran, and let datasets disappear."""
    import src.data.dataset_inventory as inv

    raw = tmp_path / "data" / "raw" / "pd"
    raw.mkdir(parents=True)
    (raw / "0042.demo.csv").write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    monkeypatch.setattr(paths, "data_roots", lambda: [tmp_path / "data"])
    monkeypatch.setattr(inv, "_processed_row_count", lambda task, dataset: None)

    assert inv.row_count("pd", "0042.demo") == 3, "header excluded, 3 data rows"
