"""Tests for per-array-task sharding of Experiment 2/3 packed results.

A cell's sweep points may now be split across many array tasks (intra-cell
parallelism); each task writes its OWN ``<method>__shard_<id>.json`` and the
skip-check / resubmit gap-scan / summariser read the UNION across shards.
"""
from __future__ import annotations


def _folds(cv: int, auc: float = 0.8, n: int | None = None) -> dict:
    n = cv if n is None else n
    return {
        i: {
            "metrics": {"AUC": auc, "Accuracy": 0.9},
            "train_time": 1.0, "predict_time": 0.1, "threshold": 0.5,
            "used_hpo": False, "hpo_n_trials": None,
            "n_clipped_below": 0, "n_clipped_above": 0,
            "info": {"task_type": "binclass"},
        }
        for i in range(n)
    }


def test_packed_shards_write_separate_files_and_union(tmp_path):
    from src.utils.result_io import (
        save_packed_point, has_complete_packed_point, complete_packed_points,
    )
    cv = 3
    kw = dict(base=tmp_path, experiment="experiment3", task="pd", dataset="d1",
              method_base="tabicl_v2")

    # Two different array tasks own disjoint points of the SAME cell.
    save_packed_point(_folds(cv), point_name="tabicl_v2__min0p15", shard="job1_0", **kw)
    save_packed_point(_folds(cv), point_name="tabicl_v2__min0p10", shard="job1_1", **kw)

    cell_dir = tmp_path / "experiment3" / "pd" / "d1"
    shard_files = sorted(p.name for p in cell_dir.glob("tabicl_v2__shard_*.json"))
    assert shard_files == [
        "tabicl_v2__shard_job1_0.json", "tabicl_v2__shard_job1_1.json",
    ]
    # No legacy unsharded file was created.
    assert not (cell_dir / "tabicl_v2.json").exists()

    # Union across shards sees both points.
    done = complete_packed_points(expected_folds=cv, **kw)
    assert done == {"tabicl_v2__min0p15", "tabicl_v2__min0p10"}

    # Skip-check finds a point regardless of which shard holds it.
    assert has_complete_packed_point(point_name="tabicl_v2__min0p10", expected_folds=cv, **kw)
    assert not has_complete_packed_point(point_name="tabicl_v2__min0p99", expected_folds=cv, **kw)

    # An incomplete point (too few folds) is NOT counted complete.
    save_packed_point(_folds(cv, n=1), point_name="tabicl_v2__min0p05", shard="job1_2", **kw)
    assert "tabicl_v2__min0p05" not in complete_packed_points(expected_folds=cv, **kw)


def test_legacy_unsharded_file_still_unioned(tmp_path):
    """shard=None keeps writing the single <method>.json; union still sees it."""
    from src.utils.result_io import save_packed_point, complete_packed_points
    cv = 2
    kw = dict(base=tmp_path, experiment="experiment2", task="lgd", dataset="d1",
              method_base="xgboost")
    save_packed_point(_folds(cv), point_name="xgboost__row1000", shard=None, **kw)   # legacy
    save_packed_point(_folds(cv), point_name="xgboost__row2000", shard="J_3", **kw)  # sharded
    assert (tmp_path / "experiment2" / "lgd" / "d1" / "xgboost.json").exists()
    done = complete_packed_points(expected_folds=cv, **kw)
    assert done == {"xgboost__row1000", "xgboost__row2000"}


def test_summarize_merges_shards_and_dedupes(tmp_path):
    from src.utils.result_io import save_packed_point
    from src.utils.result_summary import collect_fold_results
    cv = 3
    kw = dict(base=tmp_path, experiment="experiment3", task="pd", dataset="d1",
              method_base="tabicl_v2")
    save_packed_point(_folds(cv), point_name="tabicl_v2__min0p15", shard="0", **kw)
    save_packed_point(_folds(cv), point_name="tabicl_v2__min0p10", shard="1", **kw)
    # Replica race: the SAME point also landed in a second shard.
    save_packed_point(_folds(cv), point_name="tabicl_v2__min0p15", shard="2", **kw)

    df = collect_fold_results(tmp_path, "experiment3")
    n_rows = df.height if hasattr(df, "height") else len(df)
    # 2 distinct points x 3 folds = 6 rows (the duplicate is deduped, not 9).
    assert n_rows == 6
    method_fulls = set(df["method_full"].to_list() if hasattr(df, "to_list") is False
                       and hasattr(df["method_full"], "to_list") else df["method_full"])
    assert method_fulls == {"tabicl_v2__min0p15", "tabicl_v2__min0p10"}


def test_pack_work_items_splits_big_cells_only(tmp_path):
    from src.utils.slurm_generator import pack_work_items
    cap = 100
    big = [{"task": "pd", "dataset": "d1", "method": "m",
            "name": f"m__min{i}", "est_seconds": 30} for i in range(10)]   # 300s cell
    cheap = [{"task": "pd", "dataset": "d2", "method": "m",
              "name": "m", "est_seconds": 10}]                              # 10s cell

    # Whole-cell packing: the 300s cell lands in ONE slot.
    _, mx_whole = pack_work_items(big + cheap, cap_seconds=cap, max_slots=10, split_cells=False)
    assert mx_whole >= 300

    # Split packing: the big cell fans out; no slot carries the whole 300s.
    slots, mx_split = pack_work_items(big + cheap, cap_seconds=cap, max_slots=10, split_cells=True)
    assert mx_split <= cap + 30          # each sub-group is <= cap (+ at most one overflow point)
    assert mx_split < mx_whole           # genuinely more parallel
    # No points lost either way.
    assert sum(len(s) for s in slots) == 11
