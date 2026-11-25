"""Assemble a notebook result tree from the submitted benchmark and selected refits."""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from filelock import FileLock


def verify_revision_inputs(root: Path, experiment: str, task: str, dataset: str, config: dict) -> None:
    """Keep resumed revision cells on the same data, split, tuning and runtime."""
    from importlib.metadata import distribution
    from src.methods.method_runner import _fold_inputs
    from src.utils.paths import PROJECT_ROOT, find_processed_dir

    if root.resolve().is_relative_to((PROJECT_ROOT / "results").resolve()) or (
        root / "revision_sources.json"
    ).exists():
        raise ValueError("Use a separate directory for revision training")
    cells = root / experiment.lower() / task / dataset
    processed = find_processed_dir(task, dataset)
    if processed is None:
        if next(cells.glob("*.json"), None):
            raise ValueError("Cannot resume without the processed inputs")
        return  # The first fit can create the processed data.
    identity = {
        "inputs": dict(_fold_inputs(processed)),
        "split": config["split"], "training": config["training"], "tuning": config["tuning"],
        "talent": distribution("TALENT").read_text("direct_url.json"),
        "image": os.environ.get("TABPFN_CONTAINER_IMAGE"),
    }
    marker = root / ".revision" / experiment.lower() / task / (dataset + ".json")
    marker.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(marker) + ".lock"):
        if marker.is_file():
            if json.loads(marker.read_text()) != identity:
                raise ValueError(f"Inputs or settings changed for {dataset}; choose a new revision directory")
        else:
            if next(cells.glob("*.json"), None):
                raise ValueError(f"Existing results for {dataset} have no revision identity")
            with tempfile.TemporaryDirectory(dir=marker.parent) as temp:
                pending = Path(temp) / "inputs.json"
                pending.write_text(json.dumps(identity, sort_keys=True))
                os.replace(pending, marker)


def _validate_refit(source: Path, original: Path) -> None:
    """Revision cells replace complete original cells, including their predictions."""
    replacement = original.is_file()
    reference = original
    if not replacement:
        # Compare a newly benchmarked method with the uncapped linear baseline.
        reference = original.with_name("LogReg.json" if original.parent.parent.name == "pd"
                                       else "LinearRegression.json")
    old = json.loads(reference.read_bytes())
    new = json.loads(source.read_bytes())
    folds = set(old["folds"])
    if not folds or set(new["folds"]) != folds:
        raise ValueError(f"Incomplete or different folds in {source}")
    for key in ("experiment", "task", "dataset"):
        if new[key] != old[key]:
            raise ValueError(f"Different {key} in {source}")
    if new["method"] != original.stem:
        raise ValueError(f"Different method in {source}")
    with np.load(reference.with_suffix(".npz"), allow_pickle=False) as before, np.load(
        source.with_suffix(".npz"), allow_pickle=False
    ) as after:
        for fold in folds:
            prefix = f"fold_{fold}_"
            target = prefix + "y_true"
            if not np.array_equal(before[target], after[target]):
                raise ValueError(f"Different held-out targets in {source}, fold {fold}")
            for key in ("y_pred", "y_prob"):
                name = prefix + key
                if (key == "y_pred" or (replacement and name in before)) and name not in after:
                    raise ValueError(f"Missing {name} in {source}")
                if name in after and len(after[name]) != len(after[target]):
                    raise ValueError(f"Wrong number of predictions for {name} in {source}")


def assemble(baseline: Path, revision: Path, output: Path) -> None:
    baseline, revision, output = (p.resolve() for p in (baseline, revision, output))
    roots = (baseline, revision, output)
    if any(a.is_relative_to(b) for i, a in enumerate(roots)
           for j, b in enumerate(roots) if i != j):
        raise ValueError("Use separate baseline, revision and output directories")
    if output.exists():
        raise FileExistsError(output)
    if not baseline.is_dir() or not revision.is_dir():
        raise FileNotFoundError("Both input result directories must exist")
    replacements = []
    additions = []
    for experiment in ("experiment0", "experiment1"):
        for source in sorted((revision / experiment).glob("*/*/*.json")):
            relative = source.relative_to(revision)
            _validate_refit(source, baseline / relative)
            if (baseline / relative).is_file():
                replacements.append(relative)
            else:
                additions.append(relative)
    if not replacements and not additions:
        raise ValueError("No completed Experiment 0 or 1 refits found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        staging = Path(scratch) / "results"
        staging.mkdir()
        for name in ("experiment0", "experiment1", "experiment2", "experiment3"):
            if (baseline / name).is_dir():
                shutil.copytree(baseline / name, staging / name)
        for relative in replacements + additions:
            for suffix in (".json", ".npz"):
                item = relative.with_suffix(suffix)
                shutil.copy2(revision / item, staging / item)
        manifest = {"baseline": str(baseline), "revision": str(revision),
                    "replaced_cells": [p.as_posix() for p in replacements],
                    "added_cells": [p.as_posix() for p in additions]}
        (staging / "revision_sources.json").write_text(json.dumps(manifest, indent=2))
        staging.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "revision", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    assemble(args.baseline, args.revision, args.output)
