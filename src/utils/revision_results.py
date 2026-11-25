"""Assemble a notebook result tree from the submitted benchmark and selected refits."""

import argparse
import json
import os
from pathlib import Path

from src.utils.paths import experiment_results_dir, writable_results
import shutil
import tempfile

import numpy as np
from filelock import FileLock


def verify_revision_inputs(root: Path, experiment: str, task: str, dataset: str, config: dict, *, record: bool = True) -> None:
    """Keep resumed revision cells on the same data, split, tuning and runtime."""
    from importlib.metadata import distribution
    from src.methods.method_runner import _fold_inputs
    from src.utils.paths import PROJECT_ROOT, find_processed_dir

    if root.resolve().is_relative_to((PROJECT_ROOT / "results").resolve()) or (
        root / "revision_sources.json"
    ).exists():
        raise ValueError("Use a separate directory for revision training")
    writable_results(root)
    cells = experiment_results_dir(root, experiment) / task / dataset
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
    if root.name == "output":
        marker = root / experiment.lower() / "manifests" / "inputs" / task / (dataset + ".json")
    if not record:
        if marker.is_file():
            if json.loads(marker.read_text()) != identity:
                raise ValueError(f"Inputs or settings changed for {dataset}; choose a new revision directory")
        elif next(cells.glob("*.json"), None):
            raise ValueError(f"Existing results for {dataset} have no revision identity")
        return
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
    writable_results(output)
    if output.exists():
        raise FileExistsError(output)
    if not baseline.is_dir() or not revision.is_dir():
        raise FileNotFoundError("Both input result directories must exist")
    replacements = []
    additions = []
    for experiment in ("experiment0", "experiment1"):
        for source in sorted(experiment_results_dir(revision, experiment).glob("*/*/*.json")):
            relative = Path(experiment, *source.parts[-3:])
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
        overrides = {baseline / rel.with_suffix(ext) for rel in replacements + additions
                     for ext in (".json", ".npz")}

        def skip_replaced(directory, names):
            return [name for name in names if Path(directory) / name in overrides]

        for relative in replacements + additions:
            source = experiment_results_dir(revision, relative.parts[0]).joinpath(*relative.parts[1:])
            (staging / relative.parent).mkdir(parents=True, exist_ok=True)
            before = source.stat()
            for suffix in (".json", ".npz"):
                item = relative.with_suffix(suffix)
                shutil.copy2(source.with_suffix(suffix), staging / item)
            after = source.stat()
            if not os.path.samestat(before, after) or before.st_mtime_ns != after.st_mtime_ns:
                raise ValueError(f"Result changed during assembly: {source}")
            _validate_refit(staging / relative, baseline / relative)
        for name in ("experiment0", "experiment1", "experiment2", "experiment3"):
            if (baseline / name).is_dir():
                shutil.copytree(baseline / name, staging / name, ignore=skip_replaced,
                                dirs_exist_ok=True)
        manifest = {"baseline": str(baseline), "revision": str(revision),
                    "replaced_cells": [p.as_posix() for p in replacements],
                    "added_cells": [p.as_posix() for p in additions]}
        (staging / "revision_sources.json").write_text(json.dumps(manifest, indent=2))
        staging.rename(output)


def assemble_selected(baseline: Path, revision: Path, output: Path, policy: Path) -> None:
    """Publish a complete, independently movable notebook view from explicit sources."""
    import yaml

    baseline, revision, output = (p.resolve() for p in (baseline, revision, output))
    roots = (baseline, revision, output)
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise ValueError("Archive, revision and analysis directories must be separate")
    writable_results(output)
    if output.exists():
        raise FileExistsError(output)
    if not baseline.is_dir():
        raise FileNotFoundError(baseline)
    rules = yaml.safe_load(policy.read_text())
    experiments = [f"experiment{i}" for i in range(4)]
    if not isinstance(rules, dict) or set(rules) != set(experiments):
        raise ValueError("Select a source for each of experiment0 through experiment3")

    sources = {"archive": baseline, "new": revision}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / output.name
        staging.mkdir()
        origins = {}
        for exp in experiments:
            rule = rules[exp]
            if not isinstance(rule, dict) or set(rule) - {"source", "methods"}:
                raise ValueError(f"Invalid source rule for {exp}")
            default = rule.get("source")
            overrides = rule.get("methods", {})
            if default not in sources or not isinstance(overrides, dict) or any(
                value not in sources for value in overrides.values()
            ):
                raise ValueError(f"Use archive or new for {exp} and its methods")
            if exp == "experiment0" and any(value != default for value in overrides.values()):
                raise ValueError("The revision preflight and paper pilot use different split settings")
            groups = {}
            for label, base in sources.items():
                by_method = {}
                for src in sorted(experiment_results_dir(base, exp).glob("*/*/*.json")):
                    task, dataset, name = src.parts[-3:]
                    if task in ("pd", "lgd"):
                        key = (task, dataset, name.split("__")[0].removesuffix(".json"))
                        by_method.setdefault(key, []).append(src)
                groups[label] = by_method
            keys = groups["archive"].keys() | groups["new"].keys()
            if set(overrides) - {key[2] for key in keys}:
                raise ValueError(f"Unknown method override in {exp}")
            if "new" in {default, *overrides.values()} and not groups["new"]:
                raise ValueError(f"No new results available for {exp}")
            for task, dataset, method in sorted(keys):
                label = overrides.get(method, default)
                key = (task, dataset, method)
                files = groups[label].get(key, [])
                old = groups["archive"].get(key, [])
                if not files:
                    if label == "new" and exp != "experiment0":
                        raise ValueError(f"Missing new results for {exp}/{task}/{dataset}/{method}")
                    continue
                folder = experiment_results_dir(staging, exp) / task / dataset
                folder.mkdir(parents=True, exist_ok=True)
                if label == "new" and exp == "experiment1" and not {p.name for p in old} <= {p.name for p in files}:
                    raise ValueError(f"Missing result variants for {exp}/{task}/{dataset}/{method}")
                if label == "new" and exp in ("experiment2", "experiment3"):
                    def point_folds(paths):
                        merged = {}
                        for path in paths:
                            for point, entry in json.loads(path.read_bytes())["points"].items():
                                merged.setdefault(point, set()).update(entry["folds"])
                        return merged
                    previous, current = point_folds(old), point_folds(files)
                    if not current or any(current.get(point) != folds for point, folds in previous.items()):
                        raise ValueError(f"Incomplete sweep for {key}")
                for src in files:
                    dst = folder / src.name
                    stat = src.stat()
                    shutil.copy2(src, dst)
                    if exp in ("experiment0", "experiment1"):
                        shutil.copy2(src.with_suffix(".npz"), dst.with_suffix(".npz"))
                        if label == "new":
                            original = experiment_results_dir(baseline, exp) / task / dataset / src.name
                            _validate_refit(dst, dst if exp == "experiment0" else original)
                    if stat.st_mtime_ns != src.stat().st_mtime_ns or not os.path.samestat(stat, src.stat()):
                        raise ValueError(f"Result changed during assembly: {src}")
                    origins[dst.relative_to(staging).as_posix()] = label
        if not origins:
            raise ValueError("No results selected")
        manifest = {"archive": str(baseline), "new": str(revision), "policy": rules, "cells": origins}
        (staging / "revision_sources.json").write_text(json.dumps(manifest, indent=2))
        staging.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "revision", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--policy", type=Path, help="Explicit experiment and method source selections")
    args = parser.parse_args()
    if args.policy:
        assemble_selected(args.baseline, args.revision, args.output, args.policy)
    else:
        assemble(args.baseline, args.revision, args.output)
