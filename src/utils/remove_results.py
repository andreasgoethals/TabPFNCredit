# src/utils/remove_results.py
"""
Surgical tool for removing specific method results from experiment pickle files.

This utility allows selective removal of results for specific methods without
deleting entire pickle files or corrupting other experiments. Use this to:
- Re-run specific methods after fixing bugs
- Clean up results from methods that need re-evaluation
- Remove invalid/incomplete results

Usage:
    # Remove TabPFN results from all PD datasets
    python -m src.utils.remove_results --experiment experiment1 --method tabpfn --task pd

    # Remove specific method from specific dataset
    python -m src.utils.remove_results --experiment experiment1 --method tabpfn --dataset 0013.hmeq --task pd

    # Dry run (show what would be removed without making changes)
    python -m src.utils.remove_results --experiment experiment1 --method tabpfn --task pd --dry-run

Safety features:
- File locking to prevent concurrent modification
- Dry-run mode to preview changes
- Detailed logging of all modifications
- Preserves all other method results in the pickle files
"""

from __future__ import annotations
import pickle
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# File locking imports
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    try:
        import portalocker
        HAS_PORTALOCKER = True
    except ImportError:
        HAS_PORTALOCKER = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Find the project root directory dynamically."""
    current = Path(__file__).resolve().parent
    root_markers = {'src', 'setup.py', 'pyproject.toml', 'README.md', '.git'}

    for _ in range(10):
        if any((current / marker).exists() for marker in root_markers):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    return Path.cwd()


def _acquire_lock(file_handle, exclusive: bool = True) -> bool:
    """
    Acquire a file lock (cross-platform).

    Args:
        file_handle: Open file handle
        exclusive: If True, exclusive lock. If False, shared lock.

    Returns:
        True if lock acquired, False otherwise
    """
    if HAS_FCNTL:
        try:
            lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(file_handle.fileno(), lock_type)
            return True
        except IOError as e:
            logger.warning(f"Could not acquire file lock (fcntl): {e}")
            return False

    elif HAS_PORTALOCKER:
        try:
            lock_type = portalocker.LOCK_EX if exclusive else portalocker.LOCK_SH
            portalocker.lock(file_handle, lock_type)
            return True
        except Exception as e:
            logger.warning(f"Could not acquire file lock (portalocker): {e}")
            return False

    else:
        logger.warning(
            "No file locking library available. "
            "Ensure no other processes are modifying result files."
        )
        return True  # Proceed without locking


def _release_lock(file_handle) -> None:
    """Release file lock (cross-platform)."""
    if HAS_FCNTL:
        try:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass
    elif HAS_PORTALOCKER:
        try:
            portalocker.unlock(file_handle)
        except Exception:
            pass


def remove_method_from_results(
    results: Dict[str, Any],
    method: str
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Remove a specific method from results dictionary.

    Args:
        results: Results dictionary with structure {hpo_mode: {method: {fold: data}}}
        method: Method name to remove (e.g., 'tabpfn', 'mitra')

    Returns:
        Tuple of (modified results, list of removed keys)
    """
    removed_keys = []

    # Standard HPO modes in our result structure
    hpo_modes = ['NO_HPO', 'HPO']

    for hpo_mode in hpo_modes:
        if hpo_mode in results and isinstance(results[hpo_mode], dict):
            if method in results[hpo_mode]:
                del results[hpo_mode][method]
                removed_keys.append(f"{hpo_mode}/{method}")
                logger.debug(f"  Removed key: {hpo_mode}/{method}")

    return results, removed_keys


def process_pickle_file(
    filepath: Path,
    method: str,
    dry_run: bool = False
) -> Tuple[bool, List[str]]:
    """
    Process a single pickle file and remove specified method.

    Args:
        filepath: Path to the pickle file
        method: Method name to remove
        dry_run: If True, only report what would be done

    Returns:
        Tuple of (was_modified, list of removed keys)
    """
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return False, []

    removed_keys = []

    try:
        # Read the file
        with open(filepath, 'rb') as f:
            if not _acquire_lock(f, exclusive=False):
                logger.error(f"Could not acquire read lock for {filepath}")
                return False, []

            try:
                results = pickle.load(f)
            finally:
                _release_lock(f)

        # Check if method exists in results
        method_found = False
        for hpo_mode in ['NO_HPO', 'HPO']:
            if hpo_mode in results and isinstance(results[hpo_mode], dict):
                if method in results[hpo_mode]:
                    method_found = True
                    removed_keys.append(f"{hpo_mode}/{method}")

        if not method_found:
            logger.debug(f"Method '{method}' not found in {filepath.name}")
            return False, []

        if dry_run:
            logger.info(f"[DRY RUN] Would remove from {filepath.name}: {removed_keys}")
            return True, removed_keys

        # Remove the method
        results, removed_keys = remove_method_from_results(results, method)

        # Write back with exclusive lock
        with open(filepath, 'wb') as f:
            if not _acquire_lock(f, exclusive=True):
                logger.error(f"Could not acquire write lock for {filepath}")
                return False, []

            try:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
            finally:
                _release_lock(f)

        logger.info(f"Modified {filepath.name}: removed {removed_keys}")
        return True, removed_keys

    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return False, []


def remove_results(
    experiment: str,
    method: str,
    task: str,
    dataset: Optional[str] = None,
    base_dir: str = "results",
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Remove results for a specific method from experiment pickle files.

    Args:
        experiment: Experiment name/folder (e.g., 'experiment1')
        method: Method key to remove (e.g., 'tabpfn', 'mitra')
        task: Task type ('pd' or 'lgd')
        dataset: Specific dataset name. If None, apply to all datasets.
        base_dir: Base directory for results (default: 'results')
        dry_run: If True, only report what would be done

    Returns:
        Summary dictionary with stats about the operation
    """
    project_root = _find_project_root()
    results_dir = project_root / base_dir / experiment / task.lower()

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return {"error": f"Directory not found: {results_dir}"}

    # Find pickle files to process
    if dataset:
        # Specific dataset
        pkl_files = [results_dir / f"{dataset}.pkl"]
        if not pkl_files[0].exists():
            logger.error(f"Dataset file not found: {pkl_files[0]}")
            return {"error": f"File not found: {pkl_files[0]}"}
    else:
        # All datasets in the task folder
        pkl_files = sorted(results_dir.glob("*.pkl"))

    if not pkl_files:
        logger.warning(f"No pickle files found in {results_dir}")
        return {"warning": "No pickle files found"}

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Processing {len(pkl_files)} file(s) in {results_dir}")
    logger.info(f"Removing method: {method}")

    # Process each file
    summary = {
        "experiment": experiment,
        "task": task,
        "method": method,
        "dry_run": dry_run,
        "files_processed": 0,
        "files_modified": 0,
        "keys_removed": [],
        "files_skipped": [],
    }

    for pkl_file in pkl_files:
        summary["files_processed"] += 1
        was_modified, removed_keys = process_pickle_file(pkl_file, method, dry_run)

        if was_modified:
            summary["files_modified"] += 1
            summary["keys_removed"].extend([
                f"{pkl_file.stem}/{key}" for key in removed_keys
            ])
        else:
            summary["files_skipped"].append(pkl_file.stem)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment}")
    logger.info(f"Task: {task}")
    logger.info(f"Method removed: {method}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Files processed: {summary['files_processed']}")
    logger.info(f"Files modified: {summary['files_modified']}")
    logger.info(f"Total keys removed: {len(summary['keys_removed'])}")

    if summary['keys_removed']:
        logger.info("\nRemoved keys:")
        for key in summary['keys_removed']:
            logger.info(f"  - {key}")

    if summary['files_skipped'] and len(summary['files_skipped']) <= 10:
        logger.info(f"\nFiles skipped (method not found): {summary['files_skipped']}")
    elif summary['files_skipped']:
        logger.info(f"\nFiles skipped (method not found): {len(summary['files_skipped'])} files")

    logger.info("=" * 60)

    return summary


def list_methods_in_results(
    experiment: str,
    task: str,
    dataset: Optional[str] = None,
    base_dir: str = "results"
) -> Dict[str, List[str]]:
    """
    List all methods present in experiment results.

    Args:
        experiment: Experiment name/folder
        task: Task type ('pd' or 'lgd')
        dataset: Specific dataset. If None, scan all datasets.
        base_dir: Base directory for results

    Returns:
        Dictionary mapping dataset names to list of methods
    """
    project_root = _find_project_root()
    results_dir = project_root / base_dir / experiment / task.lower()

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return {}

    if dataset:
        pkl_files = [results_dir / f"{dataset}.pkl"]
    else:
        pkl_files = sorted(results_dir.glob("*.pkl"))

    methods_by_dataset = {}

    for pkl_file in pkl_files:
        if not pkl_file.exists():
            continue

        try:
            with open(pkl_file, 'rb') as f:
                results = pickle.load(f)

            methods = set()
            for hpo_mode in ['NO_HPO', 'HPO']:
                if hpo_mode in results and isinstance(results[hpo_mode], dict):
                    methods.update(results[hpo_mode].keys())

            methods_by_dataset[pkl_file.stem] = sorted(methods)

        except Exception as e:
            logger.warning(f"Could not read {pkl_file}: {e}")

    return methods_by_dataset


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Remove specific method results from experiment pickle files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove TabPFN results from all PD datasets in experiment1
  python -m src.utils.remove_results --experiment experiment1 --method tabpfn --task pd

  # Remove specific method from specific dataset
  python -m src.utils.remove_results --experiment experiment1 --method tabpfn --dataset 0013.hmeq --task pd

  # Dry run (show what would be removed)
  python -m src.utils.remove_results --experiment experiment1 --method tabpfn --task pd --dry-run

  # List all methods in results
  python -m src.utils.remove_results --experiment experiment1 --task pd --list-methods
        """
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name/folder (e.g., 'experiment1', 'pd', 'lgd')"
    )

    parser.add_argument(
        "--method",
        type=str,
        help="Method key to remove (e.g., 'tabpfn', 'mitra'). Required unless --list-methods."
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=['pd', 'lgd'],
        help="Task type ('pd' or 'lgd')"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset name. If not provided, applies to all datasets."
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="results",
        help="Base directory for results (default: 'results')"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without making changes"
    )

    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List all methods in results instead of removing"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_methods:
        # List methods mode
        methods_by_dataset = list_methods_in_results(
            experiment=args.experiment,
            task=args.task,
            dataset=args.dataset,
            base_dir=args.base_dir
        )

        if not methods_by_dataset:
            print("No results found.")
            return 1

        print(f"\nMethods in {args.experiment}/{args.task}:")
        print("=" * 60)

        # Collect all unique methods
        all_methods = set()
        for methods in methods_by_dataset.values():
            all_methods.update(methods)

        print(f"\nAll methods ({len(all_methods)}): {sorted(all_methods)}")

        if args.dataset or len(methods_by_dataset) <= 5:
            print("\nPer-dataset breakdown:")
            for dataset, methods in methods_by_dataset.items():
                print(f"  {dataset}: {methods}")

        return 0

    # Remove mode - method is required
    if not args.method:
        parser.error("--method is required unless using --list-methods")

    summary = remove_results(
        experiment=args.experiment,
        method=args.method,
        task=args.task,
        dataset=args.dataset,
        base_dir=args.base_dir,
        dry_run=args.dry_run
    )

    if "error" in summary:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
