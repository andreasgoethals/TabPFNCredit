# scripts/Experiment1.py
"""
Experiment 1: HPO Benchmark across all enabled datasets.

Supports parallelization via SLURM array jobs:
    python scripts/Experiment1.py --dataset_idx=0
    
Or run all sequentially:
    python scripts/Experiment1.py
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging
import shutil

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.utils.storage_handler import StorageHandler
from src.methods.HPO_runner import run_hpo_comparison


def _has_results(experiment_path: Path) -> bool:
    """Check if experiment folder has actual results."""
    if not experiment_path.exists():
        return False
    has_pkl = any(experiment_path.rglob("*.pkl"))
    has_metadata = any(experiment_path.rglob("*metadata.json"))
    return has_pkl or has_metadata


def _delete_experiment_folder(experiment_path: Path) -> bool:
    """Delete experiment folder completely."""
    try:
        if experiment_path.exists():
            shutil.rmtree(experiment_path, ignore_errors=True)
            return not experiment_path.exists()
        return True
    except Exception as e:
        print(f"Warning: Could not delete {experiment_path}: {e}")
        return False


def run_experiment1(
    experiment_name: str = "experiment1",
    skip_completed: bool = True,
    verbose: bool = True,
    dataset_idx: int = None
) -> None:
    """
    Run Experiment 1: HPO comparison across all enabled datasets.
    
    Args:
        experiment_name: Name of experiment (folder name in results/)
        skip_completed: If True, skip datasets with existing results.
        verbose: Whether to print detailed progress during training
        dataset_idx: If specified, only run this dataset index (for SLURM array jobs)
    """
    
    storage = StorageHandler(experiment_name)
    experiment_path = storage.get_experiment_path()
    
    # Ensure experiment directory exists
    experiment_path.mkdir(parents=True, exist_ok=True)
    (experiment_path / "pd").mkdir(exist_ok=True)
    (experiment_path / "lgd").mkdir(exist_ok=True)
    (experiment_path / "logs").mkdir(exist_ok=True)  # Create logs directory
    
    # Load configuration
    config = load_config()
    
    # Build list of all datasets
    pd_datasets = [(ds, 'pd') for ds in config['datasets']['pd'].keys()]
    lgd_datasets = [(ds, 'lgd') for ds in config['datasets']['lgd'].keys()]
    all_datasets = pd_datasets + lgd_datasets
    total_datasets = len(all_datasets)
    
    # Filter to single dataset if running in parallel mode
    if dataset_idx is not None:
        if dataset_idx < 0 or dataset_idx >= total_datasets:
            print(f"Error: dataset_idx {dataset_idx} out of range [0, {total_datasets-1}]")
            sys.exit(1)
        all_datasets = [all_datasets[dataset_idx]]
        parallel_mode = True
    else:
        parallel_mode = False
    
    # Configure logging
    if parallel_mode:
        dataset_name, task = all_datasets[0]
        # Store parallel logs in logs subfolder
        log_file = experiment_path / "logs" / f"{experiment_name}_{task}_{dataset_name}.log"
    else:
        # Store sequential log in main experiment folder
        log_file = experiment_path / f"{experiment_name}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("EXPERIMENT 1: HPO Benchmark")
    if parallel_mode:
        logger.info(f"PARALLEL MODE: Running dataset {dataset_idx + 1}/{total_datasets}")
    logger.info("="*80)
    
    logger.info(f"Results will be saved to: {experiment_path}")
    logger.info(f"Log file: {log_file}")
    
    # Save experiment metadata (only in sequential mode or for first dataset)
    if not parallel_mode or dataset_idx == 0:
        metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "skip_completed": skip_completed,
            "total_datasets": total_datasets,
            "parallel_mode": parallel_mode,
        }
        storage.save_experiment_metadata(metadata)
    
    logger.info(f"Processing {len(all_datasets)} dataset(s)")
    
    # Track results
    completed_datasets = []
    skipped_datasets = []
    failed_datasets = []
    
    # Extract common experiment parameters
    experiment_params = {
        'test_size': config['split']['test_size'],
        'val_size': config['split']['val_size'],
        'cv_splits': config['split']['cv_splits'],
        'seed': config['split']['seed'],
        'row_limit': config['split'].get('row_limit', None),
        'sampling': config['split'].get('sampling', None),
        'max_epoch': config['training']['max_epochs'],
        'batch_size': config['training']['batch_size'],
        'early_stopping': config['training']['early_stopping'],
        'early_stopping_patience': config['training']['early_stopping_patience'],
        'n_trials': config['tuning']['n_trials'],
        'config_base_dir': experiment_path,
        'verbose': verbose,
    }
    
    # Process datasets
    for idx, (dataset, task) in enumerate(all_datasets):
        global_idx = dataset_idx if parallel_mode else idx
        
        logger.info("\n" + "="*80)
        logger.info(f"Dataset {global_idx + 1}/{total_datasets}: {dataset} ({task.upper()})")
        logger.info("="*80)
        
        # Check if already completed
        if skip_completed and storage.dataset_exists(dataset, task=task):
            logger.info(f"[SKIP] Results already exist for {dataset}")
            skipped_datasets.append(f"{task}/{dataset}")
            continue
        
        try:
            logger.info("Running HPO comparison...")
            
            # Run HPO comparison
            results = run_hpo_comparison(
                task=task,
                dataset=dataset,
                **experiment_params
            )
            
            # Save results
            dataset_metadata = {
                "task": task,
                "dataset": dataset,
                "timestamp": datetime.now().isoformat(),
                "n_methods_no_hpo": len(results['NO_HPO']),
                "n_methods_hpo": len(results['HPO']),
                "methods": list(results['NO_HPO'].keys()),
            }
            
            storage.save_dataset_results(
                dataset=dataset,
                results=results,
                metadata=dataset_metadata,
                task=task,
                overwrite=True
            )
            
            completed_datasets.append(f"{task}/{dataset}")
            logger.info(f"[DONE] Completed: {dataset}")
            
        except Exception as e:
            logger.error(f"[FAIL] Failed: {dataset}", exc_info=True)
            failed_datasets.append((f"{task}/{dataset}", str(e)))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("RUN COMPLETE")
    logger.info("="*80)
    logger.info(f"Completed: {len(completed_datasets)}")
    logger.info(f"Skipped: {len(skipped_datasets)}")
    logger.info(f"Failed: {len(failed_datasets)}")
    
    if failed_datasets:
        logger.info("\nFailed datasets:")
        for ds, error in failed_datasets:
            logger.info(f"  [X] {ds}: {error}")
    
    logger.info(f"\nResults saved to: {experiment_path}")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 1: HPO Benchmark")
    parser.add_argument(
        '--dataset_idx', 
        type=int, 
        default=None,
        help='Dataset index for SLURM array jobs (0-based). If not specified, runs all datasets sequentially.'
    )
    parser.add_argument(
        '--no_skip',
        action='store_true',
        help='Re-run all datasets even if results exist'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce verbosity'
    )
    
    args = parser.parse_args()
    
    run_experiment1(
        experiment_name="experiment1",
        skip_completed=not args.no_skip,
        verbose=not args.quiet,
        dataset_idx=args.dataset_idx
    )