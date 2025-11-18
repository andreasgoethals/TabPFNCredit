# scripts/Experiment1.py
"""
Experiment 1: HPO Benchmark across all enabled datasets.

This experiment runs all enabled methods on all enabled datasets,
comparing performance with and without hyperparameter optimization.

Results structure:
- results/experiment1/pd/{dataset}.pkl
- results/experiment1/lgd/{dataset}.pkl
- results/experiment1/config_hpo/{task}/{dataset}/{method}-tuned.json
- results/experiment1/experiment1.log
"""

import sys
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
    """
    Check if experiment folder has actual results.
    
    Args:
        experiment_path: Path to experiment folder
        
    Returns:
        True if folder has .pkl files or metadata
    """
    if not experiment_path.exists():
        return False
    
    has_pkl = any(experiment_path.rglob("*.pkl"))
    has_metadata = any(experiment_path.rglob("*metadata.json"))
    
    return has_pkl or has_metadata


def _delete_experiment_folder(experiment_path: Path) -> bool:
    """
    Delete experiment folder completely.
    
    Args:
        experiment_path: Path to experiment folder
        
    Returns:
        True if successful, False if failed (OneDrive lock, etc.)
    """
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
    verbose: bool = True
) -> None:
    """
    Run Experiment 1: HPO comparison across all enabled datasets.
    
    This experiment:
    1. Loads enabled datasets from CONFIG_DATA.yaml
    2. Loads enabled methods from CONFIG_METHOD.yaml
    3. Runs each method twice: once with default params (NO_HPO) and once with HPO
    4. Saves results as {dataset}.pkl in results/experiment1/{task}/
    5. Saves tuned configs to results/experiment1/config_hpo/{task}/{dataset}/
    
    Args:
        experiment_name: Name of experiment (folder name in results/)
        skip_completed: If True, skip datasets with existing results.
                    If False, delete old results and re-run everything.
        verbose: Whether to print detailed progress during training
    """
    
    storage = StorageHandler(experiment_name)
    experiment_path = storage.get_experiment_path()
    
    # Handle existing results
    if _has_results(experiment_path):
        if skip_completed:
            print(f"Existing results found: {experiment_path}")
            print("Will skip completed datasets (set skip_completed=False to re-run)\n")
        else:
            print(f"Existing results found: {experiment_path}")
            print("Deleting old results to start fresh...\n")
            
            if _delete_experiment_folder(experiment_path):
                print("Old results deleted successfully\n")
            else:
                print("\nWARNING: Could not fully delete old results")
                print("This is usually caused by:")
                print("  - OneDrive sync locks")
                print("  - File Explorer having the folder open")
                print("  - Log files still being written")
                print("\nPlease:")
                print("  1. Close File Explorer")
                print("  2. Wait a few seconds for OneDrive to sync")
                print("  3. Try running again")
                print("\nOr manually delete: " + str(experiment_path))
                return
    
    # Ensure experiment directory exists
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging with UTF-8 encoding
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
    logger.info("="*80)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()
    
    logger.info(f"Results will be saved to: {experiment_path}")
    logger.info(f"  PD results: {experiment_path / 'pd'}")
    logger.info(f"  LGD results: {experiment_path / 'lgd'}")
    logger.info(f"  HPO configs: {experiment_path / 'config_hpo'}")
    logger.info(f"  Log file: {log_file}")
    
    # Create task-specific directories
    (experiment_path / "pd").mkdir(exist_ok=True)
    (experiment_path / "lgd").mkdir(exist_ok=True)
    
    # Save experiment metadata
    metadata = {
        "experiment_name": experiment_name,
        "start_time": datetime.now().isoformat(),
        "skip_completed": skip_completed,
        "config": config,
    }
    storage.save_experiment_metadata(metadata)
    
    # Count total datasets
    pd_datasets = list(config['datasets']['pd'].keys())
    lgd_datasets = list(config['datasets']['lgd'].keys())
    total_datasets = len(pd_datasets) + len(lgd_datasets)
    
    logger.info(f"Found {len(pd_datasets)} PD datasets and {len(lgd_datasets)} LGD datasets")
    logger.info(f"Total datasets to process: {total_datasets}")
    
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
    
    # ==================================================================================
    # Process PD Datasets
    # ==================================================================================
    for idx, dataset in enumerate(pd_datasets, 1):
        logger.info("\n" + "="*80)
        logger.info(f"Dataset {idx}/{total_datasets}: {dataset} (PD)")
        logger.info("="*80)
        
        # Check if already completed
        if skip_completed and storage.dataset_exists(dataset, task='pd'):
            logger.info(f"[SKIP] Results already exist for {dataset}")
            skipped_datasets.append(f"pd/{dataset}")
            continue
        
        try:
            logger.info("Running HPO comparison...")
            
            # Run HPO comparison
            results = run_hpo_comparison(
                task='pd',
                dataset=dataset,
                **experiment_params
            )
            
            # Save results WITH task='pd' parameter
            dataset_metadata = {
                "task": "pd",
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
                task='pd',  # ← THIS IS THE FIX!
                overwrite=True
            )
            
            completed_datasets.append(f"pd/{dataset}")
            logger.info(f"[DONE] Completed: {dataset}")
            
        except Exception as e:
            logger.error(f"[FAIL] Failed: {dataset}", exc_info=True)
            failed_datasets.append((f"pd/{dataset}", str(e)))
    
    # ==================================================================================
    # Process LGD Datasets
    # ==================================================================================
    current_idx = len(pd_datasets)
    for idx, dataset in enumerate(lgd_datasets, current_idx + 1):
        logger.info("\n" + "="*80)
        logger.info(f"Dataset {idx}/{total_datasets}: {dataset} (LGD)")
        logger.info("="*80)
        
        # Check if already completed
        if skip_completed and storage.dataset_exists(dataset, task='lgd'):
            logger.info(f"[SKIP] Results already exist for {dataset}")
            skipped_datasets.append(f"lgd/{dataset}")
            continue
        
        try:
            logger.info("Running HPO comparison...")
            
            # Run HPO comparison
            results = run_hpo_comparison(
                task='lgd',
                dataset=dataset,
                **experiment_params
            )
            
            # Save results WITH task='lgd' parameter
            dataset_metadata = {
                "task": "lgd",
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
                task='lgd',  # ← THIS IS THE FIX!
                overwrite=True
            )
            
            completed_datasets.append(f"lgd/{dataset}")
            logger.info(f"[DONE] Completed: {dataset}")
            
        except Exception as e:
            logger.error(f"[FAIL] Failed: {dataset}", exc_info=True)
            failed_datasets.append((f"lgd/{dataset}", str(e)))
    
    # ==================================================================================
    # Final Summary
    # ==================================================================================
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 1 COMPLETE")
    logger.info("="*80)
    logger.info(f"Total datasets: {total_datasets}")
    logger.info(f"Completed: {len(completed_datasets)}")
    logger.info(f"Skipped: {len(skipped_datasets)}")
    logger.info(f"Failed: {len(failed_datasets)}")
    
    if completed_datasets:
        logger.info("\nCompleted datasets:")
        for ds in completed_datasets:
            logger.info(f"  [OK] {ds}")
    
    if skipped_datasets:
        logger.info("\nSkipped datasets:")
        for ds in skipped_datasets:
            logger.info(f"  [-] {ds}")
    
    if failed_datasets:
        logger.info("\nFailed datasets:")
        for ds, error in failed_datasets:
            logger.info(f"  [X] {ds}: {error}")
    
    logger.info(f"\nResults saved to: {experiment_path}")
    logger.info(f"HPO configs saved to: {experiment_path / 'config_hpo'}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)


if __name__ == "__main__":
    """
    Run Experiment 1 with settings from config files.
    
    Usage:
        python scripts/Experiment1.py
    """
    run_experiment1(
        experiment_name="experiment1",
        skip_completed=True,  # Set to False to delete and re-run everything
        verbose=True
    )