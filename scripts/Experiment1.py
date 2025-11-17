# Experiment1.py
"""
Experiment 1: HPO Benchmark across all enabled datasets.

This experiment runs all enabled methods on all enabled datasets,
comparing performance with and without hyperparameter optimization.

Results structure:
- results/experiment1/pd/         <- PD results
    - metadata.json
    - dataset1.pkl
    - dataset2.pkl
- results/experiment1/lgd/        <- LGD results
    - metadata.json
    - dataset1.pkl
    - dataset2.pkl
- results/experiment1/config_hpo/ <- HPO configs
- results/experiment1/experiment1.log

If an experiment1 folder already exists with actual results, it will be 
automatically archived with a timestamp before starting the new run.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Goes to TabPFNCredit/
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.utils.storage_handler import StorageHandler
from src.utils.storage_archiver import StorageArchiver
from src.methods.HPO_runner import run_hpo_comparison


def _has_results(experiment_path: Path) -> bool:
    """
    Check if experiment folder has actual results (not just empty directories).
    
    Args:
        experiment_path: Path to experiment folder
        
    Returns:
        True if folder has .pkl files or metadata.json files
    """
    if not experiment_path.exists():
        return False
    
    # Check for any .pkl files or metadata.json files
    has_pkl = any(experiment_path.rglob("*.pkl"))
    has_metadata = any(experiment_path.rglob("metadata.json"))
    
    return has_pkl or has_metadata


def run_experiment1(
    experiment_name: str = "experiment1",
    skip_completed: bool = True,
    verbose: bool = True
) -> None:
    """
    Run Experiment 1: HPO comparison across all enabled datasets.
    
    Args:
        experiment_name: Name of experiment (folder name in results/)
        skip_completed: If True, skip datasets with existing results
        verbose: Whether to print detailed progress
    """
    
    # Initialize storage handler (but don't create archiver yet to avoid creating archive dir)
    storage = StorageHandler(experiment_name)
    experiment_path = storage.get_experiment_path()
    
    # Archive existing experiment if it exists AND has actual results
    if _has_results(experiment_path):
        print(f"Found existing experiment with results: {experiment_path}")
        print("Archiving old results before starting new run...")
        archiver = StorageArchiver()  # Only create archiver if we need it
        archive_path = archiver.archive_experiment(experiment_name)
        print(f"✓ Archived to: {archive_path}\n")
    
    # Ensure fresh experiment directory exists
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging to save inside experiment directory
    log_file = experiment_path / f"{experiment_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration if logging was already configured
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
    logger.info(f"Log file: {log_file}")
    logger.info(f"HPO configs will be saved to: {experiment_path / 'config_hpo'}")
    
    # Create task-specific directories
    (experiment_path / "pd").mkdir(exist_ok=True)
    (experiment_path / "lgd").mkdir(exist_ok=True)
    
    # Save experiment metadata in root
    experiment_metadata = {
        "description": "HPO comparison across all enabled datasets",
        "config": config,
        "start_time": datetime.now().isoformat(),
    }
    storage.save_experiment_metadata(experiment_metadata)
    
    # Get enabled datasets for both tasks
    pd_datasets = list(config['datasets']['pd'].keys())
    lgd_datasets = list(config['datasets']['lgd'].keys())
    
    total_datasets = len(pd_datasets) + len(lgd_datasets)
    logger.info(f"Found {len(pd_datasets)} PD datasets and {len(lgd_datasets)} LGD datasets")
    logger.info(f"Total datasets to process: {total_datasets}")
    
    # Track results
    completed_datasets = []
    failed_datasets = []
    skipped_datasets = []
    
    # Common parameters from config
    common_params = {
        'test_size': config['split']['test_size'],
        'val_size': config['split']['val_size'],
        'cv_splits': config['split']['cv_splits'],
        'seed': config['split']['seed'],
        'row_limit': config['split'].get('row_limit'),
        'max_epoch': config['training']['max_epochs'],
        'batch_size': config['training']['batch_size'],
        'n_trials': config['tuning']['n_trials'],
        'early_stopping': config['training']['early_stopping'],
        'early_stopping_patience': config['training']['early_stopping_patience'],
        'config_base_dir': experiment_path,  # HPO configs saved inside experiment folder
        'verbose': verbose,
    }
    
    # Process all datasets
    dataset_counter = 0
    
    # Process PD datasets
    for dataset in pd_datasets:
        dataset_counter += 1
        dataset_filename = f"pd/{dataset}"  # Save to pd/ subfolder
        
        logger.info("\n" + "="*80)
        logger.info(f"Dataset {dataset_counter}/{total_datasets}: {dataset} (PD)")
        logger.info("="*80)
        
        # Check if already completed
        if skip_completed and storage.is_completed(dataset_filename):
            logger.info(f"✓ Already completed, skipping...")
            skipped_datasets.append(dataset_filename)
            continue
        
        try:
            # Run HPO comparison
            logger.info(f"Running HPO comparison...")
            results = run_hpo_comparison(
                task='pd',
                dataset=dataset,
                **common_params
            )
            
            # Save results in pd/ subfolder
            dataset_metadata = {
                "task": "pd",
                "dataset": dataset,
                "timestamp": datetime.now().isoformat(),
                "n_methods_no_hpo": len(results['NO_HPO']),
                "n_methods_hpo": len(results['HPO']),
                "methods": list(results['NO_HPO'].keys()),
            }
            
            storage.save_dataset_results(
                dataset=dataset_filename,  # Will save to pd/dataset.pkl
                results=results,
                metadata=dataset_metadata,
                overwrite=True
            )
            
            completed_datasets.append(dataset_filename)
            logger.info(f"✓ Completed and saved: {dataset_filename}")
            
        except Exception as e:
            logger.error(f"✗ Failed: {dataset} - {str(e)}", exc_info=True)
            failed_datasets.append((dataset_filename, str(e)))
    
    # Process LGD datasets
    for dataset in lgd_datasets:
        dataset_counter += 1
        dataset_filename = f"lgd/{dataset}"  # Save to lgd/ subfolder
        
        logger.info("\n" + "="*80)
        logger.info(f"Dataset {dataset_counter}/{total_datasets}: {dataset} (LGD)")
        logger.info("="*80)
        
        # Check if already completed
        if skip_completed and storage.is_completed(dataset_filename):
            logger.info(f"✓ Already completed, skipping...")
            skipped_datasets.append(dataset_filename)
            continue
        
        try:
            # Run HPO comparison
            logger.info(f"Running HPO comparison...")
            results = run_hpo_comparison(
                task='lgd',
                dataset=dataset,
                **common_params
            )
            
            # Save results in lgd/ subfolder
            dataset_metadata = {
                "task": "lgd",
                "dataset": dataset,
                "timestamp": datetime.now().isoformat(),
                "n_methods_no_hpo": len(results['NO_HPO']),
                "n_methods_hpo": len(results['HPO']),
                "methods": list(results['NO_HPO'].keys()),
            }
            
            storage.save_dataset_results(
                dataset=dataset_filename,  # Will save to lgd/dataset.pkl
                results=results,
                metadata=dataset_metadata,
                overwrite=True
            )
            
            completed_datasets.append(dataset_filename)
            logger.info(f"✓ Completed and saved: {dataset_filename}")
            
        except Exception as e:
            logger.error(f"✗ Failed: {dataset} - {str(e)}", exc_info=True)
            failed_datasets.append((dataset_filename, str(e)))
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 1 COMPLETE")
    logger.info("="*80)
    logger.info(f"Total datasets: {total_datasets}")
    logger.info(f"Completed: {len(completed_datasets)}")
    logger.info(f"Skipped: {len(skipped_datasets)}")
    logger.info(f"Failed: {len(failed_datasets)}")
    
    if completed_datasets:
        logger.info(f"\nCompleted datasets:")
        for ds in completed_datasets:
            logger.info(f"  ✓ {ds}")
    
    if skipped_datasets:
        logger.info(f"\nSkipped datasets:")
        for ds in skipped_datasets:
            logger.info(f"  - {ds}")
    
    if failed_datasets:
        logger.info(f"\nFailed datasets:")
        for ds, error in failed_datasets:
            logger.info(f"  ✗ {ds}: {error}")
    
    logger.info(f"\nResults saved to: {experiment_path}")
    logger.info(f"HPO configs saved to: {experiment_path / 'config_hpo'}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)


if __name__ == "__main__":
    """
    Run Experiment 1 with settings from config files.
    
    To run:
        python Experiment1.py
    
    To run with verbose output:
        python Experiment1.py --verbose
    
    To force re-run all datasets:
        python Experiment1.py --no-skip
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Experiment 1: HPO Benchmark")
    parser.add_argument(
        "--name",
        type=str,
        default="experiment1",
        help="Experiment name (default: experiment1)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-run all datasets even if results exist"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    run_experiment1(
        experiment_name=args.name,
        skip_completed=not args.no_skip,
        verbose=args.verbose
    )