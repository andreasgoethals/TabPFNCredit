# Experiment1.py
"""
Experiment 1: HPO Benchmark across all enabled datasets.

This experiment runs all enabled methods on all enabled datasets,
comparing performance with and without hyperparameter optimization.

Results are saved per dataset in: results/experiment1/
File naming: {task}_{dataset}.pkl (e.g., pd_0001.gmsc.pkl, lgd_0001.heloc.pkl)
"""

import sys
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.utils.storage_handler import StorageHandler
from src.methods.HPO_runner import run_hpo_comparison


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
    
    print("="*80)
    print("EXPERIMENT 1: HPO Benchmark")
    print("="*80)
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    
    # Initialize storage handler
    storage = StorageHandler(experiment_name)
    print(f"Results will be saved to: {storage.get_experiment_path()}")
    
    # Save experiment metadata
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
    print(f"Found {len(pd_datasets)} PD datasets and {len(lgd_datasets)} LGD datasets")
    print(f"Total datasets to process: {total_datasets}")
    
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
        'verbose': verbose,
    }
    
    # Process all datasets
    dataset_counter = 0
    
    # Process PD datasets
    for dataset in pd_datasets:
        dataset_counter += 1
        dataset_filename = f"pd_{dataset}"
        
        print("\n" + "="*80)
        print(f"Dataset {dataset_counter}/{total_datasets}: {dataset} (PD)")
        print("="*80)
        
        # Check if already completed
        if skip_completed and storage.is_completed(dataset_filename):
            print(f"✓ Already completed, skipping...")
            skipped_datasets.append(dataset_filename)
            continue
        
        try:
            # Run HPO comparison
            print(f"Running HPO comparison...")
            results = run_hpo_comparison(
                task='pd',
                dataset=dataset,
                **common_params
            )
            
            # Save results
            dataset_metadata = {
                "task": "pd",
                "dataset": dataset,
                "timestamp": datetime.now().isoformat(),
                "n_methods_no_hpo": len(results['NO_HPO']),
                "n_methods_hpo": len(results['HPO']),
                "methods": list(results['NO_HPO'].keys()),
            }
            
            storage.save_dataset_results(
                dataset=dataset_filename,
                results=results,
                metadata=dataset_metadata,
                overwrite=True
            )
            
            completed_datasets.append(dataset_filename)
            print(f"✓ Completed and saved: {dataset_filename}")
            
        except Exception as e:
            print(f"✗ Failed: {dataset} - {str(e)}")
            failed_datasets.append((dataset_filename, str(e)))
    
    # Process LGD datasets
    for dataset in lgd_datasets:
        dataset_counter += 1
        dataset_filename = f"lgd_{dataset}"
        
        print("\n" + "="*80)
        print(f"Dataset {dataset_counter}/{total_datasets}: {dataset} (LGD)")
        print("="*80)
        
        # Check if already completed
        if skip_completed and storage.is_completed(dataset_filename):
            print(f"✓ Already completed, skipping...")
            skipped_datasets.append(dataset_filename)
            continue
        
        try:
            # Run HPO comparison
            print(f"Running HPO comparison...")
            results = run_hpo_comparison(
                task='lgd',
                dataset=dataset,
                **common_params
            )
            
            # Save results
            dataset_metadata = {
                "task": "lgd",
                "dataset": dataset,
                "timestamp": datetime.now().isoformat(),
                "n_methods_no_hpo": len(results['NO_HPO']),
                "n_methods_hpo": len(results['HPO']),
                "methods": list(results['NO_HPO'].keys()),
            }
            
            storage.save_dataset_results(
                dataset=dataset_filename,
                results=results,
                metadata=dataset_metadata,
                overwrite=True
            )
            
            completed_datasets.append(dataset_filename)
            print(f"✓ Completed and saved: {dataset_filename}")
            
        except Exception as e:
            print(f"✗ Failed: {dataset} - {str(e)}")
            failed_datasets.append((dataset_filename, str(e)))
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT 1 COMPLETE")
    print("="*80)
    print(f"Total datasets: {total_datasets}")
    print(f"Completed: {len(completed_datasets)}")
    print(f"Skipped: {len(skipped_datasets)}")
    print(f"Failed: {len(failed_datasets)}")
    
    if completed_datasets:
        print(f"\nCompleted datasets:")
        for ds in completed_datasets:
            print(f"  ✓ {ds}")
    
    if skipped_datasets:
        print(f"\nSkipped datasets:")
        for ds in skipped_datasets:
            print(f"  - {ds}")
    
    if failed_datasets:
        print(f"\nFailed datasets:")
        for ds, error in failed_datasets:
            print(f"  ✗ {ds}: {error}")
    
    print(f"\nResults saved to: {storage.get_experiment_path()}")
    print("="*80)


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