# src/methods/method_debugger.py
"""
Method Debugger: Test ALL methods on sample datasets.

Runs every method (regardless of config enabled status) on the first PD 
and first LGD dataset with reduced settings for quick debugging.

Usage:
    python src/methods/method_debugger.py
    python src/methods/method_debugger.py --quiet
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging
import time
import json
import yaml

# Setup paths - from src/methods/ go up two levels to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.utils.storage_handler import StorageHandler
from src.methods.method_runner import run_talent_method


def run_method_debugger(
    experiment_name: str = "method_debugger",
    verbose: bool = True
) -> None:
    """
    Run ALL methods on first PD and LGD datasets for debugging.
    """
    
    # Debug settings - fast runs
    DEBUG_SETTINGS = {
        'cv_splits': 1,
        'row_limit': 500,
        'n_trials': 5,
        'max_epoch': 20,
    }
    
    storage = StorageHandler(experiment_name)
    experiment_path = storage.get_experiment_path()
    
    # Ensure directories exist
    experiment_path.mkdir(parents=True, exist_ok=True)
    (experiment_path / "pd").mkdir(exist_ok=True)
    (experiment_path / "lgd").mkdir(exist_ok=True)
    
    # Load config
    config = load_config()
    
    # Load ALL methods directly from config file (bypass filtering)
    config_dir = PROJECT_ROOT / "config"
    with open(config_dir / "CONFIG_METHOD.yaml") as f:
        method_config = yaml.safe_load(f)
    
    # Get ALL methods (regardless of enabled status)
    pd_methods = list(method_config["methods"]["pd"].keys())
    lgd_methods = list(method_config["methods"]["lgd"].keys())
    
    # Get first datasets
    pd_datasets = list(config['datasets']['pd'].keys())
    lgd_datasets = list(config['datasets']['lgd'].keys())
    
    if not pd_datasets or not lgd_datasets:
        print("ERROR: No datasets in config!")
        return
    
    first_pd = pd_datasets[0]
    first_lgd = lgd_datasets[0]
    
    # Configure logging - logs go to file, minimal console output
    log_file = experiment_path / f"{experiment_name}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler() if verbose else logging.NullHandler()
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    
    # Suppress noisy loggers from method_runner internals
    logging.getLogger('TALENT').setLevel(logging.WARNING)
    
    logger.info("="*70)
    logger.info("METHOD DEBUGGER: Testing ALL methods")
    logger.info("="*70)
    logger.info(f"PD dataset: {first_pd} | LGD dataset: {first_lgd}")
    logger.info(f"PD methods: {len(pd_methods)} | LGD methods: {len(lgd_methods)}")
    logger.info(f"Settings: cv={DEBUG_SETTINGS['cv_splits']}, rows={DEBUG_SETTINGS['row_limit']}, epochs={DEBUG_SETTINGS['max_epoch']}")
    logger.info("="*70)
    
    # Console header (minimal)
    print(f"Method Debugger: {len(pd_methods)} PD + {len(lgd_methods)} LGD methods")
    print(f"Logs: {log_file}\n")
    
    # Common parameters - includes config_base_dir for storing HPO configs
    common_params = {
        'test_size': config['split']['test_size'],
        'val_size': config['split']['val_size'],
        'cv_splits': DEBUG_SETTINGS['cv_splits'],
        'seed': config['split']['seed'],
        'row_limit': DEBUG_SETTINGS['row_limit'],
        'sampling': config['split'].get('sampling', None),
        'max_epoch': DEBUG_SETTINGS['max_epoch'],
        'batch_size': config['training']['batch_size'],
        'early_stopping': config['training']['early_stopping'],
        'early_stopping_patience': config['training']['early_stopping_patience'],
        'n_trials': DEBUG_SETTINGS['n_trials'],
        'config_base_dir': experiment_path,
        'verbose': False,
    }
    
    # Track results
    results = {'pd': {'success': [], 'failed': []}, 'lgd': {'success': [], 'failed': []}}
    
    # ==========================================================================
    # Test PD methods
    # ==========================================================================
    logger.info(f"\nTESTING PD METHODS on {first_pd}")
    print(f"PD methods on {first_pd}:")
    
    for i, method in enumerate(pd_methods):
        start_time = time.time()
        
        try:
            method_results = run_talent_method(
                task='pd', dataset=first_pd, method=method, tune=False, **common_params
            )
            elapsed = time.time() - start_time
            
            if method_results and 'metrics' in method_results:
                auc = method_results['metrics'].get('AUC', method_results['metrics'].get('auc', 'N/A'))
                logger.info(f"[{i+1}/{len(pd_methods)}] ✓ {method} ({elapsed:.1f}s) AUC={auc}")
                print(f"  [{i+1:2}/{len(pd_methods)}] ✓ {method}")
                results['pd']['success'].append((method, elapsed, auc))
            else:
                logger.info(f"[{i+1}/{len(pd_methods)}] ✓ {method} ({elapsed:.1f}s)")
                print(f"  [{i+1:2}/{len(pd_methods)}] ✓ {method}")
                results['pd']['success'].append((method, elapsed, 'N/A'))
                
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)[:100]
            logger.error(f"[{i+1}/{len(pd_methods)}] ✗ {method} ({elapsed:.1f}s) {error_msg}")
            print(f"  [{i+1:2}/{len(pd_methods)}] ✗ {method}")
            results['pd']['failed'].append((method, elapsed, error_msg))
    
    # ==========================================================================
    # Test LGD methods
    # ==========================================================================
    logger.info(f"\nTESTING LGD METHODS on {first_lgd}")
    print(f"\nLGD methods on {first_lgd}:")
    
    for i, method in enumerate(lgd_methods):
        start_time = time.time()
        
        try:
            method_results = run_talent_method(
                task='lgd', dataset=first_lgd, method=method, tune=False, **common_params
            )
            elapsed = time.time() - start_time
            
            if method_results and 'metrics' in method_results:
                rmse = method_results['metrics'].get('RMSE', method_results['metrics'].get('rmse', 'N/A'))
                logger.info(f"[{i+1}/{len(lgd_methods)}] ✓ {method} ({elapsed:.1f}s) RMSE={rmse}")
                print(f"  [{i+1:2}/{len(lgd_methods)}] ✓ {method}")
                results['lgd']['success'].append((method, elapsed, rmse))
            else:
                logger.info(f"[{i+1}/{len(lgd_methods)}] ✓ {method} ({elapsed:.1f}s)")
                print(f"  [{i+1:2}/{len(lgd_methods)}] ✓ {method}")
                results['lgd']['success'].append((method, elapsed, 'N/A'))
                
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)[:100]
            logger.error(f"[{i+1}/{len(lgd_methods)}] ✗ {method} ({elapsed:.1f}s) {error_msg}")
            print(f"  [{i+1:2}/{len(lgd_methods)}] ✗ {method}")
            results['lgd']['failed'].append((method, elapsed, error_msg))
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"PD:  {len(results['pd']['success'])}/{len(pd_methods)} success, {len(results['pd']['failed'])} failed")
    logger.info(f"LGD: {len(results['lgd']['success'])}/{len(lgd_methods)} success, {len(results['lgd']['failed'])} failed")
    
    if results['pd']['failed']:
        logger.info("\nFailed PD methods:")
        for method, _, error in results['pd']['failed']:
            logger.info(f"  ✗ {method}: {error}")
    
    if results['lgd']['failed']:
        logger.info("\nFailed LGD methods:")
        for method, _, error in results['lgd']['failed']:
            logger.info(f"  ✗ {method}: {error}")
    
    # Console summary
    print(f"\nSummary: PD {len(results['pd']['success'])}/{len(pd_methods)}, LGD {len(results['lgd']['success'])}/{len(lgd_methods)}")
    if results['pd']['failed'] or results['lgd']['failed']:
        print(f"Failed: {[m for m,_,_ in results['pd']['failed']]} {[m for m,_,_ in results['lgd']['failed']]}")
    
    # Save summary
    summary = {
        "end_time": datetime.now().isoformat(),
        "debug_settings": DEBUG_SETTINGS,
        "results": {
            'pd': {
                'success': [(m, round(t, 1)) for m, t, _ in results['pd']['success']],
                'failed': [(m, e) for m, _, e in results['pd']['failed']],
            },
            'lgd': {
                'success': [(m, round(t, 1)) for m, t, _ in results['lgd']['success']],
                'failed': [(m, e) for m, _, e in results['lgd']['failed']],
            }
        }
    }
    
    summary_file = experiment_path / "debug_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to: {experiment_path}")
    print(f"Full log: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug ALL methods on sample datasets")
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    args = parser.parse_args()
    
    run_method_debugger(experiment_name="method_debugger", verbose=not args.quiet)