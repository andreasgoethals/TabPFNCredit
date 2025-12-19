"""
Method Debugger: Test ALL methods using Experiment1 configuration.

Features:
- Loads config specifically from scripts/Experiment1/config
- Cross-platform file locking (Linux/Windows compatible)
- Tests all methods defined in Experiment1's CONFIG_METHOD.yaml
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging
import time
import json
import yaml
import os

# Handle platform-specific locking
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False  # Windows

# Setup paths - from src/methods/ go up two levels to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import updated config reader
from src.utils.config_reader import load_config, get_config_dir
from src.utils.storage_handler import StorageHandler
from src.methods.method_runner import run_talent_method

def save_summary_with_lock(filepath: Path, data: dict) -> None:
    """Save JSON with file locking (Linux) or direct write (Windows)."""
    try:
        with open(filepath, 'w') as f:
            if HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                if HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Warning: Failed to save summary file: {e}")

def run_method_debugger(
    target_experiment: str = "Experiment1",
    verbose: bool = True
) -> None:
    """
    Run ALL methods defined in the target experiment's config.
    """
    
    # Debug settings - fast runs
    DEBUG_SETTINGS = {
        'cv_splits': 2,
        'row_limit': 500,
        'n_trials': 0,
        'max_epoch': 5,
    }
    
    storage = StorageHandler("method_debugger")
    experiment_path = storage.get_experiment_path()
    
    # Ensure directories exist
    experiment_path.mkdir(parents=True, exist_ok=True)
    (experiment_path / "pd").mkdir(exist_ok=True)
    (experiment_path / "lgd").mkdir(exist_ok=True)
    
    # 1. Load the specific experiment config via dispatcher
    print(f"Loading configuration from: {target_experiment}")
    try:
        config = load_config(target_experiment)
        # We also need the raw directory to read the full method list
        config_dir = get_config_dir(target_experiment)
    except Exception as e:
        print(f"CRITICAL ERROR loading config: {e}")
        return

    # 2. Load ALL methods directly from that experiment's YAML (bypass enabled=False)
    # Note: This logic assumes Experiment1 structure. If Experiment0 is different,
    # this debugger script might need its own dispatcher logic later.
    try:
        with open(config_dir / "CONFIG_METHOD.yaml") as f:
            method_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: CONFIG_METHOD.yaml not found in {config_dir}")
        return
    
    pd_methods = list(method_config["methods"]["pd"].keys())
    lgd_methods = list(method_config["methods"]["lgd"].keys())
    
    # Get first datasets from the loaded config
    pd_datasets = list(config['datasets']['pd'].keys())
    lgd_datasets = list(config['datasets']['lgd'].keys())
    
    if not pd_datasets or not lgd_datasets:
        print("ERROR: No datasets in config!")
        return
    
    first_pd = pd_datasets[0]
    first_lgd = lgd_datasets[0]
    
    # Logging setup
    log_file = experiment_path / "method_debugger.log"
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
    logging.getLogger('TALENT').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger.info("="*70)
    logger.info(f"METHOD DEBUGGER: {target_experiment}")
    logger.info("="*70)
    logger.info(f"System: {'Linux (Locking Enabled)' if HAS_FCNTL else 'Windows (Locking Disabled)'}")
    
    # Console header (minimal)
    print(f"Method Debugger: {len(pd_methods)} PD + {len(lgd_methods)} LGD methods")
    print(f"System: {'Linux (Locking Enabled)' if HAS_FCNTL else 'Windows (Locking Disabled)'}")
    print(f"Logs: {log_file}\n")
    
    # Common parameters derived from the loaded config
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
        'early_stopping_patience': 1,
        'n_trials': 0,
        'config_base_dir': experiment_path,
        'verbose': False,
    }
    
    # Track results
    results = {'pd': {'success': [], 'failed': []}, 'lgd': {'success': [], 'failed': []}}
    summary_file = experiment_path / "debug_summary.json"
    
    def update_summary():
        summary = {
            "last_update": datetime.now().isoformat(),
            "target_experiment": target_experiment,
            "results": {
                'pd': {
                    'success': [(m, round(t, 1), sc) for m, t, sc in results['pd']['success']],
                    'failed': [(m, e) for m, _, e in results['pd']['failed']],
                },
                'lgd': {
                    'success': [(m, round(t, 1), sc) for m, t, sc in results['lgd']['success']],
                    'failed': [(m, e) for m, _, e in results['lgd']['failed']],
                }
            }
        }
        save_summary_with_lock(summary_file, summary)

    # --- PD Loop ---
    logger.info(f"TESTING PD METHODS on {first_pd}")
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
                if isinstance(auc, float): auc = round(auc, 4)
                logger.info(f"[{i+1}/{len(pd_methods)}] ✓ {method} ({elapsed:.1f}s) AUC={auc}")
                print(f"  [{i+1:2}/{len(pd_methods)}] ✓ {method} (AUC={auc})")
                results['pd']['success'].append((method, elapsed, auc))
            else:
                results['pd']['success'].append((method, elapsed, 'N/A'))
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{i+1}/{len(pd_methods)}] ✗ {method} {str(e)}")
            print(f"  [{i+1:2}/{len(pd_methods)}] ✗ {method}")
            results['pd']['failed'].append((method, elapsed, str(e)))
        update_summary()

    # --- LGD Loop ---
    logger.info(f"TESTING LGD METHODS on {first_lgd}")
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
                if isinstance(rmse, float): rmse = round(rmse, 4)
                logger.info(f"[{i+1}/{len(lgd_methods)}] ✓ {method} ({elapsed:.1f}s) RMSE={rmse}")
                print(f"  [{i+1:2}/{len(lgd_methods)}] ✓ {method} (RMSE={rmse})")
                results['lgd']['success'].append((method, elapsed, rmse))
            else:
                results['lgd']['success'].append((method, elapsed, 'N/A'))
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{i+1}/{len(lgd_methods)}] ✗ {method} {str(e)}")
            print(f"  [{i+1:2}/{len(lgd_methods)}] ✗ {method}")
            results['lgd']['failed'].append((method, elapsed, str(e)))
        update_summary()

    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug ALL methods for Experiment1")
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    args = parser.parse_args()
    
    # Explicitly call Experiment1 as requested
    run_method_debugger(target_experiment="Experiment1", verbose=not args.quiet)