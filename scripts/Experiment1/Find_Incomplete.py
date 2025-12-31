#!/usr/bin/env python3
"""
Find incomplete tasks by comparing expected vs actual results.
"""

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from src.methods.method_config import NO_HPO_METHODS
import pickle

def get_expected_tasks(config):
    """Build list of all expected tasks."""
    tasks = []
    
    # PD tasks
    pd_datasets = list(config['datasets']['pd'].keys())
    pd_methods = list(config['methods']['pd'].keys())
    
    for dataset in pd_datasets:
        for method in pd_methods:
            if method in NO_HPO_METHODS:
                tasks.append((dataset, method, 'pd', 'NO_HPO'))
            else:
                tasks.append((dataset, method, 'pd', 'NO_HPO'))
                tasks.append((dataset, method, 'pd', 'HPO'))
    
    # LGD tasks
    lgd_datasets = list(config['datasets']['lgd'].keys())
    lgd_methods = list(config['methods']['lgd'].keys())
    
    for dataset in lgd_datasets:
        for method in lgd_methods:
            if method in NO_HPO_METHODS:
                tasks.append((dataset, method, 'lgd', 'NO_HPO'))
            else:
                tasks.append((dataset, method, 'lgd', 'NO_HPO'))
                tasks.append((dataset, method, 'lgd', 'HPO'))
    
    return tasks


def get_completed_tasks(experiment_dir):
    """Get list of completed tasks from result files."""
    completed = []
    
    for task_type in ['pd', 'lgd']:
        task_dir = experiment_dir / task_type
        if not task_dir.exists():
            continue
        
        for result_file in task_dir.glob('*.pkl'):
            dataset = result_file.stem
            
            try:
                with open(result_file, 'rb') as f:
                    results = pickle.load(f)
                
                # Check NO_HPO
                if 'NO_HPO' in results:
                    for method in results['NO_HPO'].keys():
                        completed.append((dataset, method, task_type, 'NO_HPO'))
                
                # Check HPO
                if 'HPO' in results:
                    for method in results['HPO'].keys():
                        completed.append((dataset, method, task_type, 'HPO'))
            
            except Exception as e:
                print(f"Warning: Could not read {result_file}: {e}")
    
    return completed


def main():
    config = load_config("Experiment1")
    experiment_dir = PROJECT_ROOT / "results" / "experiment1"
    
    print("="*70)
    print("FINDING INCOMPLETE TASKS")
    print("="*70)
    
    # Get expected and completed tasks
    expected = set(get_expected_tasks(config))
    completed = set(get_completed_tasks(experiment_dir))
    
    # Find missing
    missing = expected - completed
    
    print(f"\nExpected tasks:  {len(expected)}")
    print(f"Completed tasks: {len(completed)}")
    print(f"Missing tasks:   {len(missing)}")
    
    if missing:
        print("\n" + "="*70)
        print("MISSING TASKS (likely timed out or failed):")
        print("="*70)
        
        # Group by method
        by_method = {}
        for dataset, method, task_type, hpo_mode in sorted(missing):
            key = f"{method}/{hpo_mode}"
            if key not in by_method:
                by_method[key] = []
            by_method[key].append(f"{dataset}/{task_type}")
        
        for method_mode, datasets in sorted(by_method.items()):
            print(f"\n{method_mode}:")
            for ds in datasets:
                print(f"  - {ds}")
    else:
        print("\n✅ All tasks completed!")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()