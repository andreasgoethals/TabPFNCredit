#!/usr/bin/env python3
"""
Find incomplete tasks and categorize failures by type.

This script identifies:
1. Missing results (tasks that haven't completed)
2. Python errors (from errors.log)
3. SLURM timeouts (from .err files)
4. CUDA OOM errors (from errors.log)
"""

import sys
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

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


def parse_errors_log(experiment_dir):
    """Parse errors.log to find Python failures."""
    errors_log = experiment_dir / "logs" / "errors.log"
    
    failures = defaultdict(lambda: {'error': '', 'time': '', 'node': ''})
    
    if not errors_log.exists():
        return failures
    
    try:
        with open(errors_log, 'r') as f:
            content = f.read()
        
        # Split by error blocks
        blocks = content.split('=' * 70)
        
        for block in blocks:
            if 'FAILED:' not in block:
                continue
            
            # Extract task info
            match = re.search(r'FAILED:\s+(.+?)/([\w_]+)/(NO_HPO|HPO)', block)
            if not match:
                continue
            
            dataset = match.group(1)
            method = match.group(2)
            hpo_mode = match.group(3)
            
            # Extract error details
            error_match = re.search(r'Error:\s*(.+?)(?=\nNode:|\Z)', block, re.DOTALL)
            time_match = re.search(r'Time:\s*(.+)', block)
            node_match = re.search(r'Node:\s*(.+)', block)
            
            error_text = error_match.group(1).strip() if error_match else 'Unknown'
            time_text = time_match.group(1).strip() if time_match else ''
            node_text = node_match.group(1).strip() if node_match else ''
            
            # Categorize error type
            error_type = 'other'
            if 'CUDA out of memory' in error_text:
                error_type = 'cuda_oom'
            elif not error_text or error_text == 'Unknown':
                error_type = 'unknown'
            
            key = (dataset, method, hpo_mode)
            failures[key] = {
                'error': error_text[:200],  # Truncate long errors
                'type': error_type,
                'time': time_text,
                'node': node_text
            }
    
    except Exception as e:
        print(f"Warning: Could not parse errors.log: {e}")
    
    return failures


def find_timeout_tasks(experiment_dir):
    """Find tasks that timed out from SLURM .err files."""
    slurm_logs = experiment_dir / "logs" / "slurm"
    
    timeouts = []
    
    if not slurm_logs.exists():
        return timeouts
    
    # Search all .err files for timeout messages
    for err_file in slurm_logs.glob("*.err"):
        try:
            with open(err_file, 'r') as f:
                content = f.read()
            
            # Check for timeout indicators
            if 'TIME LIMIT' in content or 'CANCELLED' in content:
                # Get corresponding .out file to find what task was running
                out_file = err_file.with_suffix('.out')
                
                if out_file.exists():
                    with open(out_file, 'r') as f:
                        out_content = f.read()
                    
                    # Extract task info from "Running task X: dataset/method/task_type/hpo_mode"
                    match = re.search(
                        r'Running task \d+:\s+(.+?)/([\w_]+)/(pd|lgd)/(NO_HPO|HPO)',
                        out_content
                    )
                    
                    if match:
                        dataset = match.group(1)
                        method = match.group(2)
                        task_type = match.group(3)
                        hpo_mode = match.group(4)
                        
                        # Extract timestamp from error message
                        time_match = re.search(
                            r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.\d+\]',
                            content
                        )
                        timestamp = time_match.group(1) if time_match else 'Unknown'
                        
                        # Extract node name
                        node_match = re.search(r'ON ([\w\d]+)', content)
                        node = node_match.group(1) if node_match else 'Unknown'
                        
                        timeouts.append({
                            'dataset': dataset,
                            'method': method,
                            'task_type': task_type,
                            'hpo_mode': hpo_mode,
                            'time': timestamp,
                            'node': node,
                            'err_file': err_file.name
                        })
        
        except Exception as e:
            print(f"Warning: Could not read {err_file}: {e}")
    
    return timeouts


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection_header(title):
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * 70)


def group_by_method(items, key_func):
    """Group items by method/hpo_mode."""
    grouped = defaultdict(list)
    for item in items:
        key = key_func(item)
        grouped[key].append(item)
    return grouped


def main():
    config = load_config("Experiment1")
    experiment_dir = PROJECT_ROOT / "results" / "experiment1"
    
    print_section_header("EXPERIMENT 1 - COMPREHENSIVE STATUS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Get expected and completed tasks
    expected = set(get_expected_tasks(config))
    completed = set(get_completed_tasks(experiment_dir))
    missing = expected - completed
    
    # 2. Parse errors.log
    python_errors = parse_errors_log(experiment_dir)
    
    # 3. Find SLURM timeouts
    timeouts = find_timeout_tasks(experiment_dir)
    
    # 4. Categorize missing tasks
    timeout_keys = {(t['dataset'], t['method'], t['hpo_mode']) for t in timeouts}
    cuda_oom_keys = {k for k, v in python_errors.items() if v['type'] == 'cuda_oom'}
    other_error_keys = {k for k, v in python_errors.items() if v['type'] != 'cuda_oom'}
    
    # Convert missing to comparable format
    missing_with_task = [(d, m, task, h) for d, m, task, h in missing]
    missing_keys = {(d, m, h) for d, m, task, h in missing_with_task}
    
    # Categorize
    timeout_missing = [m for m in missing_with_task if (m[0], m[1], m[3]) in timeout_keys]
    cuda_oom_missing = [m for m in missing_with_task if (m[0], m[1], m[3]) in cuda_oom_keys]
    other_error_missing = [m for m in missing_with_task if (m[0], m[1], m[3]) in other_error_keys]
    unknown_missing = [m for m in missing_with_task 
                      if (m[0], m[1], m[3]) not in timeout_keys 
                      and (m[0], m[1], m[3]) not in cuda_oom_keys
                      and (m[0], m[1], m[3]) not in other_error_keys]
    
    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================
    print_section_header("SUMMARY STATISTICS")
    
    total_expected = len(expected)
    total_completed = len(completed)
    total_missing = len(missing)
    completion_rate = (total_completed / total_expected * 100) if total_expected > 0 else 0
    
    print(f"\nTotal expected tasks:  {total_expected}")
    print(f"Completed tasks:       {total_completed} ({completion_rate:.1f}%)")
    print(f"Missing tasks:         {total_missing} ({100-completion_rate:.1f}%)")
    
    print(f"\nFailure breakdown:")
    print(f"  SLURM timeouts:      {len(timeout_missing)}")
    print(f"  CUDA out of memory:  {len(cuda_oom_missing)}")
    print(f"  Other Python errors: {len(other_error_missing)}")
    print(f"  Unknown/Not started: {len(unknown_missing)}")
    
    # ADD CLARIFICATION ABOUT RESOLVED ERRORS
    if total_missing < len(timeout_keys) or total_missing < len(cuda_oom_keys):
        all_timeout_tasks = len(timeout_keys)
        all_cuda_oom_tasks = len(cuda_oom_keys)
        resolved_timeouts = all_timeout_tasks - len(timeout_missing)
        resolved_cuda = all_cuda_oom_tasks - len(cuda_oom_missing)
        
        print(f"\n📝 Note:")
        if resolved_timeouts > 0:
            print(f"   • {resolved_timeouts} timeout(s) have since completed successfully")
        if resolved_cuda > 0:
            print(f"   • {resolved_cuda} CUDA OOM error(s) have since completed successfully")
        print(f"   Only showing errors for tasks that still need attention.")
    
    # ============================================================
    # SLURM TIMEOUTS
    # ============================================================
    if timeout_missing:
        print_section_header(f"⏱️  SLURM TIMEOUTS ({len(timeout_missing)} tasks)")
        
        # Group by method/mode
        grouped = group_by_method(
            timeout_missing,
            lambda x: f"{x[1]}/{x[3]}"  # method/hpo_mode
        )
        
        for method_mode in sorted(grouped.keys()):
            tasks = grouped[method_mode]
            print_subsection_header(f"{method_mode} ({len(tasks)} datasets)")
            
            for dataset, method, task_type, hpo_mode in sorted(tasks):
                # Find corresponding timeout info
                timeout_info = next(
                    (t for t in timeouts 
                     if t['dataset'] == dataset 
                     and t['method'] == method 
                     and t['hpo_mode'] == hpo_mode),
                    None
                )
                
                if timeout_info:
                    print(f"  • {dataset}/{task_type}")
                    print(f"    Time: {timeout_info['time']}")
                    print(f"    Node: {timeout_info['node']}")
                else:
                    print(f"  • {dataset}/{task_type}")
        
        print("\n💡 Solution: Move these methods to higher-tier GPUs with longer time limits")
        print("   or add them to NO_HPO_METHODS if HPO is too slow.")
    
    # ============================================================
    # CUDA OUT OF MEMORY
    # ============================================================
    if cuda_oom_missing:
        print_section_header(f"🔥 CUDA OUT OF MEMORY ({len(cuda_oom_missing)} tasks)")
        
        grouped = group_by_method(
            cuda_oom_missing,
            lambda x: f"{x[1]}/{x[3]}"
        )
        
        for method_mode in sorted(grouped.keys()):
            tasks = grouped[method_mode]
            print_subsection_header(f"{method_mode} ({len(tasks)} datasets)")
            
            for dataset, method, task_type, hpo_mode in sorted(tasks):
                key = (dataset, method, hpo_mode)
                error_info = python_errors.get(key, {})
                
                print(f"  • {dataset}/{task_type}")
                if error_info.get('node'):
                    print(f"    Node: {error_info['node']} (P100 - 16GB VRAM)")
                if error_info.get('error'):
                    # Extract GPU memory info
                    mem_match = re.search(
                        r'(\d+\.?\d*)\s*GiB\s+total capacity.*?(\d+\.?\d*)\s*GiB\s+already allocated',
                        error_info['error']
                    )
                    if mem_match:
                        total = mem_match.group(1)
                        allocated = mem_match.group(2)
                        print(f"    Memory: {allocated}GB allocated / {total}GB total")
        
        print("\n💡 Solution: Move these methods to GPU2 (A100 - 40GB) or GPU3 (H100 - 80GB)")
    
    # ============================================================
    # OTHER PYTHON ERRORS
    # ============================================================
    if other_error_missing:
        print_section_header(f"❌ OTHER PYTHON ERRORS ({len(other_error_missing)} tasks)")
        
        grouped = group_by_method(
            other_error_missing,
            lambda x: f"{x[1]}/{x[3]}"
        )
        
        for method_mode in sorted(grouped.keys()):
            tasks = grouped[method_mode]
            print_subsection_header(f"{method_mode} ({len(tasks)} datasets)")
            
            for dataset, method, task_type, hpo_mode in sorted(tasks):
                key = (dataset, method, hpo_mode)
                error_info = python_errors.get(key, {})
                
                print(f"  • {dataset}/{task_type}")
                if error_info.get('error'):
                    error_text = error_info['error']
                    # Truncate if too long
                    if len(error_text) > 150:
                        error_text = error_text[:150] + "..."
                    print(f"    Error: {error_text}")
                if error_info.get('node'):
                    print(f"    Node: {error_info['node']}")
        
        print("\n💡 Solution: Check errors.log for full stack traces")
    
    # ============================================================
    # UNKNOWN/NOT STARTED
    # ============================================================
    if unknown_missing:
        print_section_header(f"❓ UNKNOWN/NOT STARTED ({len(unknown_missing)} tasks)")
        
        grouped = group_by_method(
            unknown_missing,
            lambda x: f"{x[1]}/{x[3]}"
        )
        
        for method_mode in sorted(grouped.keys()):
            tasks = grouped[method_mode]
            print_subsection_header(f"{method_mode} ({len(tasks)} datasets)")
            
            for dataset, method, task_type, hpo_mode in sorted(tasks):
                print(f"  • {dataset}/{task_type}")
        
        print("\n💡 These tasks have no error logs - they may not have started yet")
        print("   or may be currently running.")
    
    # ============================================================
    # COMPLETION STATUS
    # ============================================================
    if not missing:
        print_section_header("✅ ALL TASKS COMPLETED!")
        print("\n🎉 Experiment 1 has finished successfully!")
        print(f"   Total tasks completed: {total_completed}/{total_expected}")
    else:
        print_section_header("📋 NEXT STEPS")
        
        if timeout_missing:
            print("\n1. Fix timeouts:")
            print("   sbatch scripts/Experiment1/Experiment1_Retry_CPU.slurm")
            print("   sbatch scripts/Experiment1/Experiment1_Retry_GPU_A100.slurm")
            print("   sbatch scripts/Experiment1/Experiment1_Retry_GPU_H100.slurm")
        
        if cuda_oom_missing:
            print("\n2. Fix CUDA OOM:")
            print("   sbatch scripts/Experiment1/Experiment1_Retry_GPU_A100.slurm")
        
        if other_error_missing:
            print("\n3. Investigate other errors:")
            print(f"   cat {experiment_dir / 'logs' / 'errors.log'}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()