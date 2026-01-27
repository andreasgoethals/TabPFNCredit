#!/usr/bin/env python3
"""
Final Rerun Orchestrator for Experiment 1.

Reruns a single foundation model task specified via command-line arguments.
Called by Experiment1_Final0.slurm (mitra) and Experiment1_Final1.slurm (tabicl).

Usage:
    python Experiment1_Final.py --dataset 0013.home_credit --method mitra --task_type pd --hpo_mode NO_HPO
    python Experiment1_Final.py --dataset 0013.home_credit --method tabicl --task_type pd --hpo_mode NO_HPO --row_limit 50000
"""

import sys
import os
import gc
import argparse
import torch
import logging
from pathlib import Path
from time import time

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_reader import load_config
from scripts.Experiment1.Experiment1 import run_single_method

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def cleanup_gpu():
    """Aggressive GPU cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("  -> GPU Memory Cleared")


def main():
    parser = argparse.ArgumentParser(description="Experiment 1 Final Rerun")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--task_type", required=True)
    parser.add_argument("--hpo_mode", required=True)
    parser.add_argument("--row_limit", type=int, default=None)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1 - FINAL RERUN")
    print(f"  Dataset:   {args.dataset}")
    print(f"  Method:    {args.method}")
    print(f"  Task:      {args.task_type}")
    print(f"  HPO Mode:  {args.hpo_mode}")
    print(f"  Row Limit: {args.row_limit or 'None'}")
    print(f"{'='*70}\n")

    # Load base config
    base_config = load_config("Experiment1")
    config = base_config.copy()

    # Apply row limit if specified
    if args.row_limit is not None:
        if 'split' not in config:
            config['split'] = {}
        config['split']['row_limit'] = args.row_limit
        print(f"  -> Applied Row Limit: {args.row_limit}")

    # Cleanup before start
    cleanup_gpu()

    # Run
    try:
        start_time = time()
        run_single_method(
            dataset=args.dataset,
            method=args.method,
            task_type=args.task_type,
            hpo_mode=args.hpo_mode,
            config=config,
            experiment_name='experiment1',
            verbose=True
        )
        duration = time() - start_time
        print(f"\n  -> SUCCESS ({duration:.1f}s)")

    except Exception as e:
        print(f"\n  -> FAILED: {e}")
        raise

    # Final cleanup
    cleanup_gpu()

    print(f"\n{'='*70}")
    print("TASK COMPLETED")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
