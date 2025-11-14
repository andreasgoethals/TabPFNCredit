# src/utils/storage_handler.py
"""
Generic storage handler for experiment results.

Handles saving/loading results with automatic archiving of previous runs.
Agnostic to the actual structure of results - just saves whatever is passed.
"""

from __future__ import annotations
import pickle
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class StorageHandler:
    """
    Generic storage handler for experimental results.
    
    Organizes results by experiment name, with one file per dataset.
    Automatically archives previous results when re-running experiments.
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "results"):
        """
        Initialize storage handler.
        
        Args:
            experiment_name: Name of experiment (creates subfolder)
            base_dir: Base directory for all results (default: "results")
        """
        self.experiment_name = experiment_name
        
        # Ensure base_dir is relative to repo root
        repo_root = Path(__file__).resolve().parents[2]  # Go up from src/utils/
        self.base_dir = repo_root / base_dir
        
        self.experiment_dir = self.base_dir / experiment_name
        self.archive_dir = self.experiment_dir / "archive"
        
        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
    def _archive_existing_results(self) -> None:
        """
        Archive all existing results in the experiment folder.
        
        Moves existing results to archive/{timestamp}/ before saving new results.
        Only archives actual result files (.pkl), not metadata or archive folder.
        """
        # Find existing result files
        existing_files = list(self.experiment_dir.glob("*.pkl"))
        existing_metadata = list(self.experiment_dir.glob("*.json"))
        
        if not existing_files and not existing_metadata:
            return  # Nothing to archive
        
        # Create archive directory if needed
        self.archive_dir.mkdir(exist_ok=True)
        
        # Create timestamped subfolder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_subdir = self.archive_dir / timestamp
        archive_subdir.mkdir(exist_ok=True)
        
        # Move all existing files
        for file in existing_files + existing_metadata:
            shutil.move(str(file), str(archive_subdir / file.name))
        
        logger.info(f"Archived {len(existing_files)} result files to {archive_subdir}")
    
    def save_experiment_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Save experiment-level metadata (config, timestamp, versions, etc.).
        
        Archives existing metadata if present.
        
        Args:
            metadata: Dictionary with any experiment metadata
        """
        # Archive existing results before saving new metadata
        metadata_path = self.experiment_dir / "experiment_metadata.json"
        if metadata_path.exists():
            self._archive_existing_results()
        
        # Add automatic metadata
        full_metadata = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            **metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2, default=str)
        
        logger.info(f"Saved experiment metadata to {metadata_path}")
    
    def save_dataset_results(
        self, 
        dataset: str, 
        results: Any,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> None:
        """
        Save results for a single dataset.
        
        Archives existing results for this dataset before saving new ones.
        
        Args:
            dataset: Dataset name (used as filename)
            results: Any object (dict, list, etc.) - will be pickled
            metadata: Optional metadata specific to this dataset
            overwrite: If True, archives old results. If False, raises error if exists.
        """
        results_path = self.experiment_dir / f"{dataset}.pkl"
        
        # Check if results already exist
        if results_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Results for {dataset} already exist. "
                    f"Set overwrite=True to archive old results."
                )
            # Archive the existing file
            self._archive_single_dataset(dataset)
        
        # Save results as pickle
        with open(results_path, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saved results for {dataset} to {results_path}")
        
        # Save metadata as JSON (if provided)
        if metadata is not None:
            metadata_path = self.experiment_dir / f"{dataset}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata for {dataset} to {metadata_path}")
    
    def _archive_single_dataset(self, dataset: str) -> None:
        """
        Archive results for a specific dataset.
        
        Args:
            dataset: Dataset name
        """
        results_path = self.experiment_dir / f"{dataset}.pkl"
        metadata_path = self.experiment_dir / f"{dataset}_metadata.json"
        
        if not results_path.exists():
            return
        
        # Create archive structure
        self.archive_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_subdir = self.archive_dir / timestamp
        archive_subdir.mkdir(exist_ok=True)
        
        # Move files
        shutil.move(str(results_path), str(archive_subdir / results_path.name))
        if metadata_path.exists():
            shutil.move(str(metadata_path), str(archive_subdir / metadata_path.name))
        
        logger.info(f"Archived {dataset} results to {archive_subdir}")
    
    def load_dataset_results(self, dataset: str) -> Any:
        """
        Load results for a single dataset.
        
        Args:
            dataset: Dataset name
            
        Returns:
            Whatever was saved (structure-agnostic)
            
        Raises:
            FileNotFoundError: If results don't exist
        """
        results_path = self.experiment_dir / f"{dataset}.pkl"
        
        if not results_path.exists():
            raise FileNotFoundError(f"No results found for dataset {dataset}")
        
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        logger.info(f"Loaded results for {dataset} from {results_path}")
        return results
    
    def load_dataset_metadata(self, dataset: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a single dataset.
        
        Args:
            dataset: Dataset name
            
        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = self.experiment_dir / f"{dataset}_metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def is_completed(self, dataset: str) -> bool:
        """
        Check if results exist for a dataset.
        
        Args:
            dataset: Dataset name
            
        Returns:
            True if results file exists
        """
        results_path = self.experiment_dir / f"{dataset}.pkl"
        return results_path.exists()
    
    def get_completed_datasets(self) -> list[str]:
        """
        Get list of all completed datasets.
        
        Returns:
            List of dataset names that have results saved
        """
        completed = []
        for file in self.experiment_dir.glob("*.pkl"):
            dataset_name = file.stem
            completed.append(dataset_name)
        return sorted(completed)
    
    def get_archived_runs(self) -> list[str]:
        """
        Get list of archived experiment runs.
        
        Returns:
            List of archive timestamps
        """
        if not self.archive_dir.exists():
            return []
        
        archives = [d.name for d in self.archive_dir.iterdir() if d.is_dir()]
        return sorted(archives, reverse=True)  # Most recent first
    
    def get_experiment_path(self) -> Path:
        """Get path to experiment directory."""
        return self.experiment_dir


# Convenience functions for simple usage
def save_results(
    experiment_name: str,
    dataset: str,
    results: Any,
    metadata: Optional[Dict[str, Any]] = None,
    base_dir: str = "results",
    overwrite: bool = True
) -> None:
    """
    Convenience function to save dataset results.
    
    Args:
        experiment_name: Name of experiment
        dataset: Dataset name
        results: Results to save (any structure)
        metadata: Optional metadata
        base_dir: Base results directory
        overwrite: Whether to archive and overwrite existing results
    """
    handler = StorageHandler(experiment_name, base_dir)
    handler.save_dataset_results(dataset, results, metadata, overwrite=overwrite)


def load_results(
    experiment_name: str,
    dataset: str,
    base_dir: str = "results"
) -> Any:
    """
    Convenience function to load dataset results.
    
    Args:
        experiment_name: Name of experiment
        dataset: Dataset name
        base_dir: Base results directory
        
    Returns:
        Loaded results (any structure)
    """
    handler = StorageHandler(experiment_name, base_dir)
    return handler.load_dataset_results(dataset)