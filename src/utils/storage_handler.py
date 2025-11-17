# src/utils/storage_handler.py
"""
Simple storage handler for experiment results.
Save and load results only - no archiving logic.
"""

from __future__ import annotations
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class StorageHandler:
    """
    Simple storage handler for experimental results.
    Saves/loads results - archiving is handled separately by storage_archiver.py
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
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
    def save_experiment_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Save experiment-level metadata (config, timestamp, versions, etc.).
        
        Args:
            metadata: Dictionary with any experiment metadata
        """
        metadata_path = self.experiment_dir / "experiment_metadata.json"
        
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
        overwrite: bool = True
    ) -> None:
        """
        Save results for a single dataset.
        
        Args:
            dataset: Dataset name (used as filename)
            results: Any object (dict, list, etc.) - will be pickled
            metadata: Optional metadata specific to this dataset
            overwrite: If True, overwrites existing. If False, raises error if exists.
        """
        results_path = self.experiment_dir / f"{dataset}.pkl"
        
        # Check if results already exist
        if results_path.exists() and not overwrite:
            raise FileExistsError(
                f"Results for {dataset} already exist. "
                f"Set overwrite=True to replace them."
            )
        
        # Save results as pickle
        with open(results_path, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saved results for {dataset} to {results_path}")
        
        # Save metadata as JSON (if provided)
        if metadata is not None:
            metadata_path = self.experiment_dir / f"{dataset}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
    
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
    
    def get_experiment_path(self) -> Path:
        """Get path to experiment directory."""
        return self.experiment_dir


# Convenience functions
def save_results(
    experiment_name: str,
    dataset: str,
    results: Any,
    metadata: Optional[Dict[str, Any]] = None,
    base_dir: str = "results",
    overwrite: bool = True
) -> None:
    """Convenience function to save dataset results."""
    handler = StorageHandler(experiment_name, base_dir)
    handler.save_dataset_results(dataset, results, metadata, overwrite=overwrite)


def load_results(
    experiment_name: str,
    dataset: str,
    base_dir: str = "results"
) -> Any:
    """Convenience function to load dataset results."""
    handler = StorageHandler(experiment_name, base_dir)
    return handler.load_dataset_results(dataset)