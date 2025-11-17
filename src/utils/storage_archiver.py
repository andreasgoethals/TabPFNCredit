# src/utils/storage_archiver.py
"""
Archive utility for experiment results.

Provides functionality to archive completed experiments by MOVING entire experiment 
folders to a timestamped archive location. This is useful when:
- Re-running experiments and wanting to preserve old results
- Cleaning up the active results directory
- Creating versioned snapshots of experimental outputs

After archiving, the original experiment folder is REMOVED from the active results
directory and exists only in the archive.
"""

from __future__ import annotations
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class StorageArchiver:
    """
    Archive manager for experimental results.
    
    Manages the archival of experiment folders by moving them from the active results
    directory to a timestamped archive. This preserves historical results while 
    freeing up the main results directory for new runs.
    
    Directory structure:
        results/
            experiment1/              <- Active experiment (moved on archive)
            experiment2/              <- Active experiment
            archive/                  <- Archive directory
                20251117_094520_experiment1/    <- Archived experiment
                20251118_103045_experiment1/    <- Second archive of same experiment
                20251119_120000_experiment2/    <- Archived different experiment
    
    Attributes:
        base_dir (Path): Root directory containing all results
        archive_dir (Path): Subdirectory where archived experiments are stored
    """
    
    def __init__(self, base_dir: str = "results"):
        """
        Initialize the storage archiver.
        
        Sets up paths relative to the project root and ensures the archive 
        directory exists.
        
        Args:
            base_dir: Name of the base results directory (default: "results").
            This directory should be at the project root level.
        """
        # Navigate to project root: go up two levels from src/utils/
        repo_root = Path(__file__).resolve().parents[2]
        
        # Set up directory paths
        self.base_dir = repo_root / base_dir
        self.archive_dir = self.base_dir / "archive"
        
        # Ensure archive directory exists
        # parents=True: create parent directories if needed
        # exist_ok=True: don't raise error if directory already exists
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"StorageArchiver initialized with base_dir: {self.base_dir}")
    
    def archive_experiment(
        self, 
        experiment_name: str,
        custom_timestamp: Optional[str] = None
    ) -> Path:
        """
        Archive an entire experiment folder by moving it to the archive directory.
        
        This operation:
        1. Locates the experiment folder in the active results directory
        2. Creates a timestamped archive folder name
        3. MOVES (not copies) the entire experiment folder to the archive
        4. After completion, the original experiment folder no longer exists
        
        This is useful when re-running an experiment and wanting to preserve the
        old results without cluttering the active results directory.
        
        Args:
            experiment_name: Name of the experiment folder to archive.
            This should match the folder name in results/
            custom_timestamp: Optional custom timestamp string for the archive name.
            If None, uses current datetime in format YYYYMMDD_HHMMSS.
        Returns:
            Path object pointing to the newly created archive folder.
            
        Raises:
            FileNotFoundError: If the specified experiment folder doesn't exist
            in the results directory.
        Example:
            >>> archiver = StorageArchiver()
            >>> archive_path = archiver.archive_experiment("experiment1")
            >>> print(archive_path)
            /path/to/results/archive/20251117_094520_experiment1
            
            # Original results/experiment1/ folder is now gone
            # All files moved to results/archive/20251117_094520_experiment1/
        """
        # Construct path to the experiment folder to be archived
        experiment_path = self.base_dir / experiment_name
        
        # Verify the experiment folder exists
        if not experiment_path.exists():
            raise FileNotFoundError(
                f"Cannot archive: experiment folder not found at {experiment_path}"
            )
        
        # Generate timestamp for archive folder name
        if custom_timestamp is None:
            # Format: YYYYMMDD_HHMMSS (e.g., 20251117_094520)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = custom_timestamp
        
        # Create archive folder name: {timestamp}_{experiment_name}
        # Example: "20251117_094520_experiment1"
        archive_name = f"{timestamp}_{experiment_name}"
        archive_path = self.archive_dir / archive_name
        
        # Move the entire experiment folder to archive
        # This is a MOVE operation, not a copy - original folder will be removed
        logger.info(f"Archiving experiment: {experiment_name}")
        logger.info(f"  Moving from: {experiment_path}")
        logger.info(f"  Moving to:   {archive_path}")
        
        shutil.move(str(experiment_path), str(archive_path))
        
        logger.info(f"Successfully archived {experiment_name}")
        logger.info(f"  Original folder removed: {experiment_path}")
        logger.info(f"  Archive created at:      {archive_path}")
        
        return archive_path
    
    def list_archives(self, experiment_name: Optional[str] = None) -> list[str]:
        """
        List all archived experiment folders.
        
        Scans the archive directory and returns names of all archived experiments,
        optionally filtered by experiment name. Results are sorted with most recent
        archives first.
        
        Args:
            experiment_name: Optional filter to show only archives of a specific
            experiment. If None, returns all archives.
        Returns:
            List of archive folder names, sorted by timestamp (newest first).
            Each name has format: YYYYMMDD_HHMMSS_experimentname
            
        Example:
            >>> archiver = StorageArchiver()
            >>> archiver.list_archives()
            ['20251119_120000_experiment2', '20251118_103045_experiment1', 
            '20251117_094520_experiment1']
            
            >>> archiver.list_archives("experiment1")
            ['20251118_103045_experiment1', '20251117_094520_experiment1']
        """
        # Return empty list if archive directory doesn't exist yet
        if not self.archive_dir.exists():
            return []
        
        archives = []
        
        # Iterate through all items in the archive directory
        for archive_folder in self.archive_dir.iterdir():
            # Only consider directories (skip any stray files)
            if archive_folder.is_dir():
                # Apply experiment name filter if specified
                # Check if experiment_name appears in the archive folder name
                if experiment_name is None or experiment_name in archive_folder.name:
                    archives.append(archive_folder.name)
        
        # Sort by name (which includes timestamp) in reverse order
        # Reverse order means most recent (highest timestamp) first
        return sorted(archives, reverse=True)
    
    def get_archive_path(self, archive_name: str) -> Path:
        """
        Get the full path to a specific archived experiment.
        
        Convenience method to construct the path to an archive without
        needing to know the internal directory structure.
        
        Args:
            archive_name: Name of the archive folder 
            (format: YYYYMMDD_HHMMSS_experimentname)
            
        Returns:
            Path object pointing to the archive folder.
            
        Example:
            >>> archiver = StorageArchiver()
            >>> path = archiver.get_archive_path("20251117_094520_experiment1")
            >>> print(path)
            /path/to/results/archive/20251117_094520_experiment1
        """
        return self.archive_dir / archive_name
    
    def restore_from_archive(
        self,
        archive_name: str,
        target_experiment_name: Optional[str] = None,
        overwrite: bool = False
    ) -> Path:
        """
        Restore an archived experiment back to the active results directory.
        
        This operation moves an experiment from the archive back to the main results
        directory, effectively "un-archiving" it. The archive folder is removed in
        the process.
        
        Use cases:
        - Need to re-analyze old results
        - Want to continue an archived experiment
        - Made a mistake and need to recover archived data
        
        Args:
            archive_name: Name of the archive folder to restore
            (format: YYYYMMDD_HHMMSS_experimentname)
            target_experiment_name: Optional custom name for the restored experiment.
            If None, extracts the experiment name from the
            archive folder name by removing the timestamp prefix.
            overwrite: If True, overwrites any existing experiment with the same name.
            If False, raises an error if target already exists.
        Returns:
            Path object pointing to the restored experiment folder.
            
        Raises:
            FileNotFoundError: If the specified archive doesn't exist.
            FileExistsError: If the target location already exists and overwrite=False.
            
        Example:
            >>> archiver = StorageArchiver()
            # Restore with original name
            >>> path = archiver.restore_from_archive("20251117_094520_experiment1")
            # Creates: results/experiment1/
            # Removes: results/archive/20251117_094520_experiment1/
            
            # Restore with custom name
            >>> path = archiver.restore_from_archive(
            ...     "20251117_094520_experiment1",
            ...     target_experiment_name="experiment1_old"
            ... )
            # Creates: results/experiment1_old/
        """
        # Construct path to the archive
        archive_path = self.archive_dir / archive_name
        
        # Verify the archive exists
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Cannot restore: archive not found at {archive_path}"
            )
        
        # Determine the target experiment name
        if target_experiment_name is None:
            # Extract experiment name from archive folder name
            # Archive name format: YYYYMMDD_HHMMSS_experimentname
            # Split on underscore and take everything after the second underscore
            parts = archive_name.split('_', 2)  # Split on first 2 underscores only
            
            if len(parts) >= 3:
                # Successfully extracted experiment name
                target_experiment_name = parts[2]
            else:
                # Fallback: use entire archive name if format doesn't match expected
                target_experiment_name = archive_name
        
        # Construct target path in main results directory
        target_path = self.base_dir / target_experiment_name
        
        # Check if target location already exists
        if target_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Cannot restore: experiment folder already exists at {target_path}\n"
                    f"Set overwrite=True to replace it, or provide a different "
                    f"target_experiment_name."
                )
            
            # Overwrite is enabled: remove existing folder
            logger.warning(f"Overwriting existing folder: {target_path}")
            shutil.rmtree(target_path)
        
        # Move archive back to main results directory
        logger.info(f"Restoring archive: {archive_name}")
        logger.info(f"  Moving from: {archive_path}")
        logger.info(f"  Moving to:   {target_path}")
        
        shutil.move(str(archive_path), str(target_path))
        
        logger.info(f"Successfully restored experiment")
        logger.info(f"  Archive removed: {archive_path}")
        logger.info(f"  Restored at:     {target_path}")
        
        return target_path


# Convenience functions for quick operations without instantiating the class

def archive_experiment(
    experiment_name: str,
    base_dir: str = "results"
) -> Path:
    """
    Convenience function to archive an experiment.
    
    Creates a StorageArchiver instance and archives the specified experiment
    in a single function call.
    
    Args:
        experiment_name: Name of the experiment folder to archive
        base_dir: Base results directory (default: "results")
        
    Returns:
        Path to the created archive folder
        
    Example:
        >>> from src.utils.storage_archiver import archive_experiment
        >>> archive_experiment("experiment1")
        PosixPath('/path/to/results/archive/20251117_094520_experiment1')
    """
    archiver = StorageArchiver(base_dir)
    return archiver.archive_experiment(experiment_name)


def list_archives(
    experiment_name: Optional[str] = None,
    base_dir: str = "results"
) -> list[str]:
    """
    Convenience function to list archived experiments.
    
    Creates a StorageArchiver instance and lists archives in a single function call.
    
    Args:
        experiment_name: Optional filter for specific experiment
        base_dir: Base results directory (default: "results")
        
    Returns:
        List of archive folder names (most recent first)
        
    Example:
        >>> from src.utils.storage_archiver import list_archives
        >>> list_archives("experiment1")
        ['20251118_103045_experiment1', '20251117_094520_experiment1']
    """
    archiver = StorageArchiver(base_dir)
    return archiver.list_archives(experiment_name)


def restore_from_archive(
    archive_name: str,
    target_experiment_name: Optional[str] = None,
    overwrite: bool = False,
    base_dir: str = "results"
) -> Path:
    """
    Convenience function to restore an archived experiment.
    
    Creates a StorageArchiver instance and restores an archive in a single 
    function call.
    
    Args:
        archive_name: Name of archive to restore
        target_experiment_name: Optional custom name for restored experiment
        overwrite: Whether to overwrite existing experiment folder
        base_dir: Base results directory (default: "results")
        
    Returns:
        Path to restored experiment folder
        
    Example:
        >>> from src.utils.storage_archiver import restore_from_archive
        >>> restore_from_archive("20251117_094520_experiment1")
        PosixPath('/path/to/results/experiment1')
    """
    archiver = StorageArchiver(base_dir)
    return archiver.restore_from_archive(
        archive_name, 
        target_experiment_name, 
        overwrite
    )