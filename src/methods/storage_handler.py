import os
import shutil
import csv
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)

class StorageHandler:
    """
    Handles saving model results and archiving previous outputs.
    Each dataset and method combination gets its own subdirectory:
    output/results/{task_type}/{dataset_name}/{method_name}/
    """

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    BASE_OUTPUT = os.path.join(BASE_DIR, "output", "results")
    ARCHIVE_ROOT = os.path.join(BASE_DIR, "output", "output_archive")
    CURRENT_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    @classmethod
    def _ensure_dir(cls, path):
        os.makedirs(path, exist_ok=True)
        return path

    # ---------------------- #
    #  GLOBAL ARCHIVE LOGIC  #
    # ---------------------- #
    @classmethod
    def archive_previous_results(cls):
        """
        Move all existing results (both pd and lgd) into one timestamped archive folder.
        Run this ONCE at the start of each experiment run.
        """
        if not os.path.exists(cls.BASE_OUTPUT):
            return  # nothing to archive

        # Check if there is anything inside the results folder
        contents = os.listdir(cls.BASE_OUTPUT)
        if not contents:
            return  # already empty

        # Create archive folder for this timestamp
        archive_dir = os.path.join(cls.ARCHIVE_ROOT, cls.CURRENT_RUN_TIMESTAMP)
        os.makedirs(archive_dir, exist_ok=True)

        # Move entire 'results' directory contents into the archive
        for item in contents:
            src = os.path.join(cls.BASE_OUTPUT, item)
            dst = os.path.join(archive_dir, item)
            shutil.move(src, dst)
            logger.info(f"Archived previous results: {src} → {dst}")

        logger.info(f"All previous results archived under {archive_dir}")

    # -------------------------- #
    #  PER-DATASET SAVE LOGIC   #
    # -------------------------- #
    @classmethod
    def prepare_output_dir(cls, dataset_name, method_name, task_type):
        """Ensure output directory exists (called per dataset × method)."""
        output_dir = os.path.join(cls.BASE_OUTPUT, task_type, dataset_name, method_name)
        cls._ensure_dir(output_dir)
        return output_dir

    @classmethod
    def save_fold_results(cls, result_dict, dataset_name, method_name, task_type, fold_idx):
        """
        Save y_true and y_pred arrays to a fold-specific CSV file.
        Also appends a metadata row to configdata.csv.
        """
        output_dir = cls.prepare_output_dir(dataset_name, method_name, task_type)

        # Save predictions
        pred_path = os.path.join(output_dir, f"fold_{fold_idx}_predictions.csv")
        np.savetxt(pred_path, np.column_stack((result_dict["y_true"], result_dict["y_pred"])),delimiter=",", header="y_true,y_pred", comments="")
        logger.info(f"Saved predictions for fold {fold_idx} to {pred_path}")

        # Save metadata
        config_path = os.path.join(output_dir, "configdata.csv")
        write_header = not os.path.exists(config_path)
        with open(config_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "train_time_sec", "n_test", "n_features", "model"])
            writer.writerow([
                result_dict.get("timestamp", ""),
                result_dict.get("train_time_sec", ""),
                result_dict.get("n_test", ""),
                result_dict.get("n_features", ""),
                result_dict.get("model", ""),
            ])
        logger.info(f"Appended metadata for fold {fold_idx} to {config_path}")
