"""Hybrid + minimal logging for TabPFNCredit experiments.

Each run produces three files under ``results/<experiment>/logs/``:

* ``<dataset>_<method>.log``  -- per-task detail file (DEBUG level captured to disk).
* ``summary.log``             -- one line per task START / FINISH (with wall-clock and
                                 the headline metric, AUC for PD / RMSE for LGD).
* ``errors.log``              -- shared aggregated traceback file (only failures land here).

SLURM ``--output`` / ``--error`` no longer point at a ``slurm/`` subdirectory;
they point directly at ``results/<experiment>/logs/`` and the per-task
logger streams INFO+ to the per-task ``.log``. The console layer (stdout
captured by SLURM) carries the same info, so the slurm ``.out`` and the
per-task ``.log`` agree.

Verbosity policy (minimal):
* INFO    -- start, finish (with wall-clock), one headline metric, downsampling notices.
* WARNING -- LGD clipping rate > 5%, foundation-model val/test caps engaging.
* ERROR   -- traceback on failure (also appended to ``errors.log``).

Anything more granular is logged at DEBUG and only persists in the
per-task ``.log`` (set ``--verbose`` to surface it to console).
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


_FMT = "%(asctime)s [%(levelname)-5s] [%(taskid)s] %(message)s"
_DATE_FMT = "%H:%M:%S"


class _TaskFilter(logging.Filter):
    """Inject a ``taskid`` field if the LogRecord doesn't carry one."""

    def __init__(self, taskid: str = "-") -> None:
        super().__init__()
        self.taskid = taskid

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        if not hasattr(record, "taskid"):
            record.taskid = self.taskid
        return True


def configure_task_logging(
    experiment: str,
    dataset: str,
    method: str,
    task: str,
    *,
    results_root: Path,
    verbose: bool = False,
) -> Path:
    """Set up the per-task logging tree and return the per-task log path.

    Idempotent: safe to call multiple times within the same SLURM array slot.
    The root logger is reconfigured with three handlers:

    1. **Per-task FileHandler** -- detailed log for this (dataset, method).
    2. **Summary FileHandler** -- aggregates start/finish/errors across the
       whole experiment (append mode, atomic line writes).
    3. **Errors FileHandler** -- only ERROR+ lines, also append mode.

    Console output is also enabled (always INFO; DEBUG if ``verbose=True``).
    """
    logs_dir = results_root / experiment.lower() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    taskid = f"{task}/{dataset}/{method}"
    per_task_path = logs_dir / f"{dataset}_{method}.log"
    summary_path = logs_dir / "summary.log"
    errors_path = logs_dir / "errors.log"

    root = logging.getLogger()
    # Wipe handlers previously attached by another (dataset, method) in this
    # SLURM slot (CPU bundle runs hit this).
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    # Per-task detail log (DEBUG+)
    per_task_handler = logging.FileHandler(per_task_path, mode="a", encoding="utf-8")
    per_task_handler.setLevel(logging.DEBUG)
    per_task_handler.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
    per_task_handler.addFilter(_TaskFilter(taskid))
    root.addHandler(per_task_handler)

    # Summary log (INFO+, shared file)
    summary_handler = logging.FileHandler(summary_path, mode="a", encoding="utf-8")
    summary_handler.setLevel(logging.INFO)
    summary_handler.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
    summary_handler.addFilter(_TaskFilter(taskid))
    root.addHandler(summary_handler)

    # Errors-only log (ERROR+, shared file)
    errors_handler = logging.FileHandler(errors_path, mode="a", encoding="utf-8")
    errors_handler.setLevel(logging.ERROR)
    errors_handler.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
    errors_handler.addFilter(_TaskFilter(taskid))
    root.addHandler(errors_handler)

    # Console handler (INFO unless verbose)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
    console.addFilter(_TaskFilter(taskid))
    root.addHandler(console)

    # Silence chatty third-party libraries unless verbose.
    for noisy in ("matplotlib", "numexpr", "PIL", "tensorflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return per_task_path


@contextmanager
def task_timer(message: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """Context manager: log start, finish (with elapsed), and re-raise on error."""
    log = logger or logging.getLogger(__name__)
    log.info("START %s", message)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        log.exception("FAIL  %s (after %.1fs)", message, time.perf_counter() - t0)
        raise
    else:
        log.info("DONE  %s (%.1fs)", message, time.perf_counter() - t0)


__all__ = ["configure_task_logging", "task_timer"]
