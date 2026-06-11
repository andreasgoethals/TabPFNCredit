"""Centralised suppression of high-volume, non-actionable log noise.

Calling :func:`configure_quiet_runtime` once early in a process silences the
chatter observed on the cluster (tens of thousands of lines per run):

* **LightGBM** C++ ``[Info]`` / ``[Warning] No further splits`` messages --
  pinned here via :func:`lightgbm.register_logger`; the per-model
  ``verbose=-1`` injected by the method runner is the primary lever.
* **Optuna** per-trial ``INFO`` logs, the HPO tqdm progress bar, and the
  ``suggest_uniform`` / ``suggest_loguniform`` ``FutureWarning``\\s raised by
  TALENT's search space.
* **scikit-learn** deprecation (``'squared'``, ``dual``), ill-defined-metric,
  log-loss ``sum to one`` and convergence warnings.

The function is idempotent and never raises -- missing optional deps are
ignored. It is safe to call from every CLI entry point and from the method
runner.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

_CONFIGURED = False


def configure_quiet_runtime() -> None:
    """Install warning filters + third-party verbosity caps (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    # ---- Python warnings (match by message so we don't blanket-hide all
    #      FutureWarnings, only the known-noisy, non-actionable ones) ----
    warnings.filterwarnings("ignore", message=r".*suggest_(uniform|loguniform).*")
    warnings.filterwarnings("ignore", message=r".*'squared' is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*do not sum to one.*")
    warnings.filterwarnings("ignore", message=r".*default value of `dual`.*")
    # Benign chatter from TALENT's deep methods / bundled libs (TabNet device
    # banner, NODE init notice, tensor copy-construct hints, legacy indexing).
    # Deliberately NOT silenced: torch's "Using a target size ..." broadcast
    # warning -- that one signals a real loss-shape bug and must stay visible.
    warnings.filterwarnings("ignore", message=r".*Device used : .*")
    warnings.filterwarnings("ignore", message=r".*Best weights from best epoch.*")
    warnings.filterwarnings("ignore", message=r".*To copy construct from a tensor.*")
    warnings.filterwarnings("ignore", message=r".*non-tuple sequence for multidimensional indexing.*")
    warnings.filterwarnings("ignore", message=r".*Data-aware initialization is performed.*")
    try:
        from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        pass

    # ---- Optuna: drop per-trial INFO logs; disable the progress bar in
    #      non-interactive (batch) runs where it only spams the log file ----
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        pass
    try:
        if not sys.stdout.isatty():
            # tqdm reads this at bar-construction time; covers Optuna's bar
            # (TALENT calls study.optimize(show_progress_bar=True)).
            os.environ.setdefault("TQDM_DISABLE", "1")
    except Exception:
        pass

    # ---- LightGBM: route its logger through a silenced channel so the C++
    #      callback's Info/Warning lines are dropped globally. The method
    #      runner additionally sets verbose=-1 on the model itself. ----
    try:
        import lightgbm as lgb
        _silent = logging.getLogger("tabpfncredit.lightgbm")
        _silent.setLevel(logging.ERROR)
        _silent.propagate = False
        lgb.register_logger(_silent)
    except Exception:
        pass


__all__ = ["configure_quiet_runtime"]
