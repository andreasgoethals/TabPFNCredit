# src/utils/file_lock.py
"""
Cross-platform file locking utilities.

The only writer that still needs cross-process synchronisation in the
new JSON+npz storage layout is the rare HPO trial-history append (each
result file is otherwise owned by exactly one SLURM array task and needs
no lock).

Public entry point:

* :class:`FileLock` -- a context manager providing exclusive (``LOCK_EX``)
  or shared (``LOCK_SH``) locks on a target file. Works on POSIX
  (``fcntl``) and Windows (``portalocker``); degrades to a warning if
  neither backend is available.

The thin :func:`acquire_lock` / :func:`release_lock` helpers are kept for
backwards compatibility but new code should prefer :class:`FileLock`.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import BinaryIO

# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #

try:  # POSIX (Linux, macOS, VSC compute nodes)
    import fcntl  # type: ignore
    _HAS_FCNTL = True
except ImportError:
    fcntl = None  # type: ignore
    _HAS_FCNTL = False

try:  # Windows + portable fallback
    import portalocker  # type: ignore
    _HAS_PORTALOCKER = True
except ImportError:
    portalocker = None  # type: ignore
    _HAS_PORTALOCKER = False


# --------------------------------------------------------------------------- #
# Functional API (backwards-compatible with the experiment drivers)
# --------------------------------------------------------------------------- #

def acquire_lock(file_handle: BinaryIO, exclusive: bool = True) -> bool:
    """Acquire an advisory lock on ``file_handle``.

    Returns True on success or if no locking backend is available (so callers
    can still proceed in single-process setups). Returns False only when an
    IO-level error is raised by the backend.
    """
    if _HAS_FCNTL:
        try:
            lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(file_handle.fileno(), lock_type)
            return True
        except (IOError, OSError):
            return False
    if _HAS_PORTALOCKER:
        try:
            lock_type = portalocker.LOCK_EX if exclusive else portalocker.LOCK_SH
            portalocker.lock(file_handle, lock_type)
            return True
        except Exception:
            return False
    # No backend available -- warn once at import-site and proceed unlocked.
    return True


def release_lock(file_handle: BinaryIO) -> None:
    """Release an advisory lock previously acquired via :func:`acquire_lock`."""
    if _HAS_FCNTL:
        try:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
        except (IOError, OSError):
            pass
    elif _HAS_PORTALOCKER:
        try:
            portalocker.unlock(file_handle)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Context-manager API
# --------------------------------------------------------------------------- #

class FileLock:
    """Exclusive (or shared) file lock as a context manager.

    Usage::

        with FileLock(path, exclusive=True) as fh:
            # fh is open in 'a+' mode, positioned at start; safe to read/write.
            ...

    The underlying file is created if it does not exist; the parent directory
    is created lazily. On systems without either ``fcntl`` or ``portalocker``
    the lock is a no-op, a warning is emitted once, and the caller proceeds
    un-serialised -- appropriate for single-process workloads but unsafe
    under SLURM array concurrency.
    """

    _warned_missing_backend = False

    def __init__(
        self,
        filepath: os.PathLike,
        exclusive: bool = True,
        timeout: float = 30.0,
        binary: bool = False,
    ):
        self.filepath = Path(filepath)
        self.exclusive = exclusive
        self.timeout = timeout  # Reserved for future backoff-aware backends
        self.binary = binary  # Open in 'a+b' instead of 'a+' (needed for pickle).
        self._fh: BinaryIO | None = None
        self._acquired = False

    def __enter__(self) -> BinaryIO:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        # 'a+' / 'a+b' creates-if-missing, preserves existing content, allows read+write
        mode = "a+b" if self.binary else "a+"
        self._fh = open(self.filepath, mode)
        self._fh.seek(0)
        self._acquired = acquire_lock(self._fh, exclusive=self.exclusive)
        if not self._acquired and not (_HAS_FCNTL or _HAS_PORTALOCKER):
            if not FileLock._warned_missing_backend:
                warnings.warn(
                    "No file-locking backend available (install 'portalocker' on "
                    "Windows or use a POSIX system). Concurrent writers may "
                    "corrupt pickle files.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                FileLock._warned_missing_backend = True
        return self._fh

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            if self._acquired:
                release_lock(self._fh)
            self._fh.close()
            self._fh = None


__all__ = [
    "FileLock",
    "acquire_lock",
    "release_lock",
]
