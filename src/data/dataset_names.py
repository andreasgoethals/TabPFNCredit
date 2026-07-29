"""Public accessor for dataset display names + paper ordering.

Why this indirection exists
---------------------------
The actual mapping lives in :mod:`src.data.dataset_registry`, which is
**gitignored**: it pairs each proprietary dataset's real slug with its
anonymised paper name, which is exactly the information that must not be
published. Every plotting / table / caption module imports *this* module
instead, so the repository still works without the private file:

* **Registry present** (the normal case, and the only way to build the paper):
  every call delegates to it -- proprietary datasets are anonymised and
  datasets are ordered public-alphabetical-then-proprietary.
* **Registry absent** (a fresh clone): figures still render, using the raw
  on-disk slug as the label and slug order for sorting, and a single warning is
  logged. Nothing crashes, and no name can leak because there is nothing to map.

That asymmetry is deliberate: a missing registry must never silently produce
output that *looks* anonymised. The warning names the file to restore.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:  # the private mapping
    # NB: catch ImportError, not just ModuleNotFoundError -- the
    # ``from package import module`` form raises the former (not the latter)
    # when the submodule file is absent, which is exactly the fresh-clone case.
    from src.data import dataset_registry as _reg

    REGISTRY_AVAILABLE = True
except ImportError:  # pragma: no cover -- exercised on a fresh clone
    _reg = None  # type: ignore[assignment]
    REGISTRY_AVAILABLE = False
    logger.warning(
        "src/data/dataset_registry.py is missing (it is gitignored because it "
        "maps proprietary datasets to their anonymised paper names). Figures "
        "will fall back to RAW dataset slugs and slug ordering -- restore your "
        "local copy before generating anything for the paper."
    )


def _fallback_label(dataset: object) -> str:
    """Readable form of a raw slug, used only when the registry is absent."""
    return re.sub(r"^\d+[._-]", "", str(dataset)).replace("_", " ")


def display_name(dataset: object) -> str:
    """Reader-facing dataset label (anonymised when proprietary)."""
    if REGISTRY_AVAILABLE:
        return _reg.display_name(dataset)
    return _fallback_label(dataset)


def paper_id(dataset: object) -> str:
    """New paper ID (``PD3``, ``LGD1``, ...)."""
    if REGISTRY_AVAILABLE:
        return _reg.paper_id(dataset)
    return _fallback_label(dataset)


def is_proprietary(dataset: object) -> bool:
    """True iff the dataset may not be named in the paper.

    Without the registry this returns ``False``: we cannot know, and claiming
    otherwise would suppress figures for no reason.
    """
    if REGISTRY_AVAILABLE:
        return _reg.is_proprietary(dataset)
    return False


def sort_key(dataset: object) -> Tuple:
    """Paper order key: public first (alphabetical), then proprietary."""
    if REGISTRY_AVAILABLE:
        return _reg.sort_key(dataset)
    return (0, str(dataset))          # slug order


def sort_datasets(datasets: Iterable[object]) -> List[str]:
    """``datasets`` sorted into paper order."""
    if REGISTRY_AVAILABLE:
        return _reg.sort_datasets(datasets)
    return sorted(str(d) for d in datasets)


def display_names(datasets: Sequence[object]) -> List[str]:
    """Map a sequence of slugs to display labels (order preserved)."""
    return [display_name(d) for d in datasets]


def canonical_slug(dataset: object) -> Optional[str]:
    """Canonical on-disk slug for any accepted alias, else ``None``."""
    if REGISTRY_AVAILABLE:
        return _reg.canonical_slug(dataset)
    return str(dataset)


def entries_for_task(task: str) -> List:
    """Registry entries for ``task`` in paper order (empty without the registry)."""
    if REGISTRY_AVAILABLE:
        return _reg.entries_for_task(task)
    return []


def registry() -> dict:
    """``{slug: DatasetEntry}``; empty when the private registry is absent.

    Callers that need to iterate the mapping (e.g. the caption generator, which
    suppresses the slug-named twin of a proprietary figure) use this instead of
    importing the private module directly, so they keep working without it.
    """
    if REGISTRY_AVAILABLE:
        return dict(_reg.REGISTRY)
    return {}


def format_mapping_table() -> str:
    """Printable old -> new ID table."""
    if REGISTRY_AVAILABLE:
        return _reg.format_mapping_table()
    return "(src/data/dataset_registry.py not available -- no mapping to show)"


def validate_registry(known_slugs: Optional[dict] = None) -> None:
    """Validate the registry; a no-op (with a warning) when it is absent."""
    if REGISTRY_AVAILABLE:
        _reg.validate_registry(known_slugs)
        return
    logger.warning("validate_registry: registry not available -- nothing to validate.")


__all__ = [
    "REGISTRY_AVAILABLE",
    "display_name",
    "paper_id",
    "is_proprietary",
    "sort_key",
    "sort_datasets",
    "display_names",
    "canonical_slug",
    "entries_for_task",
    "registry",
    "format_mapping_table",
    "validate_registry",
]
