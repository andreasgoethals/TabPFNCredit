"""Public accessor for dataset display names + paper ordering.

Why this indirection exists
---------------------------
The historical mapping lives in the gitignored :mod:`src.data.dataset_registry`
alongside proprietary dataset names. This module records public corrections
and exposes the effective names to plots, tables, and captions:

* **Registry present:** the corrected public names and remaining proprietary
  aliases are ordered public-alphabetical-then-proprietary.
* **Registry absent:** the public corrections still resolve. Other datasets
  use the on-disk slug, with a warning to restore the registry for paper output.

That asymmetry is deliberate: a missing registry must never silently produce
output that *looks* anonymised. The warning names the file to restore.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_PUBLIC_PD_NAMES = {
    "0009.bank_status": ("Bank Loan Status Dataset", "PD13", "PropPD1"),
    "0011.loan_default": ("Loan Default Prediction - Imperial College London", "PD14", "PropPD2"),
}


def _alias_key(dataset: object) -> str:
    return str(dataset).strip().lower().replace(" ", "_")


_PUBLIC_PD_ALIASES = {
    _alias_key(alias): slug
    for slug, (name, paper, former) in _PUBLIC_PD_NAMES.items()
    for alias in (slug, slug.replace(".", "_"), slug.split(".", 1)[1], name, paper, former)
}

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
    slug = _PUBLIC_PD_ALIASES.get(_alias_key(dataset))
    if slug is not None:
        return _PUBLIC_PD_NAMES[slug][0]
    if REGISTRY_AVAILABLE:
        return _reg.display_name(dataset)
    return _fallback_label(dataset)


def paper_id(dataset: object) -> str:
    """Stable paper ID (``PD3``, ``LGD1``, ...)."""
    slug = _PUBLIC_PD_ALIASES.get(_alias_key(dataset))
    if slug is not None:
        return _PUBLIC_PD_NAMES[slug][1]
    if REGISTRY_AVAILABLE:
        return _reg.paper_id(dataset)
    return _fallback_label(dataset)


def is_proprietary(dataset: object) -> bool:
    """True iff the dataset may not be named in the paper.

    Without the registry this returns ``False``: we cannot know, and claiming
    otherwise would suppress figures for no reason.
    """
    if _alias_key(dataset) in _PUBLIC_PD_ALIASES:
        return False
    if REGISTRY_AVAILABLE:
        return _reg.is_proprietary(dataset)
    return False


def sort_key(dataset: object) -> Tuple:
    """Paper order key: public first (alphabetical), then proprietary."""
    slug = _PUBLIC_PD_ALIASES.get(_alias_key(dataset))
    if slug is not None:
        return (0, _PUBLIC_PD_NAMES[slug][0].lower())
    if REGISTRY_AVAILABLE:
        return _reg.sort_key(dataset)
    return (0, str(dataset))          # slug order


def sort_datasets(datasets: Iterable[object]) -> List[str]:
    """``datasets`` sorted into paper order."""
    return sorted((str(d) for d in datasets), key=sort_key)


def display_names(datasets: Sequence[object]) -> List[str]:
    """Map a sequence of slugs to display labels (order preserved)."""
    return [display_name(d) for d in datasets]


def canonical_slug(dataset: object) -> Optional[str]:
    """Canonical on-disk slug for any accepted alias, else ``None``."""
    slug = _PUBLIC_PD_ALIASES.get(_alias_key(dataset))
    if slug is not None:
        return slug
    if REGISTRY_AVAILABLE:
        return _reg.canonical_slug(dataset)
    return str(dataset)


def entries_for_task(task: str) -> List:
    """Registry entries for ``task`` in paper order (empty without the registry)."""
    entries = (entry for entry in registry().values() if entry.task == task.lower())
    return sorted(entries, key=lambda entry: sort_key(entry.slug))


def registry() -> dict:
    """``{slug: DatasetEntry}``; empty when the private registry is absent.

    Callers that need to iterate the mapping (e.g. the caption generator, which
    suppresses the slug-named twin of a proprietary figure) use this instead of
    importing the private module directly, so they keep working without it.
    """
    if not REGISTRY_AVAILABLE:
        return {}
    entries = dict(_reg.REGISTRY)
    for slug, (name, _, _) in _PUBLIC_PD_NAMES.items():
        if slug in entries:
            entries[slug] = replace(entries[slug], display_name=name, proprietary=False)
    return entries


def format_mapping_table() -> str:
    """Printable old -> new ID table."""
    if not REGISTRY_AVAILABLE:
        return "(src/data/dataset_registry.py not available -- no full mapping to show)"
    lines = [f"{'slug':24s} {'old':>5s} -> {'paper':<5s} {'display name':48s} source"]
    for task in ("pd", "lgd"):
        for entry in entries_for_task(task):
            lines.append(
                f"{entry.slug:24s} {entry.old_id:>5s} -> {entry.new_id:<5s} "
                f"{entry.display_name:48s} {entry.source}"
            )
    return "\n".join(lines)


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


if __name__ == "__main__":  # pragma: no cover - manual inspection
    print(format_mapping_table())
