"""Tests for the dataset display/ordering registry (src/data/dataset_registry.py).

These encode the renaming spec's verification section as fail-loud checks:

1. 14 PD + 7 LGD entries; new IDs ``PD1..PD14`` / ``LGD1..LGD7``, no gaps/dupes.
2. Every slug in the results/processed directory appears exactly once in the
   registry, and vice versa.
3. Sorting by ``(is_proprietary, display_name)`` reproduces the new-ID order.
4. No real proprietary dataset name survives in code that renders paper output.
5. The mapping table is printable for eyeball verification.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.data.dataset_names import (  # public accessor
    REGISTRY_AVAILABLE,
    canonical_slug,
    display_name,
    entries_for_task,
    format_mapping_table,
    is_proprietary,
    paper_id,
    sort_datasets,
    sort_key,
    validate_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The registry is gitignored (it de-anonymises the proprietary datasets), so on
# a fresh clone there is nothing to test -- skip rather than fail.
pytestmark = pytest.mark.skipif(
    not REGISTRY_AVAILABLE,
    reason="src/data/dataset_registry.py absent (private, gitignored)",
)

if REGISTRY_AVAILABLE:
    from src.data.dataset_registry import REGISTRY
else:  # pragma: no cover
    REGISTRY = {}

# Real names / slugs of the proprietary datasets. None of these may appear in
# rendered paper output (figure labels, generated tables, captions).
FORBIDDEN_IN_OUTPUT = [
    "Bank Status", "bank_status",
    "AXA", "axa",
    "Loss2", "loss2",
    "Base Model", "base_model",
    "Base Modelisation", "base_modelisation",
    "HELOC", "heloc",
    "Loan Default", "loan_default",
]


class TestRegistryInvariants:

    def test_validates(self):
        validate_registry()                      # spec checks 1 + 3

    def test_counts_and_ids(self):
        assert len(entries_for_task("pd")) == 14
        assert len(entries_for_task("lgd")) == 7
        assert {e.new_id for e in entries_for_task("pd")} == {f"PD{i}" for i in range(1, 15)}
        assert {e.new_id for e in entries_for_task("lgd")} == {f"LGD{i}" for i in range(1, 8)}

    def test_sort_reproduces_new_id_order(self):
        for task, prefix in (("pd", "PD"), ("lgd", "LGD")):
            entries = entries_for_task(task)      # already in sort_key order
            assert [e.new_id for e in entries] == [
                f"{prefix}{i}" for i in range(1, len(entries) + 1)
            ]

    def test_public_before_proprietary(self):
        for task in ("pd", "lgd"):
            flags = [e.proprietary for e in entries_for_task(task)]
            # once proprietary starts it never goes back to public
            assert flags == sorted(flags), f"{task}: public must precede proprietary"

    def test_public_block_is_alphabetical(self):
        for task in ("pd", "lgd"):
            pub = [e.display_name for e in entries_for_task(task) if not e.proprietary]
            assert pub == sorted(pub, key=str.lower)

    def test_proprietary_names_are_anonymised(self):
        prop = [e.display_name for e in REGISTRY.values() if e.proprietary]
        assert sorted(prop) == [
            "PropLGD1", "PropLGD2", "PropLGD3", "PropLGD4", "PropLGD5",
            "PropPD1", "PropPD2",
        ]

    def test_mapping_table_prints(self):
        table = format_mapping_table()
        assert "PropPD1" in table and "0009.bank_status" in table
        assert "PD9" in table and "PD13" in table   # old -> new both shown


class TestLookups:

    @pytest.mark.parametrize("alias", [
        "0009.bank_status", "0009_bank_status", "bank_status", "PD13", "pd13",
    ])
    def test_alias_resolution(self, alias):
        """Figure-filename and hand-typed forms must resolve to one entry."""
        assert canonical_slug(alias) == "0009.bank_status"
        assert display_name(alias) == "PropPD1"

    def test_paper_id(self):
        assert paper_id("0001.gmsc") == "PD3"
        assert paper_id("0001.heloc") == "LGD3"

    def test_proprietary_flags(self):
        assert is_proprietary("0009.bank_status")
        assert is_proprietary("0001.heloc")
        assert not is_proprietary("0001.gmsc")

    def test_unknown_dataset_degrades_gracefully(self):
        """An unregistered dataset must not crash a plot -- it gets a readable
        label and sorts last."""
        assert display_name("0099.brand_new") == "brand new"
        assert sort_key("0099.brand_new")[0] == 2      # after public + proprietary


class TestOrdering:

    def test_sort_datasets_pd(self):
        slugs = [e.slug for e in entries_for_task("pd")]
        shuffled = sorted(slugs)                        # slug order == OLD order
        assert sort_datasets(shuffled) == slugs

    def test_gmsc_is_pd3_not_pd1(self):
        """Regression guard: the OLD numbering had gmsc first."""
        order = sort_datasets([e.slug for e in entries_for_task("pd")])
        assert order[0] == "0007.cobranded"             # new PD1
        assert order.index("0001.gmsc") == 2            # new PD3

    def test_proprietary_lgd_tail_order(self):
        order = sort_datasets([e.slug for e in entries_for_task("lgd")])
        assert [display_name(d) for d in order[-5:]] == [
            "PropLGD1", "PropLGD2", "PropLGD3", "PropLGD4", "PropLGD5",
        ]


class TestAgainstDisk:

    def test_registry_matches_processed_datasets(self):
        """Spec check 2: disk <-> registry, exactly once each way."""
        from src.data.dataset_inventory import list_datasets

        found = {t: list_datasets(t) for t in ("pd", "lgd")}
        if not any(found.values()):
            pytest.skip("no processed datasets on this machine")
        validate_registry(found)


class TestNoHardCodedNames:
    """Spec check 4: no proprietary name may reach rendered paper output, and no
    display name may be hard-coded outside the registry."""

    #: modules that render paper output (labels, tables, captions)
    OUTPUT_MODULES = [
        "src/visualizations/experiment_plots.py",
        "src/visualizations/data_exploration.py",
        "src/utils/statistical_testing.py",
        "src/utils/generate_captions.py",
    ]

    def test_no_proprietary_names_in_output_modules(self):
        offenders = []
        for rel in self.OUTPUT_MODULES:
            text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
            for bad in FORBIDDEN_IN_OUTPUT:
                # Word-ish match; skip matches inside comments is overkill --
                # these strings have no business in these modules at all.
                if re.search(rf"\b{re.escape(bad)}\b", text):
                    offenders.append(f"{rel}: {bad!r}")
        assert not offenders, (
            "proprietary dataset name(s) hard-coded in output code: " + "; ".join(offenders)
        )

    def test_display_names_only_defined_in_registry(self):
        """``PropPD*`` / ``PropLGD*`` literals must exist only in the registry
        (and in this test), never in plotting/table code."""
        offenders = []
        for path in (PROJECT_ROOT / "src").rglob("*.py"):
            if path.name == "dataset_registry.py":
                continue
            text = path.read_text(encoding="utf-8")
            if re.search(r"\bProp(PD|LGD)\d", text):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
        assert not offenders, f"anonymised names hard-coded outside the registry: {offenders}"

    def test_no_adhoc_slug_prettifiers_left(self):
        """The old ``split('.')[-1]`` / ``replace('_',' ')`` dataset prettifiers
        must be gone from the plotting modules -- they bypass the registry."""
        offenders = []
        for rel in ["src/visualizations/experiment_plots.py",
                    "src/visualizations/data_exploration.py"]:
            text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
            for pat in [r'str\(d\)\.split\("\."\)', r'str\(dataset\)\.split\("\."']:
                if re.search(pat, text):
                    offenders.append(f"{rel}: {pat}")
        assert not offenders, f"ad-hoc dataset prettifier still present: {offenders}"
