"""Tests for the dataset display/ordering registry (src/data/dataset_registry.py).

These encode the renaming spec's verification section as fail-loud checks:

1. 14 PD + 7 LGD entries; new IDs ``PD1..PD14`` / ``LGD1..LGD7``, no gaps/dupes.
2. Every slug in the results/processed directory appears exactly once in the
   registry, and vice versa.
3. Sorting by ``(is_proprietary, display_name)`` reproduces the new-ID order.
4. No real proprietary dataset name survives in code that renders paper output.
5. The mapping table is printable for eyeball verification.

WHY NOTHING HERE IS HARD-CODED
------------------------------
``dataset_registry.py`` is gitignored precisely because it pairs each
proprietary dataset's REAL slug with its anonymised paper name -- it is the
de-anonymisation key. An earlier version of this file asserted those pairs
literally (``canonical_slug("0009.<real>") == ...``, ``display_name(...) ==
"PropPD1"``), which published the key in a tracked file and made gitignoring
the registry pointless.

Every case is now DERIVED from the registry at run time. That leaks nothing,
and it is strictly stronger: the checks cover all 21 datasets instead of the
two or three a human had copied in, and they cannot drift when the registry
changes. The whole module skips when the registry is absent, so a fresh clone
neither fails nor learns anything.
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


# ---------------------------------------------------------------------------
#  Registry-derived fixtures -- the only source of real names in this file
# ---------------------------------------------------------------------------

def _proprietary_entries():
    return [e for e in REGISTRY.values() if e.proprietary]


def _public_entries():
    return [e for e in REGISTRY.values() if not e.proprietary]


def _anonymised_pattern():
    """``PropPD3`` / ``PropLGD1`` and nothing else."""
    return re.compile(r"^Prop(PD|LGD)\d+$")


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

    def test_every_proprietary_display_name_is_anonymised(self):
        """No proprietary dataset may keep a recognisable display name, and the
        anonymised names must be a gap-free PropPD1..n / PropLGD1..n run."""
        pat = _anonymised_pattern()
        prop = _proprietary_entries()
        assert prop, "no proprietary datasets registered -- fixture is wrong"
        for e in prop:
            assert pat.match(e.display_name), (
                f"{e.new_id} display name is not anonymised"
            )
            # the anonymised name must not embed any part of the real slug
            slug_words = re.split(r"[^a-z0-9]+", e.slug.lower())
            for word in slug_words:
                if len(word) > 3:
                    assert word not in e.display_name.lower(), (
                        f"{e.new_id} display name leaks part of its slug"
                    )
        for task, prefix in (("pd", "PropPD"), ("lgd", "PropLGD")):
            names = sorted(e.display_name for e in prop if e.task == task)
            assert names == [f"{prefix}{i}" for i in range(1, len(names) + 1)]

    def test_public_display_names_are_not_anonymised(self):
        """Public datasets keep their real, citable names -- the anonymisation
        must not have been applied indiscriminately."""
        pat = _anonymised_pattern()
        for e in _public_entries():
            assert not pat.match(e.display_name)

    def test_mapping_table_prints_every_dataset(self):
        table = format_mapping_table()
        for e in REGISTRY.values():
            assert e.slug in table, f"{e.new_id} missing from the mapping table"
            assert e.display_name in table
            assert e.new_id in table
            assert e.old_id in table              # old -> new both shown


class TestLookups:

    def test_alias_resolution_for_every_dataset(self):
        """Every alias form a figure filename or a human might use must resolve
        to the same entry: the dotted slug, the underscore form, the bare name,
        the new paper ID (either case) and the display name."""
        for e in REGISTRY.values():
            bare = e.slug.split(".", 1)[1]
            for alias in (e.slug, e.slug.replace(".", "_"), bare,
                          e.new_id, e.new_id.lower(), e.display_name):
                assert canonical_slug(alias) == e.slug, (
                    f"alias {alias!r} did not resolve to {e.new_id}"
                )
                assert display_name(alias) == e.display_name

    def test_paper_id_round_trips(self):
        for e in REGISTRY.values():
            assert paper_id(e.slug) == e.new_id

    def test_proprietary_flags_match_the_registry(self):
        for e in REGISTRY.values():
            assert is_proprietary(e.slug) is bool(e.proprietary)
        assert any(is_proprietary(e.slug) for e in REGISTRY.values())
        assert any(not is_proprietary(e.slug) for e in REGISTRY.values())

    def test_unknown_dataset_degrades_gracefully(self):
        """An unregistered dataset must not crash a plot -- it gets a readable
        label and sorts last."""
        assert display_name("0099.brand_new") == "brand new"
        assert sort_key("0099.brand_new")[0] == 2      # after public + proprietary


class TestOrdering:

    def test_sort_datasets_matches_registry_order(self):
        for task in ("pd", "lgd"):
            slugs = [e.slug for e in entries_for_task(task)]
            shuffled = sorted(slugs)                    # slug order == OLD order
            assert sort_datasets(shuffled) == slugs

    def test_renumbering_actually_moved_something(self):
        """Regression guard for the renumbering: the new order must NOT simply
        be the old slug order, or the whole exercise was a no-op."""
        moved = 0
        for task in ("pd", "lgd"):
            slugs = [e.slug for e in entries_for_task(task)]
            if sorted(slugs) != slugs:
                moved += 1
        assert moved, "new ordering is identical to the old slug ordering"

    def test_proprietary_datasets_form_the_tail(self):
        for task in ("pd", "lgd"):
            order = sort_datasets([e.slug for e in entries_for_task(task)])
            prop_positions = [i for i, d in enumerate(order) if is_proprietary(d)]
            if not prop_positions:
                continue
            # contiguous, and running to the very end
            assert prop_positions == list(
                range(min(prop_positions), len(order))
            ), f"{task}: proprietary datasets are not a contiguous tail"
            # and numbered in order along that tail
            names = [display_name(order[i]) for i in prop_positions]
            assert names == sorted(names, key=lambda s: int(re.sub(r"\D", "", s)))


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
    display name may be hard-coded outside the registry.

    The forbidden strings are read FROM the registry rather than listed here,
    so this file never becomes the leak it is guarding against.
    """

    #: modules that render paper output (labels, tables, captions)
    OUTPUT_MODULES = [
        "src/visualizations/experiment_plots.py",
        "src/visualizations/data_exploration.py",
        "src/utils/statistical_testing.py",
        "src/utils/generate_captions.py",
    ]

    def _forbidden_strings(self):
        """Slug and slug-derived forms of every proprietary dataset."""
        out = set()
        for e in _proprietary_entries():
            bare = e.slug.split(".", 1)[1]
            out.add(bare)                        # e.g. some_bank
            out.add(bare.replace("_", " "))      # e.g. some bank
            out.add(bare.replace("_", ""))
        # single-word fragments long enough to be identifying
        return {s for s in out if len(s) > 3}

    def test_no_proprietary_names_in_output_modules(self):
        forbidden = self._forbidden_strings()
        assert forbidden, "no forbidden strings derived -- fixture is wrong"
        offenders = []
        for rel in self.OUTPUT_MODULES:
            text = (PROJECT_ROOT / rel).read_text(encoding="utf-8").lower()
            for bad in forbidden:
                if re.search(rf"\b{re.escape(bad)}\b", text):
                    offenders.append(f"{rel}: {bad!r}")
        assert not offenders, (
            "proprietary dataset name(s) hard-coded in output code: "
            + "; ".join(sorted(offenders))
        )

    def test_display_names_only_defined_in_registry(self):
        """``PropPD*`` / ``PropLGD*`` literals must exist only in the registry
        (and in this test's regex), never in plotting/table code."""
        offenders = []
        for path in (PROJECT_ROOT / "src").rglob("*.py"):
            if path.name == "dataset_registry.py":
                continue
            text = path.read_text(encoding="utf-8")
            if re.search(r"\bProp(PD|LGD)\d", text):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
        assert not offenders, f"anonymised names hard-coded outside the registry: {offenders}"

    def test_this_test_file_leaks_nothing(self):
        """Self-check: the guard must not itself contain a real proprietary slug
        or a real-to-anonymised pairing. Everything here is derived at run
        time, so the source must be clean."""
        source = Path(__file__).read_text(encoding="utf-8").lower()
        for e in _proprietary_entries():
            bare = e.slug.split(".", 1)[1]
            assert bare not in source, (
                f"this test file names the proprietary slug {bare!r} -- "
                f"publishing it would defeat gitignoring the registry"
            )


class TestNothingTrackedLeaksAProprietarySlug:
    """Publish gate: no file git would publish may name a proprietary dataset.

    This catches the class of mistake the rest of this module guards against,
    but across the WHOLE tracked tree rather than a hand-picked module list --
    committed notebook outputs and generated markdown have both leaked slugs
    before.

    There are no exemptions. Every experiment's ``CONFIG_DATA.yaml`` selects
    datasets by row count (``min_rows``) rather than by name, so no tracked file
    needs to name an on-disk dataset directory at all.
    """

    def test_no_tracked_file_names_a_proprietary_dataset(self):
        import subprocess

        forbidden = set()
        for e in _proprietary_entries():
            bare = e.slug.split(".", 1)[1]
            if len(bare) > 3:
                forbidden.add(bare.lower())
        assert forbidden, "no forbidden slugs derived -- fixture is wrong"

        tracked = subprocess.run(
            ["git", "ls-files"], cwd=PROJECT_ROOT,
            capture_output=True, text=True,
        ).stdout.split()

        offenders = []
        for rel in tracked:
            path = PROJECT_ROOT / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:                      # pragma: no cover
                continue
            for bad in forbidden:
                if re.search(rf"\b{re.escape(bad)}\b", text):
                    offenders.append(rel)
                    break
        assert not offenders, (
            "tracked file(s) name a proprietary dataset slug -- publishing them "
            f"would defeat gitignoring the registry: {sorted(set(offenders))}"
        )


class TestNothingTrackedLeaksAProprietaryColumnName:
    """Publish gate: no tracked file may disclose a proprietary dataset's schema.

    The sibling gate above guards dataset NAMES. The raw column names are just as
    sensitive -- they describe the data provider's internal schema -- and they
    live in two gitignored places (``dataset_preprocessing.py`` and each
    dataset's files under ``data/``), so a copy-paste into tracked code, a
    docstring, or a committed notebook output would publish them.

    Identifiers come from the authoritative source: each proprietary dataset's
    processed ``info.json`` plus its raw CSV header. Two subtractions keep this
    from crying wolf:

    * columns a PUBLIC dataset also uses are not secret;
    * generic credit vocabulary ("age", "term", "income") says nothing about a
      schema, and a wide proprietary CSV really does contain columns literally
      named "AUTO" and "Source".

    What remains are DISTINCTIVE identifiers -- >= 6 characters and carrying
    structure (an underscore, a digit, or camelCase) -- which is what a schema
    disclosure actually looks like.

    Skips when the datasets are not present (a fresh clone, or CI).
    """

    GENERIC = {
        "age", "term", "date", "amount", "income", "balance", "limit", "loan",
        "credit", "score", "rate", "ratio", "value", "target", "default",
        "status", "type", "code", "index", "count", "total", "purpose",
        "housing", "savings", "employment", "duration", "history", "gender",
        "region", "city", "state", "country", "year", "month", "flag", "number",
        "utilization", "principal", "interest", "price", "cost", "other",
        "unknown", "missing", "dropped", "retained", "source", "auto",
    }

    @staticmethod
    def _distinctive(name: str) -> bool:
        """Looks like a schema identifier rather than an English word."""
        if len(name) < 6:
            return False
        return ("_" in name or any(c.isdigit() for c in name)
                or re.search(r"[a-z][A-Z]", name) is not None)

    @classmethod
    def _columns_of(cls, task: str, slug: str) -> set:
        import csv as _csv
        import json as _json

        cols = set()
        info = PROJECT_ROOT / "data" / "processed" / task / slug / "info.json"
        if info.exists():
            meta = _json.loads(info.read_text(encoding="utf-8"))
            for key in ("numerical_cols", "categorical_cols"):
                cols.update(meta.get(key) or [])
        raw = PROJECT_ROOT / "data" / "raw" / task / f"{slug}.csv"
        if raw.exists():
            with raw.open(newline="", encoding="utf-8", errors="ignore") as fh:
                cols.update(h.strip() for h in next(_csv.reader(fh), []) if h.strip())
        return cols

    def _private_columns(self):
        from src.data.dataset_registry import REGISTRY, is_proprietary

        private, public = set(), set()
        for slug in REGISTRY:
            if not isinstance(slug, str):
                continue
            task = ("lgd" if (PROJECT_ROOT / "data" / "processed" / "lgd" / slug).exists()
                    else "pd")
            cols = self._columns_of(task, slug)
            (private if is_proprietary(slug) else public).update(cols)
        public_lower = {c.lower() for c in public}
        return {c for c in private
                if c.lower() not in public_lower
                and c.lower() not in self.GENERIC
                and re.fullmatch(r"[A-Za-z][\w .\-]*", c)
                and self._distinctive(c)}

    def test_no_tracked_file_names_a_proprietary_column(self):
        import subprocess

        private = self._private_columns()
        if not private:
            pytest.skip("proprietary datasets not present locally")

        pattern = re.compile(
            r"(?<![A-Za-z0-9_])(" +
            "|".join(re.escape(c) for c in sorted(private, key=len, reverse=True)) +
            r")(?![A-Za-z0-9_])", re.I)

        tracked = subprocess.run(
            ["git", "ls-files"], cwd=PROJECT_ROOT,
            capture_output=True, text=True,
        ).stdout.split()

        offenders = {}
        for rel in tracked:
            path = PROJECT_ROOT / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:                          # pragma: no cover
                continue
            found = {m.group(1).lower() for m in pattern.finditer(text)}
            if found:
                offenders[rel] = len(found)

        assert not offenders, (
            "tracked file(s) disclose proprietary raw column names -- the count "
            f"of distinct identifiers per file is shown: {offenders}"
        )

    def test_the_private_modules_are_not_tracked(self):
        """Both sources of the schema must stay out of git."""
        import subprocess

        tracked = set(subprocess.run(
            ["git", "ls-files"], cwd=PROJECT_ROOT,
            capture_output=True, text=True,
        ).stdout.split())
        for rel in ("src/data/dataset_preprocessing.py",
                    "src/data/dataset_registry.py"):
            assert rel not in tracked, (
                f"{rel} is tracked; it names proprietary schema / the "
                f"de-anonymisation key and must be gitignored")
