"""Tests for build_method_name / parse_method_name (sweep suffixes)."""

import pytest

from src.utils.result_io import build_method_name, parse_method_name


class TestBuild:

    def test_no_sweep(self):
        assert build_method_name("xgboost") == "xgboost"
        assert build_method_name("xgboost", {}) == "xgboost"

    def test_hpo_true(self):
        assert build_method_name("xgboost", {"HPO": True}) == "xgboost__HPO"

    def test_hpo_false_is_dropped(self):
        # NO_HPO is the default -- no suffix
        assert build_method_name("xgboost", {"HPO": False}) == "xgboost"

    def test_integer_row_limit(self):
        assert build_method_name("tabpfn_v3", {"row": 20000}) == "tabpfn_v3__row20000"

    def test_float_minority_proportion(self):
        # 0.15 -> 'min0p15'   (trailing zeros trimmed)
        # 0.0025 -> 'min0p0025'
        assert build_method_name("LogReg", {"min": 0.15}) == "LogReg__min0p15"
        assert build_method_name("LogReg", {"min": 0.0025}) == "LogReg__min0p0025"

    def test_multiple_axes_are_sorted(self):
        assert (
            build_method_name("xgboost", {"HPO": True, "row": 5000})
            == "xgboost__HPO__row5000"
        )


class TestParse:

    def test_no_sweep(self):
        assert parse_method_name("xgboost") == {"method": "xgboost", "sweep": {}}

    def test_hpo(self):
        assert parse_method_name("xgboost__HPO") == {
            "method": "xgboost", "sweep": {"HPO": True}
        }

    def test_integer(self):
        assert parse_method_name("tabpfn_v3__row20000") == {
            "method": "tabpfn_v3", "sweep": {"row": 20000}
        }

    def test_float(self):
        result = parse_method_name("LogReg__min0p0025")
        assert result["method"] == "LogReg"
        assert result["sweep"]["min"] == pytest.approx(0.0025)

    def test_round_trip(self):
        for method, sweep in [
            ("xgboost", {}),
            ("xgboost", {"HPO": True}),
            ("tabpfn_v3", {"row": 20000}),
            ("LogReg", {"min": 0.0025}),
        ]:
            stem = build_method_name(method, sweep)
            parsed = parse_method_name(stem)
            assert parsed["method"] == method, (method, stem, parsed)
            # HPO=False is intentionally dropped on the way in
            expected_sweep = {k: v for k, v in sweep.items() if not (k == "HPO" and not v)}
            assert parsed["sweep"] == expected_sweep, (method, stem, parsed)
