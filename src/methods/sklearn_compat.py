"""Restore ``BaseEstimator._validate_data`` for vendored estimators on sklearn >= 1.6.

Why this exists
---------------
scikit-learn 1.6 removed ``BaseEstimator._validate_data`` in favour of the
module-level ``sklearn.utils.validation.validate_data(estimator, X, y, ...)``.
TALENT vendors TabICL v1, whose ``fit``/``predict`` branch on the sklearn version
but call the METHOD in both arms::

    if OLD_SKLEARN:
        X, y = self._validate_data(X, y, dtype=None, cast_to_ndarray=False)
    else:
        X, y = self._validate_data(X, y, dtype=None, skip_check_array=True)

The ``else`` arm was clearly written for the new API -- ``skip_check_array`` is a
parameter of the new *function*, not of the old method -- but kept ``self.``. So
on sklearn >= 1.6 every TabICL v1 fit dies with::

    AttributeError: 'TabICLClassifier' object has no attribute '_validate_data'

which is what killed ``pd/0012.home_credit/tabicl``. Enabling TabFM is what
forced sklearn >= 1.6 into the cluster environment, so the two requirements
collided.

Why here and not in the TALENT fork
-----------------------------------
A fix inside the fork needs a push AND a ``pip install --force-reinstall`` on the
cluster; that second step was missed once before and cost two H100 jobs. Code in
*this* repo deploys with a plain ``git pull``, so the fix travels with the run.
The same reasoning put the TabFM chunked-inference fix in ``tabfm_chunked``.

What it does
------------
Re-attaches ``_validate_data`` to ``BaseEstimator`` as a thin forwarder to the new
free function. It restores documented pre-1.6 behaviour rather than inventing any,
and it is a no-op on sklearn < 1.6 (where the real method still exists) and on any
version that does not provide ``validate_data``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_INSTALLED = False


def install_sklearn_validate_data_shim() -> bool:
    """Ensure ``BaseEstimator._validate_data`` exists. True if a shim was added.

    Idempotent: safe to call from every method run.
    """
    global _INSTALLED
    if _INSTALLED:
        return True

    from sklearn.base import BaseEstimator

    if hasattr(BaseEstimator, "_validate_data"):
        return False                    # sklearn < 1.6 -- nothing to do

    try:
        from sklearn.utils.validation import validate_data as _validate_data_fn
    except ImportError:                 # pragma: no cover -- unknown sklearn layout
        logger.warning(
            "sklearn has no BaseEstimator._validate_data and no "
            "sklearn.utils.validation.validate_data; vendored estimators that "
            "call the former (TabICL v1) will fail."
        )
        return False

    def _validate_data(self, X="no_validation", y="no_validation", **kwargs):
        """Forward to sklearn >= 1.6's ``validate_data`` free function."""
        return _validate_data_fn(self, X, y, **kwargs)

    BaseEstimator._validate_data = _validate_data
    _INSTALLED = True
    logger.info(
        "sklearn compat: re-attached BaseEstimator._validate_data "
        "(removed in scikit-learn 1.6) for vendored estimators."
    )
    return True


# ---------------------------------------------------------------------------
#  check_X_y / check_array: force_all_finite -> ensure_all_finite
# ---------------------------------------------------------------------------
#
# The same version break, one release later. ``force_all_finite`` was renamed to
# ``ensure_all_finite`` in scikit-learn 1.6 and REMOVED in 1.8, so on the cluster
# (1.9.0) every vendored caller dies with::
#
#     TypeError: check_X_y() got an unexpected keyword argument 'force_all_finite'
#
# That killed tabpfn, tabpfn_v2, tabpfn_real and realmlp -- TALENT's wrappers all
# pass ``force_all_finite='allow-nan'``, which is load-bearing: these methods
# accept NaNs, so silently dropping the argument would make validation REJECT the
# data instead. The wrappers bind ``check_X_y`` as a module global at import time,
# so patching the sklearn module alone can be too late; already-imported modules
# are rebound too.

_VALIDATOR_NAMES = ("check_X_y", "check_array")
_VALIDATORS_PATCHED = False


def _rename_finite_kwarg(fn):
    """Wrap ``fn`` so ``force_all_finite=`` is accepted and forwarded correctly."""
    def wrapper(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return fn(*args, **kwargs)

    wrapper.__name__ = getattr(fn, "__name__", "validator")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    wrapper._tabpfncredit_compat = True          # so we never double-wrap
    return wrapper


def install_sklearn_finite_kwarg_shim() -> bool:
    """Accept the removed ``force_all_finite`` kwarg. True if anything was patched.

    Idempotent, and a no-op on scikit-learn versions that still accept it.
    """
    global _VALIDATORS_PATCHED
    if _VALIDATORS_PATCHED:
        return True

    import inspect
    import sys

    import sklearn.utils as sku
    import sklearn.utils.validation as skv

    patched = False
    for name in _VALIDATOR_NAMES:
        fn = getattr(skv, name, None)
        if fn is None or getattr(fn, "_tabpfncredit_compat", False):
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):          # pragma: no cover -- C or wrapped
            continue
        if "force_all_finite" in params:
            continue                             # sklearn < 1.8: nothing to do
        wrapped = _rename_finite_kwarg(fn)
        setattr(skv, name, wrapped)
        if getattr(sku, name, None) is fn:       # sklearn.utils re-exports it
            setattr(sku, name, wrapped)
        # Rebind in any module that already imported the name by value.
        for module in list(sys.modules.values()):
            if module is None or module is skv or module is sku:
                continue
            if getattr(module, name, None) is fn:
                try:
                    setattr(module, name, wrapped)
                except Exception:                # pragma: no cover -- exotic module
                    pass
        patched = True

    if patched:
        _VALIDATORS_PATCHED = True
        logger.info(
            "sklearn compat: check_X_y/check_array now accept force_all_finite "
            "(renamed to ensure_all_finite in 1.6, removed in 1.8)."
        )
    return patched


def install_sklearn_compat() -> None:
    """Install every sklearn compatibility shim. Call before building estimators."""
    install_sklearn_validate_data_shim()
    install_sklearn_finite_kwarg_shim()
