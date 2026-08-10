"""PySR wrapper: learn a closed-form multiplicative structure correction.

Model
-----
.. math:: R_{\\mathrm{sym}}(x) = R_c(x) \\, \\exp\\!\\big[g(\\mathbf z(x))\\big]

so PySR's regression target is the **log ratio**

.. math:: y(x) = \\log\\frac{R_{\\mathrm{obs}}(x) + \\epsilon}{R_c(x) + \\epsilon}

which is symmetric in over- and under-prediction and bounded for light rain.

Why not train on SAL
--------------------
SAL is non-local (it requires connected-component object identification) and
non-differentiable in the field, so it cannot serve as a search objective.
More importantly, using it for both fitting and evaluation would make the
headline result circular.  SAL is held out as the **evaluation instrument
only**.

Why not train on plain MSE
--------------------------
An MSE-optimal field is a blurred field.  Minimising pointwise squared error
will drive ``g`` toward a smoothing operator, improving RMSE while making the
SAL structure component ``S`` *worse* -- the classic double-penalty trap that
SAL exists to expose.  The fitting loss therefore carries an explicit
structure term,

.. math::
    \\mathcal L = \\mathrm{MSE}
      + \\lambda \\big(\\log\\sigma^{\\mathrm{sym}}_{\\mathrm{patch}}
                     - \\log\\sigma^{\\mathrm{obs}}_{\\mathrm{patch}}\\big)^2

Because PySR's per-row loss cannot see spatial neighbours, the patch-variance
term is applied through **sample weights**: rows in high-local-variance patches
where the coarse model is under-dispersed are up-weighted, which is a separable
first-order surrogate for the penalty above.  The exact non-separable form is
available via the outer selection step
(:func:`select_by_structure`), which re-ranks the PySR Pareto front by the true
patch-variance criterion computed on full 2-D fields.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as _dc_field

import numpy as np
from scipy import ndimage

from ..features import FEATURE_NAMES

__all__ = [
    "SymbolicConfig",
    "assemble_training_matrix",
    "fit_symbolic",
    "loyo_splits",
    "select_by_structure",
    "structure_weights",
    "symbolic_available",
]

EPS_MM = 0.1  # mm; regularises the log ratio for near-dry points


def symbolic_available() -> bool:
    """True if PySR (and hence a working Julia backend) can be imported."""
    try:
        import pysr  # noqa: F401
    except Exception:  # noqa: BLE001 - any import/Julia failure means "unavailable"
        return False
    return True


@dataclass
class SymbolicConfig:
    """Search settings.

    ``maxsize`` is deliberately modest: the deliverable is an *interpretable*
    closed form.  An expression of complexity 40 is a black box with extra
    steps and forfeits the only advantage this approach has over a CNN.
    """

    niterations: int = 100
    population_size: int = 33
    populations: int = 15
    maxsize: int = 20
    binary_operators: Sequence[str] = ("+", "-", "*", "/")
    unary_operators: Sequence[str] = ("exp", "log_abs", "tanh", "square")
    elementwise_loss: str = "loss(x, y, w) = w * (x - y)^2"
    model_selection: str = "best"
    parsimony: float = 3e-3
    random_state: int = 0
    max_rows: int = 200_000
    feature_names: Sequence[str] = _dc_field(default_factory=lambda: list(FEATURE_NAMES))


def assemble_training_matrix(
    features: Sequence[object],
    rain_coarse: Sequence[np.ndarray],
    rain_obs: Sequence[np.ndarray],
    wet_threshold: float = 0.2,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten per-event fields into ``(X, y, event_index)``.

    Only points that are wet in the model *or* the observations are retained.
    Including the vast dry area would let the search minimise loss by learning
    ``g = 0`` almost everywhere, which is both uninformative and numerically
    dominant.
    """
    xs, ys, idx = [], [], []
    for e, (feat, rc, ro) in enumerate(zip(features, rain_coarse, rain_obs, strict=True)):
        cols = [np.asarray(feat[name].values, float).ravel() for name in feature_names]
        X = np.column_stack(cols)
        c = np.asarray(rc, float).ravel()
        o = np.asarray(ro, float).ravel()
        y = np.log((o + EPS_MM) / (c + EPS_MM))
        keep = (np.isfinite(y) & np.isfinite(X).all(axis=1)
                & ((c >= wet_threshold) | (o >= wet_threshold)))
        xs.append(X[keep])
        ys.append(y[keep])
        idx.append(np.full(keep.sum(), e, dtype=np.int64))
    if not xs:
        raise ValueError("no events supplied")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(idx)


def structure_weights(
    rain_coarse: Sequence[np.ndarray],
    rain_obs: Sequence[np.ndarray],
    wet_threshold: float = 0.2,
    size: int = 5,
    strength: float = 1.0,
) -> np.ndarray:
    """Per-row weights that bias the search toward fixing under-dispersion.

    Weight is ``1 + strength * relu(log sigma_obs - log sigma_coarse)`` computed
    on a ``size x size`` patch, so points where the coarse field is locally too
    smooth relative to the observations dominate the fit.  Row ordering matches
    :func:`assemble_training_matrix`.
    """
    out = []
    for rc, ro in zip(rain_coarse, rain_obs, strict=True):
        c = np.asarray(rc, float)
        o = np.asarray(ro, float)
        sc = _patch_std(c, size)
        so = _patch_std(o, size)
        w = 1.0 + strength * np.clip(np.log1p(so) - np.log1p(sc), 0.0, None)
        y = np.log((o.ravel() + EPS_MM) / (c.ravel() + EPS_MM))
        keep = (np.isfinite(y) & ((c.ravel() >= wet_threshold) | (o.ravel() >= wet_threshold)))
        out.append(w.ravel()[keep])
    return np.concatenate(out)


def _patch_std(a: np.ndarray, size: int) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a, float))
    m = ndimage.uniform_filter(a, size=size, mode="reflect")
    m2 = ndimage.uniform_filter(a**2, size=size, mode="reflect")
    return np.sqrt(np.clip(m2 - m**2, 0.0, None))


def loyo_splits(years: Sequence[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-year-out train/test index pairs.

    Year-blocked rather than random: rain fields are strongly autocorrelated in
    space and time, so random point- or event-level splitting leaks the test
    signal into training and inflates apparent skill.
    """
    years = np.asarray(years)
    out = []
    for yr in np.unique(years):
        test = np.flatnonzero(years == yr)
        train = np.flatnonzero(years != yr)
        if train.size and test.size:
            out.append((train, test))
    return out


def fit_symbolic(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    cfg: SymbolicConfig | None = None,
):
    """Run PySR and return the fitted regressor.

    Raises
    ------
    RuntimeError
        If PySR is unavailable.  Guarded rather than soft-failed so that a
        missing Julia backend cannot silently degrade the experiment to a
        baseline-only comparison.
    """
    cfg = cfg or SymbolicConfig()
    if not symbolic_available():  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PySR is not importable. Install with `pip install pysr` and run "
            "`python -c 'import pysr; pysr.install()'` once to provision Julia."
        )
    from pysr import PySRRegressor

    rng = np.random.default_rng(cfg.random_state)
    if X.shape[0] > cfg.max_rows:
        sel = rng.choice(X.shape[0], cfg.max_rows, replace=False)
        X, y = X[sel], y[sel]
        weights = None if weights is None else weights[sel]
        warnings.warn(f"subsampled training rows to {cfg.max_rows}", stacklevel=2)

    kwargs = dict(
        niterations=cfg.niterations,
        population_size=cfg.population_size,
        populations=cfg.populations,
        maxsize=cfg.maxsize,
        binary_operators=list(cfg.binary_operators),
        unary_operators=list(cfg.unary_operators),
        elementwise_loss=cfg.elementwise_loss,
        model_selection=cfg.model_selection,
        parsimony=cfg.parsimony,
        random_state=cfg.random_state,
        deterministic=True,
    )
    # Serial execution is required for `deterministic=True`, and determinism is
    # non-negotiable here: the LOYO folds and the permutation control are only
    # comparable if the search is reproducible.
    #
    # PySR >= 1.0 replaced `procs` / `multithreading` with a single
    # `parallelism` argument. Probe the signature rather than pinning a version,
    # so the package works either side of that break.
    if "parallelism" in inspect.signature(PySRRegressor.__init__).parameters:
        kwargs["parallelism"] = "serial"
    else:  # pragma: no cover - legacy PySR 0.x
        kwargs["procs"] = 0
        kwargs["multithreading"] = False

    model = PySRRegressor(**kwargs)
    model.fit(X, y, weights=weights, variable_names=list(cfg.feature_names))
    return model


def make_corrector(model, feature_names: Sequence[str] = FEATURE_NAMES,
                   equation_index: int | None = None) -> Callable:
    """Wrap a fitted PySR model as ``(coarse_field, features) -> corrected_field``."""

    def _apply(rain_coarse: np.ndarray, features) -> np.ndarray:
        c = np.asarray(rain_coarse, float)
        cols = [np.asarray(features[n].values, float).ravel() for n in feature_names]
        X = np.column_stack(cols)
        g = model.predict(X, index=equation_index).reshape(c.shape)
        return np.clip(c * np.exp(np.clip(g, -10.0, 10.0)), 0.0, None)

    return _apply


def select_by_structure(
    model,
    rain_coarse: Sequence[np.ndarray],
    rain_obs: Sequence[np.ndarray],
    features: Sequence[object],
    feature_names: Sequence[str] = FEATURE_NAMES,
    size: int = 5,
) -> int:
    """Re-rank the PySR Pareto front by the true 2-D patch-variance criterion.

    PySR's own ``model_selection='best'`` trades accuracy against complexity
    under the row-wise loss, which is blind to spatial structure.  This picks
    the front member whose corrected fields best match the observed local
    variance -- the quantity ``S`` actually responds to -- while still
    preferring the simpler of two near-equal candidates.

    Returns the index into ``model.equations_``.
    """
    n_eq = len(model.equations_)
    costs = np.full(n_eq, np.inf)
    for i in range(n_eq):
        corrector = make_corrector(model, feature_names, equation_index=i)
        cost = 0.0
        for rc, ro, ft in zip(rain_coarse, rain_obs, features, strict=True):
            sym = corrector(rc, ft)
            cost += float((np.log1p(_patch_std(sym, size).mean())
                           - np.log1p(_patch_std(np.asarray(ro, float), size).mean())) ** 2)
        costs[i] = cost
    tol = 1.05 * np.nanmin(costs)
    eligible = np.flatnonzero(costs <= tol)
    complexity = np.asarray(model.equations_["complexity"], dtype=float)
    return int(eligible[np.argmin(complexity[eligible])])
