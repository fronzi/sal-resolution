"""Post-processing baselines the symbolic expression is required to beat.

Without these the experiment has no content.  In particular:

* **quantile mapping** is the real bar for the SAL amplitude component ``A``;
* **unsharp masking** is the real bar for the structure component ``S`` -- it is
  a one-parameter filter that sharpens objects, so if a 20-term symbolic
  expression cannot beat it on ``S`` there is no resolution story to tell.

Every baseline exposes the same ``fit(train) -> callable`` interface as
:mod:`sal_resolution.symbolic.fit`, so the comparison driver can treat the
symbolic model as just another entry in the arm dictionary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import ndimage

__all__ = [
    "BASELINES",
    "fit_identity",
    "fit_quantile_mapping",
    "fit_scalar_bias",
    "fit_unsharp_mask",
]

Corrector = Callable[[np.ndarray], np.ndarray]


def _stack(fields: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(f, float).ravel() for f in fields])


def fit_identity(rain_coarse: Sequence[np.ndarray], rain_obs: Sequence[np.ndarray]) -> Corrector:
    """No correction. The floor that every other arm must clear."""
    return lambda field: np.asarray(field, float).copy()


def fit_scalar_bias(rain_coarse: Sequence[np.ndarray], rain_obs: Sequence[np.ndarray]) -> Corrector:
    """Single multiplicative factor matching the domain-mean rainfall.

    Corrects SAL ``A`` by construction and leaves ``S`` and ``L`` untouched
    (``S`` is computed from volume-normalised objects, so a uniform scaling is
    very nearly a no-op for it).  Isolates how much of any apparent gain is
    just bias removal.
    """
    c = _stack(rain_coarse)
    o = _stack(rain_obs)
    ok = np.isfinite(c) & np.isfinite(o)
    denom = c[ok].mean()
    alpha = float(o[ok].mean() / denom) if denom > 0 else 1.0
    return lambda field: alpha * np.asarray(field, float)


def fit_quantile_mapping(
    rain_coarse: Sequence[np.ndarray],
    rain_obs: Sequence[np.ndarray],
    n_quantiles: int = 200,
    wet_threshold: float = 0.1,
) -> Corrector:
    """Empirical quantile mapping fitted on wet points only.

    Dry points (< ``wet_threshold`` mm) are excluded from the fit because
    including the large dry mass drags the whole transfer function toward zero
    and destroys the upper-tail mapping that matters for heavy-rain events.
    Values beyond the fitted range are extrapolated by the edge ratio rather
    than clipped, so extremes are not artificially capped.
    """
    c = _stack(rain_coarse)
    o = _stack(rain_obs)
    c = c[np.isfinite(c) & (c >= wet_threshold)]
    o = o[np.isfinite(o) & (o >= wet_threshold)]
    if c.size == 0 or o.size == 0:
        return fit_identity(rain_coarse, rain_obs)
    q = np.linspace(0.0, 1.0, n_quantiles)
    cq = np.quantile(c, q)
    oq = np.quantile(o, q)
    tail = float(oq[-1] / cq[-1]) if cq[-1] > 0 else 1.0

    def _apply(field: np.ndarray) -> np.ndarray:
        x = np.asarray(field, float)
        out = np.interp(x, cq, oq, left=oq[0], right=np.nan)
        beyond = x > cq[-1]
        out = np.where(beyond, x * tail, out)
        out = np.where(x < wet_threshold, x, out)
        return np.where(np.isfinite(x), out, np.nan)

    return _apply


def fit_unsharp_mask(
    rain_coarse: Sequence[np.ndarray],
    rain_obs: Sequence[np.ndarray],
    sigmas: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
    amounts: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.0),
) -> Corrector:
    """One-parameter-family sharpening filter, ``R + a (R - G_sigma * R)``.

    Parameters are chosen by grid search on the *log patch-variance* criterion
    rather than MSE.  Selecting on MSE would always return ``a = 0`` -- an
    MSE-optimal field is a blurred field -- which would make this a strawman
    rather than a real competitor on ``S``.
    """
    obs_var = np.mean([np.log1p(np.nanvar(np.asarray(o, float))) for o in rain_obs])
    best, best_cost = (sigmas[0], 0.0), np.inf
    for s in sigmas:
        for a in amounts:
            cost = 0.0
            for c in rain_coarse:
                arr = np.asarray(c, float)
                sharp = arr + a * (arr - ndimage.gaussian_filter(arr, sigma=s, mode="reflect"))
                cost += (np.log1p(np.nanvar(np.clip(sharp, 0.0, None))) - obs_var) ** 2
            if cost < best_cost:
                best_cost, best = cost, (s, a)
    sigma, amount = best

    def _apply(field: np.ndarray) -> np.ndarray:
        arr = np.asarray(field, float)
        blur = ndimage.gaussian_filter(np.nan_to_num(arr), sigma=sigma, mode="reflect")
        return np.where(np.isfinite(arr), np.clip(arr + amount * (arr - blur), 0.0, None), np.nan)

    _apply.params = {"sigma": sigma, "amount": amount}  # type: ignore[attr-defined]
    return _apply


#: Registry consumed by the comparison driver.
BASELINES = {
    "identity": fit_identity,
    "scalar_bias": fit_scalar_bias,
    "quantile_map": fit_quantile_mapping,
    "unsharp": fit_unsharp_mask,
}
