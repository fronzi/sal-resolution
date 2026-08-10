"""Three-arm SAL comparison driver and the recovered-gain statistic.

Headline quantity
-----------------
The fraction of the *dynamical* downscaling gain that the symbolic correction
recovers, per SAL component :math:`X \\in \\{S, A, L\\}`:

.. math::
    \\rho_X = \\frac{|X_{\\mathrm{coarse}}| - |X_{\\mathrm{sym}}|}
                    {|X_{\\mathrm{coarse}}| - |X_{\\mathrm{fine}}|}

:math:`\\rho_S` is the one that speaks to spatial resolution: ``S`` is the only
component that interrogates object size and shape.  :math:`\\rho_X = 1` means
the closed-form expression matched convection-permitting dynamics on that
component; :math:`\\rho_X < 0` means it made things worse than doing nothing.

Absolute values are used because ``S`` and ``A`` are signed and an arm that
overshoots from ``-0.4`` to ``+0.5`` has not improved.

Sign and magnitude conventions
------------------------------
``rho`` is undefined when the denominator is near zero -- i.e. when dynamical
downscaling itself brought no benefit for that event.  Those events are
returned as NaN rather than as large spurious ratios, and must be reported as
a count: a headline ``rho_S`` computed over the handful of events where the
denominator happened to be large is not a result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .sal import SALResult, compute_sal

__all__ = ["ArmResult", "bootstrap_ci", "delta_sweep", "evaluate_arms", "recovered_gain"]

COMPONENTS = ("S", "A", "L")


@dataclass
class ArmResult:
    """Per-event SAL components for one model arm on one verification grid."""

    name: str
    spacing_km: float
    results: list[SALResult]

    def component(self, comp: str) -> np.ndarray:
        return np.array([getattr(r, comp) for r in self.results], dtype=float)

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in COMPONENTS:
            v = self.component(c)
            out[f"{c}_median"] = float(np.nanmedian(v))
            out[f"{c}_mad"] = float(np.nanmedian(np.abs(v - np.nanmedian(v))))
        out["n_valid"] = int(np.isfinite(self.component("S")).sum())
        return out


def evaluate_arms(
    arms: Mapping[str, Sequence[np.ndarray]],
    obs: Sequence[np.ndarray],
    spacing_km: float,
    threshold_factor: float = 1.0 / 15.0,
    fixed_threshold: float | None = None,
    backend: str = "auto",
    area: np.ndarray | None = None,
) -> dict[str, ArmResult]:
    """Compute SAL for every arm against a common observation sequence.

    All fields must already be on the same verification grid -- this function
    deliberately does not regrid, so that the remapping step stays explicit and
    auditable upstream.
    """
    n = len(obs)
    out: dict[str, ArmResult] = {}
    for name, fields in arms.items():
        if len(fields) != n:
            raise ValueError(f"arm {name!r} has {len(fields)} events, obs has {n}")
        res = [
            compute_sal(f, o, threshold_factor=threshold_factor,
                        fixed_threshold=fixed_threshold, backend=backend, area=area)
            for f, o in zip(fields, obs, strict=True)
        ]
        out[name] = ArmResult(name=name, spacing_km=spacing_km, results=res)
    return out


def recovered_gain(
    coarse: ArmResult,
    fine: ArmResult,
    candidate: ArmResult,
    component: str = "S",
    min_gain: float = 0.02,
) -> np.ndarray:
    """Per-event :math:`\\rho_X`. NaN where dynamical downscaling gave no gain.

    ``min_gain`` is the smallest ``|X_coarse| - |X_fine|`` treated as a real
    denominator.  Events below it are masked out rather than producing ratios
    of order 1/eps.
    """
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}")
    xc = np.abs(coarse.component(component))
    xf = np.abs(fine.component(component))
    xs = np.abs(candidate.component(component))
    denom = xc - xf
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (xc - xs) / denom
    return np.where(np.abs(denom) >= min_gain, rho, np.nan)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], float] = np.nanmedian,
    random_state: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap over *events*, returning ``(stat, lo, hi)``.

    Resampling is at the event level, not the grid-point level: neighbouring
    grid points are strongly correlated, so point-level bootstrapping would
    understate the interval by roughly the square root of the number of points
    per decorrelation length.  Per-event SAL is noisy, so aim for n >= 30
    events before quoting an interval.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(random_state)
    draws = rng.choice(v, size=(n_boot, v.size), replace=True)
    stats = np.array([statistic(d) for d in draws])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(v)), float(lo), float(hi)


def amplitude_invariance_check(results_by_spacing: Mapping[float, ArmResult],
                               tol: float = 1e-9) -> tuple[bool, float]:
    """Assert that ``A`` does not drift across verification grid spacings.

    ``A`` is a ratio of domain means, so under mass-conserving remapping it is
    invariant in the grid spacing to machine precision.  If it drifts, the
    regridding is not conservative somewhere in the chain -- most likely a
    bilinear interpolation has crept in.  Run this before trusting any ``S``
    result, because the same bug would corrupt ``S`` far less visibly.

    Returns ``(passed, max_absolute_drift)``.
    """
    per_event = [r.component("A") for r in results_by_spacing.values()]
    if len(per_event) < 2:
        return True, 0.0
    stack = np.vstack(per_event)
    drift = float(np.nanmax(np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)))
    return drift <= tol, drift


def delta_sweep(
    arms_by_spacing: Mapping[float, Mapping[str, Sequence[np.ndarray]]],
    obs_by_spacing: Mapping[float, Sequence[np.ndarray]],
    coarse_name: str = "coarse",
    fine_name: str = "fine",
    candidate_names: Sequence[str] = ("symbolic",),
    component: str = "S",
    **sal_kwargs,
) -> dict[float, dict[str, tuple[float, float, float]]]:
    """Run the full verification-grid-spacing sweep.

    This sweep *is* the spatial-resolution result: it identifies the scale at
    which the symbolic correction stops adding information relative to the
    coarse model, and whether that scale approaches the effective resolution of
    the convection-permitting arm.
    """
    out: dict[float, dict[str, tuple[float, float, float]]] = {}
    for spacing, arms in arms_by_spacing.items():
        ev = evaluate_arms(arms, obs_by_spacing[spacing], spacing_km=spacing, **sal_kwargs)
        row = {}
        for cand in candidate_names:
            rho = recovered_gain(ev[coarse_name], ev[fine_name], ev[cand], component=component)
            row[cand] = bootstrap_ci(rho)
        out[spacing] = row
    return out
