"""SAL adapter, plus a self-contained reference implementation.

Primary path is ``scores.spatial`` (Marco's contribution on the ``sal-metric``
branch of github.com/fronzi/scores).  The reference implementation below exists
for two reasons:

1. tests and development can run without a ``scores`` install;
2. it gives a third independent cross-check alongside ``pysteps``, which is
   worth having before the upstream PR to ``nci/scores``.

Definitions follow Wernli et al. (2008), *MWR* **136**, 4470-4487.

Amplitude
    .. math:: A = \\frac{D(R_{\\mathrm{mod}}) - D(R_{\\mathrm{obs}})}
                       {0.5\\,[D(R_{\\mathrm{mod}}) + D(R_{\\mathrm{obs}})]}
    with :math:`D` the domain mean.  :math:`A \\in [-2, 2]`.

Structure
    .. math:: S = \\frac{V(R_{\\mathrm{mod}}) - V(R_{\\mathrm{obs}})}
                       {0.5\\,[V(R_{\\mathrm{mod}}) + V(R_{\\mathrm{obs}})]},
              \\qquad
              V = \\frac{\\sum_n R_n V_n}{\\sum_n R_n},
              \\qquad V_n = \\frac{R_n}{R_n^{\\max}}
    over objects :math:`n`.  Positive :math:`S` means objects that are too
    large or too flat; negative means too small or too peaked -- which is the
    expected signature of an under-resolved model.

Location
    .. math:: L = L_1 + L_2, \\qquad
              L_1 = \\frac{|\\mathbf x_{\\mathrm{mod}} - \\mathbf x_{\\mathrm{obs}}|}{d},
              \\qquad
              L_2 = 2\\,\\frac{|r_{\\mathrm{mod}} - r_{\\mathrm{obs}}|}{d}
    with :math:`d` the domain diagonal and :math:`r = \\sum_n R_n |\\mathbf x -
    \\mathbf x_n| / \\sum_n R_n`.  :math:`L \\in [0, 2]`.

Objects are contiguous regions exceeding :math:`R^* = f\\,R^{\\max}` with
:math:`f = 1/15` by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = ["SALResult", "compute_sal", "reference_sal", "scores_available"]


@dataclass(frozen=True)
class SALResult:
    """Structure, Amplitude and Location components for one field pair."""

    S: float
    A: float
    L: float
    L1: float
    L2: float
    n_objects_fcst: int = 0
    n_objects_obs: int = 0

    @property
    def distance(self) -> float:
        """Euclidean SAL distance ``sqrt(S^2 + A^2 + L^2)``.

        A convenience scalar for ranking arms.  It is *not* a proper score and
        weights three quantities with different natural ranges (``L`` is
        one-sided on ``[0, 2]``, ``S`` and ``A`` are signed on ``[-2, 2]``), so
        always report the components separately as well.
        """
        return float(np.sqrt(self.S**2 + self.A**2 + self.L**2))

    def as_dict(self) -> dict[str, float]:
        return {"S": self.S, "A": self.A, "L": self.L, "L1": self.L1, "L2": self.L2}


def scores_available() -> bool:
    """True if the ``scores`` SAL implementation is importable."""
    try:
        import scores.spatial  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "unavailable"
        return False
    return True


def _threshold(field: np.ndarray, factor: float, fixed: float | None) -> float:
    if fixed is not None:
        return float(fixed)
    finite = field[np.isfinite(field)]
    return float(factor * finite.max()) if finite.size else np.inf


def _objects(field: np.ndarray, thresh: float):
    mask = np.isfinite(field) & (field >= thresh) & (thresh > 0)
    labels, n = ndimage.label(mask)
    return labels, n


def _structure_and_location(field: np.ndarray, thresh: float, area: np.ndarray | None = None):
    """Return ``(V, centre_of_mass, r, n_objects)`` for one field.

    ``area`` optionally supplies per-cell areas so that domain means and object
    volumes are area-weighted.  On a lat/lon verification grid the cell area
    varies as ``cos(lat)``; ignoring that makes the amplitude component depend
    on the grid spacing, which defeats the whole cross-resolution comparison.
    """
    raw = np.where(np.isfinite(field), field, 0.0)
    w = np.ones_like(raw) if area is None else np.asarray(area, float)
    arr = raw * w
    labels, n = _objects(field, thresh)
    total = arr.sum()
    if total <= 0:
        return np.nan, np.array([np.nan, np.nan]), np.nan, 0
    com = np.array(ndimage.center_of_mass(arr))
    if n == 0:
        return np.nan, com, np.nan, 0
    idx = np.arange(1, n + 1)
    r_n = np.array(ndimage.sum_labels(arr, labels, idx), dtype=float)
    rmax_n = np.array(ndimage.maximum(raw, labels, idx), dtype=float)
    coms = np.array(ndimage.center_of_mass(arr, labels, idx), dtype=float).reshape(n, 2)
    denom = r_n.sum()
    if denom <= 0:
        return np.nan, com, np.nan, n
    with np.errstate(divide="ignore", invalid="ignore"):
        v_n = np.where(rmax_n > 0, r_n / rmax_n, 0.0)
    V = float((r_n * v_n).sum() / denom)
    dists = np.linalg.norm(coms - com[None, :], axis=1)
    r = float((r_n * dists).sum() / denom)
    return V, com, r, n


def reference_sal(
    fcst: np.ndarray,
    obs: np.ndarray,
    threshold_factor: float = 1.0 / 15.0,
    fixed_threshold: float | None = None,
    area: np.ndarray | None = None,
) -> SALResult:
    """Wernli (2008) SAL for a single pair of 2-D fields on a common grid.

    Both arrays must have identical shape -- i.e. already remapped to the
    verification grid.  ``fixed_threshold`` overrides the ``R_max/15`` rule and
    should be used for the threshold-sensitivity sweep.

    ``area`` supplies per-cell areas (see :func:`sal_resolution.regrid.cell_area`).
    Pass it whenever the verification grid spans enough latitude for ``cos(lat)``
    to vary appreciably; without it ``A`` acquires a spurious dependence on grid
    spacing.
    """
    f = np.asarray(fcst, dtype=np.float64)
    o = np.asarray(obs, dtype=np.float64)
    if f.shape != o.shape:
        raise ValueError(
            f"fcst {f.shape} and obs {o.shape} differ; remap to a common "
            "verification grid before computing SAL"
        )
    if f.ndim != 2:
        raise ValueError("reference_sal expects 2-D (lat, lon) fields")

    if area is None:
        d_f, d_o = np.nanmean(f), np.nanmean(o)
    else:
        w = np.asarray(area, float)
        if w.shape != f.shape:
            raise ValueError(f"area {w.shape} does not match field shape {f.shape}")
        def _wmean(x: np.ndarray) -> float:
            ok = np.isfinite(x)
            return float(np.sum(np.where(ok, x * w, 0.0)) / np.sum(np.where(ok, w, 0.0)))

        d_f, d_o = _wmean(f), _wmean(o)
    denom_a = 0.5 * (d_f + d_o)
    A = float((d_f - d_o) / denom_a) if denom_a > 0 else np.nan

    V_f, com_f, r_f, n_f = _structure_and_location(
        f, _threshold(f, threshold_factor, fixed_threshold), area)
    V_o, com_o, r_o, n_o = _structure_and_location(
        o, _threshold(o, threshold_factor, fixed_threshold), area)

    denom_s = 0.5 * (V_f + V_o)
    S = float((V_f - V_o) / denom_s) if np.isfinite(denom_s) and denom_s > 0 else np.nan

    diag = float(np.hypot(*(np.array(f.shape) - 1)))
    if diag > 0 and np.all(np.isfinite(com_f)) and np.all(np.isfinite(com_o)):
        L1 = float(np.linalg.norm(com_f - com_o) / diag)
    else:
        L1 = np.nan
    l2_ok = diag > 0 and np.isfinite(r_f) and np.isfinite(r_o)
    L2 = float(2.0 * abs(r_f - r_o) / diag) if l2_ok else np.nan
    L = L1 + L2

    return SALResult(S=S, A=A, L=L, L1=L1, L2=L2, n_objects_fcst=n_f, n_objects_obs=n_o)


def compute_sal(
    fcst: np.ndarray,
    obs: np.ndarray,
    threshold_factor: float = 1.0 / 15.0,
    fixed_threshold: float | None = None,
    backend: str = "auto",
    area: np.ndarray | None = None,
) -> SALResult:
    """Compute SAL via ``scores`` if available, else the reference implementation.

    ``backend`` is one of ``{"auto", "scores", "reference"}``.  Pin it to
    ``"scores"`` for the production run so the published numbers come from the
    implementation being contributed upstream; use ``"reference"`` to
    cross-check.

    .. note::
       The ``scores`` call signature is resolved dynamically because the SAL
       entry point is still on a feature branch.  Verify the returned component
       names against your ``sal_impl.py`` before trusting a production run --
       a silent mismatch in the sign convention of ``S`` would invert the
       headline conclusion.
    """
    if backend == "reference":
        return reference_sal(fcst, obs, threshold_factor, fixed_threshold, area)
    if backend in ("auto", "scores") and scores_available():
        try:
            return _sal_via_scores(fcst, obs, threshold_factor, fixed_threshold)
        except Exception as exc:
            if backend == "scores":
                raise
            import warnings
            warnings.warn(f"scores SAL failed ({exc}); falling back to reference", stacklevel=2)
    if backend == "scores":
        raise RuntimeError("`scores` is not importable but backend='scores' was requested")
    return reference_sal(fcst, obs, threshold_factor, fixed_threshold, area)


def _sal_via_scores(fcst, obs, threshold_factor, fixed_threshold) -> SALResult:  # pragma: no cover
    import scores.spatial as sp
    import xarray as xr

    fn = getattr(sp, "sal", None)
    if fn is None:
        raise AttributeError("scores.spatial has no `sal` entry point")
    dims = ("lat", "lon")
    f = xr.DataArray(np.asarray(fcst, float), dims=dims)
    o = xr.DataArray(np.asarray(obs, float), dims=dims)
    kwargs = {}
    if fixed_threshold is not None:
        kwargs["fixed_threshold"] = fixed_threshold
    else:
        kwargs["threshold_factor"] = threshold_factor
    try:
        res = fn(f, o, **kwargs)
    except TypeError:
        res = fn(f, o)

    def _get(name, default=np.nan):
        try:
            return float(res[name])
        except (KeyError, TypeError, ValueError):
            return float(default)

    l1, l2 = _get("L1"), _get("L2")
    L = _get("L", l1 + l2)
    return SALResult(S=_get("S"), A=_get("A"), L=L, L1=l1, L2=l2)
