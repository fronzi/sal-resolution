"""Predictor construction for the symbolic correction.

The correction is multiplicative,

.. math:: R_{\\mathrm{sym}}(x) = R_c(x)\\,\\exp\\!\\big[g(\\mathbf z(x))\\big]

which keeps the field non-negative and matches the approximately log-normal
distribution of rain rates.  Every element of :math:`\\mathbf z` is computable
from the *coarse* forecast plus static orography -- nothing here requires the
high-resolution model, which is the whole point: at inference time the
symbolic expression must be applicable to a 12 km field alone.

Gradient metric terms
---------------------
Horizontal derivatives on the sphere are

.. math::
    \\nabla^2 R = \\frac{1}{a^2\\cos^2\\varphi}\\,\\partial^2_\\lambda R
      + \\frac{1}{a^2\\cos\\varphi}\\,\\partial_\\varphi
        \\big(\\cos\\varphi\\,\\partial_\\varphi R\\big)
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage

from .regrid import EARTH_RADIUS_M

__all__ = [
    "FEATURE_NAMES",
    "build_features",
    "horizontal_gradient",
    "neighbourhood_stats",
    "spherical_laplacian",
    "upslope_forcing",
]

#: Canonical predictor ordering. PySR is given these names verbatim so that the
#: recovered expression is human-readable and physically interpretable.
FEATURE_NAMES = (
    "logR",        # log(1 + R_c)                 -- amplitude
    "gradR",       # |grad R_c|  [mm / 100 km]    -- local sharpness
    "lapR",        # laplacian R_c [mm / (100 km)^2] -- curvature / peakedness
    "nb3_std",     # 3x3 std of R_c               -- sub-grid variability
    "nb5_max",     # 5x5 max of R_c
    "nb9_mean",    # 9x9 mean of R_c              -- synoptic-scale envelope
    "orog",        # h [km]
    "gradh",       # |grad h| [km / 100 km]       -- terrain roughness
    "upslope",     # u . grad h [m/s per 100 km]  -- orographic forcing
)

# Gradients are reported per 100 km rather than per metre so that all
# predictors sit within a few orders of magnitude of unity. PySR's constant
# optimiser is poorly conditioned when features span 1e-8 to 1e2.
_LEN_SCALE = 1.0e5


def _radian_coords(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float))


def horizontal_gradient(
    field: np.ndarray, lat: np.ndarray, lon: np.ndarray, radius: float = EARTH_RADIUS_M
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(d/dx, d/dy)`` in units of ``field`` per metre.

    ``field`` has shape ``(nlat, nlon)`` and ``lat``/``lon`` are in degrees.
    """
    phi, lam = _radian_coords(lat, lon)
    cos_phi = np.clip(np.cos(phi), 1e-6, None)[:, None]
    d_dphi = np.gradient(field, phi, axis=0, edge_order=2)
    d_dlam = np.gradient(field, lam, axis=1, edge_order=2)
    return d_dlam / (radius * cos_phi), d_dphi / radius


def spherical_laplacian(
    field: np.ndarray, lat: np.ndarray, lon: np.ndarray, radius: float = EARTH_RADIUS_M
) -> np.ndarray:
    """Horizontal Laplacian on the sphere, units of ``field`` per metre squared."""
    phi, lam = _radian_coords(lat, lon)
    cos_phi = np.clip(np.cos(phi), 1e-6, None)[:, None]
    d_dlam = np.gradient(field, lam, axis=1, edge_order=2)
    d2_dlam2 = np.gradient(d_dlam, lam, axis=1, edge_order=2)
    d_dphi = np.gradient(field, phi, axis=0, edge_order=2)
    flux = cos_phi * d_dphi
    d_flux = np.gradient(flux, phi, axis=0, edge_order=2)
    return d2_dlam2 / (radius**2 * cos_phi**2) + d_flux / (radius**2 * cos_phi)


def neighbourhood_stats(field: np.ndarray, size: int, stat: str) -> np.ndarray:
    """Stencil statistic over a ``size x size`` window with reflected edges.

    ``stat`` is one of ``{"mean", "max", "std"}``.
    """
    if size % 2 == 0:
        raise ValueError("stencil `size` must be odd so the window is centred")
    if stat == "mean":
        return ndimage.uniform_filter(field, size=size, mode="reflect")
    if stat == "max":
        return ndimage.maximum_filter(field, size=size, mode="reflect")
    if stat == "std":
        mean = ndimage.uniform_filter(field, size=size, mode="reflect")
        mean_sq = ndimage.uniform_filter(field**2, size=size, mode="reflect")
        return np.sqrt(np.clip(mean_sq - mean**2, 0.0, None))
    raise ValueError(f"unknown stat {stat!r}; expected 'mean', 'max' or 'std'")


def upslope_forcing(
    u: np.ndarray,
    v: np.ndarray,
    orog: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    radius: float = EARTH_RADIUS_M,
) -> np.ndarray:
    """Orographic forcing :math:`\\mathbf u \\cdot \\nabla h`, units m/s per metre.

    Physically the most defensible predictor in the set: sub-grid orographic
    enhancement is precisely the process that a 4.4 km model resolves and a
    12 km model does not, so this is where a resolution-correcting expression
    ought to draw its skill from.  Use low-level winds (925 or 850 hPa).
    """
    dh_dx, dh_dy = horizontal_gradient(orog, lat, lon, radius)
    return u * dh_dx + v * dh_dy


def build_features(
    rain_coarse: xr.DataArray,
    orog: xr.DataArray,
    u: xr.DataArray | None = None,
    v: xr.DataArray | None = None,
    lat_name: str = "lat",
    lon_name: str = "lon",
    radius: float = EARTH_RADIUS_M,
) -> xr.Dataset:
    """Assemble the predictor set as a Dataset on the coarse grid.

    All inputs must already be on the *same* grid.  ``rain_coarse`` is in mm,
    ``orog`` in metres, ``u``/``v`` in m/s.  If winds are absent, ``upslope`` is
    filled with zeros so the feature matrix keeps a stable column ordering.

    Returns
    -------
    xarray.Dataset
        One variable per entry of :data:`FEATURE_NAMES`, sharing the dims of
        ``rain_coarse``.
    """
    lat = rain_coarse[lat_name].values
    lon = rain_coarse[lon_name].values
    r = np.asarray(rain_coarse.values, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("build_features expects a single 2-D (lat, lon) field")
    h = np.asarray(orog.values, dtype=np.float64)

    gx, gy = horizontal_gradient(r, lat, lon, radius)
    hx, hy = horizontal_gradient(h, lat, lon, radius)
    lap = spherical_laplacian(r, lat, lon, radius)

    if u is not None and v is not None:
        ups = upslope_forcing(np.asarray(u.values, float), np.asarray(v.values, float),
                              h, lat, lon, radius)
    else:
        ups = np.zeros_like(r)

    data = {
        "logR": np.log1p(np.clip(r, 0.0, None)),
        "gradR": np.hypot(gx, gy) * _LEN_SCALE,
        "lapR": lap * _LEN_SCALE**2,
        "nb3_std": neighbourhood_stats(r, 3, "std"),
        "nb5_max": neighbourhood_stats(r, 5, "max"),
        "nb9_mean": neighbourhood_stats(r, 9, "mean"),
        "orog": h / 1000.0,
        "gradh": np.hypot(hx, hy) * _LEN_SCALE / 1000.0,
        "upslope": ups * _LEN_SCALE,
    }
    coords = {lat_name: lat, lon_name: lon}
    return xr.Dataset(
        {k: xr.DataArray(v, dims=(lat_name, lon_name), coords=coords) for k, v in data.items()}
    )
