"""Rainfields3 radar QPE loader -- the verification truth.

Why radar rather than AGCD
--------------------------
AGCD is gauge-interpolated and therefore spatially smooth wherever the gauge
network is sparse.  Using it as truth for a *structure* study systematically
flatters the coarse model's ``S``: the coarse model and the smoothed analysis
share a lack of small-scale variance, so the 12 km arm looks better than it is
and the ceiling on any correction is artificially low.  Rainfields3 radar QPE
resolves convective objects and is the right truth here.

Cost: it needs NCI project ``rq0`` membership, has radar-specific artefacts
(beam blocking, bright band, ground clutter, range-dependent sensitivity), and
is only valid inside radar coverage.  :func:`coverage_mask` exists to enforce
the last point -- an object truncated by the edge of radar range corrupts both
``S`` and ``L``.

.. warning::
   Variable and dimension names below follow the common Rainfields3 NetCDF
   layout but should be confirmed against the actual files on ``/g/data/rq0``.
   They are exposed as arguments rather than hard-coded for that reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

__all__ = ["accumulate", "coverage_mask", "load_rainfields"]


def load_rainfields(
    paths: list[Path] | Path,
    variable: str = "precipitation",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    time_name: str = "time",
) -> xr.DataArray:
    """Open one or more Rainfields3 files as a single time-ordered DataArray.

    Latitude is normalised to **ascending** order.  Radar products and model
    output disagree on this convention, and a descending/ascending mismatch
    produces silently empty ``sel`` slices with no error raised -- one of the
    more expensive failure modes in this kind of pipeline.
    """
    paths = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    if len(paths) > 1:
        ds = xr.open_mfdataset(paths, combine="by_coords")
    else:
        ds = xr.open_dataset(paths[0])
    if variable not in ds:
        raise KeyError(f"variable {variable!r} not in file; available: {list(ds.data_vars)}")
    da = ds[variable]
    da = da.rename({lat_name: "lat", lon_name: "lon"}) if lat_name in da.dims else da
    if "lat" in da.coords and da["lat"].size > 1 and da["lat"][0] > da["lat"][-1]:
        da = da.isel(lat=slice(None, None, -1))
    if time_name in da.dims:
        da = da.sortby(time_name)
    return da


def coverage_mask(da: xr.DataArray, min_valid_fraction: float = 0.98) -> xr.DataArray:
    """Boolean mask of grid cells observed for at least ``min_valid_fraction`` of times.

    Apply this mask identically to *every* arm before computing SAL.  Masking
    only the observations would leave model objects present where the radar is
    blind, inflating both the model's object count and its apparent structure.
    """
    if "time" not in da.dims:
        return xr.ones_like(da, dtype=bool)
    frac = np.isfinite(da).mean(dim="time")
    return frac >= min_valid_fraction


def accumulate(da: xr.DataArray, window: str = "24h", closed: str = "right") -> xr.DataArray:
    """Resample instantaneous or short-interval rates into accumulations (mm).

    SAL was designed for and is most informative on daily QPF accumulations;
    at 6-minute resolution object identification is dominated by advection
    noise and ``L2`` becomes uninterpretable.
    """
    if "time" not in da.dims:
        raise ValueError("input has no `time` dimension to accumulate over")
    return da.resample(time=window, closed=closed, label=closed).sum(skipna=False)
