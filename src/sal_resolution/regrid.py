"""Exact conservative remapping between rectilinear lat/lon grids.

Rationale
---------
SAL is *not* resolution-invariant: the structure component ``S`` depends on
object identification, which depends on grid spacing and on the threshold
``R* = R_max / f``.  Comparing a 12 km model against a 4.4 km model on their
native grids therefore measures grid spacing, not model quality.  Every arm of
the experiment must be remapped to a common verification grid *first*.

The remapping must be mass-conserving.  Bilinear or nearest-neighbour
interpolation of precipitation changes the domain-mean and will corrupt the
SAL amplitude component ``A``.

Method
------
On the sphere the area element is

.. math:: \\mathrm{d}A = a^2 \\, \\mathrm{d}\\lambda \\, \\mathrm{d}(\\sin\\varphi)

so the cell-overlap area between two rectilinear lat/lon cells factorises
*exactly* into a 1-D overlap in ``sin(lat)`` and a 1-D overlap in ``lon``.
The first-order conservative remap is therefore

.. math::
    R^{\\mathrm{dst}}_{ik}
    = \\frac{\\sum_{jl} W^{\\varphi}_{ij} W^{\\lambda}_{kl} m_{jl} R^{\\mathrm{src}}_{jl}}
           {\\sum_{jl} W^{\\varphi}_{ij} W^{\\lambda}_{kl} m_{jl}}

with ``m`` a validity mask.  This is a pair of dense matrix multiplies, needs
no compiled dependency (no ESMF / xESMF), and is exact to machine precision.

Limitations
-----------
* Rectilinear (1-D coordinate) grids only.  Rotated-pole or curvilinear source
  grids must be reprojected first -- see :func:`assert_rectilinear`.
* No longitude wrapping.  Fine for regional Australian domains; would need
  branch-cut handling for global fields.
* First-order (piecewise-constant) reconstruction.  Conservative but diffusive
  when upsampling; that is the correct behaviour here since we only ever
  coarsen to the verification grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6_371_000.0

__all__ = [
    "EARTH_RADIUS_M",
    "ConservativeRegridder",
    "cell_area",
    "cell_edges",
    "make_verification_grid",
    "overlap_1d",
]


def cell_edges(centres: np.ndarray) -> np.ndarray:
    """Infer monotonic cell edges from 1-D cell centres.

    Interior edges are midpoints; the two outer edges are reflected so that the
    first and last cells have the width implied by their neighbours.

    Works for both ascending and descending coordinates.  Descending latitude
    is the ERA5/Pangu convention and silently produces empty selections
    downstream if not handled -- so it is handled here rather than assumed away.
    """
    centres = np.asarray(centres, dtype=np.float64)
    if centres.ndim != 1 or centres.size < 2:
        raise ValueError("`centres` must be 1-D with at least 2 elements")
    diffs = np.diff(centres)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError("`centres` must be strictly monotonic")
    interior = 0.5 * (centres[:-1] + centres[1:])
    first = centres[0] - 0.5 * diffs[0]
    last = centres[-1] + 0.5 * diffs[-1]
    return np.concatenate([[first], interior, [last]])


def overlap_1d(src_edges: np.ndarray, dst_edges: np.ndarray) -> np.ndarray:
    """Exact 1-D interval-overlap matrix of shape ``(n_dst, n_src)``.

    Edges may be ascending or descending; they are normalised internally.
    """
    src = np.asarray(src_edges, dtype=np.float64)
    dst = np.asarray(dst_edges, dtype=np.float64)
    src_lo = np.minimum(src[:-1], src[1:])
    src_hi = np.maximum(src[:-1], src[1:])
    dst_lo = np.minimum(dst[:-1], dst[1:])
    dst_hi = np.maximum(dst[:-1], dst[1:])
    lo = np.maximum(dst_lo[:, None], src_lo[None, :])
    hi = np.minimum(dst_hi[:, None], src_hi[None, :])
    return np.clip(hi - lo, 0.0, None)


def cell_area(lat: np.ndarray, lon: np.ndarray, radius: float = EARTH_RADIUS_M) -> np.ndarray:
    """Spherical cell areas (m^2) for a rectilinear grid, shape ``(nlat, nlon)``."""
    lat_e = np.deg2rad(cell_edges(lat))
    lon_e = np.deg2rad(cell_edges(lon))
    dsin = np.abs(np.diff(np.sin(lat_e)))
    dlon = np.abs(np.diff(lon_e))
    return radius**2 * dsin[:, None] * dlon[None, :]


@dataclass(frozen=True)
class ConservativeRegridder:
    """Precomputed first-order conservative remap operator.

    Build once per (source grid, verification grid) pair and reuse across every
    time step and every model arm -- the weight matrices are the expensive part.

    Parameters
    ----------
    w_lat, w_lon
        Overlap matrices in ``sin(lat)`` and ``lon`` (radians), shapes
        ``(n_dst_lat, n_src_lat)`` and ``(n_dst_lon, n_src_lon)``.
    dst_lat, dst_lon
        Destination cell centres, retained so the output can be re-wrapped as a
        labelled :class:`xarray.DataArray`.
    """

    w_lat: np.ndarray
    w_lon: np.ndarray
    dst_lat: np.ndarray
    dst_lon: np.ndarray
    lat_name: str = "lat"
    lon_name: str = "lon"

    @classmethod
    def from_grids(
        cls,
        src_lat: np.ndarray,
        src_lon: np.ndarray,
        dst_lat: np.ndarray,
        dst_lon: np.ndarray,
        lat_name: str = "lat",
        lon_name: str = "lon",
    ) -> ConservativeRegridder:
        w_lat = overlap_1d(np.sin(np.deg2rad(cell_edges(src_lat))),
                           np.sin(np.deg2rad(cell_edges(dst_lat))))
        w_lon = overlap_1d(np.deg2rad(cell_edges(src_lon)),
                           np.deg2rad(cell_edges(dst_lon)))
        return cls(w_lat, w_lon, np.asarray(dst_lat, float),
                   np.asarray(dst_lon, float), lat_name, lon_name)

    @property
    def coverage(self) -> np.ndarray:
        """Fraction of each destination cell covered by the source grid.

        Destination cells with coverage < 1 straddle the source domain edge;
        they are partially informed and should normally be trimmed before
        computing SAL, because a truncated object biases both ``S`` and ``L``.
        """
        num = self.w_lat.sum(axis=1)[:, None] * self.w_lon.sum(axis=1)[None, :]
        dst_full = (np.abs(np.diff(np.sin(np.deg2rad(cell_edges(self.dst_lat)))))[:, None]
                    * np.abs(np.diff(np.deg2rad(cell_edges(self.dst_lon))))[None, :])
        return num / dst_full

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """Remap a 2-D ``(lat, lon)`` array. NaNs are treated as missing."""
        src = np.asarray(field, dtype=np.float64)
        if src.shape != (self.w_lat.shape[1], self.w_lon.shape[1]):
            raise ValueError(
                f"field shape {src.shape} does not match operator source grid "
                f"{(self.w_lat.shape[1], self.w_lon.shape[1])}"
            )
        mask = np.isfinite(src).astype(np.float64)
        filled = np.where(mask > 0, src, 0.0)
        num = self.w_lat @ filled @ self.w_lon.T
        den = self.w_lat @ mask @ self.w_lon.T
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0, num / den, np.nan)
        return out

    def apply(self, da: xr.DataArray) -> xr.DataArray:
        """Remap a labelled DataArray, broadcasting over any leading dims."""
        core = [self.lat_name, self.lon_name]
        missing = [d for d in core if d not in da.dims]
        if missing:
            raise ValueError(f"DataArray is missing dimension(s) {missing}; has {da.dims}")
        out = xr.apply_ufunc(
            self.__call__,
            da,
            input_core_dims=[core],
            output_core_dims=[core],
            exclude_dims=set(core),
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={"output_sizes": {self.lat_name: self.dst_lat.size,
                                                 self.lon_name: self.dst_lon.size}},
        )
        return out.assign_coords({self.lat_name: self.dst_lat, self.lon_name: self.dst_lon})


def make_verification_grid(
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    spacing_km: float,
    radius: float = EARTH_RADIUS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular lat/lon verification grid of approximately ``spacing_km``.

    Longitude spacing is set at the domain-centre latitude so that cells are
    close to square in the middle of the domain.  The domain is *fixed across
    every arm and every value of* ``spacing_km`` -- the SAL location component
    ``L1`` is normalised by the domain diagonal, so a domain that shifts between
    arms makes ``L`` incomparable.
    """
    if spacing_km <= 0:
        raise ValueError("spacing_km must be positive")
    lat0, lat1 = sorted(lat_bounds)
    lon0, lon1 = sorted(lon_bounds)
    target_dlat = np.rad2deg(spacing_km * 1000.0 / radius)
    lat_mid = np.deg2rad(0.5 * (lat0 + lat1))
    target_dlon = target_dlat / max(np.cos(lat_mid), 1e-6)
    nlat = max(round((lat1 - lat0) / target_dlat), 1)
    nlon = max(round((lon1 - lon0) / target_dlon), 1)
    # Snap the spacing so the cells tile [lat0, lat1] x [lon0, lon1] exactly.
    # Every spacing in the sweep must span the *identical* physical domain: L1 is
    # normalised by the domain diagonal, and A is only invariant across spacings
    # if the integration domain does not move.
    dlat = (lat1 - lat0) / nlat
    dlon = (lon1 - lon0) / nlon
    lat = lat0 + (np.arange(nlat) + 0.5) * dlat
    lon = lon0 + (np.arange(nlon) + 0.5) * dlon
    return lat, lon
