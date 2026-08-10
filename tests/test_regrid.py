import numpy as np
import pytest
import xarray as xr

from sal_resolution.regrid import (
    EARTH_RADIUS_M,
    ConservativeRegridder,
    cell_area,
    cell_edges,
    make_verification_grid,
    overlap_1d,
)


def test_cell_edges_ascending_and_descending():
    asc = cell_edges(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(asc, [0.5, 1.5, 2.5, 3.5])
    desc = cell_edges(np.array([3.0, 2.0, 1.0]))
    np.testing.assert_allclose(desc, [3.5, 2.5, 1.5, 0.5])


def test_cell_edges_rejects_non_monotonic():
    with pytest.raises(ValueError, match="monotonic"):
        cell_edges(np.array([1.0, 3.0, 2.0]))


def test_overlap_1d_partitions_exactly():
    src = np.linspace(0.0, 1.0, 11)
    dst = np.linspace(0.0, 1.0, 4)
    w = overlap_1d(src, dst)
    # every source cell is fully consumed by exactly one destination partition
    np.testing.assert_allclose(w.sum(axis=0), np.diff(src), atol=1e-14)
    np.testing.assert_allclose(w.sum(axis=1), np.diff(dst), atol=1e-14)


def test_constant_field_is_preserved_exactly():
    """A conservative remap of a constant field must return the same constant."""
    src_lat = np.linspace(-35.0, -32.0, 60)
    src_lon = np.linspace(150.0, 152.0, 40)
    dst_lat, dst_lon = make_verification_grid((-35.0, -32.0), (150.0, 152.0), 25.0)
    rg = ConservativeRegridder.from_grids(src_lat, src_lon, dst_lat, dst_lon)
    out = rg(np.full((60, 40), 7.25))
    interior = rg.coverage > 0.999
    np.testing.assert_allclose(out[interior], 7.25, rtol=0, atol=1e-12)


def test_mass_is_conserved_to_machine_precision():
    """Total water must survive the remap when the destination covers the source."""
    rng = np.random.default_rng(0)
    src_lat = np.linspace(-35.0, -32.0, 90)
    src_lon = np.linspace(150.0, 152.0, 60)
    field = rng.gamma(shape=1.5, scale=4.0, size=(90, 60))
    # destination spans slightly beyond the source so nothing falls off the edge
    dst_lat = np.linspace(-35.2, -31.8, 20)
    dst_lon = np.linspace(149.8, 152.2, 14)
    rg = ConservativeRegridder.from_grids(src_lat, src_lon, dst_lat, dst_lon)
    out = rg(field)
    src_mass = np.nansum(field * cell_area(src_lat, src_lon))
    # Weight by the *overlapped* area, not the nominal destination cell area:
    # outermost destination cells stick out past the source domain, so their
    # nominal area overstates the mass they actually received. This is exactly
    # what `coverage` is for.
    dst_mass = np.nansum(out * cell_area(dst_lat, dst_lon) * rg.coverage)
    assert abs(dst_mass - src_mass) / src_mass < 1e-12


def test_nan_masking_does_not_leak():
    src_lat = np.linspace(-35.0, -32.0, 40)
    src_lon = np.linspace(150.0, 152.0, 30)
    field = np.ones((40, 30))
    field[:10, :] = np.nan
    rg = ConservativeRegridder.from_grids(src_lat, src_lon, *make_verification_grid(
        (-35.0, -32.0), (150.0, 152.0), 25.0))
    out = rg(field)
    assert np.nanmax(out) <= 1.0 + 1e-12
    assert np.nanmin(out) >= 1.0 - 1e-12  # masked cells contribute nothing, not zero


def test_shape_mismatch_is_rejected():
    rg = ConservativeRegridder.from_grids(
        np.linspace(-35, -32, 10), np.linspace(150, 152, 10),
        np.linspace(-35, -32, 4), np.linspace(150, 152, 4))
    with pytest.raises(ValueError, match="does not match operator source grid"):
        rg(np.zeros((5, 5)))


def test_apply_wraps_dataarray_with_coords():
    src_lat = np.linspace(-35.0, -32.0, 30)
    src_lon = np.linspace(150.0, 152.0, 20)
    da = xr.DataArray(np.ones((3, 30, 20)), dims=("time", "lat", "lon"),
                      coords={"lat": src_lat, "lon": src_lon})
    dst_lat, dst_lon = make_verification_grid((-35.0, -32.0), (150.0, 152.0), 50.0)
    rg = ConservativeRegridder.from_grids(src_lat, src_lon, dst_lat, dst_lon)
    out = rg.apply(da)
    assert out.dims == ("time", "lat", "lon")
    assert out.sizes["lat"] == dst_lat.size and out.sizes["lon"] == dst_lon.size
    np.testing.assert_allclose(out["lat"].values, dst_lat)


def test_verification_grid_spacing_is_close_to_requested():
    """Spacing is snapped so cells tile the domain exactly; nominal is approximate."""
    lat, _lon = make_verification_grid((-35.0, -32.0), (150.0, 152.0), 25.0)
    dlat_km = np.deg2rad(np.diff(lat)[0]) * EARTH_RADIUS_M / 1000.0
    assert abs(dlat_km - 25.0) / 25.0 < 0.1


def test_every_spacing_tiles_the_identical_domain():
    """The non-negotiable invariant: L1 is normalised by the domain diagonal,
    and A is only comparable across spacings if the domain does not move."""
    bounds_lat, bounds_lon = (-35.0, -32.0), (150.0, 152.0)
    for spacing in (5.0, 10.0, 25.0, 50.0, 100.0):
        lat, lon = make_verification_grid(bounds_lat, bounds_lon, spacing)
        lat_e, lon_e = cell_edges(lat), cell_edges(lon)
        np.testing.assert_allclose([lat_e[0], lat_e[-1]], bounds_lat, atol=1e-10)
        np.testing.assert_allclose([lon_e[0], lon_e[-1]], bounds_lon, atol=1e-10)


def test_verification_grid_rejects_nonpositive_spacing():
    with pytest.raises(ValueError, match="spacing_km must be positive"):
        make_verification_grid((-35.0, -32.0), (150.0, 152.0), 0.0)
