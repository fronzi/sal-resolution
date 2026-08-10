"""Analytic checks on the spherical differential operators.

These exist because a factor of 180/pi vs pi/180 in a horizontal gradient is a
silent ~3283x error that no smoke test would catch.
"""
import numpy as np
import xarray as xr

from sal_resolution.features import (
    FEATURE_NAMES,
    build_features,
    horizontal_gradient,
    neighbourhood_stats,
    spherical_laplacian,
    upslope_forcing,
)
from sal_resolution.regrid import EARTH_RADIUS_M

LAT = np.linspace(-40.0, -20.0, 201)
LON = np.linspace(140.0, 160.0, 201)
INT = (slice(3, -3), slice(3, -3))


def test_gradient_matches_analytic_sin_lat():
    """f = sin(phi)  =>  df/dy = cos(phi)/a,  df/dx = 0."""
    phi = np.deg2rad(LAT)[:, None]
    f = np.broadcast_to(np.sin(phi), (LAT.size, LON.size)).copy()
    gx, gy = horizontal_gradient(f, LAT, LON)
    expected_gy = np.broadcast_to(np.cos(phi) / EARTH_RADIUS_M, f.shape)
    np.testing.assert_allclose(gy[INT], expected_gy[INT], rtol=1e-6)
    assert np.abs(gx).max() < 1e-18


def test_gradient_matches_analytic_in_longitude():
    """f = sin(lambda)  =>  df/dx = cos(lambda)/(a cos(phi))."""
    phi = np.deg2rad(LAT)[:, None]
    lam = np.deg2rad(LON)[None, :]
    f = np.broadcast_to(np.sin(lam), (LAT.size, LON.size)).copy()
    gx, _ = horizontal_gradient(f, LAT, LON)
    expected = np.cos(lam) / (EARTH_RADIUS_M * np.cos(phi))
    np.testing.assert_allclose(gx[INT], expected[INT], rtol=1e-5)


def test_laplacian_matches_analytic_spherical_harmonic():
    """f = sin(phi) is Y_1^0 up to normalisation  =>  lap f = -2 sin(phi)/a^2."""
    phi = np.deg2rad(LAT)[:, None]
    f = np.broadcast_to(np.sin(phi), (LAT.size, LON.size)).copy()
    lap = spherical_laplacian(f, LAT, LON)
    expected = np.broadcast_to(-2.0 * np.sin(phi) / EARTH_RADIUS_M**2, f.shape)
    np.testing.assert_allclose(lap[INT], expected[INT], rtol=1e-5)


def test_neighbourhood_stats_on_known_window():
    a = np.zeros((5, 5))
    a[2, 2] = 9.0
    assert neighbourhood_stats(a, 3, "max")[1, 1] == 9.0
    assert neighbourhood_stats(a, 3, "max")[0, 0] == 0.0
    np.testing.assert_allclose(neighbourhood_stats(a, 3, "mean")[2, 2], 1.0)
    assert neighbourhood_stats(a, 3, "std")[2, 2] > 0


def test_neighbourhood_stats_rejects_even_window():
    import pytest
    with pytest.raises(ValueError, match="odd"):
        neighbourhood_stats(np.zeros((4, 4)), 2, "mean")
    with pytest.raises(ValueError, match="unknown stat"):
        neighbourhood_stats(np.zeros((4, 4)), 3, "median")


def test_upslope_forcing_sign():
    """Westerly flow up an east-facing slope gives positive u.grad h."""
    lat = np.linspace(-35.0, -33.0, 40)
    lon = np.linspace(150.0, 152.0, 40)
    h = np.broadcast_to(np.linspace(0.0, 1000.0, 40)[None, :], (40, 40)).copy()
    u = np.full((40, 40), 10.0)   # eastward
    v = np.zeros((40, 40))
    ups = upslope_forcing(u, v, h, lat, lon)
    assert ups[5:-5, 5:-5].mean() > 0


def test_build_features_shape_and_names():
    lat = np.linspace(-35.0, -33.0, 24)
    lon = np.linspace(150.0, 152.0, 20)
    rng = np.random.default_rng(1)
    coords = {"lat": lat, "lon": lon}
    rain = xr.DataArray(rng.gamma(1.5, 3.0, (24, 20)), dims=("lat", "lon"), coords=coords)
    orog = xr.DataArray(rng.uniform(0, 800, (24, 20)), dims=("lat", "lon"), coords=coords)
    u = xr.DataArray(np.full((24, 20), 5.0), dims=("lat", "lon"), coords=coords)
    v = xr.DataArray(np.full((24, 20), -3.0), dims=("lat", "lon"), coords=coords)
    ds = build_features(rain, orog, u, v)
    assert tuple(ds.data_vars) == FEATURE_NAMES
    for name in FEATURE_NAMES:
        assert ds[name].shape == (24, 20)
        assert np.isfinite(ds[name].values).all()


def test_build_features_without_winds_keeps_column_order():
    lat, lon = np.linspace(-35, -33, 12), np.linspace(150, 152, 10)
    coords = {"lat": lat, "lon": lon}
    rain = xr.DataArray(np.ones((12, 10)), dims=("lat", "lon"), coords=coords)
    orog = xr.DataArray(np.zeros((12, 10)), dims=("lat", "lon"), coords=coords)
    ds = build_features(rain, orog)
    assert tuple(ds.data_vars) == FEATURE_NAMES
    np.testing.assert_allclose(ds["upslope"].values, 0.0)
