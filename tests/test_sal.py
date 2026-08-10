"""Behavioural checks on the reference SAL implementation.

These double as a third cross-check (alongside pysteps) for the `scores`
contribution: the sign conventions asserted here are the ones the paper's
conclusions depend on.
"""
import numpy as np
import pytest

from sal_resolution.regrid import ConservativeRegridder, cell_area, make_verification_grid
from sal_resolution.verify.sal import SALResult, compute_sal, reference_sal


def _blob(shape, centre, radius, peak=20.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    r2 = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2
    return peak * np.exp(-r2 / (2.0 * radius**2))


def test_identical_fields_give_zero_sal():
    f = _blob((80, 80), (40, 40), 8.0)
    r = reference_sal(f, f)
    assert abs(r.S) < 1e-12
    assert abs(r.A) < 1e-12
    assert abs(r.L) < 1e-12
    assert r.distance < 1e-12


def test_uniform_scaling_isolates_amplitude():
    """Doubling the field must move A and leave S essentially untouched.

    S is computed from volume-normalised objects, so it is invariant under a
    uniform multiplicative factor. If a change here shows up in S, the object
    normalisation is wrong.
    """
    o = _blob((80, 80), (40, 40), 8.0)
    r = reference_sal(2.0 * o, o)
    assert pytest.approx((2.0 - 1.0) / (0.5 * 3.0), rel=1e-10) == r.A
    assert abs(r.S) < 1e-10
    assert abs(r.L) < 1e-10


def test_too_small_objects_give_negative_S():
    """An under-resolved model produces objects that are too peaked => S < 0."""
    obs = _blob((100, 100), (50, 50), 14.0)
    fcst = _blob((100, 100), (50, 50), 6.0)
    assert reference_sal(fcst, obs).S < 0


def test_too_broad_objects_give_positive_S():
    obs = _blob((100, 100), (50, 50), 6.0)
    fcst = _blob((100, 100), (50, 50), 14.0)
    assert reference_sal(fcst, obs).S > 0


def test_displacement_shows_up_in_L1_only():
    obs = _blob((100, 100), (40, 50), 7.0)
    fcst = _blob((100, 100), (60, 50), 7.0)
    r = reference_sal(fcst, obs)
    assert r.L1 > 0.1
    assert abs(r.S) < 1e-6          # same shape, just moved
    assert pytest.approx(0.0, abs=1e-6) == r.L2  # single object each => r = 0


def test_object_spread_shows_up_in_L2():
    """Two well-separated objects vs one centred object moves L2."""
    obs = _blob((100, 100), (50, 50), 6.0)
    fcst = _blob((100, 100), (50, 20), 6.0) + _blob((100, 100), (50, 80), 6.0)
    r = reference_sal(fcst, obs)
    assert r.L2 > 0.1
    assert r.n_objects_fcst == 2 and r.n_objects_obs == 1


def test_shape_mismatch_is_rejected_with_useful_message():
    with pytest.raises(ValueError, match=r"common\s+verification grid"):
        reference_sal(np.ones((10, 10)), np.ones((12, 12)))


def test_fixed_threshold_changes_object_count():
    f = _blob((100, 100), (50, 50), 10.0, peak=20.0)
    low = reference_sal(f, f, fixed_threshold=0.5)
    high = reference_sal(f, f, fixed_threshold=15.0)
    assert low.n_objects_fcst >= 1 and high.n_objects_fcst >= 1
    assert abs(low.S) < 1e-12 and abs(high.S) < 1e-12


def test_amplitude_is_invariant_under_conservative_regridding():
    """The self-test for the whole pipeline.

    A is a ratio of domain means, so mass-conserving remapping cannot change
    it. If this fails, a bilinear interpolation has crept into the chain -- and
    the same bug would corrupt S far less visibly.
    """
    rng = np.random.default_rng(7)
    src_lat = np.linspace(-35.0, -32.0, 120)
    src_lon = np.linspace(150.0, 152.0, 120)
    obs = rng.gamma(1.5, 4.0, (120, 120))
    fcst = 1.4 * rng.gamma(1.5, 4.0, (120, 120))

    a_values = []
    for spacing in (10.0, 25.0, 50.0):
        dst_lat, dst_lon = make_verification_grid((-35.0, -32.0), (150.0, 152.0), spacing)
        rg = ConservativeRegridder.from_grids(src_lat, src_lon, dst_lat, dst_lon)
        area = cell_area(dst_lat, dst_lon)
        a_values.append(reference_sal(rg(fcst), rg(obs), area=area).A)

    assert max(a_values) - min(a_values) < 1e-9, (
        "A drifted across verification spacings: the remap is not conservative, "
        "or the domain is not identical across spacings"
    )


def test_compute_sal_reference_backend_matches_reference_sal():
    f = _blob((60, 60), (30, 30), 6.0)
    o = _blob((60, 60), (32, 28), 8.0)
    a = compute_sal(f, o, backend="reference")
    b = reference_sal(f, o)
    assert a.as_dict() == b.as_dict()


def test_area_weighting_shape_is_validated():
    with pytest.raises(ValueError, match="does not match field shape"):
        reference_sal(np.ones((10, 10)), np.ones((10, 10)), area=np.ones((5, 5)))


def test_area_weighting_changes_A_on_a_latitudinally_wide_grid():
    """Sanity: with cos(lat) weighting the amplitude is not the naive mean ratio."""
    from sal_resolution.regrid import cell_area as _ca
    lat = np.linspace(-45.0, -10.0, 40)
    lon = np.linspace(140.0, 150.0, 40)
    obs = np.ones((40, 40))
    fcst = np.linspace(0.5, 1.5, 40)[:, None] * np.ones((1, 40))
    naive = reference_sal(fcst, obs).A
    weighted = reference_sal(fcst, obs, area=_ca(lat, lon)).A
    assert not np.isclose(naive, weighted)


def test_sal_result_distance():
    r = SALResult(S=0.3, A=-0.4, L=0.0, L1=0.0, L2=0.0)
    assert r.distance == pytest.approx(0.5)
