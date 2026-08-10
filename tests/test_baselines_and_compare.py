import numpy as np
import pytest

from sal_resolution.data.events import EventCriteria, screen_event, select_events
from sal_resolution.symbolic.baselines import (
    BASELINES,
    fit_identity,
    fit_quantile_mapping,
    fit_scalar_bias,
    fit_unsharp_mask,
)
from sal_resolution.verify.compare import (
    ArmResult,
    bootstrap_ci,
    evaluate_arms,
    recovered_gain,
)
from sal_resolution.verify.sal import SALResult


def _fields(rng, n=6, scale=1.0):
    return [scale * rng.gamma(1.5, 4.0, (40, 40)) for _ in range(n)]


def test_registry_covers_documented_baselines():
    assert set(BASELINES) == {"identity", "scalar_bias", "quantile_map", "unsharp"}


def test_identity_is_a_true_noop():
    rng = np.random.default_rng(0)
    f = _fields(rng, 3)
    corrector = fit_identity(f, f)
    np.testing.assert_array_equal(corrector(f[0]), f[0])


def test_scalar_bias_matches_the_domain_mean():
    rng = np.random.default_rng(1)
    obs = _fields(rng, 5)
    coarse = [0.5 * o for o in obs]
    corrector = fit_scalar_bias(coarse, obs)
    corrected = np.concatenate([corrector(c).ravel() for c in coarse])
    assert corrected.mean() == pytest.approx(np.concatenate([o.ravel() for o in obs]).mean(),
                                             rel=1e-10)


def test_quantile_mapping_is_monotone_and_lifts_the_tail():
    rng = np.random.default_rng(2)
    obs = _fields(rng, 5, scale=2.0)
    coarse = _fields(rng, 5, scale=1.0)
    corrector = fit_quantile_mapping(coarse, obs)
    probe = np.linspace(0.0, 60.0, 300)
    out = corrector(probe)
    assert np.all(np.diff(out) >= -1e-9), "transfer function must be non-decreasing"
    assert out[-1] > probe[-1]


def test_quantile_mapping_degenerates_gracefully_when_all_dry():
    dry = [np.zeros((10, 10))]
    corrector = fit_quantile_mapping(dry, dry)
    np.testing.assert_array_equal(corrector(dry[0]), dry[0])


def test_unsharp_mask_increases_variance_and_stays_nonnegative():
    rng = np.random.default_rng(3)
    obs = _fields(rng, 4, scale=2.0)
    from scipy import ndimage
    coarse = [ndimage.gaussian_filter(o, 2.0) for o in obs]
    corrector = fit_unsharp_mask(coarse, obs)
    out = corrector(coarse[0])
    assert np.nanvar(out) > np.nanvar(coarse[0])
    assert np.nanmin(out) >= 0.0
    assert corrector.params["amount"] > 0


def test_recovered_gain_arithmetic():
    """rho = 1 when the candidate matches the fine arm, 0 when it does nothing."""
    def arm(name, s):
        return ArmResult(name, 25.0, [SALResult(S=v, A=0.0, L=0.0, L1=0.0, L2=0.0) for v in s])

    coarse = arm("coarse", [-0.60, -0.40])
    fine = arm("fine", [-0.10, -0.10])
    exact = arm("sym", [-0.10, -0.10])
    useless = arm("sym", [-0.60, -0.40])
    np.testing.assert_allclose(recovered_gain(coarse, fine, exact), [1.0, 1.0])
    np.testing.assert_allclose(recovered_gain(coarse, fine, useless), [0.0, 0.0])


def test_recovered_gain_masks_events_with_no_dynamical_gain():
    def arm(name, s):
        return ArmResult(name, 25.0, [SALResult(S=v, A=0.0, L=0.0, L1=0.0, L2=0.0) for v in s])

    coarse = arm("coarse", [-0.30, -0.50])
    fine = arm("fine", [-0.301, -0.10])   # first event: fine model gained nothing
    cand = arm("sym", [-0.20, -0.20])
    rho = recovered_gain(coarse, fine, cand)
    assert np.isnan(rho[0])
    assert np.isfinite(rho[1])


def test_recovered_gain_rejects_bad_component():
    a = ArmResult("x", 25.0, [SALResult(0, 0, 0, 0, 0)])
    with pytest.raises(ValueError, match="component must be"):
        recovered_gain(a, a, a, component="Q")


def test_bootstrap_ci_brackets_the_statistic():
    rng = np.random.default_rng(4)
    v = rng.normal(0.5, 0.2, 200)
    stat, lo, hi = bootstrap_ci(v, n_boot=2000)
    assert lo < stat < hi
    assert stat == pytest.approx(np.median(v))


def test_bootstrap_ci_handles_all_nan():
    stat, lo, hi = bootstrap_ci(np.full(10, np.nan))
    assert np.isnan(stat) and np.isnan(lo) and np.isnan(hi)


def test_evaluate_arms_rejects_length_mismatch():
    obs = [np.ones((8, 8))] * 3
    with pytest.raises(ValueError, match="has 2 events, obs has 3"):
        evaluate_arms({"a": [np.ones((8, 8))] * 2}, obs, spacing_km=25.0)


def test_evaluate_arms_and_summary():
    rng = np.random.default_rng(5)
    obs = [rng.gamma(2.0, 5.0, (30, 30)) for _ in range(4)]
    arms = {"coarse": [0.6 * o for o in obs], "fine": obs}
    res = evaluate_arms(arms, obs, spacing_km=25.0, backend="reference")
    assert set(res) == {"coarse", "fine"}
    assert res["fine"].summary()["S_median"] == pytest.approx(0.0, abs=1e-12)
    assert res["coarse"].summary()["n_valid"] == 4


def test_event_screening_rejects_dry_and_accepts_wet():
    dry = np.zeros((50, 50))
    ok, diag = screen_event(dry)
    assert not ok and diag["reason"] in {"domain_mean", "wet_fraction"}

    yy, xx = np.mgrid[0:50, 0:50]
    wet = 40.0 * np.exp(-((yy - 25) ** 2 + (xx - 25) ** 2) / 200.0)
    ok, diag = screen_event(wet, EventCriteria(min_domain_mean=1.0))
    assert ok and diag["reason"] == "pass" and diag["n_objects"] >= 1


def test_event_screening_rejects_poor_coverage():
    field = np.full((50, 50), np.nan)
    field[:10] = 10.0
    ok, diag = screen_event(field)
    assert not ok and diag["reason"] == "coverage"


def test_select_events_returns_indices_and_full_diagnostics():
    yy, xx = np.mgrid[0:40, 0:40]
    wet = 40.0 * np.exp(-((yy - 20) ** 2 + (xx - 20) ** 2) / 150.0)
    fields = [np.zeros((40, 40)), wet, np.zeros((40, 40))]
    keep, diags = select_events(fields, EventCriteria(min_domain_mean=1.0))
    np.testing.assert_array_equal(keep, [1])
    assert len(diags) == 3  # rejection reasons are retained for the methods section
