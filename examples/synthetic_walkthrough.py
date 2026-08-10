"""End-to-end walkthrough on synthetic fields -- no data download required.

Run with::

    python examples/synthetic_walkthrough.py

It exercises the full chain (conservative remap -> SAL -> baselines -> recovered
gain -> delta sweep) so the API and the shape of the result are clear before any
real data is touched.  The "fine" arm here is simply a less-smoothed version of
truth, so the numbers are a plumbing demonstration, not a scientific result.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from sal_resolution import (
    ConservativeRegridder,
    bootstrap_ci,
    evaluate_arms,
    make_verification_grid,
    recovered_gain,
)
from sal_resolution.data.events import EventCriteria, select_events
from sal_resolution.regrid import cell_area
from sal_resolution.symbolic.baselines import BASELINES
from sal_resolution.verify.compare import amplitude_invariance_check

LAT_BOUNDS, LON_BOUNDS = (-35.0, -32.0), (150.0, 152.0)
N_NATIVE, N_EVENTS = 200, 24
SPACINGS_KM = (5.0, 10.0, 25.0, 50.0)


def synthetic_truth(rng: np.random.Generator, n: int) -> np.ndarray:
    """A few compact convective blobs -- the kind of field SAL was designed for."""
    yy, xx = np.mgrid[0:n, 0:n]
    field = np.zeros((n, n))
    for _ in range(rng.integers(3, 7)):
        cy, cx = rng.uniform(0.15 * n, 0.85 * n, 2)
        rad = rng.uniform(0.015 * n, 0.045 * n)
        field += rng.uniform(10, 45) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad**2))
    return field


def main() -> None:
    rng = np.random.default_rng(0)
    lat = np.linspace(*LAT_BOUNDS, N_NATIVE)
    lon = np.linspace(*LON_BOUNDS, N_NATIVE)

    truth = [synthetic_truth(rng, N_NATIVE) for _ in range(N_EVENTS)]
    fine = [ndimage.gaussian_filter(t, 0.02 * N_NATIVE) for t in truth]      # 4.4 km proxy
    coarse = [ndimage.gaussian_filter(t, 0.07 * N_NATIVE) for t in truth]    # 12 km proxy

    # Events are screened on OBSERVATIONS only -- screening on a model would
    # condition the sample on that model being right.
    keep, diags = select_events(truth, EventCriteria(min_domain_mean=1.0))
    rejected = len(diags) - len(keep)
    print(f"events: {len(keep)} kept, {rejected} rejected")
    truth = [truth[i] for i in keep]
    fine = [fine[i] for i in keep]
    coarse = [coarse[i] for i in keep]

    # Baselines are fitted on the coarse arm against truth.
    correctors = {name: fn(coarse, truth) for name, fn in BASELINES.items()}
    arms_native = {"coarse": coarse, "fine": fine}
    for name, fn in correctors.items():
        if name != "identity":
            arms_native[name] = [fn(c) for c in coarse]

    print(f"\n{'Delta':>7} {'arm':>13} {'S_med':>8} {'A_med':>8} {'L_med':>8} "
          f"{'rho_S':>7} {'95% CI':>18}")
    print("-" * 72)

    by_spacing = {}
    for spacing in SPACINGS_KM:
        dst_lat, dst_lon = make_verification_grid(LAT_BOUNDS, LON_BOUNDS, spacing)
        rg = ConservativeRegridder.from_grids(lat, lon, dst_lat, dst_lon)
        area = cell_area(dst_lat, dst_lon)
        arms = {k: [rg(f) for f in v] for k, v in arms_native.items()}
        obs = [rg(o) for o in truth]

        ev = evaluate_arms(arms, obs, spacing_km=spacing, backend="reference", area=area)
        by_spacing[spacing] = ev["coarse"]

        for name, res in ev.items():
            s = res.summary()
            if name in ("coarse", "fine"):
                rho_txt, ci_txt = "--", "--"
            else:
                rho = recovered_gain(ev["coarse"], ev["fine"], res, component="S")
                stat, lo, hi = bootstrap_ci(rho, n_boot=2000)
                rho_txt = f"{stat:7.3f}" if np.isfinite(stat) else "    nan"
                ci_txt = f"[{lo:6.3f}, {hi:6.3f}]" if np.isfinite(lo) else "        --"
            print(f"{spacing:6.0f}k {name:>13} {s['S_median']:8.3f} {s['A_median']:8.3f} "
                  f"{s['L_median']:8.3f} {rho_txt:>7} {ci_txt:>18}")
        print()

    # The pipeline self-test: A is a ratio of domain means, so a mass-conserving
    # remap cannot change it across verification spacings.
    passed, drift = amplitude_invariance_check(by_spacing)
    verdict = "PASS" if passed else "FAIL"
    print(f"amplitude-invariance self-test: {verdict} (max drift {drift:.3e})")
    if not passed:
        raise SystemExit("non-conservative remapping detected; S results are untrustworthy")


if __name__ == "__main__":
    main()
