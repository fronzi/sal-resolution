"""Coverage of the orchestration layer: config, CLI, loaders, sweep plumbing."""
import json

import numpy as np
import pytest
import xarray as xr

from sal_resolution.cli import build_parser, main
from sal_resolution.config import load_config
from sal_resolution.data.rainfields import accumulate, coverage_mask, load_rainfields
from sal_resolution.features import FEATURE_NAMES, build_features
from sal_resolution.regrid import ConservativeRegridder, cell_area, make_verification_grid
from sal_resolution.symbolic.fit import assemble_training_matrix, symbolic_available
from sal_resolution.verify.compare import (
    amplitude_invariance_check,
    delta_sweep,
    evaluate_arms,
)

CFG = """
domain:
  lat_bounds: [-35.0, -32.0]
  lon_bounds: [150.0, 152.0]
verification:
  spacings_km: [25, 50]
data:
  cache_dir: "~/tmp_sal_cache"
  coarse: {variable: "pr"}
  fine: {variable: "pr"}
events:
  years: [2018]
symbolic:
  parsimony: "3.0e-3"
"""


@pytest.fixture()
def cfg_path(tmp_path):
    p = tmp_path / "config.yml"
    p.write_text(CFG)
    return p


def test_load_config_coerces_and_expands(cfg_path):
    cfg = load_config(cfg_path)
    assert cfg["symbolic"]["parsimony"] == pytest.approx(3.0e-3)
    assert not cfg["data"]["cache_dir"].startswith("~")


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_fetch_dry_run_emits_urls_without_network(cfg_path, capsys):
    rc = main(["fetch", "--config", str(cfg_path), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "BARRA-R2" in out and "BARRA-C2" in out
    assert len(out.strip().splitlines()) == 24  # 2 arms x 1 year x 12 months


def test_fit_reports_missing_backend_or_missing_data(cfg_path):
    rc = main(["fit", "--config", str(cfg_path)])
    assert rc == (3 if symbolic_available() else 2)


def test_evaluate_writes_a_status_file(cfg_path, tmp_path):
    out = tmp_path / "results.json"
    rc = main(["evaluate", "--config", str(cfg_path), "--output", str(out)])
    assert rc == 3
    assert json.loads(out.read_text())["status"] == "not_run"


def _radar_dataset(tmp_path, descending_lat=True):
    lat = np.linspace(-32.0, -35.0, 12) if descending_lat else np.linspace(-35.0, -32.0, 12)
    lon = np.linspace(150.0, 152.0, 10)
    time = np.array(["2018-03-01T00:00", "2018-03-01T00:06", "2018-03-01T00:12"],
                    dtype="datetime64[ns]")
    rng = np.random.default_rng(0)
    da = xr.DataArray(rng.gamma(1.5, 2.0, (3, 12, 10)), dims=("time", "latitude", "longitude"),
                      coords={"time": time, "latitude": lat, "longitude": lon},
                      name="precipitation")
    path = tmp_path / "rf3.nc"
    da.to_dataset().to_netcdf(path)
    return path


def test_load_rainfields_normalises_descending_latitude(tmp_path):
    """Descending vs ascending latitude mismatches produce silently empty
    selections downstream, so orientation is normalised at load time."""
    da = load_rainfields(_radar_dataset(tmp_path, descending_lat=True))
    assert da["lat"][0] < da["lat"][-1]
    assert da.dims == ("time", "lat", "lon")


def test_load_rainfields_rejects_unknown_variable(tmp_path):
    with pytest.raises(KeyError, match="not in file"):
        load_rainfields(_radar_dataset(tmp_path), variable="rainrate")


def test_accumulate_conserves_total_and_is_right_labelled(tmp_path):
    """Right-closed/right-labelled: a 24 h total is stamped at the END of its
    window, matching the meteorological convention for accumulations."""
    da = load_rainfields(_radar_dataset(tmp_path))
    acc = accumulate(da, window="24h")
    np.testing.assert_allclose(acc.sum("time").values, da.sum("time").values, rtol=1e-10)
    assert acc["time"].values[-1] > da["time"].values[-1]


def test_accumulate_left_closed_puts_the_day_in_one_bin(tmp_path):
    da = load_rainfields(_radar_dataset(tmp_path))
    acc = accumulate(da, window="24h", closed="left")
    assert acc.sizes["time"] == 1
    np.testing.assert_allclose(acc.isel(time=0).values, da.sum("time").values, rtol=1e-10)


def test_accumulate_requires_a_time_dimension(tmp_path):
    da = load_rainfields(_radar_dataset(tmp_path)).isel(time=0)
    with pytest.raises(ValueError, match="no `time` dimension"):
        accumulate(da)


def test_coverage_mask_excludes_persistently_blind_cells(tmp_path):
    da = load_rainfields(_radar_dataset(tmp_path))
    da = da.copy()
    da[:, 0, 0] = np.nan          # a permanently blind cell (e.g. beam blocking)
    da[0, 1, 1] = np.nan          # a single dropout -> still 2/3 valid
    mask = coverage_mask(da, min_valid_fraction=0.98)
    assert not bool(mask.isel(lat=0, lon=0))
    assert not bool(mask.isel(lat=1, lon=1))
    assert bool(mask.isel(lat=5, lon=5))


def test_coverage_mask_without_time_is_all_true(tmp_path):
    da = load_rainfields(_radar_dataset(tmp_path)).isel(time=0)
    assert bool(coverage_mask(da).all())


def _synthetic_arms(n_events=4, seed=0, n=180):
    """Blob fields with a real structure gradient between arms.

    White noise will not do: after conservative remapping to a coarse
    verification grid the arms become indistinguishable in S and every event is
    masked out of rho by the min_gain guard.
    """
    rng = np.random.default_rng(seed)
    src_lat = np.linspace(-35.0, -32.0, n)
    src_lon = np.linspace(150.0, 152.0, n)
    from scipy import ndimage
    yy, xx = np.mgrid[0:n, 0:n]
    obs_native = []
    for _ in range(n_events):
        f = np.zeros((n, n))
        for _ in range(5):
            cy, cx = rng.uniform(0.2 * n, 0.8 * n, 2)
            rad = rng.uniform(0.02 * n, 0.05 * n)
            f += rng.uniform(10, 40) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad**2))
        obs_native.append(f)
    fine = [ndimage.gaussian_filter(o, 0.02 * n) for o in obs_native]
    coarse = [ndimage.gaussian_filter(o, 0.08 * n) for o in obs_native]
    sym = [ndimage.gaussian_filter(o, 0.045 * n) for o in obs_native]
    return src_lat, src_lon, obs_native, {"coarse": coarse, "fine": fine, "symbolic": sym}


def test_amplitude_invariance_check_passes_for_conservative_remapping():
    src_lat, src_lon, obs, arms = _synthetic_arms()
    by_spacing = {}
    for spacing in (25.0, 50.0):
        dl, dn = make_verification_grid((-35.0, -32.0), (150.0, 152.0), spacing)
        rg = ConservativeRegridder.from_grids(src_lat, src_lon, dl, dn)
        area = cell_area(dl, dn)
        res = evaluate_arms({"coarse": [rg(f) for f in arms["coarse"]]},
                            [rg(o) for o in obs], spacing_km=spacing,
                            backend="reference", area=area)
        by_spacing[spacing] = res["coarse"]
    passed, drift = amplitude_invariance_check(by_spacing)
    assert passed, f"A drifted by {drift:g}; remap is not conservative"


def test_amplitude_invariance_check_is_trivially_true_for_one_spacing():
    _, _, obs, arms = _synthetic_arms(n_events=2)
    res = evaluate_arms({"coarse": arms["coarse"]}, obs, 25.0, backend="reference")
    assert amplitude_invariance_check({25.0: res["coarse"]}) == (True, 0.0)


def test_delta_sweep_returns_a_ci_per_spacing():
    src_lat, src_lon, obs, arms = _synthetic_arms(n_events=8, seed=3)
    arms_by, obs_by = {}, {}
    for spacing in (5.0, 10.0):
        dl, dn = make_verification_grid((-35.0, -32.0), (150.0, 152.0), spacing)
        rg = ConservativeRegridder.from_grids(src_lat, src_lon, dl, dn)
        arms_by[spacing] = {k: [rg(f) for f in v] for k, v in arms.items()}
        obs_by[spacing] = [rg(o) for o in obs]
    out = delta_sweep(arms_by, obs_by, candidate_names=("symbolic",), backend="reference")
    assert set(out) == {5.0, 10.0}
    for spacing, row in out.items():
        stat, lo, hi = row["symbolic"]
        assert np.isfinite([stat, lo, hi]).all(), f"rho all-NaN at {spacing} km"
        assert lo <= stat <= hi
        # the intermediate-smoothing arm sits between coarse and fine by
        # construction, so it must recover a positive fraction of the gain
        assert 0.0 < stat < 1.5


def test_assemble_training_matrix_shapes_and_dry_point_exclusion():
    lat = np.linspace(-35.0, -33.0, 20)
    lon = np.linspace(150.0, 152.0, 18)
    coords = {"lat": lat, "lon": lon}
    rng = np.random.default_rng(11)
    feats, coarse, obs = [], [], []
    for _ in range(3):
        rain = rng.gamma(1.2, 3.0, (20, 18))
        rain[:5] = 0.0                       # a dry block that must be dropped
        orog = xr.DataArray(rng.uniform(0, 500, (20, 18)), dims=("lat", "lon"), coords=coords)
        feats.append(build_features(xr.DataArray(rain, dims=("lat", "lon"), coords=coords), orog))
        coarse.append(rain)
        o = rain * 1.3
        o[:5] = 0.0
        obs.append(o)
    X, y, idx = assemble_training_matrix(feats, coarse, obs)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert X.shape[0] == y.size == idx.size
    assert X.shape[0] < 3 * 20 * 18          # dry points excluded
    assert set(np.unique(idx)) == {0, 1, 2}
    assert np.isfinite(y).all()
    # obs = 1.3 * coarse everywhere, so the log ratio is bounded by log(1.3)
    assert 0.0 < y.max() <= np.log(1.3) + 1e-9


def test_assemble_training_matrix_rejects_empty_input():
    with pytest.raises(ValueError, match="no events supplied"):
        assemble_training_matrix([], [], [])
