import numpy as np
import pytest

from sal_resolution.config import coerce_floats
from sal_resolution.data.barra2 import BARRA_C2, BARRA_R2, build_url
from sal_resolution.symbolic.fit import EPS_MM, loyo_splits, structure_weights


def test_build_url_follows_cordex_convention():
    url = build_url(BARRA_R2, "pr", 2018, 3)
    assert "/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/pr/latest/" in url
    assert url.endswith("pr_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_1hr_201803-201803.nc")


def test_build_url_for_convective_product():
    url = build_url(BARRA_C2, "pr", 2020, 12)
    # BARRA-C2 is a 4.4 km downscaling *driven by* BARRA-R2, so BARRA-R2
    # appears in the driving-model facet, not the model facet.
    assert "/AUS-04/BOM/BARRA-R2/historical/hres/BARRA-C2/v1/" in url
    assert url.endswith(
        "pr_AUS-04_BARRA-R2_historical_hres_BOM_BARRA-C2_v1_1hr_202012-202012.nc")


def test_build_url_rejects_bad_month():
    with pytest.raises(ValueError, match="month must be"):
        build_url(BARRA_R2, "pr", 2018, 13)


def test_specs_carry_the_resolutions_under_test():
    assert BARRA_R2.approx_spacing_km == 12.0
    assert BARRA_C2.approx_spacing_km == 4.4


def test_coerce_floats_fixes_yaml_scientific_notation():
    """PyYAML leaves `3.0e-3` as a string unless the exponent is signed."""
    cfg = {"symbolic": {"parsimony": "3.0e-3", "ops": ["+", "-"]}, "n": [1, "2.5"]}
    out = coerce_floats(cfg)
    assert out["symbolic"]["parsimony"] == pytest.approx(3.0e-3)
    assert out["symbolic"]["ops"] == ["+", "-"]  # non-numeric strings untouched
    assert out["n"] == [1, 2.5]


def test_loyo_splits_are_year_blocked_and_disjoint():
    years = [2016, 2016, 2017, 2018, 2018]
    splits = loyo_splits(years)
    assert len(splits) == 3
    for train, test in splits:
        assert set(train).isdisjoint(set(test))
        assert len(set(np.array(years)[test])) == 1


def test_loyo_splits_skips_degenerate_single_year():
    assert loyo_splits([2020, 2020]) == []


def test_structure_weights_upweight_underdispersed_points():
    rng = np.random.default_rng(0)
    obs = rng.gamma(2.0, 5.0, (30, 30))
    from scipy import ndimage
    coarse = ndimage.gaussian_filter(obs, 3.0)
    w = structure_weights([coarse], [obs])
    assert w.min() >= 1.0
    assert w.max() > 1.0


def test_eps_regularises_the_log_ratio():
    assert 0 < EPS_MM < 1.0
