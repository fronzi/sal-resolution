"""BARRA2 access over NCI THREDDS, with aggressive local caching.

Why BARRA-R2 vs BARRA-C2
------------------------
The two share the same underlying model and physics and differ principally in
horizontal resolution (12 km vs 4.4 km), which makes the pairing a genuinely
resolution-controlled experiment.  ACCESS-G3 vs ACCESS-C3 would be closer to
"what BoM runs operationally", but the two operational systems also differ in
data assimilation cycling, so any SAL difference is not attributable to
resolution alone.  Both BARRA2 products are public under CC-BY-4.0, so no NCI
project membership is needed.

Caching
-------
Effective throughput from Sydney to remote object stores is dominated by
per-chunk round-trip latency, not bandwidth.  The strategy is therefore a
**single bulk download of a fixed event set to local disk**, then fully offline
work.  For ~30 events x 4 arms at 500x500 float32 this is ~120 MB -- trivially
laptop-scale, and the opposite of a streaming workload.

Path convention
---------------
BARRA2 follows CORDEX-CMIP6 archiving specifications::

    {base}/output/reanalysis/{domain}/BOM/{driving}/historical/hres/
    {model}/{version}/{freq}/{variable}/latest/
    {variable}_{domain}_{driving}_historical_hres_BOM_{model}_{version}_{freq}_{YYYYMM}-{YYYYMM}.nc

.. warning::
   The facet values for BARRA-C2 (domain ``AUS-04``, driving model
   ``BARRA-R2``) should be confirmed against the live catalogue before a
   production download -- CORDEX facets for the convective product were still
   being finalised at release.  :func:`build_url` is deliberately a pure string
   builder so the assumption is inspectable and testable offline.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

__all__ = ["BARRA_C2", "BARRA_R2", "BARRA2Spec", "build_url", "cached_download"]

NCI_THREDDS_BASE = "https://thredds.nci.org.au/thredds/fileServer/ob53"


@dataclass(frozen=True)
class BARRA2Spec:
    """Facet values identifying one BARRA2 product stream."""

    model: str
    domain: str
    driving: str
    version: str = "v1"
    approx_spacing_km: float = 12.0


#: 12 km regional reanalysis, ERA5-driven. The "coarse" arm.
BARRA_R2 = BARRA2Spec(model="BARRA-R2", domain="AUS-11", driving="ERA5", approx_spacing_km=12.0)

#: 4.4 km convection-permitting downscaling of BARRA-R2. The "fine" arm.
BARRA_C2 = BARRA2Spec(model="BARRA-C2", domain="AUS-04", driving="BARRA-R2", approx_spacing_km=4.4)


def build_url(spec: BARRA2Spec, variable: str, year: int, month: int,
              freq: str = "1hr", base: str = NCI_THREDDS_BASE) -> str:
    """Construct the THREDDS fileServer URL for one monthly file."""
    if not 1 <= month <= 12:
        raise ValueError(f"month must be in 1..12, got {month}")
    stamp = f"{year:04d}{month:02d}"
    directory = (
        f"{base}/output/reanalysis/{spec.domain}/BOM/{spec.driving}/historical/hres/"
        f"{spec.model}/{spec.version}/{freq}/{variable}/latest"
    )
    fname = (
        f"{variable}_{spec.domain}_{spec.driving}_historical_hres_BOM_"
        f"{spec.model}_{spec.version}_{freq}_{stamp}-{stamp}.nc"
    )
    return f"{directory}/{fname}"


def cached_download(url: str, cache_dir: Path, timeout: int = 600) -> Path:
    """Download ``url`` into ``cache_dir`` unless already present.

    The cache key is a hash of the URL plus the original filename, so re-running
    the acquisition script is free.  Downloads land on a ``.part`` file and are
    renamed on success, so an interrupted transfer never leaves a truncated
    NetCDF that would fail cryptically inside xarray much later.
    """
    import urllib.request

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:12]
    target = cache_dir / f"{key}_{url.rsplit('/', 1)[-1]}"
    if target.exists() and target.stat().st_size > 0:
        LOGGER.debug("cache hit: %s", target.name)
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    LOGGER.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.rename(target)
    return target
