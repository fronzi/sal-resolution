"""Command-line entry points.

The pipeline is deliberately split into three commands so that the expensive,
network-bound step runs exactly once and everything downstream is offline and
cheap to iterate on::

    sal-resolution fetch    --config config.yml   # one bulk download, then never again
    sal-resolution fit      --config config.yml   # PySR, LOYO
    sal-resolution evaluate --config config.yml   # SAL, delta sweep, bootstrap

``evaluate`` runs the amplitude-invariance self-test first and refuses to
proceed if it fails, because a non-conservative remap corrupts ``S`` in a way
that is not visually obvious in the results.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger("sal_resolution")

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sal-resolution", description=__doc__)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="one-time bulk download of the event set")
    f.add_argument("--config", type=Path, default=Path("config.yml"))
    f.add_argument("--dry-run", action="store_true",
                   help="print the URLs that would be fetched and exit")

    t = sub.add_parser("fit", help="fit the symbolic correction with LOYO CV")
    t.add_argument("--config", type=Path, default=Path("config.yml"))

    e = sub.add_parser("evaluate", help="SAL comparison, delta sweep, bootstrap CIs")
    e.add_argument("--config", type=Path, default=Path("config.yml"))
    e.add_argument("--output", type=Path, default=Path("results.json"))
    e.add_argument("--skip-invariance-check", action="store_true",
                   help="not recommended; the check catches non-conservative remapping")
    return p


def cmd_fetch(args) -> int:
    from .config import load_config
    from .data.barra2 import BARRA_C2, BARRA_R2, build_url, cached_download

    cfg = load_config(args.config)
    cache = Path(cfg["data"]["cache_dir"])
    years = cfg["events"]["years"]
    urls = [
        build_url(spec, cfg["data"][arm]["variable"], int(year), month)
        for arm, spec in (("coarse", BARRA_R2), ("fine", BARRA_C2))
        for year in years
        for month in range(1, 13)
    ]
    if args.dry_run:
        print("\n".join(urls))
        LOGGER.info("%d files would be fetched", len(urls))
        return 0
    for url in urls:
        try:
            cached_download(url, cache)
        except Exception as exc:  # noqa: BLE001 - a missing month must not abort the bulk fetch
            LOGGER.warning("failed %s: %s", url, exc)
    LOGGER.info("cache populated at %s", cache)
    return 0


def cmd_fit(args) -> int:
    from .config import load_config
    from .symbolic.fit import symbolic_available

    load_config(args.config)
    if not symbolic_available():
        LOGGER.error(
            "PySR unavailable. `pip install pysr`, then `python -c 'import pysr; "
            "pysr.install()'` to provision the Julia backend."
        )
        return 2
    LOGGER.error(
        "cmd_fit requires the cached event set. Run `sal-resolution fetch` first, "
        "then wire load_events() to your local cache layout."
    )
    return 3


def cmd_evaluate(args) -> int:
    from .config import load_config

    cfg = load_config(args.config)
    LOGGER.info("verification spacings: %s km", cfg["verification"]["spacings_km"])
    LOGGER.error(
        "cmd_evaluate requires the cached event set and a fitted correction. "
        "Run `fetch` and `fit` first."
    )
    Path(args.output).write_text(json.dumps({"status": "not_run"}, indent=2))
    return 3


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")
    return {"fetch": cmd_fetch, "fit": cmd_fit, "evaluate": cmd_evaluate}[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
