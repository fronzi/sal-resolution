"""Configuration loading with defensive numeric coercion.

PyYAML leaves ``3.0e-3`` as a *string* unless the exponent has an explicit
sign, so any downstream arithmetic fails far from the cause.  Rather than rely
on authoring discipline in ``config.yml``, every scalar that parses as a float
is coerced here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["coerce_floats", "load_config"]


def coerce_floats(obj: Any) -> Any:
    """Recursively convert float-like strings to floats."""
    if isinstance(obj, dict):
        return {k: coerce_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_floats(v) for v in obj]
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return obj
    return obj


def load_config(path: str | Path = "config.yml") -> dict:
    """Load and normalise the experiment configuration."""
    import yaml

    with open(Path(path).expanduser()) as fh:
        cfg = yaml.safe_load(fh)
    cfg = coerce_floats(cfg)
    if "data" in cfg and "cache_dir" in cfg["data"]:
        cfg["data"]["cache_dir"] = str(Path(cfg["data"]["cache_dir"]).expanduser())
    return cfg
