"""Heavy-rain event selection.

Selection is deliberately observation-based, never model-based: choosing events
where a *model* forecast heavy rain conditions the sample on that model being
right, which biases the comparison toward whichever arm was used to select.

Criteria (all evaluated on the observed field, on the verification grid):

* domain-mean accumulation above ``min_domain_mean``;
* at least ``min_wet_fraction`` of valid cells above ``wet_threshold``, so the
  domain contains an actual object rather than one isolated cell;
* at least ``min_objects`` connected objects above ``R* = R_max/15``, since SAL
  ``S`` and ``L2`` are degenerate for a single-object field;
* observation coverage above ``min_coverage`` after masking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = ["EventCriteria", "screen_event", "select_events"]


@dataclass(frozen=True)
class EventCriteria:
    min_domain_mean: float = 2.0      # mm over the accumulation window
    wet_threshold: float = 1.0        # mm
    min_wet_fraction: float = 0.02
    min_objects: int = 1
    min_coverage: float = 0.9
    threshold_factor: float = 1.0 / 15.0


def screen_event(obs: np.ndarray, criteria: EventCriteria = EventCriteria()) -> tuple[bool, dict]:
    """Return ``(passed, diagnostics)`` for one observed accumulation field."""
    arr = np.asarray(obs, float)
    valid = np.isfinite(arr)
    coverage = float(valid.mean()) if arr.size else 0.0
    if coverage < criteria.min_coverage:
        return False, {"reason": "coverage", "coverage": coverage}

    vals = arr[valid]
    domain_mean = float(vals.mean())
    wet_fraction = float((vals >= criteria.wet_threshold).mean())
    rmax = float(vals.max()) if vals.size else 0.0
    thresh = criteria.threshold_factor * rmax
    _labels, n_obj = ndimage.label(valid & (arr >= thresh) & (thresh > 0))

    diag = {
        "coverage": coverage,
        "domain_mean": domain_mean,
        "wet_fraction": wet_fraction,
        "rmax": rmax,
        "n_objects": int(n_obj),
    }
    if domain_mean < criteria.min_domain_mean:
        return False, {**diag, "reason": "domain_mean"}
    if wet_fraction < criteria.min_wet_fraction:
        return False, {**diag, "reason": "wet_fraction"}
    if n_obj < criteria.min_objects:
        return False, {**diag, "reason": "n_objects"}
    return True, {**diag, "reason": "pass"}


def select_events(obs_fields, criteria: EventCriteria = EventCriteria()):
    """Filter a sequence of observed fields, returning ``(indices, diagnostics)``.

    Keep the diagnostics: the count and reason breakdown of rejected candidates
    belongs in the paper's methods section, and a sample of 30 events drawn
    from 400 candidates needs its selection rate stated.
    """
    keep, diags = [], []
    for i, field in enumerate(obs_fields):
        ok, d = screen_event(field, criteria)
        diags.append({"index": i, **d})
        if ok:
            keep.append(i)
    return np.array(keep, dtype=int), diags
