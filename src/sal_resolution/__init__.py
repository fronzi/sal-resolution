"""Symbolic vs dynamical downscaling, judged by SAL.

Three arms are compared against radar QPE on a *common* verification grid:

===============  ==========================================================
``coarse``       BARRA-R2, 12 km
``fine``         BARRA-C2, 4.4 km -- convection-permitting, the expensive
                 control
``symbolic``     ``coarse`` plus a PySR-learned closed-form correction
===============  ==========================================================

The headline statistic is :func:`~sal_resolution.verify.compare.recovered_gain`,
the fraction of the dynamical gain in SAL ``S`` that the closed form recovers.
"""

from .features import FEATURE_NAMES, build_features
from .regrid import ConservativeRegridder, make_verification_grid
from .symbolic.baselines import BASELINES
from .symbolic.fit import SymbolicConfig, symbolic_available
from .verify.compare import bootstrap_ci, delta_sweep, evaluate_arms, recovered_gain
from .verify.sal import SALResult, compute_sal, reference_sal

__version__ = "0.1.0"

__all__ = [
    "BASELINES",
    "FEATURE_NAMES",
    "ConservativeRegridder",
    "SALResult",
    "SymbolicConfig",
    "__version__",
    "bootstrap_ci",
    "build_features",
    "compute_sal",
    "delta_sweep",
    "evaluate_arms",
    "make_verification_grid",
    "recovered_gain",
    "reference_sal",
    "symbolic_available",
]
