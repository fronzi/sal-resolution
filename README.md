# sal-resolution

Does a closed-form symbolic correction recover the spatial-structure skill that
convection-permitting dynamics buys?

Three arms are verified against radar QPE on a **common** verification grid:

| Arm | Product | Spacing |
|---|---|---|
| `coarse` | BARRA-R2 | 12 km |
| `fine` | BARRA-C2 (convection-permitting) | 4.4 km |
| `symbolic` | `coarse` + PySR-learned correction | 12 km native |
| truth | Rainfields3 radar QPE | ~1 km |

Headline statistic — the fraction of the dynamical gain recovered:

$$\rho_X = \frac{|X_{\text{coarse}}| - |X_{\text{sym}}|}{|X_{\text{coarse}}| - |X_{\text{fine}}|},\qquad X \in \{S,A,L\}$$

`rho_S` is the one that speaks to resolution: `S` is the only SAL component
that interrogates object size and shape.

## The confound this package exists to eliminate

**SAL is not resolution-invariant.** Object identification depends on grid
spacing and on the threshold `R* = R_max/15`, so computing SAL for a 12 km and
a 4.4 km field on their native grids measures grid spacing, not model quality.
Every arm is conservatively remapped to a common verification grid first, and
the sweep over that grid's spacing (5–100 km) *is* the spatial-resolution
result.

`ConservativeRegridder` implements exact spherical area-overlap remapping by
exploiting the separability of `dA = a² dλ d(sinφ)` — two dense matmuls, no
ESMF/xESMF dependency, conserving mass to machine precision.

**Self-test:** `A` is a ratio of domain means, so under mass-conserving
remapping it is invariant in the grid spacing. `amplitude_invariance_check`
enforces this. If it drifts, a bilinear interpolation has crept into the chain
— and the same bug would corrupt `S` far less visibly.

## Install

```bash
conda activate weather_ml_failure
pip install -e ".[dev,symbolic]"
python -c "import pysr; pysr.install()"   # provisions Julia, once
```

`scores` (the `sal-metric` branch) is optional. Without it, the package falls
back to the bundled `reference_sal`, a self-contained Wernli (2008)
implementation that also serves as a third cross-check alongside `pysteps`
before the upstream PR to `nci/scores`.

## Run

```bash
sal-resolution fetch    --config config.yml --dry-run   # inspect URLs first
sal-resolution fetch    --config config.yml             # one bulk download
sal-resolution fit      --config config.yml
sal-resolution evaluate --config config.yml
```

Latency from Sydney to remote object stores is dominated by per-chunk
round-trips, not bandwidth, so `fetch` is a **single bulk download** and
everything after it is offline. ~30 events × 4 arms at 500×500 float32 is
~120 MB — laptop-scale.

## See it work before downloading anything

```bash
python examples/synthetic_walkthrough.py
```

Runs the full chain on synthetic blob fields: event screening → conservative
remap → SAL → baselines → `rho_S` with bootstrap CIs → delta sweep → the
amplitude-invariance self-test (which reports drift ~4e-16).

Two things in that output are worth internalising before you look at real data:

- `scalar_bias` returns `rho_S = 0.000` *exactly*. It must — `S` is computed
  from volume-normalised objects and is therefore invariant under a uniform
  multiplicative factor. If a real run ever shows non-zero `rho_S` for this
  baseline, the object normalisation is broken.
- `quantile_map` improves `S` but drives `A` from ~0 to ~0.88. Fitting the
  transfer function on wet points only, plus tail extrapolation, inflates the
  domain mean. Quantile mapping buys structure by paying in amplitude — report
  all three components or you will mistake that trade for a gain.

## Design decisions worth knowing

**The correction is multiplicative**, `R_sym = R_c · exp(g(z))`, preserving
non-negativity and matching the roughly log-normal rain distribution. Features
`z` are computable from the coarse field plus static orography alone — nothing
requires the fine model at inference time.

**The fitting loss is not MSE.** An MSE-optimal field is a blurred field, so
plain MSE drives `g` toward a smoothing operator: RMSE improves while `S` gets
*worse*. That is the double-penalty trap SAL exists to expose. Fitting uses a
patch-variance-weighted loss, and `select_by_structure` re-ranks the PySR
Pareto front by the true 2-D local-variance criterion.

**SAL never enters the fitting objective.** It is non-local and
non-differentiable, and using it for both fitting and evaluation would make the
headline result circular. It is the held-out evaluation instrument only.

**Events are selected on observations, never on model output.** Selecting on a
model conditions the sample on that model being right.

**Cross-validation is leave-one-year-out.** Rain fields are strongly
autocorrelated in space and time; random splits leak the test signal.

## Baselines the symbolic expression must beat

| Baseline | What it tests |
|---|---|
| `identity` | the floor |
| `scalar_bias` | how much apparent gain is just bias removal |
| `quantile_map` | the real bar for `A` |
| `unsharp` | the real bar for `S` — a one-parameter sharpening filter |

Plus a permutation control: shuffle the features, refit, confirm `rho_S → 0`.
The unsharp-mask parameters are selected on log patch variance, not MSE —
selecting on MSE would always return zero sharpening and make it a strawman.

## Known limitations

- Rainfields3 needs NCI project `rq0`. AGCD is the public fallback but is
  gauge-interpolated and spatially smooth, which systematically flatters the
  coarse model's `S` and lowers the ceiling on any correction. State this if
  you use it.
- The regridder handles rectilinear lat/lon only, with no longitude wrapping.
  Fine for regional Australian domains.
- SAL is not a proper score, and `SALResult.distance` weights three quantities
  with different natural ranges. Always report components separately.
- `rho` is undefined where dynamical downscaling itself gained nothing; those
  events are returned as NaN and the count must be reported.
- Per-event SAL is noisy — aim for n ≥ 30 events before quoting bootstrap
  intervals, resampled at the **event** level (point-level resampling
  understates intervals badly).
- BARRA-C2 CORDEX facet values in `build_url` should be confirmed against the
  live NCI catalogue before a production download.

## Tests

```bash
pytest -q
pytest -q --cov=sal_resolution --cov-report=term-missing
```

The differential operators are checked against analytic solutions on the
sphere (`f = sin φ ⟹ ∇²f = −2 sin φ / a²`). This is not ceremony: a 180/π vs
π/180 slip in a horizontal gradient is a silent ~3283× error that no smoke test
would catch.
