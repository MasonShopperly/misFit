# The coupled aeroelastic wing — application result

The submission's primary application. Two served Tesseracts hold the two halves of a static
aeroelastic problem; a cross-component check qualifies the gradient providers once; the verified
coupled gradient then tailors the wing.

Every number below comes from a committed record except where the table marks it otherwise, and is
recomputed by the scripts named at the foot of this report.

## Model

Prandtl lifting line in Glauert's Fourier form, coupled to a linear finite-element cantilever
torsion beam. With `y = -(b/2) cos t` and `Γ = 2bV Σ Aₙ sin(nt)`,

```
Σ Aₙ sin(n t) [ sin(t) + n μ ] = μ sin(t) α_tot(t),      μ = a₀c/(4b)
K(s) θ = Pᵀ (w ⊙ τ),      τ = L'(y) · e · c
```

The two meshes deliberately differ — 40 cosine aerodynamic stations against 24 uniform torsion
elements — because in real aeroelasticity they never match. Transfer is conservative in both
directions, so virtual work is preserved rather than only one direction being consistent.

Reference wing: span 16 m, chord 1 m (AR 16), `GJ` = 2.0×10⁵ N·m², elastic axis 0.15c behind the
aerodynamic centre, sea level, 80 m/s, trimmed to `C_L` = 0.50.

**Design variables**: three geometric-twist control points and three log-stiffness control points.
**Objective**: induced drag plus a structural-mass proxy, at fixed trimmed lift.

The two components and what each one cannot see:
[`../aeroelastic/README.md`](../aeroelastic/README.md).

## Validation, against closed forms

| check | target | result |
|---|---|---|
| elliptic wing lift slope | `C_L = a₀α/(1 + a₀/πAR)` | 1.4e-16 relative |
| elliptic wing induced drag | `C_Di = C_L²/(πAR)` | 0.0 relative in float64 |
| cantilever torsion, tip rotation | `θ = t₀L²/(2GJ)` | 1.3e-14 relative (console; not persisted) |
| **strip-theory divergence** | `q_D = π²GJ/(4a₀ec²L²)` | **1.1e-06 relative** |
| **Glauert's rectangular δ at AR 8** | 0.0676, published | **0.0676, nothing fitted** |
| lifting line vs strip theory | should raise `q_D` | 1.291×, correct direction |

Two of these are independent corroboration rather than self-consistency: Glauert's published
monoplane table is reproduced without anything being fitted to it, and the finite-span correction
moves divergence in the direction the literature requires.

## The composition, measured

| | |
|---|---|
| partitioned fixed point vs monolithic coupled solve | **1.6e-12** relative, worst case |
| coupled adjoint across HTTP vs monolithic gradient | **9.0e-11** relative, worst case |
| observed contraction factor of the fixed point | **q/q_D**, to four digits |

Record: [`../aeroelastic/records/served_verification.json`](../aeroelastic/records/served_verification.json),
written by [`../aeroelastic/verification/run_served.py`](../aeroelastic/verification/run_served.py).

## Gradient qualification

Each component passes the platform's own gradient check on its own endpoints: `aero.A1` and
`struct.S1` together return **0 failures over 14,994 checks**
([`platform_checker_ae.json`](../aeroelastic/records/platform_checker_ae.json)). Passing alone says
nothing about the pairing, because neither component contains the assembled objective.

The pairing is therefore qualified on the assembled objective, by pointing the same
`tesseract-runtime check-gradients` at the composed `coupled.C1` Tesseract. The matched pairings pass
at all three operating points; the crossed pairing — coupled objective, rigid gradient — is rejected
on **462 of 540 sampled checks**. Verdict table, cost and coverage:
[`native_composed_check.md`](native_composed_check.md).

**Driven by the aerodynamic component's own gradient, the same served pipeline reaches a different
wing.** Its own rigid model scores that wing **0.9345** — effectively a tie with the coupled result.
The coupled model scores it **1.0851: 16.2 % worse**, carrying **19.9 % more induced drag**
at the same trimmed lift and the same structure. Both arms end at the stiffness bound with an
identical mass proxy, so the comparison isolates the aerodynamic consequence alone.

Along the straight path between the two designs the aerodynamic-only gradient fails the descent test at every one of 41 sampled points, with a worst cosine of **−0.997** against the coupled gradient. "Uphill" here means the measured-to-claimed slope ratio falls below the same η₁ = 0.01 the diagnostic uses, not merely below zero. The slope ratio
reaches −24.4 at the left endpoint, where that gradient sits at its own optimum and the ratio is
ill-conditioned.

The pairing is qualified once, before the optimizer starts, rather than re-checked at each
iteration. Nothing here measures whether re-checking would be worth its cost.

## Optimization result

L-BFGS-B with box bounds, every objective and gradient crossing the service boundary.

| | |
|---|---|
| objective | 1.3500 → **0.9336**, **30.8 % better** |
| iterations | **30** |
| trimmed lift | `C_L` = **0.5000**, held fixed |
| induced drag | `C_Di` = 0.004987 |
| built-in tip washout | **5.7°** |
| divergence margin | q/q_D = 0.747 |
| structural mass proxy | 0.4966, at the lower stiffness bound |
| wall time | 79.4 s |

| service | `apply` | `vector_jacobian_product` |
|---|---|---|
| `aero.A1` | **12,225** | **2,227** |
| `struct.S1` | **12,225** | **2,227** |

![Three stacked panels comparing two wings, red for the aerodynamic-gradient-only design and blue
for the coupled one, both trimmed to C_L = 0.50 at 80 m/s. Top: lift per span against spanwise
station, headed "same lift, same structure — and 20 % more induced drag, 0.00598 against 0.00499".
Middle: twist in degrees from root to tip, dashed as built and solid in flight, both designs at
q/q_D 0.75. Bottom: the measured-to-claimed slope ratio along the straight path from the
aerodynamic-only design to the coupled one, with all 41 sampled points inside a red band below the
0.01 threshold, annotated "no step size helps — the gradient itself must change", and clipped at
the left endpoint where the ratio reaches
−24.4.](../docs/figures/aeroelastic_wings.png)

## Limitations

Model scope, the `n` = 1 caveat, the divergence margin the optimizer spends, and the bound on what
qualification establishes are stated in full in
[`../technical_writeup.md` §8](../technical_writeup.md#8-limitations). One limitation is specific to
this report:

- **The reproduction path serves process-level.** `aero.A1` and `struct.S1` both build with
  `tesseract build` and serve as containers with the port published, returning values matching the
  source they were built from at exactly zero relative error over six checks. Every path in this
  repository nevertheless serves them process-level through `tesseract-runtime`, so the container
  route is not the one these numbers came from.

## Reproduce

```bash
.venv/bin/python scripts/aeroelastic_demo.py           # replay, under a second
.venv/bin/python scripts/aeroelastic_demo.py --live    # serve, qualify, optimize
.venv/bin/python aeroelastic/verification/validate.py               # the closed-form checks above
.venv/bin/python aeroelastic/verification/verify_adjoint.py         # the hand adjoint, three ways
.venv/bin/python aeroelastic/verification/run_native_composed_check.py  # the composed-objective qualification
.venv/bin/python aeroelastic/studies/qualification.py          # the four predeclared heuristics
.venv/bin/python aeroelastic/verification/run_served.py             # the served forward and adjoint agreements
.venv/bin/python aeroelastic/studies/run_served_optimize.py    # the two arms; rewrites aeroelastic/records/served_optimize.json
.venv/bin/python scripts/make_wing_figure.py       # the 41-point path scan and the figure above
```

Records behind this report:
[`served_optimize.json`](../aeroelastic/records/served_optimize.json) (the two arms and the optimization),
[`validation.json`](../aeroelastic/records/validation.json) (the closed forms),
[`served_verification.json`](../aeroelastic/records/served_verification.json) (the served agreements),
[`wing_figure.json`](../aeroelastic/records/wing_figure.json) (the path scan).

## Evidence

[`aeroelastic_qualification.md`](aeroelastic_qualification.md) ·
[`validity_regime.md`](validity_regime.md) ·
[`native_composed_check.md`](native_composed_check.md) ·
[sources](../technical_writeup.md#sources)
