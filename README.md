# misFit

Measuring where a rigid-wing aerodynamic gradient stops being a useful basis for designing a
flexible wing.

<!-- orient: why -->
A wing twists under the air holding the aircraft up, and the twist changes the lift, so optimising a
wing from aerodynamics alone means differentiating a shape that does not deform. That gradient is
not wrong. It is the exact derivative of the rigid-wing objective, it is a sound approximation of
the flexible one when the wing is stiff or the air is thin, and it costs a fraction of running an
aerodynamic and a structural solver to convergence against each other. The practical question is how
far into the flexible problem it can be carried before it starts doing damage.

misFit maps that boundary on one wing. It sweeps 289 designs, computes both the rigid and the
coupled gradient at each, and records where a step along the rigid one stops paying. Of those 289
designs, 31 fail. Four physically motivated warning signs were predeclared, with their thresholds
fixed before scoring, and none of them locates the 31. The paired optimization driven by the rigid
gradient converges in the same part of the design space, ending on a wing 16.2 % worse on the
coupled objective while its own model reports success.

The measurement needs a trustworthy coupled gradient to measure against. That comes from two solvers
written independently and differentiated by completely different means, JAX autodiff on the
aerodynamics and a hand-derived adjoint on the structure. Tesseract gives both a common
value-and-derivative service interface, so the coupled solve and the coupled adjoint can be
assembled from their calls without rewriting either solver into the other's differentiation stack.

[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![track](https://img.shields.io/badge/Track%2002-multi--physics%20%C2%B7%20coupled%20systems-blue.svg)](results/aeroelastic_wing.md)
[![result](https://img.shields.io/badge/boundary-inverts%20only%20when%20coupling%20and%20optimum%20combine-blue.svg)](results/validity_regime.md)

[Problem](#problem) · [Result](#result) · [Why Tesseract](#why-tesseract) ·
[Implementation](#implementation) · [Reproduce](#reproduce) · [Limitations](#limitations)

[Demo video — 4:05](docs/misfit-demo.mp4), every number drawn from the committed
records.

![Three panels under the heading "At the design condition, the air simulator alone costs 20 % more
induced drag". Top left: wing twist from root to tip, with the built-in shape falling to about minus
six degrees at the tip and the in-flight shape held near minus one, the air twisting the wing back.
Top right: lift per metre of span for two designs carrying identical total lift, the
aerodynamic-only design carrying its load further outboard. Bottom: a panel headed "Both models pass
alone. The crossed objective–gradient pairing fails.", listing verdicts from the platform's own
check-gradients tool.](docs/figures/aeroelastic_hero.png)

## Problem

The wing is trimmed to a fixed lift coefficient and the objective is induced drag plus a structural
mass proxy. Six design variables set spanwise twist and torsional stiffness.

Two gradients of that objective are available. The **coupled** gradient accounts for the fact that
changing the design changes the elastic twist, which changes the loads, which changes the twist
again. Getting it means solving the aerodynamic and structural problems against each other to a
fixed point, then transposing that loop. The **rigid** gradient omits the elastic response entirely.
It is exact in the `q → 0` limit and needs only the aerodynamic solver.

A design is counted against the rigid gradient when a step along it returns less than 1 % of the
first-order decrease that gradient predicts. That is a step-usefulness criterion, not a cosine
threshold, and it is what every comparator below is scored against.

## Result <!-- orient: outputs -->

![Two line charts of the cosine between the aerodynamic-only gradient and the coupled gradient.
Left, swept against dynamic pressure as a fraction of divergence pressure: at the optimized design
the line stays near one, then plunges through zero at the design condition and settles near minus
one, while at a generic design it barely moves and never crosses zero. Right, holding the flight
condition fixed and walking the design toward the optimum: at the design condition the line crosses
zero halfway along, while at a fifth of the divergence pressure it dips and
recovers.](docs/figures/validity_regime.png)

Sweeping 289 designs of tip twist against torsional stiffness, 31 of them (10.7 %) fail the step
criterion. In 30 of those 31 the step increases the objective, so the wing gets worse.

A natural hypothesis is that the approximation degrades as dynamic pressure `q` approaches the
wing's divergence pressure `q_D`. Two controls say otherwise. Sweeping the flight condition at a
design of the same stiffness that no optimizer chose, the two gradients agree to a cosine of 0.938
at the design condition and never fall below 0.922 anywhere up to 99 % of divergence pressure, with
no sign change and no step that ever increases the coupled objective. Coupling alone therefore does
not do it. Holding the flight condition at a fifth of divergence pressure and walking the design in
to the coupled optimum, the cosine dips to 0.67 and recovers without changing sign, so proximity to
the optimum alone does not do it either.

The inversion needs both. At the design condition the same walk crosses zero halfway along and
reaches −0.929, with the flight condition held fixed the whole way, and the rigid gradient is then
pointing uphill on the objective it is being used to minimise.

One classical result explains the interaction. Writing `e` for the error in the cheap gradient, if
`‖e‖ / ‖g_coupled‖ < 1` then that gradient is still a descent direction, which is Carter's
relative-gradient-error condition (*SIAM J. Numer. Anal.* 28, 1991). Coupling grows the numerator
and standing near the coupled optimum shrinks the denominator, so neither factor alone pushes the
ratio past one. The record bears this out: the ratio peaks at 0.75 at the generic design and never
loses descent, and passes one at `q/q_D` = 0.49 at the optimized design.

Four comparators were predeclared and scored against all 289 designs: dynamic-pressure ratio at
0.80, maximum elastic twist at 5°, twist-cancellation ratio at 1.0, and load redistribution at 0.25.
Each threshold has a stated physical motivation rather than a citation behind the number — a
near-divergence trigger leaving 20 % dynamic-pressure margin, a small-angle deformation trigger, a
25 % normalized load-redistribution trigger, and the mechanism hypothesis of this study, elastic
twist reaching geometric twist scale. None was adjusted after scoring.

None of the four discriminates. The best fires on 9 of the 31 bad designs and raises 42 false
alarms, reaching a Matthews correlation of 0.1035 against a 10.7 % base rate. The comparator taken
from this study's own account of the mechanism scored worst. All four read operating-point state,
while the governing ratio has a design-space quantity in its denominator that operating-point state
cannot observe — which accounts for these four scores without ruling out a better indicator.

Two L-BFGS-B runs were made from a common start, one driven by each gradient, both to convergence.

| | coupled gradient | aerodynamic-only gradient |
|---|---|---|
| objective, scored by both solvers together | **0.9336** | **1.0851**, 16.2 % worse |
| objective, scored by its own model | 0.9336 | 0.9345, effectively a tie |
| induced drag `C_Di` | **0.004987** | 0.005981, 19.9 % more |
| trimmed lift `C_L` | 0.5000 | 0.5000 |
| structural mass proxy | 0.4966 | 0.4966 |

Both arms end at the same stiffness bound and the same lift, so the difference is aerodynamic. The
approximate run converges normally and reports success; the failure is
invisible from inside the component that caused it.

The result is bounded to this flight regime. Re-scoring both designs across speed without
re-optimizing, the ranking reverses at about 57 m/s, below which the aerodynamic-only design is the
better of the two, by 9.6 % at 30 m/s.

![An animation of one wing becoming another. It begins red, designed by the air simulator alone at
an induced drag of 0.00598, and becomes blue, designed by both simulators together at 0.00499, with
the load moving inboard and the built-in twist deepening at the same trimmed lift and the same
structure. Every frame is a real coupled solve.](docs/figures/aeroelastic_morph.gif)

## Why Tesseract

The two solvers were written independently and get their derivatives by incompatible means. The
aerodynamics is JAX and differentiates by autodiff. The structure is a separate NumPy and SciPy
solver whose JVP and VJP were derived by hand. Neither imports the other, neither shares a process,
and there is no shared autodiff tape. Making a pair like that into one differentiable workflow
normally means porting one solver into the other's differentiation stack, or writing the
interoperability layer yourself.

Tesseract is that layer. Both components expose the same apply/JVP/VJP-style contract, and that
contract is enough to assemble the coupled forward solve and the transposed coupled adjoint from
their calls, while each solver keeps its own internals and its own way of differentiating. The
aeroelastic mathematics does not require any of this — it could be one monolithic program in a
single framework. The heterogeneous workflow is what would not survive the port.

The assembled objective goes behind the same interface. `coupled.C1` exposes the trimmed coupled
objective and its total derivative as an ordinary Tesseract, so `tesseract-runtime check-gradients`
can test that objective against the gradient offered for it. Testing the pairing at the system level
needed no numerical checking code of its own.

The contribution here is the application and the measured validity boundary, not a new aeroelastic
coupling or differentiation method.

## Implementation <!-- orient: structure -->

| Tesseract | role | differentiation |
|---|---|---|
| `aero.A1` | lifting-line aerodynamics in Glauert's Fourier form | JAX autodiff |
| `struct.S1` | cantilever torsion beam, 24 elements | hand-derived adjoint, NumPy and SciPy |
| `coupled.C1` | the assembled trimmed objective and its total derivative | none of its own; delegates to A1 and S1 |

![Design variables feed the aero.A1 Tesseract, which exchanges loads and twist both ways with the
struct.S1 Tesseract. The pair feeds a coupled objective and then the optimizer, which proposes the
next design. Off to the side in red dashes, the assembled objective is exposed as a third Tesseract,
coupled.C1, and checked once by the platform's own check-gradients before the optimizer
starts.](docs/figures/architecture.png)

The forward solve is a partitioned fixed point in the elastic twist: incidence to `aero.A1` gives
torque, torque to `struct.S1` gives twist, the twist changes the incidence, repeat until the two
agree. It reproduces a monolithic coupled solve to 1.6e-12 relative, worst case. The backward pass
is the same loop transposed, crossing the service boundary twice per iteration, and agrees with the
monolithic gradient to 9.0e-11.

The two meshes deliberately differ, 40 cosine-spaced aerodynamic stations against 24 uniform torsion
elements, and load and displacement transfer is conservative in both directions. `struct.S1`'s
hand-written JVP and VJP satisfy the dot-product identity to 1.9e-14.

The cost of the composition is set by the physics. The fixed point's observed contraction factor
tracks `q/q_D` to four digits, so the number of round trips between the two services depends on how
close the wing is to divergence: 12, 30 and 57 round trips at `q/q_D` = 0.099, 0.397 and 0.621.
These designs sit at 0.747, above that measured range. The designs where the cheap gradient fails
are also the ones where the accurate gradient costs most.

`coupled.C1` exists so that the pairing of objective and gradient can be checked at the system
level with the platform's own tool. Run against each component on its own, `tesseract-runtime
check-gradients` passes everything, 0 failures in 14,994 checks, because each component's derivative
is a correct derivative of that component. Run against the assembled objective it separates the
pairings: coupled objective with coupled gradient passes, rigid objective with rigid gradient
passes, and the coupled objective paired with the rigid gradient fails 462 of 540 checks. The
matched rows are the control, measured at the same points with the same tolerance and the same
sampling. Details, including what those resample counts do and do not mean, are in
[`results/native_composed_check.md`](results/native_composed_check.md).

## Reproduce <!-- orient: use verify -->

Python 3.11 on CPU, no Docker, no GPU. The direct Python dependencies are pinned in
`requirements.txt` to the versions used for the committed runs.

```bash
git clone https://github.com/MasonShopperly/misFit && cd misFit

# needs uv:  curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
```

```bash
# replay the committed coupled run and print the headline tables; takes under a second
.venv/bin/python scripts/aeroelastic_demo.py

# the 289-design sweep behind the 31 and the 30; prints the map, writes only with --out
.venv/bin/python aeroelastic/studies/validity_map.py

# the boundary and its two controls, in process, a few minutes
.venv/bin/python aeroelastic/studies/validity_regime.py

# the four predeclared comparators against 289 designs
.venv/bin/python aeroelastic/studies/qualification.py

# serve both components, qualify the pairing, re-optimize, and compare against the record
.venv/bin/python scripts/aeroelastic_demo.py --live
```

The replay is the one to run first. It reads committed JSON, labels every section with where its
numbers came from, and binds no ports.

`--live` starts both Tesseracts on ports 8811 and 8812, which must be free, and takes about eighteen
minutes on sixteen cores. `coupled.C1` binds nothing; it is loaded in process through
`TESSERACT_API_PATH`, the same way the platform's checker loads any component, and reaches the other
two over those ports. It writes to scratch files, so the clone is left unmodified.

It then compares what it just computed against the committed records field by field, reporting each
as `EXACT MATCH`, `NUMERICALLY CONSISTENT` or `FAILED`, and only `FAILED` exits non-zero. The middle
category exists because float64 BLAS does not fix its reduction order across runs.
`aeroelastic_demo.py --self-test` checks those comparison semantics against planted cases in about
a second.

Full reproduction route — the tolerance semantics, which figures re-render byte-identically, and
exactly what changes in your working tree when you run any of this:
[`docs/reproduce.md`](docs/reproduce.md).

## Limitations <!-- orient: limits -->

- **Reduced-order model.** Lifting line plus a torsion beam, not Navier–Stokes and not a full
  aircraft. Invalid at low aspect ratio, with sweep, in compressible flow, or near stall.
- **Static aeroelasticity only.** There is no mass axis, so no flutter. On an unswept wing flutter
  can precede divergence, and this model is silent about that.
- **Synthetic benchmark.** A defensible reference case rather than a measured aircraft, sized by a
  single 1 g cruise objective rather than by the manoeuvre, gust and strength cases that set real
  wing stiffness.
- **n = 1.** One wing, one objective, six design variables. The validity map is a measurement of one
  two-parameter family, not a law about coupled systems, and the 16.2 % comes from a single paired
  L-BFGS-B trajectory from a common start rather than from a population.
- **The optimizer spends divergence margin.** Both arms drive torsional stiffness to its lower
  bound, ending at `q/q_D` = 0.747 against 0.371 at the starting design. A box bound stops them, not
  a stability constraint.
- **Qualification is local, and consistency is not adequacy.** The check asks whether a gradient is
  a derivative of an objective at the points tested, under a tolerance the caller chooses. It cannot
  tell you the objective is the right one; two too-coarse models can be perfectly consistent with
  each other and both wrong.
- **No novelty is claimed for the solver or the check.** OpenAeroStruct and MACH-Aero already do
  coupled aerostructural optimisation with analytic coupled adjoints, and the qualification here is
  the platform's own tool run against a composition. What is new is the measurement: this wing's
  validity boundary, the two controls that isolate it, and the failure of the four predeclared
  heuristics.

## More detail

| | |
|---|---|
| [`technical_writeup.md`](technical_writeup.md) | the submission document: model, mathematics, experiment design, full results, limitations and sources |
| [`results/`](results/README.md) | the four evidence reports, one per result, each stating what it does not show |
| [`aeroelastic/`](aeroelastic/README.md) | the wing, in five directories: [`core/`](aeroelastic/core/README.md) the monolithic reference model, [`components/`](aeroelastic/components/README.md) the three Tesseracts, [`studies/`](aeroelastic/studies/README.md) the experiments, [`verification/`](aeroelastic/verification/README.md) the checks, [`records/`](aeroelastic/records/README.md) the committed evidence |
| [`scripts/`](scripts/README.md) | the demo and the two figure generators — what you run to see something |
| [`docs/`](docs/README.md) | the [reproduction route](docs/reproduce.md), the [figures](docs/figures/README.md) and the demo video |
| [`requirements.txt`](requirements.txt) | the pinned direct dependencies |

Every directory carries a README indexing its own contents, and every number in the reports is
recomputed from a committed record rather than retyped, except the few marked *console; not
persisted*.

Apache-2.0 · [Tesseract Hackathon 2026](https://pasteurlabs.ai/tesseract-hackathon-2026/)
(Pasteur Labs) · Track 02, Multi-physics and coupled systems
