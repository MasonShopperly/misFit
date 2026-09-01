# misFit

Measuring where a rigid-wing aerodynamic gradient stops being a useful basis for designing a
flexible wing.

Tesseract Hackathon 2026, Track 02 (multi-physics and coupled systems) · Apache-2.0 ·
[repository](README.md) · [reproduction route](docs/reproduce.md)

A wing deforms under the loads it carries, and the deformation changes those loads. Optimising such
a wing from aerodynamics alone means differentiating a shape that does not deform. That gradient is
the exact derivative of the rigid-wing objective and costs a fraction of converging two solvers
against each other. How far it can be carried into the flexible problem is what this study measures.

This study measures that on one reduced-order wing. Across 289 sampled designs, a step along the
rigid gradient returns less than 1 % of its own predicted first-order decrease on 31 of them, and on
30 of those the coupled objective increases instead. Neither high dynamic pressure alone nor
proximity to the coupled optimum alone produces that inversion; the two together do. Four warning
signs, predeclared with their thresholds fixed before scoring, were scored against the map and none
locates the region. A paired optimization driven by the rigid gradient converges normally and
reports success against its own model, ending on a wing 16.2 % worse on the coupled objective. The
coupled reference it is measured against comes from two solvers written independently and
differentiated by different means, composed across a Tesseract service boundary.

---

## 1. Problem and model

The design problem is preliminary-design scale. Three control points set spanwise geometric twist
and three set spanwise log torsional stiffness, six variables in all. The objective is induced drag
plus a structural mass proxy, each normalised by its value at the reference design and combined with
a fixed weight of 0.35. The aircraft is trimmed to `C_L` = 0.50 at 80 m/s rather than penalised for
missing it, which removes one hand-chosen constant from the objective. Geometric twist is bounded to
±0.10 rad and log stiffness to [−0.7, 1.0].

The wing is rectangular, 16 m span, 1 m chord, with the aerodynamic centre 0.15 chords ahead of the
elastic axis so that positive lift twists the section nose-up. Aerodynamics is Prandtl lifting line
in Glauert's Fourier form at 40 cosine-spaced stations per semispan. Structure is a linear
finite-element cantilever torsion beam at 24 uniform elements per semispan. The two discretisations
deliberately differ, as they do in practice, and load and displacement transfer is conservative in
both directions.

The coupling is the whole content of the model:

```
u = alpha + twist(t) + theta,    tau = A1(u),    theta = S1(tau, s)
```

The incidence `u` that the aerodynamics sees depends on the elastic twist `theta`, which depends on
the torque `tau`, which depends on `u`. Neither solver can produce the flying shape alone.

This is a reduced-order, static-aeroelastic, synthetic case: not Navier–Stokes, not a full aircraft,
and not validated against experiment. It is validated against closed-form theory. Elliptic
lifting-line lift reproduces to 1.4e-16 relative; Glauert's rectangular-wing induced-drag factor
reproduces across aspect ratios 4 to 10 with nothing fitted, 0.0676 at aspect ratio 8; strip-theory
divergence reproduces to 1.1e-06 once the induced angle is removed, which is the reduction the
closed form assumes, and restoring it raises the coupled divergence pressure by a factor of 1.291
as finite-span theory requires ([validation tables](results/aeroelastic_wing.md)).

Two gradients of the objective are available. The **coupled** gradient accounts for the design
changing the elastic twist, which changes the loads, which changes the twist again. The **rigid**
gradient holds the elastic twist at zero and omits its derivative entirely. It needs only the
aerodynamic solver and is exact in the `q → 0` limit.

## 2. The coupled derivative

The forward solve is a partitioned fixed point in the elastic twist: incidence to `aero.A1` gives
torque, torque to `struct.S1` gives twist, the twist changes the incidence, repeat until the two
agree. Against a monolithic coupled solve of the same equations it agrees to 1.6e-12 relative in the
worst case.

The reverse pass is that loop transposed. The total derivative requires `(I − L)⁻¹` for the loop
gain `L`, which in reverse mode is itself a fixed point, this time in the adjoint of the incidence:

```
a      = A1.vjp(cl, cdi)["alpha_tot"]
u_bar <- a + A1.vjp(torque = S1.vjp(theta_aero = u_bar)["torque"])["alpha_tot"]
```

Each iteration crosses the service boundary twice. Against the monolithic gradient the result agrees
to 9.0e-11 relative. `struct.S1`'s JVP and VJP are derived by hand rather than generated, and satisfy
the dot-product identity to 1.9e-14; they are additionally checked against finite differences and
against a JAX reference implementation.

The **number of component round trips** is governed by the fixed-point contraction, not by the
design dimension. The fixed point's observed contraction factor tracks `q/q_D` to four digits, so
how many round trips the two services need depends on how close the wing is to divergence: 12, 30
and 57 round trips at `q/q_D` = 0.099, 0.397 and 0.621. Nothing here isolates the per-call transport
cost, so this says how many calls the physics demands and not what a call costs. The optimized
designs sit at 0.747, above that measured range. The designs where the cheap gradient is least
trustworthy are therefore also the ones where the accurate gradient costs most.

## 3. Validity experiment

The criterion is step usefulness rather than gradient similarity. Let `p` be the unit descent
direction the rigid gradient proposes, along which it predicts a rate of decrease `g_rigid · p`.
Take the coupled objective's actual directional derivative along that same `p` by central difference
and form their ratio `r`. A design is counted against the rigid gradient when `r` < 0.01: the step
returns less than 1 % of the decrease it was promised. Negative `r` means the coupled objective
increases. The threshold was fixed before the map was scored, and no indicator in §4.2 was evaluated
until it was.

The map is a 17 × 17 grid over tip twist (−0.10 to +0.06 rad) and log stiffness (−1.0 to +0.6), 289
designs, all at the design flight condition. Two further sweeps separate the candidate causes: one
holds the design fixed and varies the flight condition, the other holds the flight condition fixed
and walks the design toward the coupled optimum. Each is run at two settings so that either factor
can be seen with the other held constant.

## 4. Results

### 4.1 Validity boundary

Of the 289 designs, 31 fire. On 30 of those the coupled objective increases along the direction the
rigid gradient recommends.

![Two line charts of the cosine between the aerodynamic-only and coupled gradients. Left, against
dynamic pressure: the optimized design plunges through zero, a generic design of the same stiffness
never crosses. Right, walking the design toward the optimum at fixed flight condition: crosses zero
at the design condition, dips to 0.67 and recovers at a fifth of divergence
pressure.](docs/figures/validity_regime.png)

A natural hypothesis is that increasing aeroelastic coupling alone drives the failure. The controls
do not support that.

- Varying the flight condition at a design of the same stiffness that no optimizer chose, the cosine
  between the two gradients never falls below 0.922 up to 99 % of divergence pressure, never changes
  sign, and no step along the rigid direction increases the coupled objective. At the design
  condition it is 0.938.
- Holding the flight condition at a fifth of divergence pressure and walking the design in to the
  coupled optimum, the cosine dips to 0.668 and recovers without changing sign.
- The same walk at the design condition, operating point held fixed throughout, crosses zero halfway
  along and reaches −0.929.

The two factors are the numerator and the denominator of one quantity. Writing `e = g_rigid −
g_coupled`, the condition `‖e‖ / ‖g_coupled‖ < 1` is sufficient for the rigid gradient to remain a
descent direction, which is Carter's relative-gradient-error condition (*SIAM J. Numer. Anal.* 28,
1991). Coupling grows the numerator; standing near the coupled optimum shrinks the denominator.
Measured here, the ratio reaches at most 0.7525 at the generic design and never loses descent there,
while at the optimized design it first exceeds 1 at `q/q_D` = 0.4932.

The numerator behaves that way for a concrete reason. At fixed lift the rigid objective contains no
dynamic pressure at all, so the rigid gradient returns the same vector at every flight condition:
‖g_rigid‖ = 3.7250 across nineteen speeds, spread exactly 0.0. Its error is a fixed quantity while
the coupled gradient shrinks toward stationarity, and a fixed error eventually dominates a shrinking
signal. That occurs in the last neighbourhood an optimizer visits
([sweep tables](results/validity_regime.md)).

### 4.2 Predeclared heuristics

If the boundary could be recognised from cheap operating-point state, measuring it would not be
necessary. Four physically motivated indicators were predeclared and their thresholds fixed before
the map was scored, then evaluated against all 289 designs. The motivations are stated rather than
fitted: 0.80 of divergence pressure is a near-divergence trigger leaving a 20 % dynamic-pressure
margin, 5° is a small-angle deformation trigger past which a linear model of elastic twist stops
being defensible, 0.25 is a 25 % normalized load-redistribution trigger past which a rigid
distribution is unrepresentative, and the twist-cancellation threshold of 1.0 is the mechanism
hypothesis of this study — elastic twist reaching geometric twist scale. None of the four is a
numerical threshold established by a citation, and none was adjusted after scoring.

| indicator | threshold | TP | FP | FN | TN | accuracy | MCC |
|---|---|---|---|---|---|---|---|
| dynamic-pressure ratio | 0.80 | 9 | 42 | 22 | 216 | 0.7785 | +0.1035 |
| maximum elastic twist | 5° | 12 | 68 | 19 | 190 | 0.6990 | +0.0854 |
| twist-cancellation ratio | 1.0 | 10 | 181 | 21 | 77 | 0.3010 | −0.2477 |
| load redistribution | 0.25 | 9 | 61 | 22 | 197 | 0.7128 | +0.0389 |

None of the four discriminates. Matthews correlation is the appropriate
summary at a 10.7 % positive rate, and the best of the four reaches 0.1035. Accuracy is misleading
here: the best comparator scores 0.7785 where a constant "never fires" scores 0.893. The indicator
drawn from this study's own account of the mechanism is the weakest, anti-correlated at −0.2477.

All four read operating-point state, while the governing ratio has a design-space quantity in its
denominator that no operating-point variable can observe, and every one of them needs the coupled
state whose computation the rigid gradient was supposed to avoid. That accounts for these four
scores. Nothing measured here excludes a better indicator, or a surrogate fitted to a map like this
one. What is measured is that the four obvious candidates, declared in advance, do not work on this
wing. Method and full tables:
[`results/aeroelastic_qualification.md`](results/aeroelastic_qualification.md).

### 4.3 Design consequence

Two L-BFGS-B runs were made from a common zero-twist start with identical box bounds, one driven by
each gradient, both to convergence, with every objective and gradient evaluation crossing the
service boundary.

| | coupled gradient | rigid gradient |
|---|---|---|
| objective, scored by the coupled model | 1.3500 → **0.9336** (30.8 % better, 30 iterations) | **1.0851**, 16.2 % worse |
| objective, scored by its own model | 0.9336 | 0.9345, effectively a tie |
| induced drag `C_Di` | 0.004987 | 0.005981, 19.9 % more |
| trimmed lift `C_L` | 0.5000 | 0.5000 |
| structural mass proxy | 0.4966, at the lower bound | 0.4966, at the lower bound |

Both arms end at the same stiffness bound and the same trimmed lift, so the difference between them
is aerodynamic. The coupled arm builds 8.5° of root-to-tip washout against the rigid arm's 4.1°,
which is the classically tailored answer. The optimization made 12,225 `apply` and 2,227
`vector_jacobian_product` calls to each component.

The rigid-gradient run does not fail visibly: it converges, satisfies its bounds, and reports 0.9345
against the coupled arm's 0.9336, which reads as a tie. The discrepancy is observable only from
outside the component that caused it.

The result is bounded to this flight regime. Re-scoring both designs across speed with nothing
re-optimized, the ranking reverses at about 57 m/s; below that the rigid-gradient design is the
better of the two, by 9.6 % at 30 m/s.

![Three panels: wing twist from root to tip, showing built-in against in-flight shape; lift per
metre of span for two designs at identical total lift, the aerodynamic-only design loading further
outboard; and check-gradients verdicts for each component alone and for the crossed
pairing.](docs/figures/aeroelastic_hero.png)

## 5. Why Tesseract

`aero.A1` differentiates by JAX autodiff; `struct.S1` has no autodiff framework at all, and its JVP
and VJP are hand-derived. Neither imports the other, and they share no process and no tape.
`aero.A1` is JAX and differentiates by autodiff. `struct.S1` is a separate NumPy and SciPy solver
whose JVP and VJP were derived by hand. Neither imports the other, they share no process and no
autodiff tape. Making a pair like that into one differentiable workflow normally means porting one
solver into the other's differentiation stack, or writing the interoperability layer yourself.

Tesseract is that layer. Both components expose the same value-and-derivative contract, and that
contract is enough to assemble the coupled forward solve and the transposed coupled adjoint from
their calls while each solver keeps its own internals and its own way of differentiating. The
aeroelastic mathematics does not require it — this could be one monolithic program in a single
framework. The heterogeneous workflow is what would not survive the port.

| Tesseract | role | differentiation |
|---|---|---|
| `aero.A1` | lifting-line aerodynamics, Glauert form | JAX autodiff |
| `struct.S1` | cantilever torsion beam, 24 elements | hand-derived adjoint, NumPy and SciPy |
| `coupled.C1` | the assembled trimmed objective and its total derivative | none of its own; delegates to A1 and S1 |

The assembled objective goes behind the same contract. `coupled.C1` exposes the trimmed coupled
objective as `apply` and a selected total-derivative provider on `jacobian`, `jvp` and `vjp`. It
holds no physics, only the fixed point, the two-point trim, the transposed loop and the
design-to-station map, delegating every evaluation to A1 and S1 over HTTP. Because the platform
reserves no privileged form for a composite, stock tooling that can interrogate a leaf component can
interrogate the assembled system unmodified, which is what makes §6 possible with no numerical
checking code of this project's own.

![Design variables feed aero.A1, which exchanges loads and twist both ways with struct.S1. The pair
feeds a coupled objective and then the optimizer. Off to the side, the assembled objective is
exposed as a third Tesseract, coupled.C1, checked once by check-gradients before the optimizer
starts.](docs/figures/architecture.png)

Text equivalents for all three figures are in
[`docs/figures/README.md`](docs/figures/README.md).

The contribution here is the application and the measured validity boundary. The coupled adjoint,
the derivative-checking method and the placement of a check at a system boundary are all existing
practice.

## 6. Derivative qualification

`tesseract-runtime check-gradients` compares a component's declared derivative against central
finite differences of that component's own `apply`. Run against `aero.A1` and `struct.S1`
separately, it passes everything: 0 failures in 14,994 checks. That establishes the component
contracts and nothing about the system, since component partials are not system sensitivities, so it
cannot detect the pairing studied above. The composed Tesseract gives a place to ask the question
that can.

Pointed at `coupled.C1` at its own defaults, the same tool separates the pairings:

| objective | derivative provider | verdict |
|---|---|---|
| coupled | coupled | passed, 0 of 540 |
| rigid | rigid | passed, 0 of 540 |
| coupled | **rigid** | **failed, 462 of 540** |

Three design points, three derivative endpoints each. Both derivatives are legitimate; only one is a
derivative of the assembled objective. Across the three points the crossed pairing fails on 15 of
the 18 design-index/point combinations, and at the optimized design the rejection is stark: the
rigid provider declares +2.951 in root twist where the coupled objective is stationary at −1.7e-06.
The three stiffness terms pass at that point because both models share the same dominant structural
mass contribution.

Two qualifications on reading that table. The 540 is a resample size rather than a count of
independent tests, and the columns are not symmetric evidence: at the platform's defaults a failure
means more than 10 % disagreement while a pass means only that there is less. Because
`check-gradients` compares C1 against finite differences of C1, it could in principle certify a
consistently wrong facade, so C1 is validated separately against the monolithic reference, agreeing
to 7.0e-12 on the objective and 1.4e-09 on the coupled gradient. Resampling detail, and what the
counts do and do not mean, are in
[`results/native_composed_check.md`](results/native_composed_check.md).

## 7. Reproduction

Python 3.11 on CPU, no Docker and no GPU. The direct Python dependencies are pinned in
`requirements.txt` to the versions used for the committed runs.

```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python scripts/aeroelastic_demo.py                 # replay the committed run
.venv/bin/python aeroelastic/studies/validity_map.py                 # §4.1, the 289-design sweep
.venv/bin/python aeroelastic/studies/validity_regime.py              # §4.1, the two controls
.venv/bin/python aeroelastic/studies/qualification.py                # §4.2, the four indicators
.venv/bin/python aeroelastic/verification/validate.py                     # §1, closed-form checks
.venv/bin/python aeroelastic/verification/verify_adjoint.py               # §2, the hand adjoint
.venv/bin/python aeroelastic/verification/run_native_composed_check.py    # §6, coupled.C1
.venv/bin/python scripts/aeroelastic_demo.py --live          # serve, qualify, optimize
```

The replay reads committed JSON and binds no ports. `--live` starts both components on ports 8811
and 8812, recomputes the qualification and the optimization, and compares the result against the
committed records field by field. `validity_map.py` runs the 289-design sweep behind §4.1 in about
thirteen minutes and writes nothing unless given `--out`; `qualification.py` then scores the four
indicators against the committed map. A re-run reproduces all 289 firing classifications: the cell
closest to the η₁ threshold sits 5.6e-04 from it, against a largest between-run drift of 7.5e-08.
Full route, including what changes in the working tree when these are run:
[`docs/reproduce.md`](docs/reproduce.md).

## 8. Limitations

- **Reduced-order model.** Lifting line plus a torsion beam. Invalid at low aspect ratio, with
  sweep, in compressible flow, or near stall, and not validated against experiment.
- **Static aeroelasticity only.** There is no mass axis and therefore no flutter model. On an
  unswept wing flutter can precede divergence, and this model is silent about that. Nothing here
  supports a certification statement.
- **n = 1.** One wing, one objective, six design variables, one flight condition. The validity map
  is a measurement of one two-parameter family, and the 16.2 % comes from a single paired
  trajectory from a common start rather than from a population.
- **The optimizer spends divergence margin.** Both arms drive torsional stiffness to its lower box
  bound, ending at `q/q_D` = 0.747 against 0.371 at the starting design. A box bound stops them,
  not a stability constraint.
- **Qualification is local, and consistency is not adequacy.** The check establishes whether a
  gradient is a derivative of an objective at the points tested, under a tolerance the caller
  chooses. It cannot establish that the objective is the right one. Past divergence it cannot
  evaluate the objective at all, because the partitioned fixed point does not converge.
- **The heuristics tested are not the space of heuristics.** Four predeclared indicators fail here.
  A better one, or a surrogate fitted to a map like this, is not excluded by anything measured.
- **Novelty is confined to the measurement.** OpenAeroStruct and MACH-Aero already perform coupled
  aerostructural optimization with analytic coupled adjoints, and the qualification step uses the
  platform's own tool. What is offered as new is this wing's validity boundary, the two controls
  that isolate it, and the failure of the four predeclared indicators.

## Sources

The model, the condition this study measures against, and the prior art it does not claim.

- Glauert, *The Elements of Aerofoil and Airscrew Theory*, 2nd ed., CUP 1948, §11.1–11.4 — the
  monoplane equation in the Fourier form `aero.A1` solves, and the published τ and δ tables that
  `validate.py` reproduces digit for digit.
- Multhopp, *Die Berechnung der Auftriebsverteilung von Tragflügeln*, Luftfahrtforschung 15, 1938 —
  the nodal quadrature form of the induced-angle operator.
- NACA TN 926 (1944) — replacing strip theory with lifting line moves the torsional-divergence
  velocity of an AR-6 wing by 17–40 %. Measured here, `q_D` is 1.291× the strip-theory value at
  AR 16, a divergence speed 13.6 % higher.
- Martins, Alonso & Reuther, *Optimization and Engineering* 6, 2005, DOI
  10.1023/B:OPTE.0000048536.47956.62, §5.1.1 — the frozen-deformation gradient computed exactly and
  validated independently by the complex step, reported to have "significantly lower magnitudes and
  even opposite signs for many of the design variables". That a cheap coupled-problem gradient can
  mislead is established prior art; where it starts to, on a given wing, is what is measured here.
- Bombardieri et al., 2021, Table 10 — a rigid-gradient optimum evaluated at aeroelastic equilibrium
  carries +6.87 % drag against the coupled optimum. The 16.2 % here is a different wing, objective
  and optimizer, and is not offered as a comparison against that figure.
- Carter, *SIAM J. Numer. Anal.* 28, 1991 — `‖e‖ / ‖g‖ < 1` is sufficient for an inexact gradient to
  remain a descent direction. That is the condition §4.1 attributes the boundary to.
- OpenAeroStruct (Apache-2.0, VLM plus a 6-DOF beam, analytic partials assembled by OpenMDAO) and
  MACH-Aero / ADflow / TACS — coupled aerostructural optimization with analytic coupled adjoints is
  existing practice. No solver novelty is claimed.

Reports: [validity regime](results/validity_regime.md) ·
[qualification](results/aeroelastic_qualification.md) ·
[application result](results/aeroelastic_wing.md) ·
[composed derivative check](results/native_composed_check.md). Each states what it does not show.
