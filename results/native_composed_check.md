# Composed derivative check with the platform's own checker

**Verdict: `NATIVE_DISCRIMINATES`.** Run 2026-08-06. The points, the pairings and the invocation
parameters were frozen before the composed component existed. Record:
[`../aeroelastic/records/native_composed_check.json`](../aeroelastic/records/native_composed_check.json),
re-derived line by line by
[`../aeroelastic/verification/check_native_composed.py`](../aeroelastic/verification/check_native_composed.py).

## Question

`tesseract-runtime check-gradients` compares a derivative endpoint against finite differences of the
**same module's** `apply`. The assembled coupled objective lives in neither component, so the
question is whether the platform's own checker can be asked about it at all — and if it can, whether
it separates a derivative that belongs to that objective from one that does not.

That depends entirely on where the component boundary is drawn. Drawn around each physics module,
the assembled objective is invisible to the checker.

## Construction of `coupled.C1`

`coupled.C1` draws the boundary round the composition. It exposes the assembled trimmed objective as
one Tesseract and holds **no physics**: no `aero_struct` import, no solver, no autodiff, not one copy
of either component. It holds two HTTP clients, the partitioned fixed point, the two-point trim, the
transposed adjoint loop, and the 40×3 matrix that maps three design control points onto forty
spanwise stations — the *design* parameterisation, which belongs to neither component because the
design vector does not exist inside either one.

## Invocation settings

The platform's own instrument is then pointed at it, with its own defaults: central differences,
`eps = 1e-4`, `rtol = 0.1`, its own index sampling, its own three endpoints. Three operating points ×
three declared pairings × three derivative endpoints, 60 sampled indices per endpoint, seed
`20260806`.

## Verdict table

| point | `q/q_D` | pairing | jacobian | jvp | vjp | failures |
|---|---|---|---|---|---|---|
| `before-a` | 0.371 | matched-coupled | passed | passed | passed | 0 / 180 |
| `before-a` | 0.371 | matched-rigid | passed | passed | passed | 0 / 180 |
| `before-a` | 0.371 | **crossed-rigid** | **failed** | **failed** | **failed** | **180 / 180** |
| `inside-a` | 0.612 | matched-coupled | passed | passed | passed | 0 / 180 |
| `inside-a` | 0.612 | matched-rigid | passed | passed | passed | 0 / 180 |
| `inside-a` | 0.612 | **crossed-rigid** | **failed** | **failed** | **failed** | **180 / 180** |
| `design` | 0.747 | matched-coupled | passed | passed | passed | 0 / 180 |
| `design` | 0.747 | matched-rigid | passed | passed | passed | 0 / 180 |
| `design` | 0.747 | **crossed-rigid** | **failed** | **failed** | **failed** | **102 / 180** |

`matched-rigid` — the rigid gradient against
the *rigid* objective — passes everywhere. The rigid provider is not a broken gradient; it is the
exact derivative of a different model. Only the **pairing** fails.

## What 462 of 540 means

The three crossed invocations sample 540 checks in total, 180 at each operating point, and **462 of
them are rejected**. The 78 that pass are all at `design`, where 102 of the 180 crossed checks fail;
at `before-a` and `inside-a` all 180 fail.

The platform prints the value it rejected. At the committed optimized design:

| design index | rigid gradient, declared | coupled objective, central difference | source |
|---|---|---|---|
| 0 — root twist | **+2.951** | −1.69e−06 | record |
| 1 — mid twist | **−0.844** | +1.13e−06 | record |
| 2 — tip twist | **−2.107** | +5.59e−07 | record |
| 3 — root stiffness | +0.0435 | +0.0412 | console; not persisted |
| 4 — mid stiffness | +0.0869 | +0.0891 | console; not persisted |
| 5 — tip stiffness | +0.0435 | +0.0435 | console; not persisted |

The driver records only the values the platform **rejected**, so indices 0–2 are in
`native_composed_check.json` and are re-derived by the checker under the platform's own criterion.
Indices 3–5 passed, so the platform printed them and the record does not keep them; they are quoted
from its console output and cannot be re-derived from the committed evidence. What the record does
establish independently is the count they explain: 102 of 180 at this point rather than 180.

The rigid provider asserts a strong descent direction in twist at a point where the coupled objective
is **stationary to six decimal places**. Following it moves the optimizer away from the optimum.

Three of the six components nevertheless agree. The stiffness sensitivities are dominated by the mass
term, which is identical in both models, so they land inside the platform's 10 % tolerance and pass.
That is why `design` reports 102 of 180 and not 180.

## Coverage

The discrimination at the design point comes from the twist components, so it is the *coverage* of
the platform's index sampling — not the sharpness of its criterion — that catches it there. **A check
that had sampled only the stiffness indices at this point would not have fired.** At `before-a` and
`inside-a` all six components are rejected, so that sensitivity to coverage is a property of the
design point rather than of the method.

The driver records the HTTP call count of every invocation and withholds the record if either
component was never called: a facade that answered without asking A1 and S1 would have tested
nothing, and must not be able to publish a verdict.

## Cost

Counted, not estimated — every HTTP call the facade made is recorded.

| invocation | `apply` evaluations | A1 calls | S1 calls | wall |
|---|---|---|---|---|
| `design` / matched-coupled | 13 | 15,282 | 15,282 | 74.9 s |
| `design` / **crossed-rigid** | 13 | 5,580 | 5,580 | 25.9 s |
| `design` / matched-rigid | 13 | 165 | 165 | 1.8 s |

Thirteen evaluations of the assembled objective — twelve central-difference perturbations plus the
base point — cover all six design components on all three derivative endpoints. The component call
counts differ by two orders of magnitude between pairings because the coupled objective is a fixed
point whose round-trip count is set by `q/q_D`, while the rigid objective is one call.

The 13 evaluations are set by `--max-evals`, **not** by the design dimension: the platform samples
indices, so the same six-index budget applies at six design variables or six thousand. Nothing here
supports the claim that finite differences are expensive in high dimension; that is a different
argument and this experiment does not make it.

## What this does not show

- **Nothing about containers.** `coupled.C1` is exercised through `TESSERACT_API_PATH`, exactly as
  [`run_platform_checker_ae.py`](../aeroelastic/verification/run_platform_checker_ae.py) exercises A1
  and S1. No image was built and none was served.
- **Nothing about the fourth pairing.** `crossed-frozen`, the fourth row of the committed
  qualification table, is defined by `jax.lax.stop_gradient` on the coupled state and has no served
  expression, so it is outside this experiment.
- **Nothing past divergence.** The frozen point `after` (`q/q_D = 1.0085`) is excluded and stated as
  excluded: past divergence the partitioned fixed point does not converge, so the facade cannot
  evaluate the objective there at all. That is a limitation of partitioned coupling and of this
  experiment, not of the checker.
- **Nothing about whether this route beats the alternatives.** Stock `scipy.optimize.check_grad`,
  handed the same two served callables, reaches the same three verdicts in far fewer evaluations.
  This experiment measures what the platform's own tool says about the assembled objective, not which
  instrument is cheapest.

## Reproduce

```bash
.venv/bin/python aeroelastic/verification/run_native_composed_check.py   # ~9 min, binds 8811-8812
.venv/bin/python aeroelastic/verification/check_native_composed.py       # re-derives every number above
```

The driver binds fixed ports and must not run alongside another driver that binds them. The checker
reads the committed record only and needs no network.
