# Heuristic comparison

Run 2026-08-06. The design points and every comparator threshold were frozen before any comparator
was evaluated; the thresholds and their motivations are tabulated below as they were fixed. Record:
[`../aeroelastic/records/qualification.json`](../aeroelastic/records/qualification.json). Driver:
[`../aeroelastic/studies/qualification.py`](../aeroelastic/studies/qualification.py).

The question here is whether a cheap aeroelastic state variable can predict where the rigid gradient
stops being a useful descent direction. If one could, no derivative-level check would be needed to
find that region.

This is not how the submission qualifies a gradient. That is done by pointing the platform's own
`tesseract-runtime check-gradients` at the composed `coupled.C1` Tesseract, and the evidence for it
is [`native_composed_check.md`](native_composed_check.md). The submission's headline is the validity
boundary in [`validity_regime.md`](validity_regime.md).

## Predeclared heuristics

Four candidates were predeclared and scored against the committed validity map's firing set:
**31 firing of 289 designs.**

Each threshold has a stated physical motivation; **none of the four numbers is established by a
citation**. What predeclaration buys here is that no threshold could be chosen to fit an outcome,
not that the literature nominated these values.

| comparator | formula | threshold | motivation for the number |
|---|---|---|---|
| dynamic-pressure ratio | `q / q_D(s)` | 0.80 | near-divergence trigger, leaving 20 % dynamic-pressure margin |
| maximum elastic twist | `max ｜θ_e｜` | 5° | small-angle deformation trigger, past which a linear aeroelastic model stops being defensible |
| twist cancellation ratio | `max ｜θ_e｜ / max ｜θ_geo｜` | 1.0 | **this study's own account of the mechanism** — elastic twist reaching geometric twist scale |
| load redistribution | `‖l_coupled − l_rigid‖ / ‖l_rigid‖` | 0.25 | 25 % normalized load-redistribution trigger, past which a rigid load distribution is no longer representative |

Each is evaluated on the same 289-design grid as the committed validity map and scored against that
map's firing set as ground truth. The map is produced by
[`validity_map.py`](../aeroelastic/studies/validity_map.py); a re-run reproduces all 289
classifications and both counts.

## Scores

| comparator | threshold | TP | FP | FN | TN | accuracy | needs |
|---|---|---|---|---|---|---|---|
| dynamic-pressure ratio | 0.80 | 9 | 42 | 22 | 216 | **0.7785** | coupled state |
| maximum elastic twist | 5° | 12 | 68 | 19 | 190 | 0.6990 | coupled state |
| twist cancellation ratio | 1.0 | 10 | **181** | 21 | 77 | **0.3010** | coupled state |
| load redistribution | 0.25 | 9 | 61 | 22 | 197 | 0.7128 | coupled + rigid |

None of the four discriminates. The best, dynamic-pressure ratio, misses **22 of 31** true firings
while raising 42 false alarms, at an accuracy a constant "never fires" beats outright at 0.893, since
only 10.7 % of designs fire. Because that base rate flatters every predictor, the same counts are
rescored by Matthews correlation in
[`validity_regime.md`](validity_regime.md) — no comparator reaches MCC 0.11.

The twist-cancellation ratio scores worst, at 0.3010 with 181 false positives, and it was predeclared
because the validity map's shape suggests elastic washin cancelling geometric washout is the
mechanism. As a description of where firing happens that still looks right. As a predictor it is the
weakest of the four. A correct account of a mechanism is not the same thing as a usable threshold on
it.

No threshold was adjusted after scoring, and none of the four is reported with a post-hoc best value.

## What this does and does not establish

The region where the rigid gradient stops differentiating the coupled objective is not predicted by
any of the four aeroelastic state variables tested. All four are functions of operating-point state,
while the governing quantity — Carter's relative gradient error — has a design-space norm in its
denominator that operating-point state cannot see
([`validity_regime.md`](validity_regime.md), relative-gradient-error interpretation).

Four comparators are not the space of possible heuristics. A fifth, built with knowledge of the map,
could do better; that would be a fitted threshold rather than a predeclared one, and nothing here
excludes it. The result is that four obvious candidates, chosen in advance, do not work on this wing.

## Taylor-ladder rows in the record

`qualification.json` also carries 24 rows under `part1_checker`, from a local Taylor-order ladder
run over four objective/gradient pairings at six design points, four probe directions each. It is
retained because the record is committed evidence and
[`../aeroelastic/verification/legacy_taylor_check.py`](../aeroelastic/verification/legacy_taylor_check.py)
reproduces it unchanged. Its verdicts agree with the served route: `matched-coupled` and
`matched-rigid` fit order ≈ 2 and pass, while `crossed-rigid` and `crossed-frozen` sit at order
1.000 and fail — order 1 being the analytic signature of a gradient that is not a derivative of the
objective it is paired with.

Note that `crossed-rigid` fails at every point, including points where the validity map shows the
rigid gradient is still a good descent direction (r̂ ≈ 1.73). That is not a contradiction: a
gradient can be the wrong derivative and still point downhill.

## Reproduce

```bash
.venv/bin/python aeroelastic/studies/validity_map.py    # the firing set, ~13 min
.venv/bin/python aeroelastic/studies/qualification.py   # ~4 min
```
