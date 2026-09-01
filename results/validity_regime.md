# Validity regime

Run 2026-08-06. Record:
[`../aeroelastic/records/validity_regime.json`](../aeroelastic/records/validity_regime.json).
Driver: [`../aeroelastic/studies/validity_regime.py`](../aeroelastic/studies/validity_regime.py).
Descriptive — no criterion, nothing passes or fails. The two step lengths were fixed in the driver
before any row was read.

The question is where the cheap rigid gradient stops being a safe basis for design on this wing, and
which quantity sets that boundary. Two controls separate the two candidate causes: the flight
condition, and the position of the design relative to the coupled optimum. Neither alone accounts
for the boundary.

## Control 1 — vary coupling at a generic design

Same stiffness as the committed design, so the same `q_D` and the same axis; zero twist, so no
optimizer ever chose it.

| | committed optimized design | **neutral design** |
|---|---|---|
| cosine at `q/q_D` = 0.047 | 0.99999 | 0.99912 |
| cosine at `q/q_D` = 0.747 | **0.0285** | **0.93810** |
| cosine at `q/q_D` = 0.988 | **−0.99679** | **0.92241** |
| sign change anywhere | between 0.747 and 0.843 | **none** |
| step along −g_rigid ever increases the coupled objective | yes, from 0.747 up | **never** |

Dynamic pressure alone does not destroy the rigid gradient's usefulness on this wing. Across the
entire flyable range, up to 99 % of the divergence pressure, the rigid direction at a generic design
remains a descent direction and loses only 8 % of the available first-order decrease. That is a
negative result against this project's own headline framing.

## Control 2 — approach the optimum at fixed condition

The straight line from the neutral design to the committed optimized one, at two fixed speeds.
Coupling is constant along each block.

| `t` | ‖x − x_opt‖ | cosine at `q/q_D` = 0.187 | cosine at `q/q_D` = 0.747 |
|---|---|---|---|
| 0.0 | 0.1138 | 0.98957 | 0.93810 |
| 0.2 | 0.0911 | 0.96889 | 0.85563 |
| 0.4 | 0.0683 | 0.83799 | 0.46145 |
| **0.5** | 0.0569 | 0.66759 | **−0.07321** |
| 0.6 | 0.0455 | 0.74643 | −0.56455 |
| 0.8 | 0.0228 | 0.99635 | −0.88929 |
| 1.0 | 0.0000 | 0.99978 | 0.02854 |

At the design condition the rigid gradient stops being a descent direction halfway along the
segment, with the flight condition held fixed. At a fifth of the divergence pressure the same walk
dips to 0.668 and recovers; it never changes sign.

## Combined regime

| | generic design | near the coupled optimum |
|---|---|---|
| **weak coupling** (`q/q_D` = 0.187) | cos 0.990 | cos 0.668 — degraded, still descent |
| **design condition** (`q/q_D` = 0.747) | cos 0.938 | cos **−0.073** — uphill |

Neither ingredient is sufficient. Strong coupling at a generic design keeps the rigid provider
useful; standing at the coupled optimum under weak coupling keeps it useful. Together they invert
it.

The mechanism is visible in one column of the record. At fixed lift the rigid objective contains no
dynamic pressure at all, so **‖g_rigid‖ = 3.7250 at every speed, spread exactly 0.0** — the rigid
provider returns the *same vector* for every flight condition the aircraft will ever see. Its error
is therefore a fixed quantity, while the coupled gradient shrinks as the design approaches the
coupled optimum. A fixed error dominates a shrinking signal. The place where that happens is the
last neighbourhood an optimizer visits.

## Relative-gradient-error interpretation

The two ingredients are the numerator and the denominator of a single quantity the study already
computes:

```
    e = g_cheap - g_coupled     rel = ||e|| / ||g_coupled||
    rel < 1   =>   g_cheap . g_coupled >= ||g_coupled||^2 - ||e|| ||g_coupled|| > 0
```

That is the relative-gradient-error
condition of Carter, *SIAM J. Numer. Anal.* 28(1):251–265 (1991). Coupling grows `‖e‖`; proximity to
the coupled optimum shrinks `‖g_coupled‖`. One ratio, two ways to move it.

It predicts both controls, from the committed record:

| | max relative error over the sweep | sign change |
|---|---|---|
| **generic design** | **0.7525** — never reaches 1 | **none**, and no step ever increased the objective |
| committed optimized design | 34.68 at the design condition | yes, between `q/q_D` 0.747 and 0.843 |

At the committed design the ratio first exceeds 1 at `q/q_D` = **0.4932**, well before descent is
actually lost at 0.747 — as it must, since `rel < 1` is *sufficient* and not necessary.
`rel_norm_diff` is computed in both `sweep_speed.py:75` and `validity_regime.py`; the cosine is the
more dramatic number and this ratio is the one with a theorem attached.

Direction and magnitude are different failures. At the design condition ‖g_r‖/‖g_c‖ is 34.7 at the
committed design and **0.37 at the neutral one** — the 34.7 is an artifact of standing at an
optimum, and the neutral number is the one to quote.

Following the rigid direction from the committed design at 80 m/s increases the coupled objective:
**+1.116e−03** against **−5.345e−04** for the coupled direction, on a step of 0.005.

This is a measurement of where a known sufficient condition fails on one wing, not a discovery about
coupled systems. The physics of the mechanism is aileron reversal (NACA 799, 1944; NACA 1024, 1951);
component-versus-system derivatives are the global sensitivity equations
(Sobieszczanski-Sobieski, *AIAA J.* 28(1):153–160, 1990); the condition itself is Carter's.

## Heuristic comparison

Accuracy against a **10.73 %** base rate flatters every predictor, so the four operating-point
comparators are scored by Matthews correlation from the counts the study stores:

| comparator | accuracy | **MCC** | precision | recall |
|---|---|---|---|---|
| dynamic-pressure ratio | 0.7785 | **0.1035** | 0.1765 | 0.2903 |
| maximum elastic twist | 0.6990 | 0.0854 | 0.1500 | 0.3871 |
| load redistribution | 0.7128 | 0.0389 | 0.1286 | 0.2903 |
| **twist cancellation ratio** | 0.3010 | **−0.2477** | 0.0524 | 0.3226 |

Four thresholds, each with a stated physical motivation rather than a citation behind the number,
none reaching MCC 0.11 on 289 designs, and the one predeclared from this project's own mechanism
account is anti-correlated — worse than chance. The thresholds, their formulas and their motivations
are tabulated in [`aeroelastic_qualification.md`](aeroelastic_qualification.md).

The relative-error account above explains those scores. Every one of the four is a function of
operating-point state alone, and the governing quantity has `‖g_coupled‖` in its denominator — a
design-space quantity that operating-point state cannot observe. It does not exclude a better indicator.

## Implication for qualification

The boundary says where a check would have to look. Re-checking at every optimizer iteration spends
its budget in the region where the approximate provider is still fine, and has no way of knowing
that the interesting region is the endgame. A check at the *operating condition* alone would not
have fired either, because at a generic design the provider passes at 99 % of divergence. Nothing
here measures what continuous re-checking would cost, only where it would be looking.

What the evidence supports is a check on the **assembled pairing at the design point being flown**,
repeated when the provider, the objective, the coupling or the operating condition materially
changes — which is what the composed-Tesseract route does, once, before the optimizer starts
([`native_composed_check.md`](native_composed_check.md)).

## What this does not show

- Not *"gradient validity collapses as coupling strengthens."* Control 1 refutes that as stated.
- No threshold in `q/q_D`. There is no operating-condition number at which this provider becomes
  unusable independent of where the design sits.
- Nothing about other couplings, other objectives, other providers, or more than one wing. This is
  one two-parameter family, six design variables, one lifting line, one torsion beam.
- Nothing from the cosine alone. It is a direction statement; the combined regime needed the norms
  and a finite step to say what it means.

## Reproduce

```bash
.venv/bin/python aeroelastic/studies/validity_regime.py
```

Reads [`served_optimize.json`](../aeroelastic/records/served_optimize.json) for the committed
design and rewrites
[`validity_regime.json`](../aeroelastic/records/validity_regime.json). Runs in a single process; no
component, port or container is involved.
