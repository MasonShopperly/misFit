# core

The monolithic reference model, and the common definition every study is scored against.

← [`aeroelastic/`](../README.md) · the mathematics:
[`technical_writeup.md` §1–§2](../../technical_writeup.md#1-problem-and-model)

| file | what it is |
|---|---|
| [`aero_struct.py`](aero_struct.py) | lifting line, torsion beam, load and twist transfer, trim, and the coupled fixed point — the whole aeroelastic problem in one module |
| [`model.py`](model.py) | the reference wing, the flight condition, and the trimmed objective every experiment is scored on |

**Who imports this, and who does not.** [`studies/`](../studies/README.md),
[`verification/`](../verification/README.md) and [`scripts/`](../../scripts/README.md) import both
files. The three Tesseracts in [`components/`](../components/README.md) import **neither**: each
implements its own half of the physics inside its own build context. That is the point of the
split — the components are independent implementations, and this directory is the single-process
reference they are checked against. A component that imported this file could not disagree with it,
and the disagreement is the measurement.

`model.py` holds the constants a result depends on — the flight speed `V`, the trim target
`CL_TARGET`, the drag/mass exchange rate `W_MASS` and the finite-difference step `H_FD`. Changing
one changes every number this repository reports, which is why nothing else defines them locally.
