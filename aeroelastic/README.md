# aeroelastic

The wing. Five directories, one concept each.

← [repository root](../README.md) · model and mathematics:
[`technical_writeup.md` §1–§2](../technical_writeup.md#1-problem-and-model) · how to run any of
this: [`docs/reproduce.md`](../docs/reproduce.md)

| | |
|---|---|
| [`core/`](core/README.md) | the monolithic reference model, and the objective every experiment is scored on |
| [`components/`](components/README.md) | the three Tesseracts: `aero.A1`, `struct.S1`, `coupled.C1` |
| [`studies/`](studies/README.md) | the experiments that produce the reported science |
| [`verification/`](verification/README.md) | the checks that the implementation and its derivatives are right |
| [`records/`](records/README.md) | the ten committed records — nine written by the drivers above, one by [`scripts/make_wing_figure.py`](../scripts/make_wing_figure.py) |

The studies/verification split is a claim about what each file is for. A study asks a question whose
answer is a result; a check asks whether the code computes what it says, and its answer is a pass or
a failure. Both write records, and both are reported in [`results/`](../results/README.md).

`served/` appears here during a live run, holds server logs and call counts, and is ignored by git —
tracked evidence and runtime scratch never share a directory. Three drivers bind ports 8811 and
8812; they are marked in [`studies/`](studies/README.md) and
[`verification/`](verification/README.md), and only one may run at a time.
