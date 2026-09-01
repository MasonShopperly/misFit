# scripts

What a reader deliberately runs to *see* something. Everything here consumes committed records and
renders them — as terminal output or as a published surface. The experiments and checks that produce
the evidence live in [`aeroelastic/`](../aeroelastic/README.md).

← [repository root](../README.md) · full route: [`docs/reproduce.md`](../docs/reproduce.md)

| file | role | what it does |
|---|---|---|
| [`aeroelastic_demo.py`](aeroelastic_demo.py) | **run this first** | replays the committed run in under a second and prints every headline table, labelled with where its numbers came from. `--live` serves both components and recomputes, comparing field by field; `--self-test` checks the comparison semantics against planted cases |
| [`make_media.py`](make_media.py) | regenerate | redraws four of the five published figures from the records. `--check` is a developer byte-comparison, not part of the reproduction route |
| [`make_wing_figure.py`](make_wing_figure.py) | regenerate | draws `docs/figures/aeroelastic_wings.png` and writes [`wing_figure.json`](../aeroelastic/records/wing_figure.json), the 41-point scan its third panel plots |

`make_wing_figure.py` is the one file here that writes a record. It sits with the other figure
generator rather than with the studies because drawing the figure is why it exists; what each figure
plots and the record behind it are in [`docs/figures/README.md`](../docs/figures/README.md).

Nothing here is a check. The record verifier is
[`aeroelastic/verification/check_native_composed.py`](../aeroelastic/verification/check_native_composed.py)
and the Taylor-ladder library two records were produced with is
[`aeroelastic/verification/legacy_taylor_check.py`](../aeroelastic/verification/legacy_taylor_check.py),
with the rest of the checking.
