# Reproduce

Everything here runs from a fresh clone, on CPU, without Docker. Python 3.11; the direct Python
dependencies are pinned in `requirements.txt` to the versions used for the committed runs.

```bash
git clone https://github.com/MasonShopperly/misFit && cd misFit
uv venv .venv --python 3.11                                    # or: python3.11 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt   # or: .venv/bin/pip install -r requirements.txt
```

Run everything from the repository root. Nothing writes outside the clone except where noted.

## The headline measurement

The leading result is the validity boundary: where the cheap rigid-wing gradient stops being safe to
design with, the two controls that isolate it, and the four predeclared comparators that fail to
predict it.

```bash
.venv/bin/python aeroelastic/studies/validity_map.py      # the 289-design sweep, ~13 min
.venv/bin/python aeroelastic/studies/validity_regime.py   # the boundary and its two controls
.venv/bin/python aeroelastic/studies/qualification.py     # the four predeclared comparators
```

`validity_map.py` is the sweep behind the headline counts: 289 designs on a 17×17 grid of tip twist
against uniform log stiffness, 31 firing, 30 of those with the coupled objective increasing along
the rigid descent direction. It prints the map and the summary and **writes nothing unless given
`--out PATH`**, so a reproduction cannot overwrite the committed record by accident. To regenerate
and compare:

```bash
.venv/bin/python aeroelastic/studies/validity_map.py --out /tmp/validity_map.json
.venv/bin/python aeroelastic/studies/validity_map.py --out aeroelastic/records/validity_map.json
```

The second form is the only way the committed record is rewritten, which is why it is spelled out
rather than left as a default.

Re-run on the machine of record, all 289 firing classifications and both counts came back
identical. The `r̂` cells are ratios of central differences and move in the 8th significant figure —
largest absolute drift 7.5e-08, against 5.6e-04 from the nearest cell to the `η₁ = 0.01` threshold,
so no cell is close to reclassifying.

`validity_regime.py` sweeps flight condition at two designs and walks the design segment at two
fixed conditions, writing the record the front-page figure is rendered from. `qualification.py`
scores the four comparators against those 289 designs at thresholds frozen before any of them was
evaluated; the thresholds and their motivations are tabulated in
[`results/aeroelastic_qualification.md`](../results/aeroelastic_qualification.md). All three run in
process against the committed model; none binds a port. Reports:
[`results/validity_regime.md`](../results/validity_regime.md) ·
[`results/aeroelastic_qualification.md`](../results/aeroelastic_qualification.md).

## The application

```bash
.venv/bin/python scripts/aeroelastic_demo.py          # replay the committed coupled run, ~0.02 s
.venv/bin/python scripts/aeroelastic_demo.py --live   # serve both components and recompute, ~18 min
```

**The replay is the one command to run first.** It prints the coupled problem, the composed
qualification table, the optimized wing, the aerodynamic-only comparison and the endpoint counts,
labelling every section with where its numbers came from. It reads committed JSON: no solver, no
server, no port, and safe to run repeatedly.

**`--live` binds ports 8811 and 8812**, starts both Tesseracts, qualifies the provider and re-runs
the optimization, then compares what it just computed against the committed record field by field.
It writes to a scratch file, so it leaves the clone unmodified. The composed-Tesseract qualification
is most of the eighteen minutes.

The comparison reports each field as `EXACT MATCH`, `NUMERICALLY CONSISTENT` or `FAILED`, and only
`FAILED` produces a non-zero exit status. The middle category exists because float64 BLAS does not
fix its reduction order across runs, so a recomputation on your hardware need not land on the same
bits. Measured on one machine at one commit: one run reproduced 295/295 and 43/43 bit for bit, and
two later runs differed in 51 fields, all in the 9th to 13th significant digit. Fields within
`rtol` 1e-6 and `atol` 1e-9 are reported as consistent; call counts, exit codes, verdicts and other
discrete fields are compared by equality with no tolerance at all. Every field that is not
bit-identical is printed with its deviation, whichever side of the tolerance it falls on.

`atol` carries the quantities that are zero for reporting purposes: the coupled gradient at the
optimized design, which is stationary, and the residuals measuring agreement with the monolithic
solve. Those are central differences at `eps` = 1e-4 on an objective of order 1, so their absolute
noise floor is about 2e-12 and a re-run moves them by 4e-12 to 1.6e-11. 1e-9 sits above that and
three orders below 1.7e-06, the smallest quantity this project reports.

`.venv/bin/python scripts/aeroelastic_demo.py --self-test` checks those comparison semantics
against planted cases in about a second, without serving anything.

## The composition, checked

```bash
.venv/bin/python aeroelastic/verification/validate.py                   # closed-form checks
.venv/bin/python aeroelastic/verification/verify_adjoint.py             # the hand adjoint, three ways
.venv/bin/python aeroelastic/verification/run_platform_checker_ae.py    # check-gradients on each component alone
.venv/bin/python aeroelastic/verification/run_served.py                 # the served vertical, forward and back
.venv/bin/python aeroelastic/verification/run_native_composed_check.py  # coupled.C1 + check-gradients
.venv/bin/python aeroelastic/verification/check_native_composed.py          # verify the committed record
```

`verify_adjoint.py` checks `struct.S1`'s hand-derived adjoint against the dot-product identity,
against finite differences, and against a JAX reference. `run_served.py` runs the coupled forward
solve and the transposed coupled adjoint across HTTP and compares both against a monolithic
in-process reference — the 1.6e-12 and 9.0e-11 agreements. `run_native_composed_check.py` serves
`coupled.C1` and runs the platform's own `tesseract-runtime check-gradients` against it, the
experiment behind [`results/native_composed_check.md`](../results/native_composed_check.md), with
its points and settings fixed before the composed component existed. `check_native_composed.py`
re-derives that report's verdicts, counts, rejected values and wall times from the committed
record and fails if any disagrees. The three values the report marks *console; not persisted* are
the exception, and they are marked because the record does not hold them. It needs no ports and
takes under a second.

## Regenerate what is published

```bash
.venv/bin/python aeroelastic/studies/run_served_optimize.py   # rewrites aeroelastic/records/served_optimize.json
.venv/bin/python scripts/make_media.py                # every figure but the wing panel
.venv/bin/python scripts/make_wing_figure.py      # docs/figures/aeroelastic_wings.png
```

`run_served_optimize.py` binds ports 8811 and 8812 and takes several minutes. The figure generators
read only committed records and take about ninety seconds, and write only the surfaces this
repository publishes. What each figure plots, and the record behind it, are in
[`docs/figures/README.md`](figures/README.md).

Re-rendering leaves three of the five figures modified. Two are anti-alias
rounding: `aeroelastic_hero.png` and `aeroelastic_wings.png` differ on 43 and 14 pixels
respectively, by one intensity level. `aeroelastic_morph.gif` is different in kind, differing on
6,612 of its 8.3 million pixels by up to 51 levels: it is palettized, its 256-entry palette is
re-derived on each render, and a sub-level change in the raster can move a pixel to a palette entry
far away in intensity. No plotted quantity moves in any of the three. The other two
(`validity_regime.png` and `architecture.png`) come back byte-identical, and `make_wing_figure.py`
also rewrites `aeroelastic/records/wing_figure.json` in the last digits of seven numeric fields.

The rendering itself is deterministic: three consecutive runs on one machine produced byte-identical
output on every surface. What differs is the environment, so these three files will differ on your
machine and match on the one they were rendered on.

## What a reproduction run changes in your clone

The measurement scripts rewrite the record they own. After running the set above, `git status` shows
four modified files, and `git diff` is not empty. That is not a failure — it is float64 arithmetic
not landing on the same bits twice. Measured on a fresh clone against the committed records:

| record | differing fields | worst relative difference |
|---|---|---|
| `records/qualification.json` | 1,167 of them numeric | 7.1e-03, on a fitted Taylor slope |
| `records/validity_regime.json` | 557 | 1.5e-11 |
| `records/speed_sweep.json` | 82 | 1.5e-11 |
| `records/validation.json` | 4 | 1.4e-14 |

**Across all four: zero non-numeric changes and zero key-set changes.** Every verdict, count,
classification and label is identical. The 7.1e-03 outlier has a cause: those fields are
least-squares *fitted convergence orders* over finite-difference remainders near machine precision,
where cancellation dominates. They move in the third decimal — 1.9642 against 1.9782 — around an
expected value of 2 and against a PASS threshold of 1.7, so they are nowhere near changing a
verdict. Every headline number, including the 289-design counts and the four comparators' confusion
matrices, reproduces exactly.

`records/platform_checker_ae.json` is the fifth record a run rewrites, and it is the one record that
rewrites *identically*: its checker is seeded and its counts come from the declared shapes, so
re-running `run_platform_checker_ae.py` reproduces the committed file byte-for-byte.

`git checkout -- .` restores the committed records if you want a clean tree back.

## Ports

The live paths bind **8811** and **8812** and refuse to start if either is in use. `coupled.C1`
binds nothing: it is loaded in process through `TESSERACT_API_PATH`, exactly as the platform's own
checker loads any component, and reaches `aero.A1` and `struct.S1` over those two ports like any
other client. Run one live path at a time.

## Repository map

Every directory carries a README indexing its own contents.

| path | what it is |
|---|---|
| [`aeroelastic/core/`](../aeroelastic/core/README.md) | the monolithic reference model and the objective every experiment is scored on; the components do not import it |
| [`aeroelastic/components/`](../aeroelastic/components/README.md) | the three Tesseracts: `aero.A1`, `struct.S1`, `coupled.C1` |
| [`aeroelastic/studies/`](../aeroelastic/studies/README.md) | the experiments that produce the reported science |
| [`aeroelastic/verification/`](../aeroelastic/verification/README.md) | the checks, the record verifier, and the Taylor-ladder library two committed records were produced with |
| [`aeroelastic/records/`](../aeroelastic/records/README.md) | the ten committed records, and only those |
| [`scripts/`](../scripts/README.md) | the demo and the two figure generators |
| [`results/`](../results/README.md) | one report per result, each stating what it does not show |
| [`docs/figures/`](figures/README.md) | the five published figures, with the record and generator behind each |
| `docs/misfit-demo.mp4` | the demo video, 4:05 |
