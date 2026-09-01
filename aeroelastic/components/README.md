# components

The three Tesseracts. Each directory is a build context: `tesseract build` copytrees it, so it holds
an API, a config and a requirements file and nothing else.

← [`aeroelastic/`](../README.md) · why three:
[`technical_writeup.md` §5](../../technical_writeup.md#5-why-tesseract)

| directory | identity | differentiation |
|---|---|---|
| [`aero/`](aero/README.md) | `aero.A1`, lifting-line aerodynamics | JAX autodiff |
| [`structure/`](structure/README.md) | `struct.S1`, cantilever torsion beam | hand-derived adjoint, NumPy and SciPy |
| [`coupled/`](coupled/README.md) | `coupled.C1`, the assembled objective | none of its own; delegates to A1 and S1 over HTTP |

The directory names are for humans; the identities `aero.A1`, `struct.S1` and `coupled.C1` are what the
records, the reports and the running services use, and they are set in each `tesseract_config.yaml`.

None of the three imports [`core/`](../core/README.md), and none declares any `package_data`: every
build context is exactly the four files listed in its own README. `core/` holds the monolithic
reference these are *checked against*, not a library they share.
