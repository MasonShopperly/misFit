# coupled.C1

The assembled aeroelastic objective exposed as one Tesseract. A facade over `aero.A1` and
`struct.S1`, holding no physics of its own.

← [`aeroelastic/`](../README.md)

**Contract.** `apply` returns the trimmed coupled objective; `jacobian`, `jacobian_vector_product`
and `vector_jacobian_product` return whichever total derivative the selected provider offers, so a
coupled objective can be paired with a rigid gradient on purpose.

**What it holds: no solver, no autodiff, no copy of either component.** Two HTTP clients, the
partitioned fixed point, the two-point trim, the transposed adjoint loop, and the 40×3 matrix
mapping three design control points onto forty spanwise stations — the design parameterisation,
which belongs to neither component because the design vector does not exist inside either one.
Every quantity with physics in it arrives over a socket.

**Why it exists.** `check-gradients` compares a component's derivative against finite differences of
that same component's `apply`, so pointed at A1 or S1 it can only ask a component-local question.
Drawing the component boundary around the composition instead lets the platform's own tool be asked
whether a derivative belongs to the *assembled* objective. That experiment is
[`results/native_composed_check.md`](../../../results/native_composed_check.md).

| file | what it is |
|---|---|
| [`tesseract_api.py`](tesseract_api.py) | the facade, the fixed point, the trim and the transposed loop |
| [`tesseract_config.yaml`](tesseract_config.yaml) | build metadata; no `package_data`, and — unlike A1 and S1 — no physics either |
| [`tesseract_requirements.txt`](tesseract_requirements.txt) | this component's own dependencies |

**It binds no port.** It is loaded in process through `TESSERACT_API_PATH`, exactly as the
platform's checker loads any component, and reaches A1 and S1 over 8811 and 8812 like any other
client. The call counter it writes is load-bearing: a run that reaches zero calls to either
component is void and the record is withheld.
