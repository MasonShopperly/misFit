# aero.A1

The aerodynamic half of the coupled pair: Prandtl lifting line in Glauert's Fourier form, 40
cosine-spaced stations per semispan.

← [`aeroelastic/`](../README.md)

**Contract.** Given the total local incidence at every station, returns the aerodynamic torque about
the elastic axis, the lift coefficient and the induced-drag coefficient. `apply`, `jacobian`,
`jacobian_vector_product` and `vector_jacobian_product`.

**Differentiation.** JAX autodiff, in float64.

**It does not know a structure exists.** It is handed an incidence and returns a load; the elastic
part of that incidence belongs to `struct.S1`. That is why neither component can produce the coupled
state alone.

| file | what it is |
|---|---|
| [`tesseract_api.py`](tesseract_api.py) | the component: schemas, `apply`, and the derivative endpoints |
| [`tesseract_config.yaml`](tesseract_config.yaml) | build metadata; no `package_data` — the aerodynamics is implemented here, not imported |
| [`tesseract_requirements.txt`](tesseract_requirements.txt) | this component's own dependencies |

This directory is the build context for `tesseract build aeroelastic/components/aero`, which is why it holds
nothing else. Served on port 8811 by the drivers in [`studies/`](../../studies/README.md) and
[`verification/`](../../verification/README.md).

Where the composition is explained:
[`technical_writeup.md` §5](../../../technical_writeup.md#5-why-tesseract).
