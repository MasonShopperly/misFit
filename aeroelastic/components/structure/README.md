# struct.S1

The structural half of the coupled pair: a linear finite-element cantilever torsion beam, 24 uniform
elements per semispan.

← [`aeroelastic/`](../README.md)

**Contract.** Given the applied torque and the stiffness design variables, returns the elastic twist
— resolved onto the *aerodynamic* stations, because that is the only form `aero.A1` can consume, so
the mesh transfer lives with the mesh that owns it. `apply`, `jacobian`,
`jacobian_vector_product` and `vector_jacobian_product`.

**Differentiation: a hand-derived adjoint in NumPy and SciPy, deliberately not JAX.** A composed
system whose components all differentiate the same way tests nothing about composition. The
interesting boundary is the one where an autodiff component must exchange derivatives with a solver
that has none, which is the realistic case for structural codes. The derivation is in the module
docstring; [`verify_adjoint.py`](../../verification/verify_adjoint.py) checks it three ways — against the
dot-product identity to 1.9e-14, against central differences, and against a JAX reference.

| file | what it is |
|---|---|
| [`tesseract_api.py`](tesseract_api.py) | the component, and the adjoint derivation it implements |
| [`tesseract_config.yaml`](tesseract_config.yaml) | build metadata; carries no autodiff framework at all |
| [`tesseract_requirements.txt`](tesseract_requirements.txt) | this component's own dependencies |

This directory is the build context for `tesseract build aeroelastic/components/structure`. Served on port
8812 by the drivers in [`studies/`](../../studies/README.md) and
[`verification/`](../../verification/README.md).

Where the composition is explained:
[`technical_writeup.md` §5](../../../technical_writeup.md#5-why-tesseract).
