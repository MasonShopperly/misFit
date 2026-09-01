# SPDX-License-Identifier: Apache-2.0
"""aero.A1 — lifting-line aerodynamics. Differentiated by JAX autodiff.

Given the total local incidence at each spanwise station, returns the aerodynamic torque about the
elastic axis, the lift coefficient and the induced-drag coefficient. It does not know that a
structure exists: it is handed an incidence and returns a load. The elastic contribution to that
incidence is somebody else's business, which is exactly why the coupled state cannot be produced
here.

The Fourier form is Glauert's monoplane equation,

    sum_n A_n sin(n t) [ sin(t) + n mu ]  =  mu sin(t) alpha_tot(t),    mu = a0 c / (4 b)

collocated at N cosine stations on the semispan, odd harmonics only for a symmetric wing.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel

# BEFORE any array is built. Without it this component serves float32 while declaring Float64, and
# the failure is quiet: the partitioned fixed point stalls at ~1e-7 instead of converging, and the
# coupled gradient agrees with the monolithic one to 4e-04 rather than to machine precision. Both
# look like "close enough" until the residual is plotted against iteration count.
jax.config.update("jax_enable_x64", True)

from tesseract_core.runtime import Array, Differentiable, Float64      # noqa: E402
from tesseract_core.runtime.jax_recipes import (                       # noqa: E402
    jax_abstract_eval, jax_apply, jax_jacobian, jax_jvp, jax_vjp,
)

B_SPAN, CHORD, E_OFF, N_AERO = 16.0, 1.0, 0.15, 40
A0, RHO = 2.0 * jnp.pi, 1.225
SEMISPAN = 0.5 * B_SPAN
AREA = B_SPAN * CHORD
AR = B_SPAN ** 2 / AREA

_K = jnp.arange(1, N_AERO + 1)
_T = (_K - 0.5) * jnp.pi / (2.0 * N_AERO)
_N = 2 * _K - 1
_MU = A0 * CHORD / (4.0 * B_SPAN)
_SIN = jnp.sin(jnp.outer(_T, _N))
_M = _SIN * (jnp.sin(_T)[:, None] + _N[None, :] * _MU)
_RHS = _MU * jnp.sin(_T)


class InputSchema(BaseModel):
    alpha_tot: Differentiable[Array[(N_AERO,), Float64]]
    v: Array[(), Float64]


class OutputSchema(BaseModel):
    torque: Differentiable[Array[(N_AERO,), Float64]]
    cl: Differentiable[Array[(), Float64]]
    cdi: Differentiable[Array[(), Float64]]


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    alpha_tot, v = inputs["alpha_tot"], inputs["v"]
    a_n = jnp.linalg.solve(_M, _RHS * alpha_tot)
    gamma = 2.0 * B_SPAN * v * (_SIN @ a_n)
    lift_per_span = RHO * v * gamma
    return {"torque": lift_per_span * E_OFF * CHORD,
            "cl": jnp.pi * AR * a_n[0],
            "cdi": jnp.pi * AR * jnp.sum(_N * a_n ** 2)}


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**jax_apply(apply_jit, inputs))


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(inputs: InputSchema, jvp_inputs: set[str],
                            jvp_outputs: set[str], tangent_vector):
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str],
                            vjp_outputs: set[str], cotangent_vector):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)


def abstract_eval(abstract_inputs):
    return jax_abstract_eval(apply_jit, abstract_inputs)
