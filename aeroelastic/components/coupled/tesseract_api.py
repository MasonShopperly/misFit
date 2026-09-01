# SPDX-License-Identifier: Apache-2.0
"""coupled.C1 — the assembled aeroelastic objective, exposed as one Tesseract. A facade, not a model.

WHY THIS EXISTS. `tesseract-runtime check-gradients` compares a component's declared derivative
against central finite differences of THAT component's own `apply`. Run on `aero.A1` and `struct.S1` it passes both, which settles nothing about the ASSEMBLED
objective, because that objective lives in neither module.

That is a statement about where the boundary was drawn, not about the checker. Draw it here instead
and the platform can ask the question directly.

WHAT IT HOLDS: nothing. No physics, no solver, no autodiff, no copy of either component. It holds
HTTP clients, a fixed point, a trim, the transposed adjoint loop, and the (40 x 3) matrix that maps
three design control points to forty spanwise stations -- the DESIGN parameterisation, which belongs
to neither component and has to live at the composition boundary because that is where the design
vector first exists. Every number with physics in it arrives over a socket. The counter written to
MISFIT_C1_COUNTS is the proof: a run that reaches zero calls to either component is void.

    apply(x)   ->  J(x) = CDi/cdi_ref + 0.35 mass/mass_ref, at the trimmed state
    jacobian / jvp / vjp  ->  the gradient OF THE DECLARED PAIRING

`pairing` is an ordinary input field, so the crossed case is visible in the payload the checker is
handed rather than hidden in an environment variable:

    matched-coupled   apply = coupled J     derivative = coupled adjoint    should pass
    matched-rigid     apply = RIGID  J      derivative = rigid gradient     should pass
    crossed-rigid     apply = coupled J     derivative = RIGID gradient     the question

Contract frozen before this file existed.

Not served standalone: it is exercised through TESSERACT_API_PATH, exactly as
run_platform_checker_ae.py exercises A1 and S1, and it needs A1 and S1 already serving.
"""
import atexit
import json
import os

import numpy as np
from pydantic import BaseModel

from tesseract_core import Tesseract
from tesseract_core.runtime import Array, Differentiable, Float64

N_AERO, N_CTRL, N_DESIGN = 40, 3, 6
W_MASS = 0.35                     # the drag/mass exchange rate of the committed objective, carried
FP_TOL, FP_MAX = 1e-13, 400       # same tolerance as the committed served vertical slice
H_ALPHA = 1e-3                    # trim sensitivity step; CDi is exactly quadratic in alpha

AERO_URL = os.environ.get("MISFIT_AERO_URL", "http://127.0.0.1:8811")
STRUCT_URL = os.environ.get("MISFIT_STRUCT_URL", "http://127.0.0.1:8812")

PAIRINGS = ("matched-coupled", "matched-rigid", "crossed-rigid")

# --- design parameterisation: three control points -> forty aero stations -------------------
# eta = cos(t) at the same collocation stations both components already use. The map is linear, so
# its matrix is exact and its transpose moves gradients back without a derivative being taken.
_T = (np.arange(1, N_AERO + 1) - 0.5) * np.pi / (2.0 * N_AERO)
_ETA = np.cos(_T)
_KNOTS = np.array([0.0, 0.5, 1.0])
_B = np.stack([np.interp(_ETA, _KNOTS, np.eye(N_CTRL)[i]) for i in range(N_CTRL)], axis=1)

COUNTS: dict[str, int] = {}
_CLIENTS: dict[str, Tesseract] = {}

# The runtime imports this module more than once per invocation, so more than one COUNTS dict
# exists and each registers its own atexit handler. A handler that WROTE the file would clobber the
# working copy's totals with an empty dict from a copy that never ran anything, and the invocation
# would report zero HTTP calls. Each instance instead owns a key and the driver sums across keys,
# which is idempotent however many times it is called.
_TOKEN = f"{os.getpid()}:{id(COUNTS):x}"


def _dump_counts() -> None:
    path = os.environ.get("MISFIT_C1_COUNTS")
    if not path:
        return
    try:
        data = json.loads(open(path).read())
    except Exception:
        data = {}
    data[_TOKEN] = COUNTS
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)


atexit.register(_dump_counts)


def call(url: str, endpoint: str, *args, **kw):
    """Every physics evaluation in this module goes through here, and every one is counted."""
    if url not in _CLIENTS:
        _CLIENTS[url] = Tesseract.from_url(url, timeout=(5.0, 300.0))
    key = f"{'aero.A1' if url == AERO_URL else 'struct.S1'}.{endpoint}"
    COUNTS[key] = COUNTS.get(key, 0) + 1
    COUNTS["_total"] = COUNTS.get("_total", 0) + 1
    if COUNTS["_total"] % 256 == 0:        # atexit is the real write; this survives a kill
        _dump_counts()
    return getattr(_CLIENTS[url], endpoint)(*args, **kw)


class InputSchema(BaseModel):
    x: Differentiable[Array[(N_DESIGN,), Float64]]
    v: Array[(), Float64]
    cl_target: Array[(), Float64]
    cdi_ref: Array[(), Float64]
    mass_ref: Array[(), Float64]
    pairing: str = "matched-coupled"


class OutputSchema(BaseModel):
    objective: Differentiable[Array[(), Float64]]


# --- the coupled state neither component can produce ----------------------------------------
def coupled_forward(t_ctrl, s_ctrl, alpha, v):
    twist = _B @ np.asarray(t_ctrl, np.float64)
    theta = np.zeros(N_AERO)
    for it in range(FP_MAX):
        ao = call(AERO_URL, "apply", {"alpha_tot": alpha + twist + theta, "v": np.float64(v)})
        so = call(STRUCT_URL, "apply", {"torque": np.asarray(ao["torque"], np.float64),
                                        "s_ctrl": np.asarray(s_ctrl, np.float64)})
        new = np.asarray(so["theta_aero"], np.float64)
        d = float(np.max(np.abs(new - theta)))
        theta = new
        if d < FP_TOL:
            break
    else:
        raise RuntimeError(f"partitioned fixed point did not converge: residual {d:.3e} "
                           f"after {FP_MAX} round trips -- q/q_D is at or above 1")
    return {"u": alpha + twist + theta, "cl": float(ao["cl"]), "cdi": float(ao["cdi"]),
            "mass": float(so["mass"]), "torque": np.asarray(ao["torque"], np.float64),
            "round_trips": it + 1}


def rigid_forward(t_ctrl, s_ctrl, alpha, v):
    """The same wing with the structure switched off. The aerodynamic component alone is enough
    for the loads; the structural component is still asked for the mass, because the objective has
    a mass term whichever aerodynamic model produced the drag."""
    twist = _B @ np.asarray(t_ctrl, np.float64)
    u = alpha + twist
    ao = call(AERO_URL, "apply", {"alpha_tot": u, "v": np.float64(v)})
    so = call(STRUCT_URL, "apply", {"torque": np.asarray(ao["torque"], np.float64),
                                    "s_ctrl": np.asarray(s_ctrl, np.float64)})
    return {"u": u, "cl": float(ao["cl"]), "cdi": float(ao["cdi"]), "mass": float(so["mass"]),
            "torque": np.asarray(ao["torque"], np.float64), "round_trips": 1}


def trimmed(t_ctrl, s_ctrl, v, cl_target, forward):
    """CL is affine in alpha in both models, so two solves fix the trim line exactly."""
    s0 = forward(t_ctrl, s_ctrl, 0.0, v)
    s1 = forward(t_ctrl, s_ctrl, 1.0, v)
    alpha = (cl_target - s0["cl"]) / (s1["cl"] - s0["cl"])
    st = forward(t_ctrl, s_ctrl, alpha, v)
    st["alpha"] = alpha
    st["dcl_dalpha"] = s1["cl"] - s0["cl"]
    return st


def objective_value(x, v, cl_target, cdi_ref, mass_ref, model):
    forward = coupled_forward if model == "coupled" else rigid_forward
    st = trimmed(np.asarray(x[:3]), np.asarray(x[3:]), v, cl_target, forward)
    return st["cdi"] / cdi_ref + W_MASS * st["mass"] / mass_ref, st


# --- the same loop, transposed ---------------------------------------------------------------
def _trim_multiplier(t_ctrl, s_ctrl, v, st, cdi_ref, forward):
    """Trim holds CL fixed, so the free lift sensitivity is projected out with the multiplier of
    the constraint CL(x, alpha(x)) = CL*. CDi is exactly quadratic in alpha, so the central
    difference below is exact rather than approximate."""
    a = st["alpha"]
    sp = forward(t_ctrl, s_ctrl, a + H_ALPHA, v)
    sm = forward(t_ctrl, s_ctrl, a - H_ALPHA, v)
    dcdi_da = (sp["cdi"] - sm["cdi"]) / (2.0 * H_ALPHA)
    return (dcdi_da / cdi_ref) / st["dcl_dalpha"]


def coupled_gradient(x, v, cl_target, cdi_ref, mass_ref):
    t_ctrl, s_ctrl = np.asarray(x[:3]), np.asarray(x[3:])
    st = trimmed(t_ctrl, s_ctrl, v, cl_target, coupled_forward)
    lam = _trim_multiplier(t_ctrl, s_ctrl, v, st, cdi_ref, coupled_forward)
    ain = {"alpha_tot": st["u"], "v": np.float64(v)}
    sin = {"torque": st["torque"], "s_ctrl": s_ctrl}

    a = np.asarray(call(AERO_URL, "vector_jacobian_product", ain,
                        vjp_inputs=["alpha_tot"], vjp_outputs=["cl", "cdi"],
                        cotangent_vector={"cl": np.float64(-lam),
                                          "cdi": np.float64(1.0 / cdi_ref)})["alpha_tot"],
                   np.float64)
    u_bar = a.copy()
    for _ in range(FP_MAX):
        tau_bar = call(STRUCT_URL, "vector_jacobian_product", sin,
                       vjp_inputs=["torque"], vjp_outputs=["theta_aero"],
                       cotangent_vector={"theta_aero": u_bar})["torque"]
        back = np.asarray(call(AERO_URL, "vector_jacobian_product", ain,
                               vjp_inputs=["alpha_tot"], vjp_outputs=["torque"],
                               cotangent_vector={"torque": np.asarray(tau_bar, np.float64)}
                               )["alpha_tot"], np.float64)
        new = a + back
        d = float(np.max(np.abs(new - u_bar)))
        u_bar = new
        if d < FP_TOL:
            break
    s_bar = np.asarray(call(STRUCT_URL, "vector_jacobian_product", sin,
                            vjp_inputs=["s_ctrl"], vjp_outputs=["theta_aero", "mass"],
                            cotangent_vector={"theta_aero": u_bar,
                                              "mass": np.float64(W_MASS / mass_ref)})["s_ctrl"],
                       np.float64)
    return np.concatenate([_B.T @ u_bar, s_bar])


def rigid_gradient(x, v, cl_target, cdi_ref, mass_ref):
    """The exact derivative of the RIGID objective -- the q -> 0 limit of the one above. There is
    no loop to transpose: the structure does not respond, so the adjoint is one aerodynamic VJP.
    The mass term still comes from the structural component."""
    t_ctrl, s_ctrl = np.asarray(x[:3]), np.asarray(x[3:])
    st = trimmed(t_ctrl, s_ctrl, v, cl_target, rigid_forward)
    lam = _trim_multiplier(t_ctrl, s_ctrl, v, st, cdi_ref, rigid_forward)
    u_bar = np.asarray(call(AERO_URL, "vector_jacobian_product",
                            {"alpha_tot": st["u"], "v": np.float64(v)},
                            vjp_inputs=["alpha_tot"], vjp_outputs=["cl", "cdi"],
                            cotangent_vector={"cl": np.float64(-lam),
                                              "cdi": np.float64(1.0 / cdi_ref)})["alpha_tot"],
                       np.float64)
    s_bar = np.asarray(call(STRUCT_URL, "vector_jacobian_product",
                            {"torque": st["torque"], "s_ctrl": s_ctrl},
                            vjp_inputs=["s_ctrl"], vjp_outputs=["theta_aero", "mass"],
                            cotangent_vector={"theta_aero": np.zeros(N_AERO),
                                              "mass": np.float64(W_MASS / mass_ref)})["s_ctrl"],
                       np.float64)
    return np.concatenate([_B.T @ u_bar, s_bar])


# --- endpoints --------------------------------------------------------------------------------
def _models(pairing: str) -> tuple[str, str]:
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
    return ("rigid", "rigid") if pairing == "matched-rigid" else \
           ("coupled", "coupled" if pairing == "matched-coupled" else "rigid")


def apply(inputs: InputSchema) -> OutputSchema:
    obj_model, _ = _models(inputs.pairing)
    j, _ = objective_value(np.asarray(inputs.x, np.float64), float(inputs.v),
                           float(inputs.cl_target), float(inputs.cdi_ref),
                           float(inputs.mass_ref), obj_model)
    return OutputSchema(objective=np.float64(j))


def _gradient(inputs) -> np.ndarray:
    _, grad_model = _models(inputs.pairing)
    fn = coupled_gradient if grad_model == "coupled" else rigid_gradient
    return fn(np.asarray(inputs.x, np.float64), float(inputs.v), float(inputs.cl_target),
              float(inputs.cdi_ref), float(inputs.mass_ref))


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    g = _gradient(inputs)
    return {o: {i: g for i in jac_inputs} for o in jac_outputs}


def jacobian_vector_product(inputs: InputSchema, jvp_inputs: set[str],
                            jvp_outputs: set[str], tangent_vector):
    g = _gradient(inputs)
    t = np.asarray(tangent_vector["x"], np.float64)
    return {o: np.float64(g @ t) for o in jvp_outputs}


def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str],
                            vjp_outputs: set[str], cotangent_vector):
    g = _gradient(inputs)
    c = float(np.asarray(cotangent_vector["objective"]))
    return {i: c * g for i in vjp_inputs}


def abstract_eval(abstract_inputs):
    return {"objective": {"shape": (), "dtype": "float64"}}
