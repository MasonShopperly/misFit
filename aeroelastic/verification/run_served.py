#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""The vertical slice: two Tesseracts over HTTP, coupled forward AND coupled backward.

Binds 8811-8812 and must be serialised against any other driver that binds them.

FORWARD, a partitioned fixed point. Neither component can produce the coupled state:

    u   = alpha + twist(t) + theta          total local incidence
    tau = A1.apply(u)                       aerodynamics: incidence -> torque, CL, CDi
    theta = S1.apply(tau, s)                structure:    torque    -> twist at the aero stations

BACKWARD, the same loop transposed. Writing L = dS/dtau . dA_tau/du for the loop gain, the total
derivative needs (I - L)^-1, and in reverse mode that is a fixed point in the adjoint of u:

    a       = A1.vjp(cl = dF/dcl, cdi = dF/dcdi)["alpha_tot"]
    u_bar  <- a + A1.vjp(torque = S1.vjp(theta_aero = u_bar)["torque"])["alpha_tot"]
    dJ/dt   = (d twist/dt)^T u_bar
    dJ/ds   = S1.vjp(theta_aero = u_bar, mass = dF/dmass)["s_ctrl"]

Both loops contract at q/q_D, so the adjoint costs about what the primal costs, and neither has a
route that avoids the other component. A1 never learns that a structure exists; S1 never learns
what a lift coefficient is. That is the property the composition claim rests on, and it is checked
here against the monolithic in-process gradient rather than asserted.

    .venv/bin/python aeroelastic/verification/run_served.py
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
AE = HERE.parent
COMPONENTS = AE / "components"
sys.path.insert(0, str(AE / "core"))

import jax                                                                # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
import numpy as np                                                        # noqa: E402
from tesseract_core import Tesseract                                      # noqa: E402

import aero_struct as A                                                   # noqa: E402
from aero_struct import RHO, Wing                                         # noqa: E402
from model import W_MASS, refs, CL_TARGET                                 # noqa: E402

P_AERO, P_STRUCT = 8811, 8812
RUNTIME = Path(sys.executable).parent / "tesseract-runtime"
RESULTS = AE / "served"
W = Wing()
FP_TOL, FP_MAX = 1e-13, 400


def assert_ports_free(ports):
    busy = [p for p in ports
            if socket.socket().connect_ex(("127.0.0.1", int(p))) == 0]
    if busy:
        raise SystemExit(f"ports already in use: {busy}. Stop the other driver first.")


def start_server(api: Path, port: int):
    env = dict(os.environ)
    env["TESSERACT_API_PATH"] = str(api)
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = open(RESULTS / f"serve_{port}.log", "w")
    return subprocess.Popen(
        [str(RUNTIME), "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        cwd=tempfile.mkdtemp(prefix=f"ae{port}_"))


def clean_pycache(api: Path) -> None:
    """Remove the __pycache__ a served component's import leaves inside its own build context.

    check_packaging.py refuses a __pycache__ in a component directory, correctly: `tesseract build`
    copytrees that directory and the compiled cache would enter the image. But serving the
    component creates one, so any live run left the repository check red afterwards, and the
    discovery came at the worst possible moment. Cleaned by whoever created it."""
    import shutil
    cache = api.parent / "__pycache__"
    if cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)


def wait_healthy(url: str, timeout: float = 240.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.4)
    raise RuntimeError(f"{url} never became healthy")


class Calls:
    def __init__(self):
        self.n = {}

    def bump(self, key):
        self.n[key] = self.n.get(key, 0) + 1


CALLS = Calls()


AERO = f"http://127.0.0.1:{P_AERO}"
STRUCT = f"http://127.0.0.1:{P_STRUCT}"
_CLIENTS: dict = {}


def client(url: str):
    """The SDK client, as the rest of this repository uses it: it owns the array encoding."""
    if url not in _CLIENTS:
        _CLIENTS[url] = Tesseract.from_url(url, timeout=(5.0, 300.0))
    return _CLIENTS[url]


def call(url: str, endpoint: str, *args, **kw):
    CALLS.bump(f"{'aero.A1' if url == AERO else 'struct.S1'}.{endpoint}")
    return getattr(client(url), endpoint)(*args, **kw)


def coupled_forward(t_ctrl, s_ctrl, alpha, v):
    """The state neither component can produce alone."""
    twist = np.asarray(A.twist_profile(W, jnp.asarray(t_ctrl)))
    theta = np.zeros(twist.size)
    for it in range(FP_MAX):
        u = alpha + twist + theta
        ao = call(AERO, "apply", {"alpha_tot": u, "v": np.float64(v)})
        so = call(STRUCT, "apply", {"torque": np.asarray(ao["torque"], np.float64),
                                    "s_ctrl": np.asarray(s_ctrl, np.float64)})
        new = np.asarray(so["theta_aero"], dtype=np.float64)
        d = float(np.max(np.abs(new - theta)))
        theta = new
        if d < FP_TOL:
            break
    return {"theta": theta, "u": alpha + twist + theta, "cl": float(ao["cl"]),
            "cdi": float(ao["cdi"]), "mass": float(so["mass"]),
            "torque": np.asarray(ao["torque"], np.float64), "iters": it + 1}


def coupled_adjoint(state, s_ctrl, v, dF_dcl, dF_dcdi, dF_dmass):
    """The same loop transposed. Every iteration crosses the boundary twice."""
    ain = {"alpha_tot": state["u"], "v": np.float64(v)}
    sin = {"torque": state["torque"], "s_ctrl": np.asarray(s_ctrl, np.float64)}
    a = np.asarray(call(AERO, "vector_jacobian_product", ain,
                        vjp_inputs=["alpha_tot"], vjp_outputs=["cl", "cdi"],
                        cotangent_vector={"cl": np.float64(dF_dcl),
                                          "cdi": np.float64(dF_dcdi)})["alpha_tot"], np.float64)

    u_bar = a.copy()
    for it in range(FP_MAX):
        tau_bar = call(STRUCT, "vector_jacobian_product", sin,
                       vjp_inputs=["torque"], vjp_outputs=["theta_aero"],
                       cotangent_vector={"theta_aero": u_bar})["torque"]
        back = np.asarray(call(AERO, "vector_jacobian_product", ain,
                               vjp_inputs=["alpha_tot"], vjp_outputs=["torque"],
                               cotangent_vector={"torque": np.asarray(tau_bar, np.float64)}
                               )["alpha_tot"], np.float64)
        new = a + back
        d = float(np.max(np.abs(new - u_bar)))
        u_bar = new
        if d < FP_TOL:
            break

    s_bar = np.asarray(call(STRUCT, "vector_jacobian_product", sin,
                            vjp_inputs=["s_ctrl"], vjp_outputs=["theta_aero", "mass"],
                            cotangent_vector={"theta_aero": u_bar,
                                              "mass": np.float64(dF_dmass)})["s_ctrl"], np.float64)
    return u_bar, s_bar, it + 1


def served_gradient(x, v, cdi_ref, mass_ref, cl_target):
    """dJ/dx for J = CDi/CDi_ref + W_MASS mass/mass_ref, at the TRIMMED state, entirely over HTTP."""
    t_ctrl, s_ctrl = np.asarray(x[:3]), np.asarray(x[3:])
    # Trim: CL is affine in alpha, so two coupled solves give the trimmed incidence exactly.
    s0 = coupled_forward(t_ctrl, s_ctrl, 0.0, v)
    s1 = coupled_forward(t_ctrl, s_ctrl, 1.0, v)
    alpha = (cl_target - s0["cl"]) / (s1["cl"] - s0["cl"])
    st = coupled_forward(t_ctrl, s_ctrl, alpha, v)
    j = st["cdi"] / cdi_ref + W_MASS * st["mass"] / mass_ref

    # Trim holds CL fixed, so the free lift sensitivity must be projected out. With the constraint
    # CL(x, alpha(x)) = CL*, the trimmed gradient is the gradient of the Lagrangian
    # L = J - lambda (CL - CL*) taken at FIXED alpha, with lambda = (dJ/dalpha)/(dCL/dalpha).
    # dCL/dalpha is exact from the two-point trim line because CL is affine in alpha. CDi is
    # exactly QUADRATIC in alpha (the Fourier coefficients are affine and CDi is their weighted
    # square), so a central difference in alpha is likewise exact, not approximate.
    dcl_da = s1["cl"] - s0["cl"]
    h = 1e-3
    sp = coupled_forward(t_ctrl, s_ctrl, alpha + h, v)
    sm = coupled_forward(t_ctrl, s_ctrl, alpha - h, v)
    dcdi_da = (sp["cdi"] - sm["cdi"]) / (2.0 * h)
    lam = (dcdi_da / cdi_ref) / dcl_da
    u_bar, s_bar, it = coupled_adjoint(st, s_ctrl, v, dF_dcl=-lam, dF_dcdi=1.0 / cdi_ref,
                                       dF_dmass=W_MASS / mass_ref)
    dtwist_dt = np.asarray(jax.jacobian(lambda tc: A.twist_profile(W, tc))(jnp.asarray(t_ctrl)))
    return j, np.concatenate([dtwist_dt.T @ u_bar, s_bar]), alpha, st, it


def main() -> int:
    assert_ports_free([P_AERO, P_STRUCT])
    RESULTS.mkdir(parents=True, exist_ok=True)
    procs = []
    try:
        procs.append(start_server(COMPONENTS / "aero" / "tesseract_api.py", P_AERO))
        procs.append(start_server(COMPONENTS / "structure" / "tesseract_api.py", P_STRUCT))
        wait_healthy(AERO)
        wait_healthy(STRUCT)
        print(f"aero.A1 on {P_AERO} (JAX autodiff)   struct.S1 on {P_STRUCT} (hand adjoint)\n")

        fwd_rows, bwd_rows = [], []
        print("FORWARD -- the partitioned fixed point over HTTP vs the monolithic solve")
        t_ctrl = np.array([0.05, 0.01, -0.06])
        s_ctrl = np.array([0.1, -0.2, -0.4])
        for v in (40.0, 80.0, 100.0):
            st = coupled_forward(t_ctrl, s_ctrl, 0.04, v)
            th_ref, _, cl_ref, cdi_ref = A.solve_coupled(W, jnp.asarray(t_ctrl),
                                                         jnp.asarray(s_ctrl), 0.04, v)
            th_ref_a = np.asarray(A.transfer_matrix(W) @ th_ref)
            rel = float(np.max(np.abs(st["theta"] - th_ref_a)) / np.max(np.abs(th_ref_a)))
            qd = float(A.divergence_q(W, jnp.asarray(s_ctrl)))
            print(f"  V={v:>5.0f}  q/q_D={0.5 * RHO * v ** 2 / qd:.3f}  round trips={st['iters']:>3d}  "
                  f"theta rel={rel:.2e}  CL {st['cl']:.9f} vs {float(cl_ref):.9f}  "
                  f"CDi {st['cdi']:.9f} vs {float(cdi_ref):.9f}")
            fwd_rows.append({"v": v, "q_over_qd": 0.5 * RHO * v ** 2 / qd,
                             "round_trips": st["iters"], "theta_rel_err": rel,
                             "cl_served": st["cl"], "cl_mono": float(cl_ref)})

        print("\nBACKWARD -- the coupled adjoint over HTTP vs the monolithic JAX gradient")
        cdi_ref_v, mass_ref_v = refs()

        def j_mono(xx, v):
            solver = A.solve_coupled
            _, _, _, cdi, _ = A.solve_trimmed(W, xx[:3], xx[3:], v, CL_TARGET, solver)
            return cdi / cdi_ref_v + W_MASS * A.mass_proxy(W, xx[3:]) / mass_ref_v

        x = np.concatenate([t_ctrl, s_ctrl])
        for v in (60.0, 80.0):
            g_mono = np.asarray(jax.grad(lambda xx: j_mono(xx, v))(jnp.asarray(x)))
            j_served, g_served, alpha, st, it = served_gradient(
                x, v, cdi_ref_v, mass_ref_v, CL_TARGET)
            rel = float(np.max(np.abs(g_served - g_mono)) / np.max(np.abs(g_mono)))
            print(f"  V={v:>5.0f}  adjoint round trips={it:>3d}  J {j_served:.10f} vs "
                  f"{float(j_mono(jnp.asarray(x), v)):.10f}   grad rel err={rel:.3e}")
            bwd_rows.append({"v": v, "adjoint_round_trips": it, "grad_rel_err": rel})
            print(f"        served {np.array2string(g_served, precision=6)}")
            print(f"        mono   {np.array2string(g_mono, precision=6)}")

        print("\nENDPOINT INVENTORY -- what each component was actually asked for")
        for k in sorted(CALLS.n):
            print(f"  {k:<40s} {CALLS.n[k]:>6d}")
        (AE / "records" / "served_verification.json").write_text(json.dumps(
            {"forward": fwd_rows, "backward": bwd_rows, "calls": CALLS.n}, indent=1) + "\n")
        print(f"  wrote {AE / 'records' / 'served_verification.json'}")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        for _api in (COMPONENTS / "aero" / "tesseract_api.py",
                     COMPONENTS / "structure" / "tesseract_api.py"):
            clean_pycache(_api)
    return 0


if __name__ == "__main__":
    sys.exit(main())
