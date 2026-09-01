# verification

The checks. Each asks whether the code computes what it claims, and answers with a pass or a
failure rather than with a result.

← [`aeroelastic/`](../README.md) · how to run any of these:
[`docs/reproduce.md`](../../docs/reproduce.md)

## Active verification

| file | what it checks | record |
|---|---|---|
| [`validate.py`](validate.py) | the solvers against closed-form theory | [`validation.json`](../records/validation.json) |
| [`verify_adjoint.py`](verify_adjoint.py) | `struct.S1`'s hand adjoint three ways: dot-product identity, finite differences, a JAX reference | console |
| [`run_served.py`](run_served.py) ⚠ | the coupled forward solve and transposed adjoint over HTTP against a monolithic reference | [`served_verification.json`](../records/served_verification.json) |
| [`run_platform_checker_ae.py`](run_platform_checker_ae.py) | `tesseract-runtime check-gradients` on each component alone: 0 failures in 14,994 | [`platform_checker_ae.json`](../records/platform_checker_ae.json) |
| [`run_native_composed_check.py`](run_native_composed_check.py) ⚠ | the same platform checker against the assembled `coupled.C1` | [`native_composed_check.json`](../records/native_composed_check.json) |
| [`check_native_composed.py`](check_native_composed.py) | re-derives [its report](../../results/native_composed_check.md)'s verdicts, counts, rejected values and wall times from the record and fails if any disagrees | reads only |

⚠ binds ports 8811 and 8812. Serialise against each other and against
[`studies/run_served_optimize.py`](../studies/run_served_optimize.py).

## Record-compatibility support

[`legacy_taylor_check.py`](legacy_taylor_check.py) is a library, not an entrypoint: it is imported
by [`studies/qualification.py`](../studies/qualification.py) and
[`studies/run_served_optimize.py`](../studies/run_served_optimize.py) because two committed records
carry its historical Taylor-ladder output —
[`served_optimize.json`](../records/served_optimize.json) step 1 and Part 1 of
[`qualification.json`](../records/qualification.json) — and reproducing those records byte-for-byte
requires it unchanged. This submission qualifies through the platform's own `check-gradients`
against `coupled.C1` ([report](../../results/native_composed_check.md)).
