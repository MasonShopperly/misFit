# records

The ten committed evidence records. Every number in [`results/`](../../results/README.md) and in the
README is recomputed from one of these, not retyped.

← [`aeroelastic/`](../README.md) · what each result means:
[`results/`](../../results/README.md) · how to regenerate any of them:
[`docs/reproduce.md`](../../docs/reproduce.md)

These are **tracked evidence**. The `served/` directory that appears in `aeroelastic/` during a live
run holds server logs and call counts, is regenerated every time, and is ignored by git.

| record | written by |
|---|---|
| [`validity_map.json`](validity_map.json) | [`studies/validity_map.py`](../studies/validity_map.py) |
| [`validity_regime.json`](validity_regime.json) | [`studies/validity_regime.py`](../studies/validity_regime.py) |
| [`qualification.json`](qualification.json) | [`studies/qualification.py`](../studies/qualification.py) |
| [`speed_sweep.json`](speed_sweep.json) | [`studies/sweep_speed.py`](../studies/sweep_speed.py) |
| [`served_optimize.json`](served_optimize.json) | [`studies/run_served_optimize.py`](../studies/run_served_optimize.py) |
| [`served_verification.json`](served_verification.json) | [`verification/run_served.py`](../verification/run_served.py) |
| [`native_composed_check.json`](native_composed_check.json) | [`verification/run_native_composed_check.py`](../verification/run_native_composed_check.py) |
| [`platform_checker_ae.json`](platform_checker_ae.json) | [`verification/run_platform_checker_ae.py`](../verification/run_platform_checker_ae.py) |
| [`validation.json`](validation.json) | [`verification/validate.py`](../verification/validate.py) |
| [`wing_figure.json`](wing_figure.json) | [`scripts/make_wing_figure.py`](../../scripts/make_wing_figure.py) |

Four records do not share a stem with their producer —
`speed_sweep.json` comes from `sweep_speed.py`, `served_verification.json` from `run_served.py`,
`validation.json` from `validate.py`, `platform_checker_ae.json` from `run_platform_checker_ae.py` —
and `wing_figure.json` is the one record written from outside `aeroelastic/`, because the file that
writes it exists to draw a figure.

Re-running any driver rewrites only its own record, in the last digits, for the reasons and to the
tolerances measured in [`docs/reproduce.md`](../../docs/reproduce.md).
