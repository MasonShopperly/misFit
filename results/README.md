# results

One evidence report per result. Each states its record, the driver that produced it, and what it
does not show.

← [repository root](../README.md) · interpretation and mathematics:
[`technical_writeup.md`](../technical_writeup.md)

| question | report | record |
|---|---|---|
| Where does the cheap rigid gradient stop being a safe basis for design, and what sets that boundary? | [`validity_regime.md`](validity_regime.md) | [`validity_regime.json`](../aeroelastic/records/validity_regime.json) |
| Can a cheap operating-point heuristic predict that boundary? | [`aeroelastic_qualification.md`](aeroelastic_qualification.md) | [`qualification.json`](../aeroelastic/records/qualification.json) |
| What does using the cheap gradient cost on the actual design problem? | [`aeroelastic_wing.md`](aeroelastic_wing.md) | [`served_optimize.json`](../aeroelastic/records/served_optimize.json) |
| Can the platform's own checker be asked about the *assembled* objective? | [`native_composed_check.md`](native_composed_check.md) | [`native_composed_check.json`](../aeroelastic/records/native_composed_check.json) |

The headline is the first: 31 of 289 designs fail the step criterion, 30 of them uphill, and the
inversion needs both strong coupling and proximity to the coupled optimum rather than either alone.

Every number in these reports is recomputed from a committed record rather than retyped, except the
few marked *console; not persisted* where the producing script prints a value it does not store. How
to re-run any of them: [`docs/reproduce.md`](../docs/reproduce.md).
