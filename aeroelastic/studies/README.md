# studies

The experiments. Each asks a question whose answer is a reported result, writes one record into
[`records/`](../records/README.md), and is written up in [`results/`](../../results/README.md).

← [`aeroelastic/`](../README.md) · how to run any of these:
[`docs/reproduce.md`](../../docs/reproduce.md)

| driver | record | reported in |
|---|---|---|
| [`validity_map.py`](validity_map.py) | [`validity_map.json`](../records/validity_map.json) | the 289-design sweep behind the 31 and the 30 |
| [`validity_regime.py`](validity_regime.py) | [`validity_regime.json`](../records/validity_regime.json) | [validity regime](../../results/validity_regime.md) |
| [`qualification.py`](qualification.py) | [`qualification.json`](../records/qualification.json) | [qualification](../../results/aeroelastic_qualification.md) |
| [`sweep_speed.py`](sweep_speed.py) | [`speed_sweep.json`](../records/speed_sweep.json) | the re-scoring across speed, and the ~57 m/s reversal |
| [`run_served_optimize.py`](run_served_optimize.py) ⚠ | [`served_optimize.json`](../records/served_optimize.json) | [application result](../../results/aeroelastic_wing.md) |

⚠ binds ports 8811 and 8812. Serialise it against the two marked drivers in
[`verification/`](../verification/README.md).

`validity_map.py` writes nothing without `--out PATH`, so a reproduction cannot overwrite the
committed record by accident. Every other driver here rewrites the record it owns.
