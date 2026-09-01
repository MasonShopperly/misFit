# figures

The five published figures: what each is drawn from, what draws it, and where it appears.

← [`docs/`](../README.md) · what the figures mean:
[`technical_writeup.md`](../../technical_writeup.md)

| asset | record(s) | generator | embedded in |
|---|---|---|---|
| [`aeroelastic_hero.png`](aeroelastic_hero.png) | [`served_optimize`](../../aeroelastic/records/served_optimize.json), [`platform_checker_ae`](../../aeroelastic/records/platform_checker_ae.json), [`native_composed_check`](../../aeroelastic/records/native_composed_check.json) | [`make_media.py`](../../scripts/make_media.py) | [README](../../README.md), [writeup §4.3](../../technical_writeup.md#43-design-consequence) |
| [`validity_regime.png`](validity_regime.png) | [`validity_regime`](../../aeroelastic/records/validity_regime.json) | [`make_media.py`](../../scripts/make_media.py) | [README](../../README.md), [writeup §4.1](../../technical_writeup.md#41-validity-boundary) |
| [`architecture.png`](architecture.png) | drawn, not measured | [`make_media.py`](../../scripts/make_media.py) | [README](../../README.md), [writeup §5](../../technical_writeup.md#5-why-tesseract) |
| [`aeroelastic_morph.gif`](aeroelastic_morph.gif) | [`served_optimize`](../../aeroelastic/records/served_optimize.json); every frame a real coupled solve | [`make_media.py`](../../scripts/make_media.py) | [README](../../README.md) |
| [`aeroelastic_wings.png`](aeroelastic_wings.png) | [`served_optimize`](../../aeroelastic/records/served_optimize.json), [`wing_figure`](../../aeroelastic/records/wing_figure.json) | [`make_wing_figure.py`](../../scripts/make_wing_figure.py) | [application result](../../results/aeroelastic_wing.md) |

```bash
.venv/bin/python scripts/make_media.py            # the first four
.venv/bin/python scripts/make_wing_figure.py      # aeroelastic_wings.png
```

Re-rendering returns two of the five byte-identical and changes three without moving any plotted
quantity; the pixel counts and the reason are in [`docs/reproduce.md`](../reproduce.md).

## Accessibility

Each figure carries a full text description in the alt text where it is embedded. Two sets of
quantities are rendered *into* the images and stated nowhere else in text, so they are given here:

- `aeroelastic_hero.png`, bottom panel — `check-gradients` failures of checks: **0 of 9,000** for
  the aerodynamic model against itself, **0 of 5,994** for the structural model, **462 of 540** for
  the aerodynamic gradient against the coupled aircraft. The first two sum to the 14,994 reported
  elsewhere, and all three are read from the records above rather than retyped.
- `aeroelastic_wings.png`, middle panel — built-in twist runs about **+2.8°** at the root to
  **−5.7°** at the tip; in flight the same wing sits near **+2.8°** at the root and about **−1.1°**
  at the tip. That ~4.6° of elastic wash-in at the tip is the effect the study is about.
