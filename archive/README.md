# Archive

Superseded exploratory work from earlier in the project, kept for provenance
-- none of this is deleted, but **none of it is the one to use going
forward**. The current, general, validated comparison tooling lives in
`research/` (code) and `report/` (curated results). Everything here predates
that and duplicates what it does, in less general/parameterized ways.

## What's here and why it was superseded

### `plc_analysis/`
`PLC_Analysis.py` (from `parcnet-is2/pretrained_models/`) -- an early
standalone LPC-vs-NN comparison script with its own `LPCConcealer` class.
Superseded by `research/concealment.py`, which wraps the real production
`PARCnet` class directly instead of reimplementing AR-only logic separately.
`analysis_output/` is its figure/audio output.

### `irmas_tests/`
`test_irmas.py` and `test_irmas_batch.py` -- used `live_sim_engine.py`'s
stepper classes (`LiveConcealer`/`LiveConcealerNN`) to test against IRMAS
files paired with real captured traces. Validated bit-identical to the real
`PARCnet` class at the time, but it's a second code path, not the general
one. Superseded by `research/run_metrics_comparison.py`, which calls
`PARCnet` directly and runs on the native `parcnet-is2/example_test_set`
(genuine clean/lossy/trace ground truth) instead of an external dataset
paired with an arbitrary trace. `irmas_test_output/` and
`irmas_batch_output/` are their results -- the most polished individual
burst waveform/spectrogram figures were copied into
`report/figures/example_bursts/` since the current `research/` scripts only
produce aggregate charts, not single-burst visualizations.

### `notebook/`
`plc_challenge.py` -- a copy of the Colab notebook source (the original
stays in `~/Downloads`, still the live upload source for Colab).
`plc_challenge_test.py` -- a locally-path-adjusted working copy that was
rescued from a temp scratchpad directory (it would have been lost
otherwise). Both were notebook-style exploration scripts; `plc_test_figures/`
and `plc_test_audio/` are their output. Superseded by the same
`research/run_metrics_comparison.py` / `run_hparam_sweeps.py` the IRMAS
tests above were superseded by.

### `ar_model_diagnostics/`
`ar_model_output/` -- figures from `parcnet-is2/ar_model.py`'s own
`__main__` block (a one-off burst-comparison diagnostic). Note:
**`ar_model.py` itself is NOT archived** -- it's production code, used by
`parcnet.py` and therefore by `research/concealment.py` too. Only this one
script's example output plots are here, superseded by
`research/run_hparam_sweeps.py`'s burst-length sweep.

### `configs/`
`config_ar_only.yaml` -- a separate config file for running AR-only mode,
made redundant once `PARCnet` gained a `disable_nn` constructor flag (one
class, one config, both modes).

## What's NOT here (intentionally)

- `parcnet-is2/parcnet.py`, `ar_model.py`, `nn_model.py`, `data.py`,
  `loss.py`, `metrics.py`, `train.py`, `inference.py` -- all production code,
  still in active use.
- `live_sim_engine.py`, `server.py`, `static/` -- the live web simulator, a
  separate interactive tool, not report/comparison material. Out of scope
  for this cleanup entirely.
