# PLC Research Suite

Generates the figures and per-file data behind a research-level comparison of
three causal concealment methods -- AR-only (LPC), AR+NN (PARCnet), and Sparse
AR (Severi 2025, a from-scratch reimplementation -- see `research/
concealment.py` and `parcnet-is2/sparse_ar_model.py`'s module docstrings for
the algorithm and, importantly, two points where this implementation had to
depart from the published paper's literal description to behave sensibly on
real audio, with the measured evidence for each). This is tooling that
produces figures/CSVs for a report, not the report itself.

## Dataset

Everything here runs on `parcnet-is2/example_test_set`: a real
(clean, lossy, trace) triplet per file, where the trace's loss pattern was
actually applied to that lossy file from that clean file -- genuine ground
truth, not a synthetic or arbitrarily-paired loss model. `research/dataset_utils.py`
loads and matches these by filename.

## Metrics (`research/metrics.py`)

**Physical** (waveform/spectrum error, no model of hearing): SNR and
multi-resolution STFT distance, both at the signal's native sample rate --
no downsampling. SI-SDR and log-spectral distance are also implemented
(still usable directly from `metrics.py`) but dropped from the default
scripts' output: across a 40-file run they tracked SNR and MR-STFT so
closely (same ranking, same win-rate) that reporting both pairs was
redundant, not more informative.

**Psychoacoustic** (models of human hearing):
- PLCMOS (Microsoft, INTERSPEECH 2023) -- reference-free, purpose-built for PLC.
- ViSQOL (Google) -- full-band "audio" mode MOS-LQO, via the pure-Python
  `visqol-python` package (no WSL/Docker required; a native C++ ViSQOL build
  was considered but has no Windows wheel -- the pure-Python reimplementation
  works identically for our purposes and is far simpler to install). ViSQOL
  needs roughly 1.5s of audio to produce a score at all (its patch-based
  algorithm errors out below that) -- too long for individual burst gaps,
  so it (like PLCMOS) is only ever scored on whole files. Per-burst analysis
  uses gap SNR only.

PEAQ (ITU-R BS.1387) was considered and dropped: there is no maintained
Python implementation and the reference tools are MATLAB-only.

## Concealment (`research/concealment.py`)

Three independent functions: `conceal_lpc` (pure AR recursive prediction, no
torch/NN dependency at all), `conceal_nn` (the real production `PARCnet`
class from `parcnet-is2/parcnet.py`, NN always on), and `conceal_sparse_ar`
(a from-scratch implementation of Severi 2025's OMP-based sparse AR
concealer, `parcnet-is2/sparse_ar_model.py`). At the AR level, `conceal_lpc`
and `conceal_nn` were verified bit-identical (MSE = 0.0); `conceal_sparse_ar`
is a separate algorithm and doesn't participate in that check, but reuses the
same per-packet recursive loop and external crossfade so its output is
directly comparable. Every hyperparameter is a plain function argument,
never hardcoded: AR order, diagonal loading, context length and extra/
crossfade dimension for the first two; max. long-term lags, max. lag
searched, Burg order and regularization for Sparse AR (context length and
crossfade dimension are shared by all three).

## Run order

```
python -m research.run_metrics_comparison --n-files 300   # expensive (VM)
python -m research.run_hparam_sweeps --n-files 20        # expensive (VM)
python -m research.plot_burst_comparison                 # a few minutes
python -m research.run_statistical_analysis              # cheap; reads the CSVs above
python -m research.build_report                          # Word report from all of the above
```

`build_report.py` writes `output/report/PLC_LPC_vs_PARCnet_Report.docx` (needs
`pip install python-docx`). All numbers and conclusions in it are read from
the outputs above, so re-running it after new experiments updates the whole
report. It adds a "preliminary results" notice while the main comparison has
fewer than 30 files or a sweep has fewer than 10.

## Scripts

### `run_metrics_comparison.py`
Samples N files (default 300) **stratified by signal complexity** (spectral
flatness quantile buckets, via `dataset_utils.sample_diverse_test_files`) --
a plain random sample clumps wherever this dataset happens to be dense
(mostly tonal/simple content), so stratifying is what actually makes the
"vs. complexity" analysis meaningful. Runs all three methods at default
hyperparameters, computes every metric above, and produces:
- `output/metrics_comparison/per_file_results.csv` -- every metric, plus
  concealment time per method and per concealed packet
- `output/metrics_comparison/per_burst_results.csv` -- gap SNR of every
  method for every real burst in every evaluated file
- `output/metrics_comparison/run_info.json` -- machine and hparams, so timings
  stay traceable to the hardware they were measured on
- `output/metrics_comparison/win_rate_summary.png` -- how often each
  non-baseline method beats LPC (the baseline), per metric
- `output/metrics_comparison/complexity_grid.png` -- quality vs. signal complexity,
  one panel per metric, log-scaled x-axis, with a per-method trend line (linear
  fit in log-x space) so the relationship is readable without eyeballing a dot cloud

### `run_hparam_sweeps.py`
Eight sweeps total, each varying one hyperparameter while holding the others
at their defaults, and each evaluating only the method(s) that hyperparameter
actually affects (see the module's own docstring for the full breakdown):
AR order and diagonal loading (LPC + Hybrid only); context length and
extra/crossfade dim (all three, since Sparse AR shares that structural
plumbing); and max. long-term lags, max. lag searched, Burg order, and
regularization (Sparse AR only -- `--skip-sparse-ar-sweeps` to omit these
four). Every sampled file is concealed whole, against its own real trace --
no synthetic bursts. Measures SNR, MR-STFT, PLCMOS, ViSQOL and compute time;
one five-panel figure plus the raw CSV per sweep, in `output/hparam_sweeps/`.
Burst length is deliberately not swept: in a causal, recursive concealer it
is an outcome of the network, not a design choice.

An earlier synthetic-burst version of the LPC/Hybrid sweeps found that AR
order and context length are not free once the NN is in the loop (moving
them away from the trained defaults made the NN fall below AR-only). That
finding still needs confirming with the whole-file sweeps.

### `plot_burst_comparison.py`
Network-loss statistics for the whole dataset, plus waveform/spectrogram
(3-column, one per method), pairwise-difference (one panel per method pair)
and crossfade-boundary plots of real bursts at chosen lengths.

### `run_statistical_analysis.py`
Reads the CSVs above (no new concealment, except the failure-case plots)
and writes `output/statistical_analysis/`:
1. `significance_tests.csv/.md` -- paired Wilcoxon signed-rank test for
   every (metric, method-pair) combination -- 4 metrics x 3 pairs = 12 tests,
   Holm-corrected together as one family, with rank-biserial effect size and
   a bootstrap 95% CI of the mean difference for each
2. `metric_distributions.png`, `paired_differences.png` -- per-file values
   for all three methods, not just win rates
3. `burst_length_quality.png/.csv` -- gap SNR vs. real burst length, all
   three methods, plus each non-baseline method's advantage over LPC; CIs
   are cluster-bootstrapped over files, because bursts from the same file
   are not independent
4. `realtime_feasibility.png` -- time to conceal one packet divided by the
   packet's duration, for every method; above 1.0 the method can't run live
   on that machine
5. `sweep_summary.csv/.md` -- best value per hyperparameter, method and
   metric; each sweep only lists the methods it actually evaluated
6. `failure_cases/<method>/` -- one subfolder per non-baseline method, with
   plots of the files where that method loses to LPC by the most

## Style

Plain functions over classes, every magic number is a keyword argument with
a documented default, nothing is hardcoded to one file/value -- the scripts
are meant to be re-run with different arguments, not edited.
`research/plotting.py` fixes one color per method (orange=LPC, green=NN,
purple=Sparse AR) reused across every figure in this suite -- LPC/NN match
the colors already used elsewhere in the project (`test_irmas.py`,
`plc_challenge_test.py`); `plot_burst_comparison.py` keeps its own
longer-standing blue/green LPC/NN palette instead (a pre-existing
inconsistency between the two scripts), but shares the same purple for
Sparse AR.
`research/safe_io.py` writes every CSV/PNG via a temp-file-then-rename, since
Windows has repeatedly locked these exact output files mid-run (antivirus/
indexer scans, or the file simply being open in a viewer). This clears most
locks, but not a file held open exclusively (e.g. open in Excel) -- that
still raises, so the run doesn't silently report stale results; close
whatever has the file open and re-run, or save under a different filename.
