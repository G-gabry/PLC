"""
Hyperparameter tradeoff sweeps for all four concealment methods (AR-only
"LPC", AR+NN "Hybrid", Sparse AR, and Endless Sounds -- see
research/concealment.py): how does quality and compute time change as you
vary a single design-choice knob, holding everything else fixed?

Each sweep only evaluates the method(s) its own hyperparameter actually
affects -- never all four for every sweep, since most of these parameters
are specific to one or two methods:

  Shared by LPC and Hybrid only (their common AR fit):
    - AR order              (parcnet-is2/config.yaml default: 256)
    - AR diagonal loading    (regularization term; default: 0.001)

  Shared by all four (structural parameters every method's per-packet loop
  uses, even though Sparse AR's and Endless Sounds' own models inside that
  loop are unrelated):
    - Context length         (packets of past audio fed to the predictor; default: 8)
    - Extra/crossfade dim    (samples of predicted overlap beyond the lost packet
                              used for the fade-in/fade-out splice; default: 256)

  Sparse AR only (its own model, see parcnet-is2/sparse_ar_model.py):
    - max_k_lags             (max. non-contiguous long-term lags; paper default: 3)
    - max_lag_l              (max. lag searched; paper default: 1024)
    - burg_order_pb          (short-term Burg model order; paper default: 8,
                              but see that sweep's own docstring for why this
                              one is swept well past the paper's own cap)
    - lambda_reg              (Tikhonov regularization, as a fraction of r0; paper default: 0.01)

  Endless Sounds only (its own model, see parcnet-is2/endless_sounds_model.py):
    - boundary_blend         (samples of real-to-synthetic equal-power crossfade at
                              the start of each prediction; no paper-given default,
                              see that sweep's docstring)

Evaluating a method against a hyperparameter that doesn't affect it would
waste compute on an identical, flat result every time (e.g. Sparse AR has no
`ar_order` of its own at all), so run_hparam_sweeps.py's sweep functions
each declare which methods they run.

Method: whole-file, real-trace evaluation -- for each swept value, EVERY
sampled file is conceeled against its OWN real lossy/trace (whatever bursts
it naturally contains, whatever real loss rate it has), using exactly
run_metrics_comparison.py's METHODS -- only the swept hyperparameter changes.
This is deliberately NOT a synthetic single-burst injection: an earlier
version of this script cropped a window of clean audio and injected one
controlled burst, which cleanly isolates the hyperparameter's effect but
doesn't reflect what you'd actually see in deployment on real traffic. This
version answers "what do I actually get at this setting on real data", at
the cost of needing to process a whole file (hundreds of packets, each a
possible NN forward pass) per data point instead of a small cropped window
-- meaningfully more compute per point, but no synthetic construction
anywhere.

Burst length is deliberately NOT swept here. Because the system is causal
and recursive, a burst of length N is just the same single-packet prediction
mechanism invoked N times in a row, each time relying on its own prior
output instead of real audio -- the resulting degradation is close to a
direct consequence of causality itself, not an independent design question
like AR order or context length are (which you actually choose). It's also
not something you can meaningfully "sweep" on real data: a real trace's
burst lengths aren't a knob, they're whatever that trace happens to contain.

Nothing is hardcoded: which values to sweep, how many files to average over,
and the fixed defaults for the other hyperparameters are all arguments.

Usage:
    python -m research.run_hparam_sweeps --n-files 20
"""
import os

# Must happen before numpy/scipy/numba are imported by anything (including
# research.dataset_utils below) -- see run_metrics_comparison.py's identical
# block for why: without this, --workers > 1 oversubscribes the machine by
# a factor of workers, since each worker's own BLAS/OpenMP pool otherwise
# tries to use every core on the machine.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import argparse
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from research import dataset_utils, metrics, complexity, plotting, safe_io
from research.run_metrics_comparison import METHODS

OUT_DIR = Path(__file__).resolve().parent / "output" / "hparam_sweeps"

DEFAULT_HPARAMS = dict(ar_order=256, diagonal_load=0.001, context_dim_packets=8, extra_dim=256,
                       max_k_lags=3, max_lag_l=1024, burg_order_pb=8, lambda_reg=0.01,
                       boundary_blend=64, es_seed=0)

CONCEAL_FN = {key: fn for key, _, fn, _ in METHODS}
METHOD_LABEL = {key: label for key, label, _, _ in METHODS}
METHOD_COLOR = {key: color for key, _, _, color in METHODS}

ALL_METRICS = [
    ("snr", "SNR (dB)", False, "mean"),
    ("mrstft", "MR-STFT dist.", True, "mean"),
    ("plcmos", "PLCMOS", False, "mean"),
    ("visqol", "ViSQOL MOS-LQO", False, "mean"),
    ("time_ms", "Compute time, median (ms)", True, "median"),
]


def evaluate_file_with_timing(tf: dataset_utils.TestFile, hparams: dict, methods: list[str]) -> dict:
    """Same metrics as run_metrics_comparison.evaluate_file, plus wall-clock
    timing for each method's single concealment call, for only the given
    `methods`. No repeated warmup/timing runs here (unlike
    concealment.time_conceal_ms): whole-file NN concealment on a real trace
    is expensive enough -- potentially hundreds of forward passes per file
    -- that repeating it 5-7x per file per hyperparameter value, as the old
    cropped-burst sweep did, would be prohibitively expensive at this scale.
    One timing per call is noisier but tractable.
    """
    row = {
        "file": tf.stem,
        "n_packets": tf.n_packets,
        "loss_rate_pct": tf.loss_rate * 100,
        "complexity_flatness": complexity.spectral_flatness(tf.clean),
    }
    for key in methods:
        t0 = time.perf_counter()
        out = CONCEAL_FN[key](tf.lossy, tf.trace, **hparams)
        row[f"time_ms_{key}"] = (time.perf_counter() - t0) * 1000
        row[f"snr_{key}"] = metrics.snr_db(tf.clean, out)
        row[f"mrstft_{key}"] = metrics.multires_stft_distance(tf.clean, out)
        row[f"plcmos_{key}"] = metrics.plcmos_score(out, tf.sample_rate)
        row[f"visqol_{key}"] = metrics.visqol_score(tf.clean, out, tf.sample_rate)
    return row


def _init_worker(single_thread: bool) -> None:
    if single_thread:
        import torch
        torch.set_num_threads(1)


def _evaluate_task(task) -> dict:
    stem, hparams, param_name, value, methods = task
    row = evaluate_file_with_timing(dataset_utils.load_test_file(stem), hparams, methods)
    row["param"] = param_name
    row["value"] = value
    return row


def sweep_hparam(stems: list[str], param_name: str, values: list, apply_value, methods: list[str],
                 workers: int = 1, checkpoint: safe_io.Checkpoint | None = None) -> pd.DataFrame:
    """Whole-file, real-trace sweep: `apply_value(value)` maps one swept
    value to the full hparams dict to evaluate every file with, and
    `methods` is the (possibly 1-element) subset of METHODS keys this
    hyperparameter actually affects. Every (value, file) pair is an
    independent task, spread over `workers` processes. With a checkpoint,
    each finished pair is saved at once and pairs already in it are not
    recomputed.
    """
    if checkpoint is None:
        checkpoint = safe_io.Checkpoint(None)
    key = lambda value, stem: f"{param_name}|{value!r}|{stem}"
    todo = [(stem, apply_value(value), param_name, value, methods) for value in values for stem in stems
            if key(value, stem) not in checkpoint]
    left = {value: sum(key(value, stem) not in checkpoint for stem in stems) for value in values}
    if len(todo) < len(values) * len(stems):
        print(f"  {param_name}: resuming, {len(values) * len(stems) - len(todo)} evaluations already saved, "
              f"{len(todo)} to go", flush=True)

    results = dataset_utils.resilient_map(_evaluate_task, todo, workers,
                                          initializer=_init_worker, initargs=(workers > 1,))
    for task_index, row in results:
        stem, _, _, value, _ = todo[task_index]
        checkpoint.add(key(value, stem), row)
        left[value] -= 1
        if left[value] == 0:
            print(f"  {param_name} = {value}: done ({len(stems)} files, {', '.join(methods)})", flush=True)
    return pd.DataFrame([checkpoint.records[key(value, stem)] for value in values for stem in stems])


# --------------------------------------------------------------------------
# LPC + Hybrid only (their shared AR fit)
# --------------------------------------------------------------------------
def sweep_ar_order(stems, values=(32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 3840),
                   workers=1, checkpoint=None):
    """Pushed to 3840 -- right up against the hard structural limit (order
    must stay below the AR context length in samples, 4096 here with the
    default context_dim_packets=8). Verified empirically (in an earlier
    version of this sweep) that nothing crashes even this close to the
    limit, and that SNR plateaus near 0 dB by ~768-1024. Sparse AR has no
    `ar_order` of its own, so it's not evaluated here.
    """
    return sweep_hparam(stems, "ar_order", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "ar_order": v}, ["lpc", "nn"], workers, checkpoint)


def sweep_diagonal_load(stems, values=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0), workers=1, checkpoint=None):
    """Sparse AR has its own, separate Tikhonov regularization (lambda_reg,
    see sweep_lambda_reg below) and is not evaluated here."""
    return sweep_hparam(stems, "diagonal_load", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "diagonal_load": v}, ["lpc", "nn"], workers, checkpoint)


# --------------------------------------------------------------------------
# Shared by all three methods (structural, not model-specific)
# --------------------------------------------------------------------------
def sweep_context_length(stems, values=(1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256), workers=1, checkpoint=None):
    """Pushed to 256 packets (~3s of context for predicting one packet) to
    see whether the plateau past the trained default (8) is a hard ceiling
    or keeps slowly drifting. Genuinely shared: this is the context window
    LPC/Hybrid fit their AR model on, and the same window Sparse AR fits its
    Burg + long-term lag model on.
    """
    return sweep_hparam(stems, "context_length_packets", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "context_dim_packets": v},
                         ["lpc", "nn", "sparse_ar"], workers, checkpoint)


def sweep_extra_dim(stems, values=(16, 32, 64, 128, 192, 256, 384, 512, 768, 1024), workers=1, checkpoint=None):
    """0 is deliberately excluded: the production PARCnet class (parcnet.py)
    has a latent edge-case bug there -- `prediction[-self.extra_dim:]` with
    extra_dim=0 slices as `[-0:]`, which Python/NumPy treats as the *whole*
    array rather than empty, so it crashes multiplying against an
    empty fade_out. conceal_lpc guards this case (`if extra_dim > 0`) but
    the real NN class doesn't, and patching production code for a degenerate
    "no crossfade at all" config is out of scope here -- 16 is the lowest
    value that exercises a genuinely short crossfade without hitting it.
    Values above the trained default (256) test whether a longer predicted
    overlap keeps helping or starts hurting once the predictor is
    extrapolating well past the lost packet's own length (512 samples).
    Genuinely shared: all three methods use the same external crossfade
    logic over this many samples (research/concealment.py).
    """
    return sweep_hparam(stems, "extra_dim", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "extra_dim": v},
                         ["lpc", "nn", "sparse_ar"], workers, checkpoint)


# --------------------------------------------------------------------------
# Sparse AR only (its own model -- see parcnet-is2/sparse_ar_model.py)
# --------------------------------------------------------------------------
def sweep_max_k_lags(stems, values=(1, 2, 3, 4, 5, 6), workers=1, checkpoint=None):
    """Paper default is 3. Tests whether allowing more long-term lags
    recovers any of the quality gap found in the main comparison (see
    sparse_ar_model.py's module docstring), or whether the bottleneck is
    elsewhere (the short-term Burg stage decaying before a long-term lag is
    even found -- see that same docstring).
    """
    return sweep_hparam(stems, "max_k_lags", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "max_k_lags": v}, ["sparse_ar"], workers, checkpoint)


def sweep_max_lag_l(stems, values=(128, 256, 512, 1024, 2048), workers=1, checkpoint=None):
    """Paper default is 1024. Kept below the context window in samples
    (context_dim_packets * packet_dim, 4096 at the shared default) so a
    candidate lag can never reach past the real context available to fit it.
    """
    return sweep_hparam(stems, "max_lag_l", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "max_lag_l": v}, ["sparse_ar"], workers, checkpoint)


def sweep_burg_order_pb(stems, values=(2, 4, 8, 16, 32, 64, 128), workers=1, checkpoint=None):
    """Paper caps this at <=8 (it's meant to be a short, cheap lead-in, not
    the main predictor). Swept well past that cap because of a concrete,
    measured finding during this implementation: with this project's
    continuous short-term + long-term cascade (see sparse_ar_model.py's
    module docstring -- the paper's own two-phase Burg-then-SAR split had to
    be replaced with one), the order-8 short-term filter recursed over a
    whole packet decays to numerical silence well before the prediction
    ends, and a one-off check found order 64 scored noticeably better (0.37
    dB vs. -0.05 dB on one test burst) than order 8. This sweep turns that
    single spot-check into a proper measurement across real files instead
    of a hand-picked number, and reports honestly whether going past the
    paper's own stated range actually helps.
    """
    return sweep_hparam(stems, "burg_order_pb", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "burg_order_pb": v}, ["sparse_ar"], workers, checkpoint)


def sweep_lambda_reg(stems, values=(0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0), workers=1, checkpoint=None):
    """Paper default is 0.01 (as a fraction of r0, the zero-lag
    autocorrelation/residual energy). 0.0 (no Tikhonov regularization at
    all) is included to see whether the regularized re-optimization
    (equation 3) actually matters, or whether the skip radius around chosen
    lags already keeps the Gram matrix well-conditioned on its own."""
    return sweep_hparam(stems, "lambda_reg", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "lambda_reg": v}, ["sparse_ar"], workers, checkpoint)


def sweep_boundary_blend(stems, values=(0, 16, 32, 64, 128, 256, 512), workers=1, checkpoint=None):
    """Endless Sounds' only quality-affecting hyperparameter: how many samples
    the real-to-synthetic equal-power crossfade spans at the start of each
    prediction (see parcnet-is2/endless_sounds_model.py). The paper
    establishes that this blend exists (its Section III-C) but never gives it
    a number, so the default used here (64) is this project's own choice --
    which is exactly why it is worth sweeping rather than asserting. 0 is
    included as the degenerate "no blend at all" case, to show what the
    crossfade is actually buying."""
    return sweep_hparam(stems, "boundary_blend", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "boundary_blend": v},
                         ["endless_sounds"], workers, checkpoint)


def plot_sweep(df: pd.DataFrame, title: str, xlabel: str, out_path: Path, x_log: bool = False) -> Path:
    """One figure per sweep, one panel per metric (SNR, MR-STFT, PLCMOS,
    ViSQOL, compute time) vs. the swept value -- five separate panels, never
    combined into one dual-axis chart, since none of these share a scale.
    Only plots the methods actually present in `df` (one line each).

    SNR/MR-STFT/PLCMOS/ViSQOL use mean +/-1 standard deviation across files.
    Compute time uses median with an IQR (25th-75th percentile) band instead:
    wall-clock timing on a whole real file is vulnerable to one-off OS stalls
    (antivirus/indexer activity, observed directly in this project), and
    mean+-std is not robust to that kind of heavy-tailed noise.
    """
    methods = [k for k in CONCEAL_FN if f"snr_{k}" in df.columns]
    agg_spec = {}
    for key, _, _, stat in ALL_METRICS:
        for method in methods:
            col = f"{key}_{method}"
            if stat == "mean":
                agg_spec[f"{col}_mean"] = (col, "mean")
                agg_spec[f"{col}_std"] = (col, "std")
            else:
                agg_spec[f"{col}_median"] = (col, "median")
                agg_spec[f"{col}_q25"] = (col, lambda s: s.quantile(0.25))
                agg_spec[f"{col}_q75"] = (col, lambda s: s.quantile(0.75))
    summary = df.groupby("value").agg(**agg_spec).reset_index().sort_values("value")

    fig, axes = plotting.new_axes(figsize=(5, 4), ncols=3, nrows=2)
    for ax, (key, ylabel, lower_is_better, stat) in zip(axes.flat, ALL_METRICS):
        for method in methods:
            col = f"{key}_{method}"
            color, label = METHOD_COLOR[method], METHOD_LABEL[method]
            if stat == "mean":
                ax.errorbar(summary["value"], summary[f"{col}_mean"], yerr=summary[f"{col}_std"],
                            color=color, marker="o", markersize=4, linewidth=1.8, capsize=3,
                            elinewidth=1, alpha=0.9, label=label)
            else:
                lower = summary[f"{col}_median"] - summary[f"{col}_q25"]
                upper = summary[f"{col}_q75"] - summary[f"{col}_median"]
                ax.errorbar(summary["value"], summary[f"{col}_median"], yerr=[lower, upper],
                            color=color, marker="o", markersize=4, linewidth=1.8, capsize=3,
                            elinewidth=1, alpha=0.9, label=label)
        if x_log:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        plotting.style_axis(ax, grid_axis="both")
    axes.flat[0].legend(fontsize=8, frameon=False)
    axes.flat[-1].axis("off")  # 6th grid cell unused (5 metrics)
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    actual_path = safe_io.save_fig_safe(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return actual_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-files", type=int, default=20,
                         help="How many files to average each sweep point over (whole-file real-trace "
                              "evaluation is expensive per file, so this defaults lower than "
                              "run_metrics_comparison.py's 40).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel processes. Use about the number of physical cores. Compute-time "
                             "results are then single-threaded and measured under load.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore results saved by an earlier (possibly crashed) run and start over.")
    parser.add_argument("--skip-sparse-ar-sweeps", action="store_true",
                        help="Skip the four Sparse-AR-only sweeps (max_k_lags, max_lag_l, burg_order_pb, "
                             "lambda_reg), running only the original LPC/Hybrid sweeps.")
    parser.add_argument("--skip-endless-sounds-sweeps", action="store_true",
                        help="Skip the Endless-Sounds-only sweep (boundary_blend).")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated sweep parameter name(s) to run instead of all of them, e.g. "
                             "'lambda_reg' or 'ar_order,context_length_packets'.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = safe_io.Checkpoint(args.out_dir / "checkpoint.jsonl", fresh=args.fresh)

    stems = dataset_utils.select_diverse_stems(args.n_files, seed=args.seed, workers=args.workers)
    print(f"Sampled {len(stems)} files from example_test_set (seed={args.seed}), "
          f"stratified by spectral flatness for genuine content diversity.\n", flush=True)

    sweep_fns = [
        ("AR order", "ar_order", sweep_ar_order, False),
        ("Context length", "context_length_packets", sweep_context_length, False),
        ("AR diagonal loading", "diagonal_load", sweep_diagonal_load, True),
        ("Extra/crossfade dim", "extra_dim", sweep_extra_dim, False),
    ]
    if not args.skip_sparse_ar_sweeps:
        sweep_fns += [
            ("Sparse AR: max lags (k)", "max_k_lags", sweep_max_k_lags, False),
            ("Sparse AR: max lag (L)", "max_lag_l", sweep_max_lag_l, False),
            ("Sparse AR: Burg order", "burg_order_pb", sweep_burg_order_pb, False),
            # x_log=False: the sweep now includes 0.0, which can't be placed on a log axis.
            ("Sparse AR: regularization", "lambda_reg", sweep_lambda_reg, False),
        ]
    if not args.skip_endless_sounds_sweeps:
        sweep_fns += [
            ("Endless Sounds: boundary blend", "boundary_blend", sweep_boundary_blend, False),
        ]
    if args.only:
        all_names = [s[1] for s in sweep_fns]
        wanted = {p.strip().lower() for p in args.only.split(",")}
        unknown = wanted - {n.lower() for n in all_names}
        if unknown:
            raise SystemExit(f"--only: unknown sweep name(s) {sorted(unknown)}. Valid names: {all_names}")
        sweep_fns = [s for s in sweep_fns if s[1].lower() in wanted]

    for title, param_name, sweep_fn, x_log in sweep_fns:
        df = sweep_fn(stems, workers=args.workers, checkpoint=checkpoint)
        csv_path = args.out_dir / f"sweep_{param_name}.csv"
        png_path = args.out_dir / f"sweep_{param_name}.png"
        actual_csv_path = safe_io.save_csv_safe(df, csv_path, index=False)
        actual_png_path = plot_sweep(df, f"Tradeoff: {title}", param_name, png_path, x_log=x_log)
        print(f"\n{title}: saved {actual_csv_path.name} and {actual_png_path.name}")
        if actual_csv_path != csv_path or actual_png_path != png_path:
            print(f"  (NOTE: {csv_path.name}/{png_path.name} are locked by something -- "
                  f"saved under a fallback name instead; close whatever has them open)")

    print(f"\nAll sweeps saved to {args.out_dir}")


if __name__ == "__main__":
    main()
