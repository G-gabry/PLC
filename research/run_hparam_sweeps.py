"""
Hyperparameter tradeoff sweeps for AR-only (LPC) vs AR+NN (PARCnet)
concealment: how does quality and compute time change as you vary a single
design-choice knob, holding everything else fixed?

Four sweeps, each isolating one hyperparameter:
  - AR order              (parcnet-is2/config.yaml default: 256)
  - Context length        (packets of past audio fed to the predictor; default: 8)
  - AR diagonal loading   (regularization term; default: 0.001)
  - Extra/crossfade dim   (samples of predicted overlap beyond the lost packet
                            used for the fade-in/fade-out splice; parcnet-is2/
                            config.yaml default: 256)

Method: whole-file, real-trace evaluation -- for each swept value, EVERY
sampled file is conceeled against its OWN real lossy/trace (whatever bursts
it naturally contains, whatever real loss rate it has) by both methods,
using exactly run_metrics_comparison.py's evaluate_file() -- only the swept
hyperparameter changes. This is deliberately NOT a synthetic single-burst
injection: an earlier version of this script cropped a window of clean audio
and injected one controlled burst, which cleanly isolates the hyperparameter's
effect but doesn't reflect what you'd actually see in deployment on real
traffic. This version answers "what do I actually get at this setting on
real data", at the cost of needing to process a whole file (hundreds of
packets, each a possible NN forward pass) per data point instead of a small
cropped window -- meaningfully more compute per point, but no synthetic
construction anywhere.

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

from research import dataset_utils, concealment, metrics, complexity, plotting, safe_io

OUT_DIR = Path(__file__).resolve().parent / "output" / "hparam_sweeps"

DEFAULT_HPARAMS = dict(ar_order=256, diagonal_load=0.001, context_dim_packets=8, extra_dim=256)

ALL_METRICS = [
    ("snr", "SNR (dB)", False, "mean"),
    ("mrstft", "MR-STFT dist.", True, "mean"),
    ("plcmos", "PLCMOS", False, "mean"),
    ("visqol", "ViSQOL MOS-LQO", False, "mean"),
    ("time_ms", "Compute time, median (ms)", True, "median"),
]


def evaluate_file_with_timing(tf: dataset_utils.TestFile, hparams: dict) -> dict:
    """Same metrics as run_metrics_comparison.evaluate_file, plus wall-clock
    timing for each method's single concealment call. No repeated warmup/
    timing runs here (unlike concealment.time_conceal_ms): whole-file NN
    concealment on a real trace is expensive enough -- potentially hundreds
    of forward passes per file -- that repeating it 5-7x per file per
    hyperparameter value, as the old cropped-burst sweep did, would be
    prohibitively expensive at this scale. One timing per call is noisier
    but tractable.
    """
    t0 = time.perf_counter()
    this_lpc = concealment.conceal_lpc(tf.lossy, tf.trace, **hparams)
    time_lpc = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    this_nn = concealment.conceal_nn(tf.lossy, tf.trace, **hparams)
    time_nn = (time.perf_counter() - t0) * 1000

    return {
        "file": tf.stem,
        "n_packets": tf.n_packets,
        "loss_rate_pct": tf.loss_rate * 100,
        "complexity_flatness": complexity.spectral_flatness(tf.clean),
        "snr_lpc": metrics.snr_db(tf.clean, this_lpc), "snr_nn": metrics.snr_db(tf.clean, this_nn),
        "mrstft_lpc": metrics.multires_stft_distance(tf.clean, this_lpc),
        "mrstft_nn": metrics.multires_stft_distance(tf.clean, this_nn),
        "plcmos_lpc": metrics.plcmos_score(this_lpc, tf.sample_rate),
        "plcmos_nn": metrics.plcmos_score(this_nn, tf.sample_rate),
        "visqol_lpc": metrics.visqol_score(tf.clean, this_lpc, tf.sample_rate),
        "visqol_nn": metrics.visqol_score(tf.clean, this_nn, tf.sample_rate),
        "time_ms_lpc": time_lpc, "time_ms_nn": time_nn,
    }


def _init_worker(single_thread: bool) -> None:
    if single_thread:
        import torch
        torch.set_num_threads(1)


def _evaluate_task(task) -> dict:
    stem, hparams, param_name, value = task
    row = evaluate_file_with_timing(dataset_utils.load_test_file(stem), hparams)
    row["param"] = param_name
    row["value"] = value
    return row


def sweep_hparam(stems: list[str], param_name: str, values: list, apply_value, workers: int = 1,
                 checkpoint: safe_io.Checkpoint | None = None) -> pd.DataFrame:
    """Whole-file, real-trace sweep: `apply_value(value)` maps one swept
    value to the full hparams dict to evaluate every file with. Every
    (value, file) pair is an independent task, spread over `workers`
    processes. With a checkpoint, each finished pair is saved at once and
    pairs already in it are not recomputed.
    """
    if checkpoint is None:
        checkpoint = safe_io.Checkpoint(None)
    key = lambda value, stem: f"{param_name}|{value!r}|{stem}"
    todo = [(stem, apply_value(value), param_name, value) for value in values for stem in stems
            if key(value, stem) not in checkpoint]
    left = {value: sum(key(value, stem) not in checkpoint for stem in stems) for value in values}
    if len(todo) < len(values) * len(stems):
        print(f"  {param_name}: resuming, {len(values) * len(stems) - len(todo)} evaluations already saved, "
              f"{len(todo)} to go", flush=True)

    results = dataset_utils.resilient_map(_evaluate_task, todo, workers,
                                          initializer=_init_worker, initargs=(workers > 1,))
    for task_index, row in results:
        stem, _, _, value = todo[task_index]
        checkpoint.add(key(value, stem), row)
        left[value] -= 1
        if left[value] == 0:
            print(f"  {param_name} = {value}: done ({len(stems)} files)", flush=True)
    return pd.DataFrame([checkpoint.records[key(value, stem)] for value in values for stem in stems])


def sweep_ar_order(files, values=(32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 3840), workers=1, checkpoint=None):
    """Pushed to 3840 -- right up against the hard structural limit (order
    must stay below the AR context length in samples, 4096 here with the
    default context_dim_packets=8). Verified empirically (in the earlier
    synthetic-burst version of this sweep) that nothing crashes even this
    close to the limit, and that SNR plateaus near 0 dB by ~768-1024.
    """
    return sweep_hparam(files, "ar_order", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "ar_order": v}, workers, checkpoint)


def sweep_context_length(files, values=(1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256), workers=1, checkpoint=None):
    """Pushed to 256 packets (~3s of context for predicting one packet) to
    see whether the plateau past the trained default (8) is a hard ceiling
    or keeps slowly drifting.
    """
    return sweep_hparam(files, "context_length_packets", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "context_dim_packets": v}, workers, checkpoint)


def sweep_diagonal_load(files, values=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0), workers=1, checkpoint=None):
    return sweep_hparam(files, "diagonal_load", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "diagonal_load": v}, workers, checkpoint)


def sweep_extra_dim(files, values=(16, 32, 64, 128, 192, 256, 384, 512, 768, 1024), workers=1, checkpoint=None):
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
    """
    return sweep_hparam(files, "extra_dim", list(values),
                         lambda v: {**DEFAULT_HPARAMS, "extra_dim": v}, workers, checkpoint)


def plot_sweep(df: pd.DataFrame, title: str, xlabel: str, out_path: Path, x_log: bool = False) -> Path:
    """One figure per sweep, one panel per metric (SNR, MR-STFT, PLCMOS,
    ViSQOL, compute time) vs. the swept value -- five separate panels, never
    combined into one dual-axis chart, since none of these share a scale.

    SNR/MR-STFT/PLCMOS/ViSQOL use mean +/-1 standard deviation across files.
    Compute time uses median with an IQR (25th-75th percentile) band instead:
    wall-clock timing on a whole real file is vulnerable to one-off OS stalls
    (antivirus/indexer activity, observed directly in this project), and
    mean+-std is not robust to that kind of heavy-tailed noise.
    """
    agg_spec = {}
    for key, _, _, stat in ALL_METRICS:
        for method in ("lpc", "nn"):
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
        for method, color, label in [("lpc", plotting.LPC_COLOR, "This LPC"), ("nn", plotting.NN_COLOR, "This NN")]:
            col = f"{key}_{method}"
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
