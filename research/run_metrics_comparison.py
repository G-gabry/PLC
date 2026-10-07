"""
Research-level comparison of AR-only (LPC) vs AR+NN (PARCnet) concealment,
using the real parcnet-is2/example_test_set dataset -- every (clean, lossy,
trace) triplet there is the genuine ground truth, not a synthetic/arbitrary
pairing, so this is the most rigorous dataset available in this project.

Two families of metrics are computed for every file, matching what a PLC
paper would report:

  PHYSICAL (waveform/spectrum error -- see research/metrics.py docstring):
    SNR, SI-SDR, log-spectral distance (LSD), multi-resolution STFT distance

  PSYCHOACOUSTIC (models of human hearing):
    PLCMOS (Microsoft, reference-free), ViSQOL (Google, full-band "audio" mode)

Nothing here is fixed to one file or one hyperparameter setting -- how many
files, which hyperparameters, and where results are saved are all arguments.
Produces:
  - a per-file CSV with every metric for both methods, plus wall-clock
    concealment time per method (and per concealed packet)
  - a per-burst CSV: for every real burst in every evaluated file, the SNR
    of each method measured only inside that gap -- the input to
    run_statistical_analysis.py's quality-vs-burst-length analysis, collected
    here so the expensive NN pass runs once, not twice
  - run_info.json: machine/hparam details, so timing results are always
    traceable to the hardware they were measured on
  - a win-rate summary chart across all 4 metrics
  - a signal-complexity grid: for each metric, how LPC vs NN quality varies
    with the clean signal's spectral flatness (low = tonal/simple, high =
    noise-like/complex) -- a continuous, measured axis instead of a manual
    genre/instrument label

Usage:
    python -m research.run_metrics_comparison --n-files 40 --seed 0
"""
import os

# Must happen before numpy/scipy/numba are imported by anything (including
# research.dataset_utils below): each one spawns its own BLAS/OpenMP thread
# pool sized to the machine's full core count unless told otherwise. With
# --workers > 1, that means every one of N worker processes independently
# tries to use all cores for its linear algebra (the AR model's
# autocorrelation fit), so N processes oversubscribe the machine by a factor
# of N -- observed on a 32-vCPU VM as a load average above 400 and each
# worker taking 20+ minutes of CPU time to finish one file. Each worker is
# already a unit of parallelism (one file at a time); the math inside it
# should run single-threaded.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from research import dataset_utils, concealment, metrics, complexity, plotting, safe_io

OUT_DIR = Path(__file__).resolve().parent / "output" / "metrics_comparison"

# (column prefix, display label, lower_is_better)
# SI-SDR and LSD were dropped: across 40 files they tracked SNR and MR-STFT so
# closely (same win-rate, same ranking) that they added clutter, not
# information -- metrics.py still has them if a future analysis wants them.
ALL_METRICS = [
    ("snr", "SNR (dB)", False),
    ("mrstft", "MR-STFT dist.", True),
    ("plcmos", "PLCMOS", False),
    ("visqol", "ViSQOL MOS-LQO", False),
]
PHYSICAL_KEYS = {"snr", "mrstft"}


def warmup(hparams: dict) -> None:
    """Loads the NN checkpoint and runs one throwaway forward pass per method
    before anything is timed -- otherwise the first file's NN time would
    include checkpoint loading and torch's first-call overhead."""
    dummy = np.zeros(concealment.PACKET_SIZE * 16, dtype=np.float32)
    dummy_trace = np.zeros(16, dtype=int)
    dummy_trace[10] = 1
    concealment.conceal_lpc(dummy, dummy_trace, **hparams)
    concealment.conceal_nn(dummy, dummy_trace, **hparams)


def burst_rows(tf: dataset_utils.TestFile, this_lpc: np.ndarray, this_nn: np.ndarray) -> list[dict]:
    """Gap-local quality for every real burst in the file: SNR measured only
    over the lost samples. Waveform SNR is the only metric here because the
    perceptual models (PLCMOS, ViSQOL) need seconds of audio, not a
    single-packet gap. The clean gap's level is stored so near-silent gaps,
    where SNR is meaningless, can be filtered out later."""
    rows = []
    for start, length in dataset_utils.find_bursts(tf.trace):
        a, b = start * concealment.PACKET_SIZE, (start + length) * concealment.PACKET_SIZE
        clean_gap = tf.clean[a:b]
        rows.append({
            "file": tf.stem,
            "start_packet": start,
            "burst_length": length,
            "clean_gap_rms_db": float(20 * np.log10(np.sqrt(np.mean(clean_gap ** 2)) + 1e-12)),
            "gap_snr_lpc": metrics.snr_db(clean_gap, this_lpc[a:b]),
            "gap_snr_nn": metrics.snr_db(clean_gap, this_nn[a:b]),
        })
    return rows


def evaluate_file(tf: dataset_utils.TestFile, hparams: dict) -> tuple[dict, list[dict]]:
    """Runs both concealment methods on one file (timed), computes every
    metric, and returns (per-file row, per-burst rows)."""
    t0 = time.perf_counter()
    this_lpc = concealment.conceal_lpc(tf.lossy, tf.trace, **hparams)
    time_lpc = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    this_nn = concealment.conceal_nn(tf.lossy, tf.trace, **hparams)
    time_nn = (time.perf_counter() - t0) * 1000

    n_lost = int(tf.trace.sum())
    row = {
        "file": tf.stem,
        "n_packets": tf.n_packets,
        "n_lost_packets": n_lost,
        "sample_rate": tf.sample_rate,
        "loss_rate_pct": tf.loss_rate * 100,
        "complexity_flatness": complexity.spectral_flatness(tf.clean),
        "time_ms_lpc": time_lpc,
        "time_ms_nn": time_nn,
        # Both methods only do work on lost packets, so total time / lost
        # packets is the cost of concealing one packet.
        "ms_per_lost_packet_lpc": time_lpc / n_lost if n_lost else np.nan,
        "ms_per_lost_packet_nn": time_nn / n_lost if n_lost else np.nan,
        "snr_lpc": metrics.snr_db(tf.clean, this_lpc),
        "snr_nn": metrics.snr_db(tf.clean, this_nn),
        "mrstft_lpc": metrics.multires_stft_distance(tf.clean, this_lpc),
        "mrstft_nn": metrics.multires_stft_distance(tf.clean, this_nn),
        "plcmos_lpc": metrics.plcmos_score(this_lpc, tf.sample_rate),
        "plcmos_nn": metrics.plcmos_score(this_nn, tf.sample_rate),
        "visqol_lpc": metrics.visqol_score(tf.clean, this_lpc, tf.sample_rate),
        "visqol_nn": metrics.visqol_score(tf.clean, this_nn, tf.sample_rate),
    }
    return row, burst_rows(tf, this_lpc, this_nn)


def init_worker(hparams: dict, single_thread: bool) -> None:
    """Per-process setup: one torch thread per worker when running in
    parallel (so workers don't fight over cores), then the warm-up."""
    if single_thread:
        import torch
        torch.set_num_threads(1)
    warmup(hparams)


def evaluate_stem(task) -> tuple[dict, list[dict]]:
    stem, hparams = task
    return evaluate_file(dataset_utils.load_test_file(stem), hparams)


def save_run_info(out_dir: Path, hparams: dict, args) -> None:
    import torch
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "workers": args.workers,
        "torch_threads": 1 if args.workers > 1 else torch.get_num_threads(),
        "hparams": hparams,
        "n_files": args.n_files,
        "seed": args.seed,
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))


def plot_win_rates(df: pd.DataFrame, out_path: Path) -> dict:
    win_rates = {}
    print(f"\n===== Aggregate results over {len(df)} files =====")
    for key, label, lower_is_better in ALL_METRICS:
        lpc_col, nn_col = f"{key}_lpc", f"{key}_nn"
        nn_wins = (df[nn_col] < df[lpc_col]).sum() if lower_is_better else (df[nn_col] > df[lpc_col]).sum()
        win_rates[label] = 100 * nn_wins / len(df)
        print(f"\n{label} ({'lower=better' if lower_is_better else 'higher=better'}):")
        print(f"  NN better on {nn_wins}/{len(df)} files ({win_rates[label]:.1f}%)")
        print(f"  Mean  -- LPC: {df[lpc_col].mean():.4f}   NN: {df[nn_col].mean():.4f}")
        print(f"  Median-- LPC: {df[lpc_col].median():.4f}   NN: {df[nn_col].median():.4f}")

    fig, ax = plotting.new_axes(figsize=(10, 4.5))
    labels, values = list(win_rates.keys()), list(win_rates.values())
    colors = [plotting.NN_COLOR if key in PHYSICAL_KEYS else "#6D5BD0" for key, _, _ in ALL_METRICS]
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    ax.axhline(50, color=plotting.TIE_COLOR, linewidth=1.5, linestyle="--", zorder=2)
    ax.text(len(labels) - 0.4, 51.5, "50% = coin flip", color=plotting.TIE_COLOR, fontsize=9, ha="right")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}%",
                ha="center", va="bottom", fontsize=10, color="#1f2937")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Files where NN beats LPC (%)")
    ax.set_title(f"NN vs. LPC win rate across {len(df)} files (example_test_set)")
    plotting.style_axis(ax)
    fig.tight_layout()
    plotting.save_fig(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return win_rates


def plot_complexity_grid(df: pd.DataFrame, out_path: Path) -> None:
    """One scatter panel per metric: x = signal complexity (spectral flatness
    of the clean file), y = metric value, one dot per file per method. Shows
    whether the NN's advantage (or disadvantage) depends on how complex the
    content is, instead of collapsing everything into one aggregate number.

    Log-scaled x-axis: this dataset's files are mostly low-flatness
    (tonal/melodic) content, so a linear axis crams nearly every point into
    the leftmost few percent of the plot. A dashed trend line (linear fit in
    log-x space) is drawn per method so the direction of the relationship is
    readable at a glance instead of having to eyeball a dot cloud.
    """
    fig, axes = plotting.new_axes(figsize=(6, 4.5), ncols=2, nrows=2)
    for ax, (key, label, _) in zip(axes.flat, ALL_METRICS):
        log_x = np.log10(df["complexity_flatness"])
        for method, color, series_label in [("lpc", plotting.LPC_COLOR, "This LPC"),
                                             ("nn", plotting.NN_COLOR, "This NN")]:
            y = df[f"{key}_{method}"]
            ax.scatter(df["complexity_flatness"], y, color=color, label=series_label,
                       s=24, alpha=0.75, zorder=3)
            slope, intercept = np.polyfit(log_x, y, 1)
            x_line = np.linspace(log_x.min(), log_x.max(), 50)
            ax.plot(10 ** x_line, slope * x_line + intercept, color=color,
                    linestyle="--", linewidth=1.5, alpha=0.9, zorder=2)
        ax.set_xscale("log")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Spectral flatness, log scale (0=tonal, 1=noise-like)", fontsize=9)
        plotting.style_axis(ax, grid_axis="both")
    axes.flat[0].legend(fontsize=9, frameon=False)
    fig.suptitle("Concealment quality vs. signal complexity\n(dots = files, dashed lines = trend)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    plotting.save_fig(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-files", type=int, default=300, help="How many files to sample from example_test_set.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the file sample.")
    parser.add_argument("--ar-order", type=int, default=256)
    parser.add_argument("--diagonal-load", type=float, default=0.001)
    parser.add_argument("--context-dim-packets", type=int, default=8)
    parser.add_argument("--extra-dim", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel processes (one file each). Use about the number of physical cores. "
                             "Per-file timings are then single-threaded and measured under load.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hparams = dict(ar_order=args.ar_order, diagonal_load=args.diagonal_load,
                   context_dim_packets=args.context_dim_packets, extra_dim=args.extra_dim)

    stems = dataset_utils.select_diverse_stems(args.n_files, seed=args.seed, workers=args.workers)
    print(f"Sampled {len(stems)} files from example_test_set (seed={args.seed}), "
          f"stratified by spectral flatness for genuine content diversity.", flush=True)

    save_run_info(args.out_dir, hparams, args)

    rows, all_burst_rows = [], []
    results = dataset_utils.parallel_map(evaluate_stem, [(stem, hparams) for stem in stems], args.workers,
                                         initializer=init_worker, initargs=(hparams, args.workers > 1))
    for i, (row, file_burst_rows) in enumerate(results, start=1):
        rows.append(row)
        all_burst_rows.extend(file_burst_rows)
        print(f"[{i}/{len(stems)}] {row['file']} ({row['n_packets']} pkts, {row['loss_rate_pct']:.1f}% lost): "
              f"SNR lpc={row['snr_lpc']:.2f} nn={row['snr_nn']:.2f} dB | "
              f"PLCMOS lpc={row['plcmos_lpc']:.2f} nn={row['plcmos_nn']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    safe_io.save_csv(df, args.out_dir / "per_file_results.csv", index=False)
    safe_io.save_csv(pd.DataFrame(all_burst_rows), args.out_dir / "per_burst_results.csv", index=False)

    plot_win_rates(df, args.out_dir / "win_rate_summary.png")
    plot_complexity_grid(df, args.out_dir / "complexity_grid.png")

    print(f"\nPer-file results: {args.out_dir / 'per_file_results.csv'}")
    print(f"Per-burst results: {args.out_dir / 'per_burst_results.csv'}")
    print(f"Win-rate summary: {args.out_dir / 'win_rate_summary.png'}")
    print(f"Complexity grid:  {args.out_dir / 'complexity_grid.png'}")


if __name__ == "__main__":
    main()
