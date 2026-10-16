"""
Research-level comparison of four causal concealment methods, using the real
parcnet-is2/example_test_set dataset -- every (clean, lossy, trace) triplet
there is the genuine ground truth, not a synthetic/arbitrary pairing, so this
is the most rigorous dataset available in this project:

  AR-only (LPC)    -- pure AR(order) recursive linear prediction
  AR+NN (Hybrid)   -- PARCnet (AR + a trained neural residual)
  Sparse AR        -- Severi (2025)'s OMP-based sparse AR concealer -- see
                       research/concealment.py and parcnet-is2/
                       sparse_ar_model.py's module docstrings for the
                       algorithm and, importantly, where this implementation
                       had to depart from the paper's literal text
  Endless Sounds   -- Svento (2026)'s spectral phase-randomisation
                       inpainting, causally re-derived -- see research/
                       concealment.py and parcnet-is2/endless_sounds_model.py's
                       module docstrings for the algorithm and which parts of
                       the paper's own two-sided pipeline had to be dropped
                       or re-derived for one-sided (past-only) prediction

LPC is treated as the baseline the other three are compared against, so
every win-rate figure here reads as "how often does method X beat plain
LPC", not every pairwise combination -- the full pairwise statistics (every
method pair) are run_statistical_analysis.py's job, not this script's.

Two families of metrics are computed for every file, matching what a PLC
paper would report:

  PHYSICAL (waveform/spectrum error -- see research/metrics.py docstring):
    SNR, SI-SDR, log-spectral distance (LSD), multi-resolution STFT distance

  PSYCHOACOUSTIC (models of human hearing):
    PLCMOS (Microsoft, reference-free), ViSQOL (Google, full-band "audio" mode)

Nothing here is fixed to one file or one hyperparameter setting -- how many
files, which hyperparameters, and where results are saved are all arguments.
Produces:
  - a per-file CSV with every metric for all three methods, plus wall-clock
    concealment time per method (and per concealed packet)
  - a per-burst CSV: for every real burst in every evaluated file, the SNR
    of each method measured only inside that gap -- the input to
    run_statistical_analysis.py's quality-vs-burst-length analysis, collected
    here so each method's expensive pass runs once, not once per analysis
  - run_info.json: machine/hparam details, so timing results are always
    traceable to the hardware they were measured on
  - a win-rate summary chart (each non-baseline method vs. LPC) across all 4 metrics
  - a signal-complexity grid: for each metric, how quality varies with the
    clean signal's spectral flatness (low = tonal/simple, high =
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

# (key, display label, concealment function, plot color)
METHODS = [
    ("lpc", "This LPC", concealment.conceal_lpc, plotting.LPC_COLOR),
    ("nn", "This NN", concealment.conceal_nn, plotting.NN_COLOR),
    ("sparse_ar", "This Sparse AR", concealment.conceal_sparse_ar, plotting.SPARSE_AR_COLOR),
    ("endless_sounds", "This Endless Sounds", concealment.conceal_endless_sounds, plotting.ENDLESS_SOUNDS_COLOR),
]
BASELINE_KEY = "lpc"
NON_BASELINE = [m for m in METHODS if m[0] != BASELINE_KEY]

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
    """Runs one throwaway call per method on dummy input before anything is
    timed -- otherwise the first file's time would include the NN's
    checkpoint load, torch's first-call overhead, and (for Sparse AR) numba
    JIT compilation, instead of just the real per-file cost."""
    dummy = np.zeros(concealment.PACKET_SIZE * 16, dtype=np.float32)
    dummy_trace = np.zeros(16, dtype=int)
    dummy_trace[10] = 1
    for _, _, fn, _ in METHODS:
        fn(dummy, dummy_trace, **hparams)


def burst_rows(tf: dataset_utils.TestFile, outputs: dict[str, np.ndarray]) -> list[dict]:
    """Gap-local quality for every real burst in the file: SNR measured only
    over the lost samples, for every method. Waveform SNR is the only metric
    here because the perceptual models (PLCMOS, ViSQOL) need seconds of
    audio, not a single-packet gap. The clean gap's level is stored so
    near-silent gaps, where SNR is meaningless, can be filtered out later."""
    rows = []
    for start, length in dataset_utils.find_bursts(tf.trace):
        a, b = start * concealment.PACKET_SIZE, (start + length) * concealment.PACKET_SIZE
        clean_gap = tf.clean[a:b]
        row = {
            "file": tf.stem,
            "start_packet": start,
            "burst_length": length,
            "clean_gap_rms_db": float(20 * np.log10(np.sqrt(np.mean(clean_gap ** 2)) + 1e-12)),
        }
        for key, _, _, _ in METHODS:
            row[f"gap_snr_{key}"] = metrics.snr_db(clean_gap, outputs[key][a:b])
        rows.append(row)
    return rows


def evaluate_file(tf: dataset_utils.TestFile, hparams: dict) -> tuple[dict, list[dict]]:
    """Runs all three concealment methods on one file (each timed), computes
    every metric, and returns (per-file row, per-burst rows)."""
    outputs, timings = {}, {}
    for key, _, fn, _ in METHODS:
        t0 = time.perf_counter()
        outputs[key] = fn(tf.lossy, tf.trace, **hparams)
        timings[key] = (time.perf_counter() - t0) * 1000

    n_lost = int(tf.trace.sum())
    row = {
        "file": tf.stem,
        "n_packets": tf.n_packets,
        "n_lost_packets": n_lost,
        "sample_rate": tf.sample_rate,
        "loss_rate_pct": tf.loss_rate * 100,
        "complexity_flatness": complexity.spectral_flatness(tf.clean),
    }
    for key, _, _, _ in METHODS:
        row[f"time_ms_{key}"] = timings[key]
        # Every method only does work on lost packets, so total time / lost
        # packets is the cost of concealing one packet.
        row[f"ms_per_lost_packet_{key}"] = timings[key] / n_lost if n_lost else np.nan
    for key, _, _, _ in METHODS:
        out = outputs[key]
        row[f"snr_{key}"] = metrics.snr_db(tf.clean, out)
        row[f"mrstft_{key}"] = metrics.multires_stft_distance(tf.clean, out)
        row[f"plcmos_{key}"] = metrics.plcmos_score(out, tf.sample_rate)
        row[f"visqol_{key}"] = metrics.visqol_score(tf.clean, out, tf.sample_rate)

    return row, burst_rows(tf, outputs)


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
    """Each non-baseline method's win rate against LPC, per metric (grouped
    bars: one group per metric, one bar per non-baseline method). Not every
    pairwise combination -- see the module docstring for why LPC is the
    baseline here, and run_statistical_analysis.py for the full pairwise
    tests.
    """
    win_rates: dict[tuple[str, str], float] = {}
    print(f"\n===== Aggregate results over {len(df)} files =====")
    for key, label, lower_is_better in ALL_METRICS:
        base_col = f"{key}_{BASELINE_KEY}"
        print(f"\n{label} ({'lower=better' if lower_is_better else 'higher=better'}):")
        print(f"  Mean  -- " + "   ".join(f"{m[1]}: {df[f'{key}_{m[0]}'].mean():.4f}" for m in METHODS))
        print(f"  Median-- " + "   ".join(f"{m[1]}: {df[f'{key}_{m[0]}'].median():.4f}" for m in METHODS))
        for mkey, mlabel, _, _ in NON_BASELINE:
            col = f"{key}_{mkey}"
            wins = (df[col] < df[base_col]).sum() if lower_is_better else (df[col] > df[base_col]).sum()
            pct = 100 * wins / len(df)
            win_rates[(mkey, key)] = pct
            print(f"  {mlabel} beats This LPC on {wins}/{len(df)} files ({pct:.1f}%)")

    fig, ax = plotting.new_axes(figsize=(10, 4.5))
    n_metrics, n_methods = len(ALL_METRICS), len(NON_BASELINE)
    width = 0.8 / n_methods
    x = np.arange(n_metrics)
    for i, (mkey, mlabel, _, color) in enumerate(NON_BASELINE):
        offset = (i - (n_methods - 1) / 2) * width
        values = [win_rates[(mkey, key)] for key, _, _ in ALL_METRICS]
        bars = ax.bar(x + offset, values, width=width * 0.9, color=color, label=mlabel, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=8, color="#1f2937")
    ax.axhline(50, color=plotting.TIE_COLOR, linewidth=1.5, linestyle="--", zorder=2)
    ax.text(n_metrics - 0.5, 51.5, "50% = coin flip", color=plotting.TIE_COLOR, fontsize=9, ha="right")
    ax.set_xticks(x, [label for _, label, _ in ALL_METRICS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Files where method beats This LPC (%)")
    ax.set_title(f"Win rate vs. This LPC across {len(df)} files (example_test_set)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=n_methods, frameon=False, fontsize=9)
    plotting.style_axis(ax)
    fig.tight_layout()
    plotting.save_fig(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return win_rates


def plot_complexity_grid(df: pd.DataFrame, out_path: Path) -> None:
    """One scatter panel per metric: x = signal complexity (spectral flatness
    of the clean file), y = metric value, one dot per file per method. Shows
    whether each method's quality depends on how complex the content is,
    instead of collapsing everything into one aggregate number.

    Log-scaled x-axis: this dataset's files are mostly low-flatness
    (tonal/melodic) content, so a linear axis crams nearly every point into
    the leftmost few percent of the plot. A dashed trend line (linear fit in
    log-x space) is drawn per method so the direction of the relationship is
    readable at a glance instead of having to eyeball a dot cloud.
    """
    fig, axes = plotting.new_axes(figsize=(6, 4.5), ncols=2, nrows=2)
    for ax, (key, label, _) in zip(axes.flat, ALL_METRICS):
        log_x = np.log10(df["complexity_flatness"])
        for mkey, mlabel, _, color in METHODS:
            y = df[f"{key}_{mkey}"]
            ax.scatter(df["complexity_flatness"], y, color=color, label=mlabel,
                       s=20, alpha=0.7, zorder=3)
            slope, intercept = np.polyfit(log_x, y, 1)
            x_line = np.linspace(log_x.min(), log_x.max(), 50)
            ax.plot(10 ** x_line, slope * x_line + intercept, color=color,
                    linestyle="--", linewidth=1.5, alpha=0.9, zorder=2)
        ax.set_xscale("log")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Spectral flatness, log scale (0=tonal, 1=noise-like)", fontsize=9)
        plotting.style_axis(ax, grid_axis="both")
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Concealment quality vs. signal complexity\n(dots = files, dashed lines = trend)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    plotting.save_fig(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-files", type=int, default=300, help="How many files to sample from example_test_set.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the file sample.")
    parser.add_argument("--ar-order", type=int, default=256, help="LPC/Hybrid only.")
    parser.add_argument("--diagonal-load", type=float, default=0.001, help="LPC/Hybrid only.")
    parser.add_argument("--context-dim-packets", type=int, default=8, help="Shared by all three methods.")
    parser.add_argument("--extra-dim", type=int, default=256, help="Shared by all three methods.")
    parser.add_argument("--max-k-lags", type=int, default=3, help="Sparse AR only.")
    parser.add_argument("--max-lag-l", type=int, default=1024, help="Sparse AR only.")
    parser.add_argument("--burg-order-pb", type=int, default=8, help="Sparse AR only.")
    parser.add_argument("--lambda-reg", type=float, default=0.01, help="Sparse AR only.")
    parser.add_argument("--boundary-blend", type=int, default=64, help="Endless Sounds only.")
    parser.add_argument("--es-seed", type=int, default=0,
                        help="Endless Sounds only: its phase-randomisation RNG seed "
                             "(separate from --seed, which samples files).")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel processes (one file each). Use about the number of physical cores. "
                             "Per-file timings are then single-threaded and measured under load.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore results saved by an earlier (possibly crashed) run and start over.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hparams = dict(ar_order=args.ar_order, diagonal_load=args.diagonal_load,
                   context_dim_packets=args.context_dim_packets, extra_dim=args.extra_dim,
                   max_k_lags=args.max_k_lags, max_lag_l=args.max_lag_l,
                   burg_order_pb=args.burg_order_pb, lambda_reg=args.lambda_reg,
                   boundary_blend=args.boundary_blend, es_seed=args.es_seed)

    stems = dataset_utils.select_diverse_stems(args.n_files, seed=args.seed, workers=args.workers)
    print(f"Sampled {len(stems)} files from example_test_set (seed={args.seed}), "
          f"stratified by spectral flatness for genuine content diversity.", flush=True)

    save_run_info(args.out_dir, hparams, args)

    # Every finished file goes straight to the checkpoint, so a crash loses
    # only the files in flight and re-running the same command resumes.
    checkpoint = safe_io.Checkpoint(args.out_dir / "checkpoint.jsonl", fresh=args.fresh)
    key_of = {stem: f"{stem}|{json.dumps(hparams, sort_keys=True)}" for stem in stems}
    todo = [stem for stem in stems if key_of[stem] not in checkpoint]
    n_done = len(stems) - len(todo)
    if n_done:
        print(f"Resuming: {n_done} files already in {checkpoint.path.name}, {len(todo)} to go "
              f"(use --fresh to start over).", flush=True)

    results = dataset_utils.resilient_map(evaluate_stem, [(stem, hparams) for stem in todo], args.workers,
                                          initializer=init_worker, initargs=(hparams, args.workers > 1))
    for task_index, (row, file_burst_rows) in results:
        checkpoint.add(key_of[todo[task_index]], {"row": row, "bursts": file_burst_rows})
        n_done += 1
        snr_bits = " ".join(f"{key}={row[f'snr_{key}']:.2f}" for key, _, _, _ in METHODS)
        print(f"[{n_done}/{len(stems)}] {row['file']} ({row['n_packets']} pkts, {row['loss_rate_pct']:.1f}% lost): "
              f"SNR {snr_bits} dB", flush=True)

    rows, all_burst_rows = [], []
    for stem in stems:
        record = checkpoint.records[key_of[stem]]
        rows.append(record["row"])
        all_burst_rows.extend(record["bursts"])

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
