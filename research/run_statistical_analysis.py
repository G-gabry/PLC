"""
Statistical analysis layer for the LPC vs. NN comparison: turns the raw CSVs
from the other scripts into report-ready evidence. Apart from the
failure-case plots (which re-conceal a handful of files), it does no
concealment of its own, so it is cheap to re-run after every VM run.

Inputs (each section is skipped, with a message, if its input is missing):
  output/metrics_comparison/per_file_results.csv   run_metrics_comparison.py
  output/metrics_comparison/per_burst_results.csv  run_metrics_comparison.py
  output/metrics_comparison/run_info.json          run_metrics_comparison.py
  output/hparam_sweeps/sweep_*.csv                 run_hparam_sweeps.py

Outputs (output/statistical_analysis/):
  1. significance_tests.csv/.md   Paired Wilcoxon signed-rank test per metric,
                                  Holm-corrected for testing 4 metrics at once,
                                  rank-biserial effect size, bootstrap 95% CI
                                  of the mean difference.
  2. metric_distributions.png     Per-file values of both methods (box + paired lines).
     paired_differences.png       Per-file NN-minus-LPC difference, oriented so
                                  positive always means "NN better".
  3. burst_length_quality.png/.csv  Gap SNR vs. real burst length. CIs use a
                                  cluster bootstrap over files, since bursts in
                                  the same file are not independent samples.
  4. realtime_feasibility.png     Real-time factor (time to conceal one packet
                                  / duration of one packet) per method; above
                                  1.0 the method cannot keep up with live audio
                                  on the machine that ran the evaluation.
  5. sweep_summary.csv/.md        Best value per hyperparameter, method and
                                  metric vs. the default, with compute cost.
  6. failure_cases/               Waveform/spectrogram/diff/crossfade plots of
                                  the files where LPC beats the NN by the most.

Usage:
    python -m research.run_statistical_analysis
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from research import dataset_utils, concealment, plotting, safe_io
from research.run_metrics_comparison import ALL_METRICS, OUT_DIR as METRICS_DIR
from research.run_hparam_sweeps import DEFAULT_HPARAMS, OUT_DIR as SWEEPS_DIR

OUT_DIR = Path(__file__).resolve().parent / "output" / "statistical_analysis"
ALPHA = 0.05

# Sweep CSV "param" name -> concealment hparam key it controls.
SWEEP_PARAM_TO_HPARAM = {
    "ar_order": "ar_order",
    "context_length_packets": "context_dim_packets",
    "diagonal_load": "diagonal_load",
    "extra_dim": "extra_dim",
}

BURST_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 7, "4-7"), (8, None, "8+")]


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def oriented_diff(df: pd.DataFrame, key: str, lower_is_better: bool, suffix: str = "") -> pd.Series:
    """NN minus LPC, sign-flipped for lower-is-better metrics so that
    positive always means "NN better"."""
    diff = df[f"{key}_nn{suffix}"] - df[f"{key}_lpc{suffix}"]
    return -diff if lower_is_better else diff


def bootstrap_mean_ci(values, groups=None, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap 95% CI. With `groups`, resamples whole
    groups (cluster bootstrap) instead of individual values."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        boot_means = values[idx].mean(axis=1)
    else:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        sums = np.array([values[groups == g].sum() for g in unique])
        counts = np.array([(groups == g).sum() for g in unique])
        pick = rng.integers(0, len(unique), size=(n_boot, len(unique)))
        boot_means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(boot_means, [ALPHA / 2, 1 - ALPHA / 2])
    return float(values.mean()), float(lo), float(hi)


def wilcoxon_with_effect(diff) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank p-value and matched-pairs rank-biserial
    correlation r (-1..1, positive = NN better). Zero differences (ties,
    e.g. files with no lost packets) are dropped, as the test requires."""
    nonzero = np.asarray(diff, dtype=float)
    nonzero = nonzero[nonzero != 0]
    if len(nonzero) < 2:
        return np.nan, np.nan
    p = float(stats.wilcoxon(nonzero).pvalue)
    ranks = stats.rankdata(np.abs(nonzero))
    w_pos, w_neg = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    return p, float((w_pos - w_neg) / (w_pos + w_neg))


def holm_correction(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the chance of any false
    positive across the whole family of tests, not just each one)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running_max
    return adjusted


def effect_size_label(r: float) -> str:
    if np.isnan(r):
        return "n/a"
    r = abs(r)
    return "negligible" if r < 0.1 else "small" if r < 0.3 else "medium" if r < 0.5 else "large"


def write_markdown_table(df: pd.DataFrame, path: Path, float_fmt: str = "{:.4g}") -> None:
    def fmt(v):
        return float_fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# 1. Significance tests
# --------------------------------------------------------------------------
def significance_tests(per_file: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for key, label, lower_is_better in ALL_METRICS:
        diff = oriented_diff(per_file, key, lower_is_better)
        mean_diff, ci_lo, ci_hi = bootstrap_mean_ci(diff)
        p, r = wilcoxon_with_effect(diff)
        rows.append({
            "metric": label,
            "n_files": len(diff),
            "mean_lpc": per_file[f"{key}_lpc"].mean(),
            "mean_nn": per_file[f"{key}_nn"].mean(),
            "mean_nn_advantage": mean_diff,
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
            "median_nn_advantage": float(np.median(diff)),
            "nn_win_rate_pct": 100 * float((diff > 0).mean()),
            "p_wilcoxon": p,
            "rank_biserial_r": r,
            "effect_size": effect_size_label(r),
        })
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_correction(table["p_wilcoxon"].fillna(1.0))
    table["verdict"] = [
        ("NN better" if r > 0 else "LPC better") if p < ALPHA else "no significant difference"
        for p, r in zip(table["p_holm"], table["rank_biserial_r"])
    ]

    safe_io.save_csv_safe(table, out_dir / "significance_tests.csv", index=False)
    write_markdown_table(table[["metric", "n_files", "mean_lpc", "mean_nn", "mean_nn_advantage",
                                "ci95_low", "ci95_high", "nn_win_rate_pct", "p_holm",
                                "rank_biserial_r", "effect_size", "verdict"]],
                         out_dir / "significance_tests.md")

    print("\n===== 1. Significance tests (paired Wilcoxon, Holm-corrected) =====")
    print("(nn_advantage > 0 means NN better, for every metric)")
    for row in table.itertuples():
        print(f"  {row.metric:<16} advantage {row.mean_nn_advantage:+.4f} "
              f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}]  p_holm={row.p_holm:.3g}  "
              f"r={row.rank_biserial_r:+.2f} ({row.effect_size}) -> {row.verdict}")
    return table


# --------------------------------------------------------------------------
# 2. Distribution plots
# --------------------------------------------------------------------------
def plot_distributions(per_file: pd.DataFrame, tests: pd.DataFrame, out_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = len(per_file)

    fig, axes = plotting.new_axes(figsize=(3.6, 4.4), ncols=len(ALL_METRICS))
    for ax, (key, label, lower_is_better) in zip(axes, ALL_METRICS):
        lpc, nn = per_file[f"{key}_lpc"].to_numpy(), per_file[f"{key}_nn"].to_numpy()
        for a, b in zip(lpc, nn):
            ax.plot([0, 1], [a, b], color=plotting.TIE_COLOR, alpha=0.3, linewidth=0.7, zorder=1)
        box = ax.boxplot([lpc, nn], positions=[0, 1], widths=0.5, patch_artist=True,
                         showfliers=False, zorder=2, medianprops={"color": "#1f2937"})
        for patch, color in zip(box["boxes"], [plotting.LPC_COLOR, plotting.NN_COLOR]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for x, values, color in [(0, lpc, plotting.LPC_COLOR), (1, nn, plotting.NN_COLOR)]:
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(values)), values, s=10,
                       color=color, alpha=0.7, zorder=3)
        ax.set_xticks([0, 1], ["LPC", "NN"])
        ax.set_title(f"{label}\n({'lower' if lower_is_better else 'higher'} = better)", fontsize=10)
        plotting.style_axis(ax)
    fig.suptitle(f"Per-file metric distributions ({n} files; gray lines connect the same file)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    safe_io.save_fig_safe(fig, out_dir / "metric_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plotting.new_axes(figsize=(3.6, 4.4), ncols=len(ALL_METRICS))
    for ax, (key, label, lower_is_better), test in zip(axes, ALL_METRICS, tests.itertuples()):
        diff = oriented_diff(per_file, key, lower_is_better).to_numpy()
        parts = ax.violinplot(diff, positions=[0], widths=0.7, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#6D5BD0")
            body.set_alpha(0.3)
        ax.scatter(rng.uniform(-0.06, 0.06, len(diff)), diff, s=10, color="#6D5BD0", alpha=0.7, zorder=3)
        ax.axhline(0, color=plotting.TIE_COLOR, linestyle="--", linewidth=1.2, zorder=1)
        ax.errorbar([0.42], [test.mean_nn_advantage],
                    yerr=[[test.mean_nn_advantage - test.ci95_low], [test.ci95_high - test.mean_nn_advantage]],
                    fmt="o", color="#1f2937", capsize=4, zorder=4)
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.7)
        ax.set_title(f"{label}\np_holm={test.p_holm:.2g}, r={test.rank_biserial_r:+.2f}", fontsize=10)
        plotting.style_axis(ax)
    axes[0].set_ylabel("NN advantage over LPC\n(> 0 = NN better)")
    fig.suptitle("Per-file NN-vs-LPC difference (black dot = mean with bootstrap 95% CI)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    safe_io.save_fig_safe(fig, out_dir / "paired_differences.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n===== 2. Saved metric_distributions.png and paired_differences.png =====")


# --------------------------------------------------------------------------
# 3. Quality vs. real burst length
# --------------------------------------------------------------------------
def burst_length_analysis(per_burst: pd.DataFrame, out_dir: Path, silence_db: float, seed: int) -> None:
    audible = per_burst[per_burst["clean_gap_rms_db"] > silence_db]
    n_dropped = len(per_burst) - len(audible)

    rows = []
    for lo, hi, label in BURST_BINS:
        mask = audible["burst_length"] >= lo
        if hi is not None:
            mask &= audible["burst_length"] <= hi
        in_bin = audible[mask]
        if len(in_bin) == 0:
            continue
        files = in_bin["file"].to_numpy()
        lpc = bootstrap_mean_ci(in_bin["gap_snr_lpc"], groups=files, seed=seed)
        nn = bootstrap_mean_ci(in_bin["gap_snr_nn"], groups=files, seed=seed)
        diff = bootstrap_mean_ci(in_bin["gap_snr_nn"] - in_bin["gap_snr_lpc"], groups=files, seed=seed)
        rows.append({
            "burst_length_bin": label, "n_bursts": len(in_bin), "n_files": len(np.unique(files)),
            "gap_snr_lpc_mean": lpc[0], "gap_snr_lpc_ci_low": lpc[1], "gap_snr_lpc_ci_high": lpc[2],
            "gap_snr_nn_mean": nn[0], "gap_snr_nn_ci_low": nn[1], "gap_snr_nn_ci_high": nn[2],
            "nn_advantage_mean": diff[0], "nn_advantage_ci_low": diff[1], "nn_advantage_ci_high": diff[2],
            "nn_win_rate_pct": 100 * float((in_bin["gap_snr_nn"] > in_bin["gap_snr_lpc"]).mean()),
        })
    table = pd.DataFrame(rows)
    safe_io.save_csv_safe(table, out_dir / "burst_length_quality.csv", index=False)

    x = np.arange(len(table))
    tick_labels = [f"{r.burst_length_bin}\n(n={r.n_bursts})" for r in table.itertuples()]
    fig, (ax1, ax2) = plotting.new_axes(figsize=(6, 4.4), ncols=2)
    for method, color, label, offset in [("lpc", plotting.LPC_COLOR, "This LPC", -0.06),
                                         ("nn", plotting.NN_COLOR, "This NN", 0.06)]:
        mean = table[f"gap_snr_{method}_mean"]
        ax1.errorbar(x + offset, mean,
                     yerr=[mean - table[f"gap_snr_{method}_ci_low"], table[f"gap_snr_{method}_ci_high"] - mean],
                     color=color, marker="o", linewidth=1.8, capsize=4, label=label)
    ax1.set_xticks(x, tick_labels, fontsize=8)
    ax1.set_xlabel("Real burst length (packets)")
    ax1.set_ylabel("Gap SNR (dB, higher = better)")
    ax1.set_title("Quality inside the gap vs. burst length", fontsize=11)
    ax1.legend(frameon=False, fontsize=9)
    plotting.style_axis(ax1)

    mean = table["nn_advantage_mean"]
    ax2.errorbar(x, mean, yerr=[mean - table["nn_advantage_ci_low"], table["nn_advantage_ci_high"] - mean],
                 color="#6D5BD0", marker="o", linewidth=1.8, capsize=4)
    ax2.axhline(0, color=plotting.TIE_COLOR, linestyle="--", linewidth=1.2)
    ax2.set_xticks(x, tick_labels, fontsize=8)
    ax2.set_xlabel("Real burst length (packets)")
    ax2.set_ylabel("NN minus LPC gap SNR (dB, > 0 = NN better)")
    ax2.set_title("NN advantage vs. burst length", fontsize=11)
    plotting.style_axis(ax2)

    fig.suptitle(f"Gap SNR on every real burst ({len(audible)} bursts, {audible['file'].nunique()} files; "
                 f"{n_dropped} near-silent gaps below {silence_db:.0f} dBFS excluded)\n"
                 f"error bars = 95% CI, cluster-bootstrapped over files", fontsize=11, y=1.06)
    fig.tight_layout()
    safe_io.save_fig_safe(fig, out_dir / "burst_length_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n===== 3. Quality vs. real burst length (gap SNR, dB) =====")
    for r in table.itertuples():
        print(f"  {r.burst_length_bin:>4} pkts (n={r.n_bursts:>5}): LPC {r.gap_snr_lpc_mean:6.2f}  "
              f"NN {r.gap_snr_nn_mean:6.2f}  NN advantage {r.nn_advantage_mean:+.2f} "
              f"[{r.nn_advantage_ci_low:+.2f}, {r.nn_advantage_ci_high:+.2f}]")


# --------------------------------------------------------------------------
# 4. Real-time feasibility
# --------------------------------------------------------------------------
def realtime_analysis(per_file: pd.DataFrame, run_info: dict | None, out_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    timed = per_file.dropna(subset=["ms_per_lost_packet_lpc", "ms_per_lost_packet_nn"])
    packet_ms = concealment.PACKET_SIZE / timed["sample_rate"] * 1000

    fig, ax = plotting.new_axes(figsize=(7, 4.6))
    print("\n===== 4. Real-time feasibility =====")
    for x, (method, color, label) in enumerate([("lpc", plotting.LPC_COLOR, "This LPC"),
                                                ("nn", plotting.NN_COLOR, "This NN")]):
        rtf = (timed[f"ms_per_lost_packet_{method}"] / packet_ms).to_numpy()
        box = ax.boxplot([rtf], positions=[x], widths=0.5, patch_artist=True, showfliers=False,
                         medianprops={"color": "#1f2937"})
        box["boxes"][0].set_facecolor(color)
        box["boxes"][0].set_alpha(0.35)
        ax.scatter(x + rng.uniform(-0.08, 0.08, len(rtf)), rtf, s=10, color=color, alpha=0.7, zorder=3)
        median = float(np.median(rtf))
        ax.text(x + 0.3, median, f"median {median:.3g}x", fontsize=9, va="center", color=color)
        median_ms = float(timed[f"ms_per_lost_packet_{method}"].median())
        print(f"  {label}: median {median_ms:.2f} ms per concealed packet = {median:.3g}x real time "
              f"({'OK' if median < 1 else 'TOO SLOW'} for live use)")
    ax.axhline(1.0, color="#DC2626", linestyle="--", linewidth=1.5)
    ax.text(1.45, 1.0, "real-time limit", color="#DC2626", fontsize=9, va="bottom", ha="right")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if lo <= t <= hi]
    ax.set_yticks(ticks, [f"{t:g}x" for t in ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks([0, 1], ["This LPC", "This NN"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel("Real-time factor (log scale)\n= time to conceal 1 packet / packet duration")
    machine = ""
    if run_info:
        machine = (f"\nmeasured on: {run_info.get('processor') or run_info.get('platform')}, "
                   f"{run_info.get('cpu_count')} CPUs, torch threads={run_info.get('torch_threads')}")
    ax.set_title(f"Can each method keep up with live audio? (packet = {packet_ms.median():.1f} ms; "
                 f"below the red line = yes){machine}", fontsize=10)
    plotting.style_axis(ax)
    fig.tight_layout()
    safe_io.save_fig_safe(fig, out_dir / "realtime_feasibility.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Hyperparameter sweep summary
# --------------------------------------------------------------------------
def sweep_summary(sweeps_dir: Path, out_dir: Path) -> None:
    rows = []
    for csv_path in sorted(sweeps_dir.glob("sweep_*.csv")):
        df = pd.read_csv(csv_path)
        param = df["param"].iloc[0]
        default = DEFAULT_HPARAMS.get(SWEEP_PARAM_TO_HPARAM.get(param))
        n_files = df["file"].nunique()
        by_value = df.groupby("value").agg(
            **{f"{k}_{m}": (f"{k}_{m}", "mean") for k, _, _ in ALL_METRICS for m in ("lpc", "nn")},
            **{f"time_{m}": (f"time_ms_{m}", "median") for m in ("lpc", "nn")},
        )
        default_rows = by_value[np.isclose(by_value.index.astype(float), float(default))] if default is not None else []
        for method in ("lpc", "nn"):
            for key, label, lower_is_better in ALL_METRICS:
                col = f"{key}_{method}"
                best_value = by_value[col].idxmin() if lower_is_better else by_value[col].idxmax()
                at_default = default_rows[col].iloc[0] if len(default_rows) else np.nan
                at_best = by_value.loc[best_value, col]
                gain = (at_default - at_best) if lower_is_better else (at_best - at_default)
                rows.append({
                    "hyperparameter": param, "method": "LPC" if method == "lpc" else "NN", "metric": label,
                    "n_files": n_files, "default_value": default, "metric_at_default": at_default,
                    "best_value": best_value, "metric_at_best": at_best, "gain_over_default": gain,
                    "median_time_ms_at_default": default_rows[f"time_{method}"].iloc[0] if len(default_rows) else np.nan,
                    "median_time_ms_at_best": by_value.loc[best_value, f"time_{method}"],
                })
    if not rows:
        print("\n===== 5. Sweep summary: skipped (no sweep_*.csv found) =====")
        return
    table = pd.DataFrame(rows)
    safe_io.save_csv_safe(table, out_dir / "sweep_summary.csv", index=False)
    write_markdown_table(table, out_dir / "sweep_summary.md")
    print("\n===== 5. Hyperparameter sweep summary (best value per metric; gain > 0 = better than default) =====")
    for (param, method), group in table.groupby(["hyperparameter", "method"], sort=False):
        best = ", ".join(f"{r.metric}: {r.best_value:g} ({r.gain_over_default:+.3g})" for r in group.itertuples())
        print(f"  {param:<24} {method:<4} {best}")
    thin = table[table["n_files"] < 10]["hyperparameter"].unique()
    if len(thin):
        print(f"  WARNING: {', '.join(thin)} averaged over fewer than 10 files -- "
              f"too few to draw conclusions; re-run the sweeps with more --n-files.")


# --------------------------------------------------------------------------
# 6. Failure cases
# --------------------------------------------------------------------------
def failure_cases(per_file: pd.DataFrame, per_burst: pd.DataFrame | None, metric_key: str,
                  n_cases: int, hparams: dict, silence_db: float, out_dir: Path) -> None:
    from research import plot_burst_comparison

    _, label, lower_is_better = next(m for m in ALL_METRICS if m[0] == metric_key)
    ranked = per_file.assign(nn_advantage=oriented_diff(per_file, metric_key, lower_is_better))
    worst = ranked[ranked["nn_advantage"] < 0].nsmallest(n_cases, "nn_advantage")
    print(f"\n===== 6. Failure cases (files where LPC beats NN by the most on {label}) =====")
    if worst.empty:
        print(f"  NN never loses to LPC on {label} -- no failure cases to plot.")
        return

    cases_dir = out_dir / "failure_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for k, row in enumerate(worst.itertuples(), start=1):
        tf = dataset_utils.load_test_file(row.file)
        candidates = None
        if per_burst is not None:
            candidates = per_burst[(per_burst["file"] == row.file) & (per_burst["clean_gap_rms_db"] > silence_db)]
        if candidates is not None and len(candidates):
            burst = candidates.loc[(candidates["gap_snr_nn"] - candidates["gap_snr_lpc"]).idxmin()]
            start, length, reason = int(burst.start_packet), int(burst.burst_length), "burst with largest LPC gap-SNR lead"
        else:
            bursts = dataset_utils.find_bursts(tf.trace)
            if not bursts:
                continue
            start, length = max(bursts, key=lambda b: b[1])
            reason = "longest burst (no per-burst data)"
        tag = f"failure{k}_{row.file}"
        plot_burst_comparison.plot_burst_comparison(tf, start, length, hparams, cases_dir, tag=tag)
        records.append({"case": k, "file": row.file, f"{metric_key}_lpc": getattr(row, f"{metric_key}_lpc"),
                        f"{metric_key}_nn": getattr(row, f"{metric_key}_nn"), "nn_advantage": row.nn_advantage,
                        "complexity_flatness": row.complexity_flatness, "loss_rate_pct": row.loss_rate_pct,
                        "plotted_burst_start": start, "plotted_burst_length": length, "burst_choice": reason})
        print(f"  case {k}: {row.file}  NN advantage {row.nn_advantage:+.3f} -> plotted "
              f"{length}-packet burst at packet {start}")
    safe_io.save_csv_safe(pd.DataFrame(records), cases_dir / "failure_cases.csv", index=False)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR,
                        help="Folder with run_metrics_comparison.py outputs.")
    parser.add_argument("--sweeps-dir", type=Path, default=SWEEPS_DIR,
                        help="Folder with run_hparam_sweeps.py outputs.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--silence-db", type=float, default=-60.0,
                        help="Bursts whose clean gap is quieter than this (dBFS RMS) are excluded from the "
                             "burst-length analysis: SNR of near-silence is meaningless.")
    parser.add_argument("--failure-metric", default="snr", choices=[m[0] for m in ALL_METRICS])
    parser.add_argument("--n-failure-cases", type=int, default=3)
    parser.add_argument("--skip-failure-cases", action="store_true",
                        help="Skip the failure-case plots (the only part that re-runs concealment).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_file_path = args.metrics_dir / "per_file_results.csv"
    per_burst_path = args.metrics_dir / "per_burst_results.csv"
    run_info_path = args.metrics_dir / "run_info.json"
    per_file = pd.read_csv(per_file_path) if per_file_path.exists() else None
    per_burst = pd.read_csv(per_burst_path) if per_burst_path.exists() else None
    run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else None
    hparams = (run_info or {}).get("hparams", DEFAULT_HPARAMS)
    missing_msg = f"(missing {{}} -- run `python -m research.run_metrics_comparison` first)"

    if per_file is not None:
        tests = significance_tests(per_file, args.out_dir)
        plot_distributions(per_file, tests, args.out_dir, args.seed)
    else:
        print(f"\n1-2. Significance tests / distributions: skipped {missing_msg.format(per_file_path.name)}")

    if per_burst is not None:
        burst_length_analysis(per_burst, args.out_dir, args.silence_db, args.seed)
    else:
        print(f"\n3. Burst-length analysis: skipped {missing_msg.format(per_burst_path.name)}")

    if per_file is not None and "ms_per_lost_packet_nn" in per_file.columns:
        realtime_analysis(per_file, run_info, args.out_dir, args.seed)
    else:
        print("\n4. Real-time feasibility: skipped (per_file_results.csv has no timing columns -- "
              "it predates timing; re-run run_metrics_comparison)")

    sweep_summary(args.sweeps_dir, args.out_dir)

    if per_file is not None and not args.skip_failure_cases:
        failure_cases(per_file, per_burst, args.failure_metric, args.n_failure_cases, hparams,
                      args.silence_db, args.out_dir)

    print(f"\nAll outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
