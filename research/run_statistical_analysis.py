"""
Statistical analysis layer for the three-way LPC / NN (Hybrid) / Sparse AR
comparison: turns the raw CSVs from the other scripts into report-ready
evidence. Apart from the failure-case plots (which re-conceal a handful of
files), it does no concealment of its own, so it is cheap to re-run after
every VM run.

Inputs (each section is skipped, with a message, if its input is missing):
  output/metrics_comparison/per_file_results.csv   run_metrics_comparison.py
  output/metrics_comparison/per_burst_results.csv  run_metrics_comparison.py
  output/metrics_comparison/run_info.json          run_metrics_comparison.py
  output/hparam_sweeps/sweep_*.csv                 run_hparam_sweeps.py

Outputs (output/statistical_analysis/):
  1. significance_tests.csv/.md   Paired Wilcoxon signed-rank test for every
                                  (metric, method pair) combination -- 4
                                  metrics x 3 pairs (LPC-NN, LPC-SparseAR,
                                  NN-SparseAR) = 12 tests, Holm-corrected
                                  together as one family, with rank-biserial
                                  effect size and a bootstrap 95% CI of the
                                  mean difference for each.
  2. metric_distributions.png     Per-file values of all three methods (box + paired lines).
     paired_differences.png       Per-file A-minus-B difference for every pair,
                                  oriented so positive always means "A better".
  3. burst_length_quality.png/.csv  Gap SNR vs. real burst length, all three
                                  methods, plus each non-baseline method's
                                  advantage over LPC. CIs use a cluster
                                  bootstrap over files, since bursts in the
                                  same file are not independent samples.
  4. realtime_feasibility.png     Real-time factor (time to conceal one packet
                                  / duration of one packet) per method; above
                                  1.0 the method cannot keep up with live audio
                                  on the machine that ran the evaluation.
  5. sweep_summary.csv/.md        Best value per hyperparameter, method and
                                  metric vs. the default, with compute cost.
                                  Each sweep only lists the methods that CSV
                                  actually evaluated (run_hparam_sweeps.py
                                  only evaluates a method against a
                                  hyperparameter that actually affects it).
  6. failure_cases/               Waveform/spectrogram/diff/crossfade plots of
                                  the files where each non-baseline method
                                  loses to LPC by the most, one subfolder per
                                  method.

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
from research.run_metrics_comparison import ALL_METRICS, METHODS, BASELINE_KEY, OUT_DIR as METRICS_DIR
from research.run_hparam_sweeps import DEFAULT_HPARAMS, OUT_DIR as SWEEPS_DIR

OUT_DIR = Path(__file__).resolve().parent / "output" / "statistical_analysis"
ALPHA = 0.05

METHOD_KEYS = [m[0] for m in METHODS]
METHOD_LABELS = {m[0]: m[1] for m in METHODS}
METHOD_COLORS = {m[0]: m[3] for m in METHODS}
METHOD_PAIRS = [(METHODS[i][0], METHODS[j][0]) for i in range(len(METHODS)) for j in range(i + 1, len(METHODS))]

# Sweep CSV "param" name -> concealment hparam key it controls.
SWEEP_PARAM_TO_HPARAM = {
    "ar_order": "ar_order",
    "context_length_packets": "context_dim_packets",
    "diagonal_load": "diagonal_load",
    "extra_dim": "extra_dim",
    "max_k_lags": "max_k_lags",
    "max_lag_l": "max_lag_l",
    "burg_order_pb": "burg_order_pb",
    "lambda_reg": "lambda_reg",
}

BURST_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 7, "4-7"), (8, None, "8+")]


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def oriented_diff(df: pd.DataFrame, key: str, lower_is_better: bool, method_a: str, method_b: str,
                  suffix: str = "") -> pd.Series:
    """method_a minus method_b, sign-flipped for lower-is-better metrics so
    that positive always means "method_a better"."""
    diff = df[f"{key}_{method_a}{suffix}"] - df[f"{key}_{method_b}{suffix}"]
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
    """Every (metric, method pair) combination is one test in the SAME
    family: Holm correction is applied once, across all of them together
    (4 metrics x 3 pairs = 12 tests here), not separately per pair -- so
    adding a third method doesn't quietly make each individual test easier
    to pass.
    """
    rows = []
    for key, label, lower_is_better in ALL_METRICS:
        for a, b in METHOD_PAIRS:
            diff = oriented_diff(per_file, key, lower_is_better, a, b)
            mean_diff, ci_lo, ci_hi = bootstrap_mean_ci(diff)
            p, r = wilcoxon_with_effect(diff)
            rows.append({
                "metric": label,
                "pair": f"{METHOD_LABELS[a]} vs {METHOD_LABELS[b]}",
                "method_a": METHOD_LABELS[a],
                "method_b": METHOD_LABELS[b],
                "n_files": len(diff),
                "mean_a": per_file[f"{key}_{a}"].mean(),
                "mean_b": per_file[f"{key}_{b}"].mean(),
                "mean_a_advantage": mean_diff,
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "median_a_advantage": float(np.median(diff)),
                "a_win_rate_pct": 100 * float((diff > 0).mean()),
                "p_wilcoxon": p,
                "rank_biserial_r": r,
                "effect_size": effect_size_label(r),
            })
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_correction(table["p_wilcoxon"].fillna(1.0))
    table["verdict"] = [
        (f"{a} better" if r > 0 else f"{b} better") if p < ALPHA else "no significant difference"
        for p, r, a, b in zip(table["p_holm"], table["rank_biserial_r"], table["method_a"], table["method_b"])
    ]

    safe_io.save_csv_safe(table, out_dir / "significance_tests.csv", index=False)
    write_markdown_table(table[["metric", "pair", "n_files", "mean_a", "mean_b", "mean_a_advantage",
                                "ci95_low", "ci95_high", "a_win_rate_pct", "p_holm",
                                "rank_biserial_r", "effect_size", "verdict"]],
                         out_dir / "significance_tests.md")

    print(f"\n===== 1. Significance tests (paired Wilcoxon, Holm-corrected over all "
          f"{len(table)} metric x pair tests) =====")
    print("(a_advantage > 0 means the first-named method is better, for every metric)")
    for pair, group in table.groupby("pair", sort=False):
        print(f"\n  {pair}:")
        for row in group.itertuples():
            print(f"    {row.metric:<16} advantage {row.mean_a_advantage:+.4f} "
                  f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}]  p_holm={row.p_holm:.3g}  "
                  f"r={row.rank_biserial_r:+.2f} ({row.effect_size}) -> {row.verdict}")
    return table


# --------------------------------------------------------------------------
# 2. Distribution plots
# --------------------------------------------------------------------------
def plot_distributions(per_file: pd.DataFrame, tests: pd.DataFrame, out_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = len(per_file)
    x_pos = list(range(len(METHOD_KEYS)))

    fig, axes = plotting.new_axes(figsize=(3.6, 4.4), ncols=len(ALL_METRICS))
    for ax, (key, label, lower_is_better) in zip(axes, ALL_METRICS):
        values = [per_file[f"{key}_{m}"].to_numpy() for m in METHOD_KEYS]
        for row in zip(*values):
            ax.plot(x_pos, row, color=plotting.TIE_COLOR, alpha=0.25, linewidth=0.6, zorder=1)
        box = ax.boxplot(values, positions=x_pos, widths=0.5, patch_artist=True,
                         showfliers=False, zorder=2, medianprops={"color": "#1f2937"})
        for patch, m in zip(box["boxes"], METHOD_KEYS):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.35)
        for x, vals, m in zip(x_pos, values, METHOD_KEYS):
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(vals)), vals, s=8,
                       color=METHOD_COLORS[m], alpha=0.7, zorder=3)
        ax.set_xticks(x_pos, [METHOD_LABELS[m].replace("This ", "") for m in METHOD_KEYS], fontsize=8)
        ax.set_title(f"{label}\n({'lower' if lower_is_better else 'higher'} = better)", fontsize=10)
        plotting.style_axis(ax)
    fig.suptitle(f"Per-file metric distributions ({n} files; gray lines connect the same file across methods)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    safe_io.save_fig_safe(fig, out_dir / "metric_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plotting.new_axes(figsize=(3.2, 3.2), ncols=len(ALL_METRICS), nrows=len(METHOD_PAIRS))
    for row_i, (a, b) in enumerate(METHOD_PAIRS):
        for col_i, (key, label, lower_is_better) in enumerate(ALL_METRICS):
            ax = axes[row_i, col_i]
            diff = oriented_diff(per_file, key, lower_is_better, a, b).to_numpy()
            test = tests[(tests["metric"] == label) & (tests["method_a"] == METHOD_LABELS[a]) &
                        (tests["method_b"] == METHOD_LABELS[b])].iloc[0]
            parts = ax.violinplot(diff, positions=[0], widths=0.7, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor("#6D5BD0")
                body.set_alpha(0.3)
            ax.scatter(rng.uniform(-0.06, 0.06, len(diff)), diff, s=8, color="#6D5BD0", alpha=0.7, zorder=3)
            ax.axhline(0, color=plotting.TIE_COLOR, linestyle="--", linewidth=1.2, zorder=1)
            ax.errorbar([0.42], [test.mean_a_advantage],
                        yerr=[[test.mean_a_advantage - test.ci95_low], [test.ci95_high - test.mean_a_advantage]],
                        fmt="o", color="#1f2937", capsize=4, markersize=4, zorder=4)
            ax.set_xticks([])
            ax.set_xlim(-0.6, 0.7)
            if row_i == 0:
                ax.set_title(label, fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f"{METHOD_LABELS[a]}\nvs {METHOD_LABELS[b]}", fontsize=7.5)
            ax.tick_params(labelsize=7)
            plotting.style_axis(ax)
    fig.suptitle("Per-file A-vs-B difference, every pair (> 0 = A better; black dot = mean with bootstrap 95% CI)",
                 fontsize=11, y=1.01)
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
    present = [m for m in METHOD_KEYS if f"gap_snr_{m}" in audible.columns]
    non_baseline_present = [m for m in present if m != BASELINE_KEY]

    rows = []
    for lo, hi, label in BURST_BINS:
        mask = audible["burst_length"] >= lo
        if hi is not None:
            mask &= audible["burst_length"] <= hi
        in_bin = audible[mask]
        if len(in_bin) == 0:
            continue
        files = in_bin["file"].to_numpy()
        row = {"burst_length_bin": label, "n_bursts": len(in_bin), "n_files": len(np.unique(files))}
        for m in present:
            mean, lo_ci, hi_ci = bootstrap_mean_ci(in_bin[f"gap_snr_{m}"], groups=files, seed=seed)
            row[f"gap_snr_{m}_mean"], row[f"gap_snr_{m}_ci_low"], row[f"gap_snr_{m}_ci_high"] = mean, lo_ci, hi_ci
        for m in non_baseline_present:
            diff = in_bin[f"gap_snr_{m}"] - in_bin[f"gap_snr_{BASELINE_KEY}"]
            mean, lo_ci, hi_ci = bootstrap_mean_ci(diff, groups=files, seed=seed)
            row[f"{m}_advantage_mean"], row[f"{m}_advantage_ci_low"], row[f"{m}_advantage_ci_high"] = mean, lo_ci, hi_ci
            row[f"{m}_win_rate_pct"] = 100 * float((diff > 0).mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    safe_io.save_csv_safe(table, out_dir / "burst_length_quality.csv", index=False)

    x = np.arange(len(table))
    tick_labels = [f"{r.burst_length_bin}\n(n={r.n_bursts})" for r in table.itertuples()]
    fig, (ax1, ax2) = plotting.new_axes(figsize=(6, 4.4), ncols=2)
    offsets = np.linspace(-0.08, 0.08, len(present)) if len(present) > 1 else [0.0]
    for m, offset in zip(present, offsets):
        mean = table[f"gap_snr_{m}_mean"]
        ax1.errorbar(x + offset, mean,
                     yerr=[mean - table[f"gap_snr_{m}_ci_low"], table[f"gap_snr_{m}_ci_high"] - mean],
                     color=METHOD_COLORS[m], marker="o", linewidth=1.8, capsize=4, label=METHOD_LABELS[m])
    ax1.set_xticks(x, tick_labels, fontsize=8)
    ax1.set_xlabel("Real burst length (packets)")
    ax1.set_ylabel("Gap SNR (dB, higher = better)")
    ax1.set_title("Quality inside the gap vs. burst length", fontsize=11)
    ax1.legend(frameon=False, fontsize=9)
    plotting.style_axis(ax1)

    offsets2 = np.linspace(-0.06, 0.06, len(non_baseline_present)) if len(non_baseline_present) > 1 else [0.0]
    for m, offset in zip(non_baseline_present, offsets2):
        mean = table[f"{m}_advantage_mean"]
        ax2.errorbar(x + offset, mean,
                     yerr=[mean - table[f"{m}_advantage_ci_low"], table[f"{m}_advantage_ci_high"] - mean],
                     color=METHOD_COLORS[m], marker="o", linewidth=1.8, capsize=4, label=METHOD_LABELS[m])
    ax2.axhline(0, color=plotting.TIE_COLOR, linestyle="--", linewidth=1.2)
    ax2.set_xticks(x, tick_labels, fontsize=8)
    ax2.set_xlabel("Real burst length (packets)")
    ax2.set_ylabel(f"Method minus {METHOD_LABELS[BASELINE_KEY]} gap SNR\n(dB, > 0 = method better)")
    ax2.set_title(f"Advantage over {METHOD_LABELS[BASELINE_KEY]} vs. burst length", fontsize=11)
    if len(non_baseline_present) > 1:
        ax2.legend(frameon=False, fontsize=9)
    plotting.style_axis(ax2)

    fig.suptitle(f"Gap SNR on every real burst ({len(audible)} bursts, {audible['file'].nunique()} files; "
                 f"{n_dropped} near-silent gaps below {silence_db:.0f} dBFS excluded)\n"
                 f"error bars = 95% CI, cluster-bootstrapped over files", fontsize=11, y=1.06)
    fig.tight_layout()
    safe_io.save_fig_safe(fig, out_dir / "burst_length_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n===== 3. Quality vs. real burst length (gap SNR, dB) =====")
    for r in table.itertuples():
        means = "  ".join(f"{METHOD_LABELS[m]} {getattr(r, f'gap_snr_{m}_mean'):6.2f}" for m in present)
        advs = "  ".join(f"{METHOD_LABELS[m]} advantage {getattr(r, f'{m}_advantage_mean'):+.2f} "
                         f"[{getattr(r, f'{m}_advantage_ci_low'):+.2f}, {getattr(r, f'{m}_advantage_ci_high'):+.2f}]"
                         for m in non_baseline_present)
        print(f"  {r.burst_length_bin:>4} pkts (n={r.n_bursts:>5}): {means}  |  {advs}")


# --------------------------------------------------------------------------
# 4. Real-time feasibility
# --------------------------------------------------------------------------
def realtime_analysis(per_file: pd.DataFrame, run_info: dict | None, out_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    present = [m for m in METHOD_KEYS if f"ms_per_lost_packet_{m}" in per_file.columns]
    timed = per_file.dropna(subset=[f"ms_per_lost_packet_{m}" for m in present])
    packet_ms = concealment.PACKET_SIZE / timed["sample_rate"] * 1000

    fig, ax = plotting.new_axes(figsize=(7, 4.6))
    print("\n===== 4. Real-time feasibility =====")
    for x, method in enumerate(present):
        color, label = METHOD_COLORS[method], METHOD_LABELS[method]
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
    ax.text(len(present) - 0.55, 1.0, "real-time limit", color="#DC2626", fontsize=9, va="bottom", ha="right")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if lo <= t <= hi]
    ax.set_yticks(ticks, [f"{t:g}x" for t in ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks(range(len(present)), [METHOD_LABELS[m] for m in present])
    ax.set_xlim(-0.5, len(present) - 0.4)
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
        # Only the methods this particular sweep actually evaluated: a
        # hyperparameter sweep only evaluates the method(s) it actually
        # affects (run_hparam_sweeps.py), e.g. an ar_order sweep has no
        # sparse_ar columns at all, since ar_order doesn't affect it.
        present = [m for m in METHOD_KEYS if f"snr_{m}" in df.columns]
        by_value = df.groupby("value").agg(
            **{f"{k}_{m}": (f"{k}_{m}", "mean") for k, _, _ in ALL_METRICS for m in present},
            **{f"time_{m}": (f"time_ms_{m}", "median") for m in present},
        )
        default_rows = by_value[np.isclose(by_value.index.astype(float), float(default))] if default is not None else []
        for method in present:
            for key, label, lower_is_better in ALL_METRICS:
                col = f"{key}_{method}"
                best_value = by_value[col].idxmin() if lower_is_better else by_value[col].idxmax()
                at_default = default_rows[col].iloc[0] if len(default_rows) else np.nan
                at_best = by_value.loc[best_value, col]
                gain = (at_default - at_best) if lower_is_better else (at_best - at_default)
                rows.append({
                    "hyperparameter": param, "method": METHOD_LABELS[method], "metric": label,
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
def failure_cases_for_method(per_file: pd.DataFrame, per_burst: pd.DataFrame | None, metric_key: str,
                             method: str, n_cases: int, hparams: dict, silence_db: float, out_dir: Path) -> None:
    """Failure cases for one non-baseline method against LPC -- the files
    where it loses by the most, each plotted via plot_burst_comparison.py's
    full three-method figure (not just the losing pair), so the third
    method's behavior on that same burst is visible for free.
    """
    from research import plot_burst_comparison

    _, label, lower_is_better = next(m for m in ALL_METRICS if m[0] == metric_key)
    method_label = METHOD_LABELS[method]
    advantage_col = f"{method}_advantage"
    ranked = per_file.assign(**{advantage_col: oriented_diff(per_file, metric_key, lower_is_better,
                                                             method, BASELINE_KEY)})
    worst = ranked[ranked[advantage_col] < 0].nsmallest(n_cases, advantage_col)
    print(f"\n===== 6. Failure cases: {method_label} vs {METHOD_LABELS[BASELINE_KEY]} "
          f"(largest losses on {label}) =====")
    if worst.empty:
        print(f"  {method_label} never loses to {METHOD_LABELS[BASELINE_KEY]} on {label} -- no failure cases to plot.")
        return

    cases_dir = out_dir / "failure_cases" / method
    cases_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for k, row in enumerate(worst.itertuples(), start=1):
        tf = dataset_utils.load_test_file(row.file)
        candidates = None
        if per_burst is not None and f"gap_snr_{method}" in per_burst.columns:
            candidates = per_burst[(per_burst["file"] == row.file) & (per_burst["clean_gap_rms_db"] > silence_db)]
        if candidates is not None and len(candidates):
            burst = candidates.loc[(candidates[f"gap_snr_{method}"] - candidates[f"gap_snr_{BASELINE_KEY}"]).idxmin()]
            start, length = int(burst.start_packet), int(burst.burst_length)
            reason = f"burst with largest {METHOD_LABELS[BASELINE_KEY]} gap-SNR lead"
        else:
            bursts = dataset_utils.find_bursts(tf.trace)
            if not bursts:
                continue
            start, length = max(bursts, key=lambda b: b[1])
            reason = "longest burst (no per-burst data)"
        tag = f"failure{k}_{row.file}"
        plot_burst_comparison.plot_burst_comparison(tf, start, length, hparams, cases_dir, tag=tag)
        advantage = getattr(row, advantage_col)
        records.append({"case": k, "file": row.file, f"{metric_key}_{method}": getattr(row, f"{metric_key}_{method}"),
                        f"{metric_key}_{BASELINE_KEY}": getattr(row, f"{metric_key}_{BASELINE_KEY}"),
                        f"{method}_advantage": advantage,
                        "complexity_flatness": row.complexity_flatness, "loss_rate_pct": row.loss_rate_pct,
                        "plotted_burst_start": start, "plotted_burst_length": length, "burst_choice": reason})
        print(f"  case {k}: {row.file}  {method_label} advantage {advantage:+.3f} -> plotted "
              f"{length}-packet burst at packet {start}")
    safe_io.save_csv_safe(pd.DataFrame(records), cases_dir / "failure_cases.csv", index=False)


def failure_cases(per_file: pd.DataFrame, per_burst: pd.DataFrame | None, metric_key: str,
                  n_cases: int, hparams: dict, silence_db: float, out_dir: Path) -> None:
    for method in METHOD_KEYS:
        if method == BASELINE_KEY or f"{metric_key}_{method}" not in per_file.columns:
            continue
        failure_cases_for_method(per_file, per_burst, metric_key, method, n_cases, hparams, silence_db, out_dir)


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
