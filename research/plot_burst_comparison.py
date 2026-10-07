"""
Waveform + spectrogram + diff visualizations for a single real burst, on
both methods -- the thing the aggregate charts in run_metrics_comparison.py
and run_hparam_sweeps.py deliberately don't show: what the concealment
actually looks like, not just a win-rate percentage or a sweep curve. Also
produces the dataset-wide network-loss analysis and the crossfade-boundary
zoom -- between this script and the other two, every plot made over the
course of this project has a home.

For one (file, burst) pair, produces three figures. Colors/font sizes/
figsize style are carried over from the project's original
plc_challenge_test.py / PLC_Analysis.py comparison plots, but the column
layout reflects this project's actual two methods -- not the original's
3-column layout (This LPC (recursive) / LPC without NN (pipeline) / LPC
with NN (PARCnet)), whose first two columns are now provably bit-identical
(conceal_lpc vs PARCnet's own AR branch, MSE=0.0 -- see concealment.py).
Showing that duplicate as a third column added nothing once that was
proven, so it's dropped:
  - A 2-column grid (This LPC (recursive) / LPC with NN (PARCnet)),
    waveform on top and spectrogram below.
  - A single difference figure (this_lpc vs lpc_with_nn) with the pair's
    Gap MSE in its title.
  - A crossfade-boundary zoom (new, not in the old scripts): tightly cropped
    on the handback point where concealment splices back into real audio,
    with a dashed line at the exact splice and a shaded box for the
    crossfade region -- the static-plot equivalent of the live simulator's
    "Crossfade boundary, zoomed" panels.

Also, once per run (not per burst): a network-loss analysis across every
real trace in the dataset -- global burst-length distribution and per-file
packet-loss-rate distribution -- ported from PLC_Analysis.py's own seaborn
whitegrid design, reusing dataset_utils.find_bursts instead of a separate counter.
(seaborn's styling is applied via a scoped `with sns.axes_style(...)` block,
not the original's global sns.set_theme() -- that would leak into every
other figure this script draws afterward, since this script runs multiple
plot functions in one process, unlike the original one-shot notebook cells.)

Uses the real, standalone conceal_lpc/conceal_nn functions -- not a
reimplementation -- so these plots are guaranteed consistent with every
other number in this project.

Usage:
    python -m research.plot_burst_comparison --n-examples 3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from research import dataset_utils, concealment, plotting, safe_io

OUT_DIR = Path(__file__).resolve().parent / "output" / "burst_comparisons"
PACKET_SIZE = concealment.PACKET_SIZE


def plot_network_analysis(out_dir: Path) -> None:
    """Dataset-wide network-loss characteristics across every real trace in
    example_test_set: global burst-length distribution and per-file
    packet-loss-rate distribution. Ported from the project's original
    PLC_Analysis.py (run_network_analysis), reusing dataset_utils.find_bursts
    instead of a separate burst-counting implementation. Runs once per
    script invocation, not once per burst example -- it's a property of the
    whole dataset, not of any one file.
    """
    stems = dataset_utils.list_available_stems()
    traces_dir = dataset_utils.EXAMPLE_TEST_SET_DIR / "traces"

    all_packets = []
    all_bursts = []
    per_file_plr = []
    for stem in stems:
        trace = np.loadtxt(traces_dir / f"{stem}.txt", dtype=int).flatten()
        if len(trace) == 0:
            continue
        all_packets.append(trace)
        per_file_plr.append(float(trace.mean()))
        all_bursts.extend(length for _, length in dataset_utils.find_bursts(trace))

    combined = np.concatenate(all_packets)
    plr = combined.mean()
    print(f"Overall Packet Loss Rate (PLR): {plr * 100:.2f}%")
    if all_bursts:
        print(f"Average Loss Burst Length     : {np.mean(all_bursts):.2f} packets")

    # Exact design from PLC_Analysis.py's run_network_analysis: seaborn
    # whitegrid theme, these specific colors/figsize/font sizes. Scoped via
    # `with sns.axes_style(...)` rather than the original's global
    # sns.set_theme() so it doesn't leak into this script's other figures.
    df_micro = pd.DataFrame({"packet_loss_rate": per_file_plr})
    with sns.axes_style("whitegrid"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sns.histplot(all_bursts, bins=range(1, max(all_bursts) + 2),
                     discrete=True, ax=ax1, color="#2ecc71")
        ax1.set_yscale("log")
        ax1.set_title("Global Burst Length Distribution", fontsize=14, pad=15)
        ax1.set_xlabel("Consecutive Packets Lost (Burst Length)", fontsize=12)
        ax1.set_ylabel("Count (Log Scale)", fontsize=12)
        ax1.set_xticks(range(1, max(all_bursts) + 1))

        sns.histplot(data=df_micro, x="packet_loss_rate", bins=20, ax=ax2, color="#e74c3c")
        ax2.set_title("Distribution of Packet Loss Rates Across Files", fontsize=14, pad=15)
        ax2.set_xlabel("Average Packet Loss Rate per File", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        global_avg = df_micro["packet_loss_rate"].mean()
        ax2.axvline(global_avg, color="black", linestyle="--", label=f"Global Avg ({global_avg * 100:.1f}%)")
        ax2.legend()

        fig.tight_layout()
    path = out_dir / "network_statistics.png"
    actual_path = safe_io.save_fig_safe(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {actual_path}")


def plot_crossfade_boundary(tf: dataset_utils.TestFile, burst_start_packet: int, burst_length: int,
                             this_lpc: np.ndarray, this_nn: np.ndarray, hparams: dict,
                             out_dir: Path, tag: str, zoom_ms: float = 15) -> Path:
    """Zoomed view of the handback boundary -- the moment concealment
    splices back into real audio -- with a dashed line at the exact splice
    point and a shaded box for the crossfade (fade-in) region's actual
    extent (hparams['extra_dim'] samples). Mirrors the live web simulator's
    "Crossfade boundary, zoomed" panels as a static plot. Takes this_lpc/
    this_nn as arguments rather than recomputing them, since the caller
    (plot_burst_comparison) already has them.
    """
    sr = tf.sample_rate
    extra_dim = hparams.get("extra_dim", 256)
    gap_end = (burst_start_packet + burst_length) * PACKET_SIZE
    crossfade_ms = extra_dim / sr * 1000

    zoom = int(sr * zoom_ms / 1000)
    win_start = max(0, gap_end - zoom)
    win_end = min(len(tf.clean), gap_end + zoom)
    t_ms = (np.arange(win_start, win_end) - gap_end) / sr * 1000

    signals = [
        ("clean", "Clean (ground truth)", tf.clean, "#999999"),
        ("this_lpc", "This LPC (recursive)", this_lpc, "#2a78d6"),
        ("this_nn", "LPC with NN (PARCnet)", this_nn, "#1baf7a"),
    ]
    y_max = max(np.max(np.abs(sig[win_start:win_end])) for _, _, sig, _ in signals) * 1.1

    fig, axes = plotting.new_axes(figsize=(5, 3.4), ncols=3)
    for ax, (_, label, sig, color) in zip(axes, signals):
        ax.plot(t_ms, sig[win_start:win_end], color=color, linewidth=1.2)
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.2, zorder=2)
        ax.axvspan(0, crossfade_ms, color=plotting.TIE_COLOR, alpha=0.25, zorder=1)
        ax.set_title(label, fontsize=10, color=color)
        ax.set_ylim(-y_max, y_max)
        ax.set_xlabel("Time rel. to splice point (ms)", fontsize=8)
        plotting.style_axis(ax, grid_axis="both")
    axes[0].set_ylabel("Amplitude")
    fig.suptitle(f"{tag} -- crossfade boundary (handback), {burst_length}-packet burst\n"
                 f"dashed = splice point, shaded = {crossfade_ms:.1f}ms crossfade region",
                 fontsize=11, y=1.1)
    fig.tight_layout()
    path = out_dir / f"crossfade_{tag}_{burst_length}pkt.png"
    actual_path = safe_io.save_fig_safe(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return actual_path


def plot_burst_comparison(tf: dataset_utils.TestFile, burst_start_packet: int, burst_length: int,
                           hparams: dict, out_dir: Path, context_ms: float = 40,
                           tag: str | None = None) -> dict:
    """Runs both methods on tf's real lossy/trace (not a synthetic burst --
    this visualizes an actual loss event from the dataset), then saves the
    waveform/spectrogram grid and the pairwise-diff figure for the given
    burst. Returns the pairwise gap-MSE table as a dict.
    """
    this_lpc_recursive = concealment.conceal_lpc(tf.lossy, tf.trace, **hparams)
    lpc_with_nn = concealment.conceal_nn(tf.lossy, tf.trace, **hparams)

    gap_start = burst_start_packet * PACKET_SIZE
    gap_end = gap_start + burst_length * PACKET_SIZE
    sr = tf.sample_rate
    tag = tag or tf.stem

    # Colors carried over from plc_challenge_test.py's compare_all_methods.
    signals = [
        ("this_lpc", "This LPC (recursive)", this_lpc_recursive, "#2a78d6"),
        ("lpc_with_nn", "LPC with NN (PARCnet)", lpc_with_nn, "#1baf7a"),
    ]

    context = int(sr * context_ms / 1000)
    win_start = max(0, gap_start - context)
    win_end = min(len(tf.clean), gap_end + context)
    t_ms = (np.arange(win_start, win_end) - gap_start) / sr * 1000
    gap_end_ms = (gap_end - gap_start) / sr * 1000
    y_max = max(np.max(np.abs(sig[win_start:win_end])) for _, _, sig, _ in signals) * 1.1

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1.15]})
    for col, (_, label, sig, color) in enumerate(signals):
        ax = axes[0, col]
        ax.plot(t_ms, sig[win_start:win_end], color=color, linewidth=1.1)
        ax.axvspan(0, gap_end_ms, color="black", alpha=0.08)
        ax.set_title(label, fontsize=10, color=color)
        ax.set_ylim(-y_max, y_max)
        ax.set_xlabel("Time relative to gap start (ms)")
        if col == 0:
            ax.set_ylabel("Amplitude")

    specs_db = {}
    for name, _, sig, _ in signals:
        stft = librosa.stft(sig[win_start:win_end], n_fft=1024, hop_length=64)
        specs_db[name] = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    vmin = min(s.min() for s in specs_db.values())
    vmax = max(s.max() for s in specs_db.values())

    for col, (name, label, _, color) in enumerate(signals):
        ax = axes[1, col]
        img = librosa.display.specshow(specs_db[name], sr=sr, hop_length=64, x_axis="time", y_axis="hz",
                                        ax=ax, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"Spectrogram -- {label}", fontsize=9, color=color)
        ax.set_ylim(0, 8000)
        ax.set_xlabel("Time (s)")
        if col == 0:
            ax.set_ylabel("Frequency (Hz)")

    fig.colorbar(img, ax=axes[1, :], format="%+2.0f dB", fraction=0.03, pad=0.02, label="Magnitude (dB)")
    fig.suptitle(f"{tag} -- {burst_length}-packet burst", fontsize=13, y=1.0)
    fig.subplots_adjust(hspace=0.65, top=0.88, right=0.88)
    compare_path = out_dir / f"compare_{tag}_{burst_length}pkt.png"
    safe_io.save_fig(fig, compare_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    name_a, label_a, sig_a, _ = signals[0]
    name_b, label_b, sig_b, _ = signals[1]
    diff = sig_a[win_start:win_end] - sig_b[win_start:win_end]
    gap_mse = float(np.mean((sig_a[gap_start:gap_end] - sig_b[gap_start:gap_end]) ** 2))
    rows = {f"{name_a}_vs_{name_b}": gap_mse}

    diff_fig, diff_ax = plt.subplots(figsize=(6, 4))
    diff_ax.plot(t_ms, diff, color="#7d5ba6", linewidth=1.0)
    diff_ax.axvspan(0, gap_end_ms, color="black", alpha=0.08)
    diff_ax.set_title(f"{label_a} minus {label_b}\nGap MSE: {gap_mse:.3e}", fontsize=10)
    diff_ax.set_xlabel("Time relative to gap start (ms)")
    diff_fig.suptitle(f"{tag} -- difference, {burst_length}-packet burst", fontsize=12, y=1.02)
    diff_fig.tight_layout()
    diff_path = out_dir / f"diff_{tag}_{burst_length}pkt.png"
    safe_io.save_fig(diff_fig, diff_path, dpi=150, bbox_inches="tight")
    plt.close(diff_fig)

    plot_crossfade_boundary(tf, burst_start_packet, burst_length, this_lpc_recursive, lpc_with_nn,
                             hparams, out_dir, tag)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-burst-lengths", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                         help="Burst lengths (packets) to find a real example of and plot.")
    parser.add_argument("--n-examples", type=int, default=1,
                         help="How many example files/bursts to plot per target length.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ar-order", type=int, default=256)
    parser.add_argument("--diagonal-load", type=float, default=0.001)
    parser.add_argument("--context-dim-packets", type=int, default=8)
    parser.add_argument("--extra-dim", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-network-analysis", action="store_true",
                         help="Skip the dataset-wide network-loss analysis (it's cheap -- just reads "
                              "trace files, no audio/concealment -- so there's rarely a reason to skip it).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hparams = dict(ar_order=args.ar_order, diagonal_load=args.diagonal_load,
                   context_dim_packets=args.context_dim_packets, extra_dim=args.extra_dim)

    if not args.skip_network_analysis:
        plot_network_analysis(args.out_dir)
        print()

    stems = dataset_utils.list_available_stems()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(stems)

    remaining = {length: args.n_examples for length in args.target_burst_lengths}
    all_rows = []
    for stem in stems:
        if not remaining:
            break
        tf = dataset_utils.load_test_file(stem)
        for start_packet, length in dataset_utils.find_bursts(tf.trace):
            if remaining.get(length, 0) <= 0:
                continue
            print(f"burst_length={length}: using {stem} (packet {start_packet})")
            gap_mse = plot_burst_comparison(tf, start_packet, length, hparams, args.out_dir)
            all_rows.append({"file": stem, "burst_length": length, "start_packet": start_packet, **gap_mse})
            remaining[length] -= 1
            if remaining[length] <= 0:
                del remaining[length]

    for length in remaining:
        print(f"No example found for burst_length={length} (ran out of files).")

    import pandas as pd
    df = pd.DataFrame(all_rows)
    safe_io.save_csv(df, args.out_dir / "gap_mse_summary.csv", index=False)
    print(f"\nSaved {len(all_rows)} burst comparison(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
