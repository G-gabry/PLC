"""
Tests AR-only and AR+NN concealment (live_sim_engine.py) on a real IRMAS music
file, damaged with an ACTUAL captured network trace -- not a synthetic loss
model. Uses the same batch/matplotlib plotting style as plc_challenge.py's
compare_all_methods (waveform+spectrogram grid, difference figure, gap MSE,
full-file SNR), not the live web simulator.
"""
import time
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from live_sim_engine import SimConfig, LiveConcealer, LiveConcealerNN, load_nn_model


def save_fig(fig, path, **kwargs):
    """fig.savefig with retries -- Windows intermittently holds a brief lock on
    freshly-written files in this folder (antivirus/indexer scanning them),
    which raises OSError(22, 'Invalid argument') on the very next write."""
    for attempt in range(5):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.3)

# ---------------------------------------------------------------------------
#IRMAS_FILE = r"D:\IRMAS dataset\IRMAS-TestingData-Part3\IRMAS-TestingData-Part3\Part3\02 bwv 1068 air on g string-1.wav"
IRMAS_FILE = r"D:\IRMAS dataset\IRMAS-TestingData-Part3\IRMAS-TestingData-Part3\Part3\02. School Boy-1.wav"
TRACE_FILE = r"C:\Users\g\PLC\PLC_NN\2024-music-plc-challenge\parcnet-is2\example_test_set\traces\1UaHCp-394-87506-140741-186260-68344-3434-1214.txt"
CHECKPOINT = r"C:\Users\g\PLC\PLC_NN\2024-music-plc-challenge\parcnet-is2\pretrained_models\parcnet-is2_baseline_checkpoint.ckpt"
OUT_DIR = Path(r"C:\Users\g\PLC\PLC_NN\2024-music-plc-challenge\irmas_test_output")
OUT_DIR.mkdir(exist_ok=True)

PACKET_SIZE = 512
SAMPLE_RATE = 44100

config = SimConfig(packet_size=PACKET_SIZE, order=256, diagonal_load=0.001,
                    context_dim=4096, extra_dim=256, nn_context_dim=4096,
                    nn_fade_dim=64, equal_power=False, sample_rate=SAMPLE_RATE)

# --- Load + align: same sample rate, audio packet count matched to the trace ---
clean, sr = librosa.load(IRMAS_FILE, sr=SAMPLE_RATE, mono=True)
trace = np.loadtxt(TRACE_FILE, dtype=int).flatten()

n_packets_audio = len(clean) // PACKET_SIZE
n_packets = min(len(trace), n_packets_audio)
print(f"IRMAS file packets available: {n_packets_audio}, trace packets: {len(trace)} -> using {n_packets}")

clean = clean[:n_packets * PACKET_SIZE].astype(np.float64)
trace = trace[:n_packets]

lossy = clean.copy()
loss_mask = np.repeat(trace, PACKET_SIZE).astype(bool)
lossy[loss_mask] = 0.0

sf.write(OUT_DIR / "irmas_clean.wav", clean, SAMPLE_RATE)
sf.write(OUT_DIR / "irmas_lossy.wav", lossy, SAMPLE_RATE)

# --- Run both concealment methods, packet by packet (batch-equivalent) ---
print("Loading PARCnet NN checkpoint...")
nn_model = load_nn_model(CHECKPOINT, packet_dim=PACKET_SIZE, extra_pred_dim=config.extra_dim, lite=True, device="cpu")

concealer_ar = LiveConcealer(config)
concealer_nn = LiveConcealerNN(config, nn_model, device="cpu")

this_lpc = np.zeros_like(clean)
this_nn = np.zeros_like(clean)

print("Running concealment over the file...")
for i in range(n_packets):
    idx = i * PACKET_SIZE
    lost = bool(trace[i])
    real_chunk = None if lost else lossy[idx:idx + PACKET_SIZE]
    this_lpc[idx:idx + PACKET_SIZE] = concealer_ar.step(real_chunk)
    this_nn[idx:idx + PACKET_SIZE] = concealer_nn.step(real_chunk)
    if i % 200 == 0:
        print(f"  packet {i}/{n_packets}")

sf.write(OUT_DIR / "irmas_this_lpc.wav", this_lpc, SAMPLE_RATE)
sf.write(OUT_DIR / "irmas_this_nn.wav", this_nn, SAMPLE_RATE)


def snr_db(ref, out):
    err = ref - out
    return 10 * np.log10(np.sum(ref ** 2) / np.sum(err ** 2))


print(f"\nFull-file SNR -- This LPC (AR-only): {snr_db(clean, this_lpc):.2f} dB")
print(f"Full-file SNR -- This NN (AR+PARCnet): {snr_db(clean, this_nn):.2f} dB")

# PLCMOS (Microsoft, INTERSPEECH 2023) -- a neural Mean Opinion Score predictor
# purpose-built for packet-loss concealment quality. Unlike SNR, it's
# reference-free (no clean file needed, works even on the challenge's own
# no-ground-truth example_test_set) and predicts how natural/listenable the
# audio sounds to a human, on roughly a 1-5 scale, rather than raw error energy.
import speechmos.plcmos

print("\nPLCMOS (higher = more natural-sounding to a human listener):")
for label, path in [("Clean (ceiling)", "irmas_clean.wav"),
                     ("Transmitted (raw gaps)", "irmas_lossy.wav"),
                     ("This LPC (AR-only)", "irmas_this_lpc.wav"),
                     ("This NN (AR+PARCnet)", "irmas_this_nn.wav")]:
    score = speechmos.plcmos.run(str(OUT_DIR / path), sr=16000, verbose=False)["plcmos"]
    print(f"  {label}: {score:.3f}")


def extract_bursts(seq):
    bursts, i = [], 0
    while i < len(seq):
        if seq[i]:
            start = i
            while i < len(seq) and seq[i]:
                i += 1
            bursts.append((start, i - start))
        else:
            i += 1
    return bursts


bursts = extract_bursts(trace)
print(f"\nBursts found in the (truncated) trace: {len(bursts)}")


def plot_burst(start_packet, burst_len):
    gap_start = start_packet * PACKET_SIZE
    gap_size = burst_len * PACKET_SIZE
    gap_end = gap_start + gap_size

    signals = [
        ("clean", "Clean (ground truth)", clean, "#999999"),
        ("transmitted", "Transmitted (silence = lost)", lossy, "#4f9dff"),
        ("this_lpc", "This LPC (AR-only, recursive)", this_lpc, "#e8871e"),
        ("this_nn", "This NN (AR + PARCnet)", this_nn, "#1baf7a"),
    ]

    context_ms = 40
    context = int(SAMPLE_RATE * context_ms / 1000)
    win_start = max(0, gap_start - context)
    win_end = min(len(clean), gap_end + context)
    t_ms = (np.arange(win_start, win_end) - gap_start) / SAMPLE_RATE * 1000
    gap_end_ms = gap_size / SAMPLE_RATE * 1000
    y_max = max(np.max(np.abs(sig[win_start:win_end])) for _, _, sig, _ in signals) * 1.1

    fig, axes = plt.subplots(2, 4, figsize=(19, 8), gridspec_kw={"height_ratios": [1, 1.15]})
    for col, (name, label, sig, color) in enumerate(signals):
        ax = axes[0, col]
        ax.plot(t_ms, sig[win_start:win_end], color=color, linewidth=1.0)
        ax.axvspan(0, gap_end_ms, color="black", alpha=0.08)
        # "transmitted" has no meaningful SNR in the gap (it's just zeroed
        # silence there), so only annotate the two actual concealment methods.
        if name in ("this_lpc", "this_nn"):
            gap_snr = snr_db(clean[gap_start:gap_end], sig[gap_start:gap_end])
            title = f"{label}\nGap SNR: {gap_snr:.2f} dB"
        else:
            title = label
        ax.set_title(title, fontsize=11, color=color)
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
        img = librosa.display.specshow(specs_db[name], sr=SAMPLE_RATE, hop_length=64, x_axis="time", y_axis="hz",
                                        ax=ax, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"Spectrogram \u2014 {label}", fontsize=9, color=color)
        ax.set_ylim(0, 8000)
        ax.set_xlabel("Time (s)")
        if col == 0:
            ax.set_ylabel("Frequency (Hz)")

    fig.colorbar(img, ax=axes[1, :], format="%+2.0f dB", fraction=0.02, pad=0.02, label="Magnitude (dB)")
    fig.suptitle(f"IRMAS test file \u2014 {burst_len}-packet burst at packet {start_packet}", fontsize=13, y=1.0)
    save_fig(fig, OUT_DIR / f"irmas_compare_{burst_len}pkt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    diff_pairs = [(1, 2), (1, 3), (2, 3)]  # transmitted-vs-lpc, transmitted-vs-nn, lpc-vs-nn (skip "clean")
    diffs = [signals[i][2][win_start:win_end] - signals[j][2][win_start:win_end] for i, j in diff_pairs]
    diff_y_max = max(np.max(np.abs(d)) for d in diffs) * 1.1

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    rows = []
    for ax, (i, j), diff in zip(axes2, diff_pairs, diffs):
        name_a, label_a, sig_a, _ = signals[i]
        name_b, label_b, sig_b, _ = signals[j]
        gap_mse = np.mean((sig_a[gap_start:gap_end] - sig_b[gap_start:gap_end]) ** 2)
        rows.append((f"{name_a} vs {name_b}", gap_mse))
        ax.plot(t_ms, diff, color="#9b6bff", linewidth=1.0)
        ax.axvspan(0, gap_end_ms, color="black", alpha=0.08)
        ax.set_title(f"{label_a} minus {label_b}\nGap MSE: {gap_mse:.3e}", fontsize=9)
        ax.set_ylim(-diff_y_max, diff_y_max)
        ax.set_xlabel("Time relative to gap start (ms)")
    fig2.tight_layout()
    save_fig(fig2, OUT_DIR / f"irmas_diff_{burst_len}pkt.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nGap-region MSE ({burst_len}-packet burst at packet {start_packet}):")
    for name, mse in rows:
        print(f"  {name}: {mse:.6e}")

    # The MSE table above only compares the methods against each other (or
    # against the zeroed "transmitted" signal) -- it never checks against the
    # actual clean ground truth. This is the localized version of the
    # full-file SNR, restricted to just this burst's gap, so quality can be
    # compared burst-by-burst instead of only as one aggregate number.
    print(f"Gap-region SNR vs. clean ({burst_len}-packet burst):")
    print(f"  this_lpc: {snr_db(clean[gap_start:gap_end], this_lpc[gap_start:gap_end]):.2f} dB")
    print(f"  this_nn:  {snr_db(clean[gap_start:gap_end], this_nn[gap_start:gap_end]):.2f} dB")


TARGET_BURST_LENGTHS = [2, 4, 8, 16]
for target_len in TARGET_BURST_LENGTHS:
    match = next((b for b in bursts if b[1] == target_len), None)
    if match is None:
        print(f"\nNo {target_len}-packet burst found in this (truncated) trace -- skipping.")
        continue
    start_packet, burst_len = match
    print(f"\nPlotting a {burst_len}-packet burst: packet {start_packet}")
    plot_burst(start_packet, burst_len)

print(f"\nAll outputs saved to {OUT_DIR}")
