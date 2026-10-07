"""
Batch-tests AR-only (LPC) vs AR+NN (PARCnet) concealment across many
(IRMAS file, real captured trace) pairs, using full-band metrics that do NOT
require downsampling -- unlike PLCMOS, which forces everything to 16kHz and
throws away all content above 8kHz (a real problem for music: cymbals, string
harmonics, and other high-frequency detail live well above that).

Metrics (all computed at the file's native 44.1kHz, full-file):
  - SNR       (time domain, higher = better)              -- already used elsewhere
  - SI-SDR    (scale-invariant SNR, higher = better)       -- robust to gain mismatch
  - LSD       (log-spectral distance in dB, lower = better) -- spectral-domain error
  - MR-STFT   (multi-resolution STFT distance, lower = better) -- same family of
              loss used to judge vocoder/synthesis quality in audio ML

Pairs the first N sorted IRMAS files with the first N sorted real captured
traces (1:1, arbitrary but deterministic pairing -- there's no natural
correspondence between a music file and a trace, we just want variety of both),
runs both concealment methods packet-by-packet (batch-equivalent to real-time),
then reports win-rate statistics: on how many files does NN actually beat LPC,
per metric.
"""
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import speechmos.plcmos

from live_sim_engine import SimConfig, LiveConcealer, LiveConcealerNN, load_nn_model

# ---------------------------------------------------------------------------
IRMAS_DIR = Path(r"D:\IRMAS dataset\IRMAS-TestingData-Part3\IRMAS-TestingData-Part3\Part3")
TRACES_DIR = Path(r"C:\Users\g\PLC\PLC_NN\2024-music-plc-challenge\parcnet-is2\example_test_set\traces")
CHECKPOINT = Path(r"C:\Users\g\PLC\PLC_NN\2024-music-plc-challenge\parcnet-is2\pretrained_models\parcnet-is2_baseline_checkpoint.ckpt")
OUT_DIR = Path(r"C:\Users\g\PLC\PLC_NN\2024-music-plc-challenge\irmas_batch_output")
OUT_DIR.mkdir(exist_ok=True)

PACKET_SIZE = 512
SAMPLE_RATE = 44100
N_FILES = 40

config = SimConfig(packet_size=PACKET_SIZE, order=256, diagonal_load=0.001,
                    context_dim=4096, extra_dim=256, nn_context_dim=4096,
                    nn_fade_dim=64, equal_power=False, sample_rate=SAMPLE_RATE)

print("Loading PARCnet NN checkpoint...")
nn_model = load_nn_model(CHECKPOINT, packet_dim=PACKET_SIZE, extra_pred_dim=config.extra_dim,
                          lite=True, device="cpu")

irmas_files = sorted(IRMAS_DIR.glob("*.wav"))[:N_FILES]
trace_files = sorted(TRACES_DIR.glob("*.txt"))[:N_FILES]
print(f"Pairing {len(irmas_files)} IRMAS files with {len(trace_files)} trace files (1:1, sorted order).")


# --- Full-band metrics -- none of these downsample or need a pretrained model ---
def snr_db(ref, out, eps=1e-12):
    err = ref - out
    return 10 * np.log10((np.sum(ref ** 2) + eps) / (np.sum(err ** 2) + eps))


def si_sdr_db(ref, out, eps=1e-12):
    """Scale-invariant SDR -- projects out onto ref, scores what's left as noise."""
    alpha = np.dot(out, ref) / (np.dot(ref, ref) + eps)
    ref_scaled = alpha * ref
    noise = out - ref_scaled
    return 10 * np.log10((np.sum(ref_scaled ** 2) + eps) / (np.sum(noise ** 2) + eps))


def log_spectral_distance(ref, out, n_fft=1024, hop=256, eps=1e-8):
    """LSD in dB at the native sample rate -- lower is better."""
    s_ref = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop))
    s_out = np.abs(librosa.stft(out, n_fft=n_fft, hop_length=hop))
    diff_db = 20 * np.log10(np.maximum(s_ref, eps)) - 20 * np.log10(np.maximum(s_out, eps))
    return np.mean(np.sqrt(np.mean(diff_db ** 2, axis=0)))


def multires_stft_distance(ref, out, fft_sizes=(512, 1024, 2048), eps=1e-8):
    """Mean of (spectral convergence + log-magnitude L1) across several FFT
    sizes, at the native sample rate -- lower is better."""
    total = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        s_ref = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop))
        s_out = np.abs(librosa.stft(out, n_fft=n_fft, hop_length=hop))
        sc = np.linalg.norm(s_ref - s_out) / (np.linalg.norm(s_ref) + eps)
        log_mag = np.mean(np.abs(np.log(np.maximum(s_ref, eps)) - np.log(np.maximum(s_out, eps))))
        total += sc + log_mag
    return total / len(fft_sizes)


def plcmos_score(audio, native_sr):
    """Reference-free PLCMOS score. The model only operates at 16kHz, so this
    resamples internally -- unlike the other metrics above, which stay at the
    native rate. Clip first: concealment can occasionally overshoot [-1, 1]
    slightly, which PLCMOS rejects outright."""
    audio_16k = librosa.resample(audio, orig_sr=native_sr, target_sr=16000)
    audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32)
    return speechmos.plcmos.run(audio_16k, sr=16000)["plcmos"]


rows = []
for i, (irmas_path, trace_path) in enumerate(zip(irmas_files, trace_files), start=1):
    trace = np.loadtxt(trace_path, dtype=int).flatten()
    clean, _ = librosa.load(irmas_path, sr=SAMPLE_RATE, mono=True)

    n_packets = min(len(trace), len(clean) // PACKET_SIZE)
    if n_packets < 20:
        print(f"[{i}/{len(irmas_files)}] SKIP {irmas_path.name}: only {n_packets} packets, too short.")
        continue

    clean = clean[:n_packets * PACKET_SIZE]
    trace = trace[:n_packets]
    lossy = clean.copy()
    lossy[np.repeat(trace, PACKET_SIZE).astype(bool)] = 0.0

    # Fresh stepper state per file -- the NN checkpoint itself is loaded once
    # and reused (that's the expensive part), but each file is an independent
    # stream so the AR/crossfade state must not leak across files.
    concealer_ar = LiveConcealer(config)
    concealer_nn = LiveConcealerNN(config, nn_model, device="cpu")
    this_lpc = np.zeros_like(clean)
    this_nn = np.zeros_like(clean)
    for p in range(n_packets):
        idx = p * PACKET_SIZE
        real_chunk = None if trace[p] else lossy[idx:idx + PACKET_SIZE]
        this_lpc[idx:idx + PACKET_SIZE] = concealer_ar.step(real_chunk)
        this_nn[idx:idx + PACKET_SIZE] = concealer_nn.step(real_chunk)

    row = {
        "file": irmas_path.name,
        "trace": trace_path.name,
        "n_packets": n_packets,
        "loss_rate_pct": trace.mean() * 100,
        "snr_lpc": snr_db(clean, this_lpc), "snr_nn": snr_db(clean, this_nn),
        "sisdr_lpc": si_sdr_db(clean, this_lpc), "sisdr_nn": si_sdr_db(clean, this_nn),
        "lsd_lpc": log_spectral_distance(clean, this_lpc), "lsd_nn": log_spectral_distance(clean, this_nn),
        "mrstft_lpc": multires_stft_distance(clean, this_lpc), "mrstft_nn": multires_stft_distance(clean, this_nn),
        "plcmos_lpc": plcmos_score(this_lpc, SAMPLE_RATE), "plcmos_nn": plcmos_score(this_nn, SAMPLE_RATE),
    }
    rows.append(row)
    print(f"[{i}/{len(irmas_files)}] {irmas_path.name} ({n_packets} pkts, "
          f"{row['loss_rate_pct']:.1f}% lost): SNR lpc={row['snr_lpc']:.2f} nn={row['snr_nn']:.2f} dB")

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "batch_results.csv", index=False)

METRICS = [("snr", "SNR (dB)", False), ("sisdr", "SI-SDR (dB)", False),
           ("lsd", "LSD (dB)", True), ("mrstft", "MR-STFT dist.", True),
           ("plcmos", "PLCMOS", False)]
CHART_METRICS = {"SNR (dB)", "PLCMOS"}

print(f"\n===== Aggregate results over {len(df)} files =====")
win_rates = {}
for key, label, lower_is_better in METRICS:
    lpc_col, nn_col = f"{key}_lpc", f"{key}_nn"
    nn_wins = (df[nn_col] < df[lpc_col]).sum() if lower_is_better else (df[nn_col] > df[lpc_col]).sum()
    total = len(df)
    win_rates[label] = 100 * nn_wins / total
    print(f"\n{label} ({'lower=better' if lower_is_better else 'higher=better'}):")
    print(f"  NN better on {nn_wins}/{total} files ({win_rates[label]:.1f}%)")
    print(f"  Mean  -- LPC: {df[lpc_col].mean():.4f}   NN: {df[nn_col].mean():.4f}")
    print(f"  Median-- LPC: {df[lpc_col].median():.4f}   NN: {df[nn_col].median():.4f}")

# --- Summary chart: how often does NN actually beat LPC (SNR vs. PLCMOS only) ---
chart_win_rates = {k: v for k, v in win_rates.items() if k in CHART_METRICS}
fig, ax = plt.subplots(figsize=(6, 4.5))
labels = list(chart_win_rates.keys())
values = list(chart_win_rates.values())
bar_color = "#3B82C4"      # single series -> one categorical hue, no rainbow
tie_color = "#9CA3AF"      # neutral gray reference line at 50%
bars = ax.bar(labels, values, color=bar_color, width=0.55, zorder=3)
ax.axhline(50, color=tie_color, linewidth=1.5, linestyle="--", zorder=2)
ax.text(len(labels) - 0.4, 51.5, "50% = coin flip", color=tie_color, fontsize=9, ha="right")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.0f}%",
            ha="center", va="bottom", fontsize=10, color="#1f2937")
ax.set_ylim(0, 100)
ax.set_ylabel("Files where NN beats LPC (%)")
ax.set_title(f"NN vs. LPC win rate across {len(df)} IRMAS files, by metric")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
fig.tight_layout()
fig.savefig(OUT_DIR / "win_rate_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nPer-file results:  {OUT_DIR / 'batch_results.csv'}")
print(f"Win-rate summary:  {OUT_DIR / 'win_rate_summary.png'}")
