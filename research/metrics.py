"""
Audio quality metrics for comparing PLC concealment methods.

Two families, matching what any PLC paper would report:

  PHYSICAL (waveform/spectrum error, no model of hearing):
    - snr_db                 time-domain signal-to-noise ratio
    - si_sdr_db              scale-invariant SNR (tolerant of a gain mismatch)
    - log_spectral_distance  per-frame spectral error, in dB
    - multires_stft_distance multi-resolution spectral error (vocoder-style loss)

  PSYCHOACOUSTIC (models of human hearing, score meant to track what a
  listener would actually perceive):
    - plcmos_score            Microsoft's PLCMOS (INTERSPEECH 2023), reference-free,
                               purpose-built for packet-loss concealment quality
    - visqol_score             Google ViSQOL MOS-LQO, full-band "audio" mode
                               (via the pure-Python `visqol-python` package --
                               no WSL/Docker needed, runs natively on Windows)

All physical metrics operate at whatever sample rate the audio already is --
no forced downsampling. PLCMOS and ViSQOL have fixed operating rates dictated
by their underlying models (16kHz and 48kHz respectively); that resampling
happens only inside those two functions, not anywhere else in this file.
"""
import numpy as np
import librosa


# --------------------------------------------------------------------------
# Physical metrics
# --------------------------------------------------------------------------
def snr_db(ref: np.ndarray, out: np.ndarray, eps: float = 1e-12) -> float:
    """Time-domain signal-to-noise ratio, in dB. Higher is better."""
    err = ref - out
    return 10 * np.log10((np.sum(ref ** 2) + eps) / (np.sum(err ** 2) + eps))


def si_sdr_db(ref: np.ndarray, out: np.ndarray, eps: float = 1e-12) -> float:
    """Scale-invariant SDR, in dB. Higher is better.

    Rescales `out` to best match `ref`'s amplitude before scoring the
    remaining error, so a pure volume mismatch isn't punished the way plain
    SNR would punish it.
    """
    alpha = np.dot(out, ref) / (np.dot(ref, ref) + eps)
    ref_scaled = alpha * ref
    noise = out - ref_scaled
    return 10 * np.log10((np.sum(ref_scaled ** 2) + eps) / (np.sum(noise ** 2) + eps))


def log_spectral_distance(ref: np.ndarray, out: np.ndarray, n_fft: int = 1024,
                           hop: int = 256, eps: float = 1e-8) -> float:
    """Log-spectral distance, in dB, at the signal's native sample rate.
    Lower is better; 0 = identical magnitude spectra.
    """
    s_ref = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop))
    s_out = np.abs(librosa.stft(out, n_fft=n_fft, hop_length=hop))
    diff_db = 20 * np.log10(np.maximum(s_ref, eps)) - 20 * np.log10(np.maximum(s_out, eps))
    return float(np.mean(np.sqrt(np.mean(diff_db ** 2, axis=0))))


def multires_stft_distance(ref: np.ndarray, out: np.ndarray,
                            fft_sizes=(512, 1024, 2048), eps: float = 1e-8) -> float:
    """Mean of (spectral convergence + log-magnitude L1) across several FFT
    sizes, at the signal's native sample rate. Lower is better; 0 = identical.
    Same family of loss used to judge vocoder/synthesis audio quality.
    """
    total = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        s_ref = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop))
        s_out = np.abs(librosa.stft(out, n_fft=n_fft, hop_length=hop))
        sc = np.linalg.norm(s_ref - s_out) / (np.linalg.norm(s_ref) + eps)
        log_mag = np.mean(np.abs(np.log(np.maximum(s_ref, eps)) - np.log(np.maximum(s_out, eps))))
        total += sc + log_mag
    return total / len(fft_sizes)


PHYSICAL_METRICS = {
    "snr": (snr_db, False),
    "sisdr": (si_sdr_db, False),
    "lsd": (log_spectral_distance, True),
    "mrstft": (multires_stft_distance, True),
}  # name -> (function, lower_is_better)


# --------------------------------------------------------------------------
# Psychoacoustic metrics
# --------------------------------------------------------------------------
_plcmos_model = None


def plcmos_score(audio: np.ndarray, native_sr: int) -> float:
    """Reference-free PLCMOS score (roughly 1-5, higher = more natural-sounding
    to a human listener). The model only operates at 16kHz; that resampling is
    local to this function and doesn't affect any other metric.
    """
    import speechmos.plcmos
    audio_16k = librosa.resample(audio, orig_sr=native_sr, target_sr=16000)
    audio_16k = np.clip(audio_16k, -1.0, 1.0).astype(np.float32)
    return speechmos.plcmos.run(audio_16k, sr=16000)["plcmos"]


_visqol_api = None


def visqol_score(ref: np.ndarray, out: np.ndarray, native_sr: int) -> float:
    """Google ViSQOL MOS-LQO score (full-band "audio" mode, roughly 1-5,
    higher is better). ViSQOL's audio mode operates at 48kHz; that resampling
    is local to this function. The underlying VisqolApi instance (loading the
    SVR model) is created once and reused -- it's the expensive part.
    """
    import visqol as _visqol
    global _visqol_api
    if _visqol_api is None:
        _visqol_api = _visqol.VisqolApi()
        _visqol_api.create(mode="audio")

    ref_48k = librosa.resample(ref, orig_sr=native_sr, target_sr=48000).astype(np.float64)
    out_48k = librosa.resample(out, orig_sr=native_sr, target_sr=48000).astype(np.float64)
    n = min(len(ref_48k), len(out_48k))
    result = _visqol_api.measure_from_arrays(ref_48k[:n], out_48k[:n], 48000)
    return result.moslqo


PSYCHOACOUSTIC_METRICS = {
    "plcmos": (plcmos_score, False),   # signature differs (no ref) -- handled by caller
    "visqol": (visqol_score, False),
}
