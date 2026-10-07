"""
Signal complexity, operationalized as spectral flatness: a standard DSP
measure of how tonal/predictable (near 0) vs. noise-like/complex (near 1) a
signal is -- the ratio of the geometric mean to the arithmetic mean of the
power spectrum. Used as the x-axis for "does concealment quality depend on
how complex the signal is" plots, instead of a manual genre/instrument label.
"""
import numpy as np
import librosa


def spectral_flatness(audio: np.ndarray, n_fft: int = 1024, hop: int = 256) -> float:
    """Mean spectral flatness over the whole signal, in [0, 1].
    0 = perfectly tonal (e.g. a pure sine), 1 = perfectly flat/noise-like.
    """
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop)
    return float(np.mean(flatness))
