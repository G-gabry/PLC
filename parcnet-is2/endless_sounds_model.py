"""
Causal "Endless Sounds" packet-loss concealment, adapted from Svento (2026),
"Audio Inpainting with Endless Sounds" (EEICT 2026) -- a spectral
phase-randomisation technique originally built for *offline*, two-sided audio
inpainting (real context both before AND after the gap). This module is a
from-scratch, CAUSAL reimplementation built to the same calling convention as
ar_model.py's ARModel / sparse_ar_model.py's SparseARModel, so it drops into
the same per-packet recursive loop research/concealment.py already uses for
the other three methods.

WHY THIS IS NOT A PORT OF THE PAPER'S OWN PIPELINE:

  The paper's full inpainting pipeline (Fig. 2/3, Section III) generates TWO
  endless extensions -- yL grown forward from the left context, yR grown
  BACKWARD (reversed time) from the right context -- normalises and
  time-aligns each to its neighbouring real audio, then crossfades yL and yR
  together inside the gap (equation 6). That needs the real audio AFTER the
  gap, which a real-time PLC system does not have: the lost packet hasn't
  arrived yet (that is the entire problem being solved), and nothing
  downstream of "now" is available at concealment time.

  This implementation only ever builds the equivalent of yL (Section III-C's
  "Context-Extension Blending", part (a)) -- the one operation in the
  paper's own pipeline that doesn't need future audio. There is no yR, no
  crossfade-point optimisation, and no right-boundary blend: that role is
  played by the surrounding concealment pipeline's own outbound crossfade
  (extra_dim-based fade_in/fade_out in research/concealment.py), exactly the
  same substitution already made for Sparse AR -- see sparse_ar_model.py's
  module docstring, point 2. This keeps every method's output comparable
  through one shared external crossfade instead of four different
  boundary-handling schemes.

  One further, causality-forced departure: the paper's own left-side blend
  (Fig. 2, "Left-side crossfade sL") crossfades the TAIL of the real left
  context itself into the synthetic extension -- i.e. it retroactively
  reshapes a few already-real samples. In a real-time stream those samples
  have already been received (and, in a live system, already played); they
  cannot be rewritten, and predict() has no mechanism to do so anyway -- it
  can only ever return NEW samples, never mutate `valid`. The adaptation
  here keeps the paper's equal-power crossfade shape (its equation 5) but
  applies it entirely within the returned prediction: the first
  `boundary_blend` samples of the output fade from a verbatim copy of the
  context's own last `boundary_blend` real samples into the new synthetic
  continuation, instead of fading the stored context itself.

ALGORITHM (see EndlessSoundsModel.predict for the exact steps):

  1. Zero-pad the causal context (samples already received, or this
     method's own earlier predictions within the same burst -- see
     predict()'s docstring) by exactly `steps` samples. Not a larger FFT
     than that: the magnitude spectrum is already fixed by the context
     itself, so padding further spends compute on a finer frequency grid
     without adding any real information about the signal.
  2. Real FFT of the zero-padded context; keep the magnitude spectrum,
     replace every bin's phase with an independent U(-pi, pi) draw -- except
     the DC bin and (for an even FFT size) the Nyquist bin, which are copied
     through unchanged, since randomising them would make the inverse FFT
     produce a complex (non-real) signal. The paper's Section II does not
     discuss these two edge bins at all; this is this implementation's own
     explicit choice, not something the source specifies.
  3. Inverse real FFT back to the time domain. Because the context was
     zero-padded *after* its own samples, the tail of this resynthesis
     (beyond where the real context ended) is newly-generated material with
     the same magnitude spectrum as the real context, but no longer aligned
     to it sample-for-sample -- the "endless sounds" trick from Valimaki et
     al. (2018, the paper's own reference [7]) applied to a context-sized
     window instead of an arbitrarily long one.
  4. Energy-normalise the synthetic tail to the context's own RMS (paper's
     "Normalization & Alignment" stage, Fig. 3/Section III-B -- a slice of a
     phase-randomised IFFT preserves the *whole*-window energy (Parseval)
     but not a sub-slice's local energy, so without this the prediction can
     come out noticeably quieter or louder than the real audio around it).
  5. Equal-power crossfade (paper's equation 5: w1 = cos(pi*t/2),
     w2 = sin(pi*t/2)) into the first `boundary_blend` samples of that
     normalised tail. The paper's own version blends xL[-Le+k] with yL[k]
     walking forward together (both already anchored at the same real
     boundary). That alignment isn't available causally -- this
     implementation instead plays the context's own last `boundary_blend`
     samples in REVERSE (so blend position 0 lands exactly on valid[-1], the
     true last real sample, and blend position Le-1 lands on valid[-Le]) and
     crossfades that into the synthetic tail. This guarantees the prediction
     starts at the real signal's actual last value instead of jumping to an
     arbitrary point `boundary_blend` samples in the past -- a short
     reverse-and-fade is itself a standard, independently-known PLC
     technique (not from this paper), used here specifically because it is
     the simplest way to satisfy "begins at the true boundary" under a
     one-sided causal constraint.

Two hyperparameters, both specific to this method (not shared with the other
three beyond context_dim_packets/extra_dim, which are structural and shared
by all four):
  - `boundary_blend`: length of the real-to-synthetic crossfade (samples).
    The paper establishes that this blend exists (Section III-C) but never
    gives it a numeric default; this implementation's choice, exposed here
    so it can be swept like any other hyperparameter in this project.
  - `seed`: this instance's phase-randomisation RNG is seeded once at
    construction, not reset on every predict() call, so one model instance
    produces a reproducible sequence of packets across one file (useful for
    fair, repeatable hyperparameter sweeps) without repeating the exact same
    random phase pattern packet to packet, which would sound like a comb
    filter / metallic repetition.
"""
import warnings

import numpy as np


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n (n >= 1), for efficient FFT sizing."""
    n = max(1, n)
    return 1 << (n - 1).bit_length()


class EndlessSoundsModel:
    """Causal spectral phase-randomisation concealer -- see this module's
    docstring for the algorithm and its causal departures from Svento
    (2026)'s own two-sided inpainting pipeline. Same calling convention as
    ar_model.py's ARModel / sparse_ar_model.py's SparseARModel
    (`.predict(valid, steps)`), so it drops into the same per-packet
    recursive loop research/concealment.py already uses for the other three
    methods.
    """

    def __init__(self, boundary_blend: int = 64, seed: int = 0):
        """
            :param boundary_blend: Le, length (samples) of the equal-power
                crossfade between the real context's own tail and the new
                synthetic continuation (paper: Section III-C-a; see module
                docstring for why there is no paper-given numeric default)
            :param seed: seeds this instance's own RNG once, at construction
                -- see module docstring for why this isn't reseeded per call
        """
        self.boundary_blend = boundary_blend
        self._rng = np.random.default_rng(seed)

    def predict(self, valid: np.ndarray, steps: int) -> np.ndarray:
        """Generates `steps` new samples continuing `valid`.

        `valid` is the causal context: samples already received, or (for the
        second and later packets of a multi-packet burst, via the shared
        per-packet loop in research/concealment.py) this same model's own
        earlier predictions within this burst -- never the hidden/future
        ground truth. This mirrors ARModel/SparseARModel's existing
        recursive convention exactly; see concealment.py's `output` array,
        which every method reads its context from and writes its own
        predictions into before the next packet is processed.

            :param valid: ndarray of past samples
            :param steps: the number of samples to be predicted
            :return: ndarray of predicted samples, length == steps
        """
        valid = np.asarray(valid, dtype=np.float64)
        n_ctx = len(valid)
        if n_ctx == 0:
            return np.zeros(steps)

        fft_size = _next_pow2(n_ctx + steps)
        padded = np.zeros(fft_size)
        padded[:n_ctx] = valid

        spectrum = np.fft.rfft(padded)
        magnitude = np.abs(spectrum)

        phase = self._rng.uniform(-np.pi, np.pi, size=len(spectrum))
        new_spectrum = magnitude * np.exp(1j * phase)
        # DC and (for an even fft_size) Nyquist must stay exactly real for a
        # real-valued inverse FFT -- copied through unchanged rather than
        # phase-randomised. See module docstring: the paper doesn't address
        # these two edge bins.
        new_spectrum[0] = spectrum[0]
        if fft_size % 2 == 0:
            new_spectrum[-1] = spectrum[-1]

        resynth = np.fft.irfft(new_spectrum, n=fft_size)

        # The tail beyond where the real context ended is new, spectrally-
        # consistent-but-phase-randomised material -- the actual prediction.
        synth_tail = resynth[n_ctx:n_ctx + steps]

        # Energy normalisation (paper's Fig. 3 "Normalization & Alignment"):
        # a sub-slice of a phase-randomised IFFT doesn't preserve the
        # context's own local RMS, only the whole-window energy.
        ctx_rms = np.sqrt(np.mean(valid ** 2))
        tail_rms = np.sqrt(np.mean(synth_tail ** 2))
        if tail_rms > 1e-12:
            synth_tail = synth_tail * (ctx_rms / tail_rms)

        out = synth_tail.copy()
        blend = min(self.boundary_blend, steps, n_ctx)
        if blend > 0:
            t = np.linspace(0.0, 1.0, blend)
            w_real = np.cos(0.5 * np.pi * t)    # fades out: context's own real tail
            w_synth = np.sin(0.5 * np.pi * t)   # fades in: new synthetic continuation
            real_component = valid[-1:-blend - 1:-1]  # last `blend` real samples, REVERSED:
            # position 0 == valid[-1] (the true boundary), position blend-1 == valid[-blend]
            out[:blend] = w_real * real_component + w_synth * synth_tail[:blend]

        if np.any(~np.isfinite(out)):
            warnings.warn('Endless Sounds prediction contains NaN/Inf', RuntimeWarning)
            return np.zeros(steps)

        return out
