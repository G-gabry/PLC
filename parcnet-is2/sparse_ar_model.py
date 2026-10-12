"""
Sparse AR packet-loss concealment, implemented from Severi (2025), "Orthogonal
Matching Pursuit based Linear prediction for Real-time Packet Loss Concealment"
-- not a port of the author's own code (none was available), but a from-scratch
implementation of the paper's equations and pseudocode, built to the same
calling convention as ar_model.py's ARModel so it can be swapped in wherever
that is used.

ONE DELIBERATE, EMPIRICALLY-JUSTIFIED DEPARTURE FROM THE PAPER'S LITERAL TEXT,
documented here because it materially changes behavior:

  Algorithm 1, applied literally (global argmax|rho(d)| over d=1..L on the RAW
  signal's autocorrelation, with the paper's own stopping threshold of 0.55),
  was tested against 60 real single-packet losses across 10 files from this
  project's dataset. It selected lag=1 in 60/60 cases and scored ~0 dB SNR
  (near-silence) in nearly all of them, against AR(256)'s mean of 2.2 dB. The
  cause: ordinary audio sampled well above its bandwidth has very high
  sample-to-sample correlation purely from being smooth/bandlimited, which
  has nothing to do with musical periodicity -- and that correlation sits at
  lag 1, almost always above the paper's 0.55 early-stop threshold, so the
  search fires and stops before ever looking far enough out to find a real
  pitch period. A single lag-1 coefficient below 1 is also, by construction,
  a pure exponential decay to zero, which is a poor model for anything but a
  literally decaying signal.

  The fix implemented here is the standard short-term/long-term prediction
  split used throughout speech and audio coding (including OPUS/SILK -- the
  exact codec this paper's own Burg citation, Vos (2013), comes from): the
  Burg model first whitens the context (removes its short-term, non-periodic
  structure), and the OMP lag search runs on THAT RESIDUAL instead of the raw
  signal. Periodicity shows up as real peaks in a whitened signal's
  autocorrelation; in raw audio those peaks are swamped by the smoothness-
  driven decay described above. Verified on the same 60 bursts: real,
  plausible pitch-period lags are selected (e.g. a lag of 672 samples at
  44.1kHz is ~66Hz) and Sparse AR wins or matches AR(256) on roughly half of
  them, instead of 0/60.

  This also removes the paper's own two-phase structure (Burg for the first
  m_b samples, then SAR for the rest, spliced together via a search for the
  best crossfade point c* in equation 6): with the long-term correction
  defined on the Burg residual, the natural and standard way to synthesize is
  a single continuous recursion -- at every step, predict the short-term part
  from the Burg filter and the long-term part from the sparse lags, and add
  them -- which needs no splice or crossfade search at all. See
  SparseARModel.predict for the full recursion.

Two further, smaller choices, where the paper's description was
underspecified:

  1. The Burg fit itself (Section II-E-1) cites Vos (2013)'s fast,
     autocorrelation-based approximation of Burg's method, but that
     algorithm's details aren't in this paper. This module instead
     implements the classical, exact Burg recursion (Numerical Recipes'
     "memcof", a standard reference algorithm) directly on the time-domain
     history. Verified against a pure sinusoid (order 4 recovers it to
     <0.001 RMSE over a 100-sample continuation).

  2. The paper's own "right boundary" crossfade (w=32, Section II-E-3) that
     splices the prediction back into the next received packet is not
     implemented here. That role is instead played by the surrounding
     concealment pipeline's own crossfade (extra_dim-based fade_in/fade_out
     in research/concealment.py), which every method in this project already
     shares -- so predictions from this model are directly comparable to
     conceal_lpc/conceal_nn's, rather than being crossfaded twice.

Everything else (OMP lag selection with a skip radius around chosen lags, the
remaining stopping criteria, Tikhonov-regularized coefficient
re-optimization, the signal-conditioning compressor/limiter) follows the
paper's Sections II and V directly; see each function's docstring for the
exact equation or sub-section it implements.
"""
import warnings

import numpy as np
import scipy.signal
from numba import njit

# Fixed internal constants from the paper's Section V-A/V-B -- not exposed as
# sweep axes because the paper treats them as implementation choices, not the
# model's primary hyperparameters (those are k, L, p_b and lambda).
MIN_CORR_FRAC = 0.2      # minimum |rho|/r0 to accept a candidate lag (V-A)
EARLY_STOP_FRAC = 0.55   # stop after 1 lag if its |rho|/r0 exceeds this (V-A)
SKIP_RADIUS = 80         # samples excluded around each chosen lag (V-A)
COMPRESSOR_DECAY = 0.995 # per-sample damping once the hard limiter first triggers (V-B)


@njit
def _burg_coeffs(data: np.ndarray, order: int) -> np.ndarray:
    """Classical Burg recursion (Numerical Recipes' "memcof"): returns AR
    coefficients d[0..order-1] such that x[n] ~= sum_i d[i]*x[n-1-i]. Ported
    directly from that reference algorithm's 1-indexed pseudocode.
    """
    n = len(data)
    wk1 = np.empty(n)
    wk2 = np.empty(n)
    wkm = np.zeros(order)
    d = np.zeros(order)

    wk1[0] = data[0]
    wk2[n - 2] = data[n - 1]
    for j in range(1, n - 1):
        wk1[j] = data[j]
        wk2[j - 1] = data[j]

    for k in range(order):
        num, denom = 0.0, 0.0
        for j in range(n - k - 1):
            num += wk1[j] * wk2[j]
            denom += wk1[j] * wk1[j] + wk2[j] * wk2[j]
        dk = 2.0 * num / denom if denom != 0.0 else 0.0
        d[k] = dk
        for i in range(k):
            d[i] = wkm[i] - dk * wkm[k - 1 - i]
        if k == order - 1:
            return d
        for i in range(k + 1):
            wkm[i] = d[i]
        for j in range(n - k - 2):
            new_wk1_j = wk1[j] - wkm[k] * wk2[j]
            wk2[j] = wk2[j + 1] - wkm[k] * wk1[j + 1]
            wk1[j] = new_wk1_j
    return d


@njit
def _burg_residual(data: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """The Burg (short-term) model's own prediction error on its own fitting
    data: e[n] = data[n] - sum_i coeff[i]*data[n-1-i], for n >= order. This
    is what the OMP lag search runs on -- see the module docstring for why.
    """
    order = len(coeff)
    n = len(data)
    e = np.zeros(n - order)
    for t in range(order, n):
        pred = 0.0
        for i in range(order):
            pred += coeff[i] * data[t - 1 - i]
        e[t - order] = data[t] - pred
    return e


def _select_lags(acf: np.ndarray, max_lag_l: int, max_k_lags: int, lambda_reg: float) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1 (OMP lag selection) + Section II-C's efficient candidate-
    correlation formula (equation 2) + Section II-D's regularized
    coefficient re-optimization (equation 3), with the three stopping
    criteria from Section V-A (minimum correlation, early termination after
    one strong lag, and a skip radius around every chosen lag so nearby,
    near-redundant lags can't all be picked). Runs on the autocorrelation of
    the Burg residual (see module docstring), not the raw signal.
    """
    r0 = acf[0]
    if r0 <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0)

    excluded = np.zeros(max_lag_l + 1, dtype=bool)
    lags: list[int] = []
    phis = np.empty(0)

    for j in range(max_k_lags):
        best_d, best_rho, best_abs = -1, 0.0, -1.0
        for d in range(1, max_lag_l + 1):
            if excluded[d]:
                continue
            rho = acf[d]
            for i, lag in enumerate(lags):
                rho -= phis[i] * acf[abs(d - lag)]
            if abs(rho) > best_abs:
                best_d, best_rho, best_abs = d, rho, abs(rho)

        if best_d == -1 or best_abs <= MIN_CORR_FRAC * r0:
            break

        lags.append(best_d)
        lo, hi = max(1, best_d - SKIP_RADIUS), min(max_lag_l, best_d + SKIP_RADIUS)
        excluded[lo:hi + 1] = True

        k = len(lags)
        R = np.empty((k, k))
        rvec = np.empty(k)
        for a in range(k):
            rvec[a] = acf[lags[a]]
            for b in range(k):
                R[a, b] = acf[abs(lags[a] - lags[b])]
        R[np.diag_indices(k)] += lambda_reg * r0
        try:
            phis = np.linalg.solve(R, rvec)
        except np.linalg.LinAlgError:
            lags.pop()
            break

        if j == 0 and best_abs > EARLY_STOP_FRAC * r0:
            break

    return np.array(lags, dtype=np.int64), phis


@njit
def _cascade_synthesis(s_tail: np.ndarray, e_tail: np.ndarray, burg_coeff: np.ndarray,
                       lags: np.ndarray, phis: np.ndarray, steps: int) -> np.ndarray:
    """Short-term + long-term synthesis recursion, the standard structure
    behind LPC-based speech/audio codecs (short-term = Burg, long-term =
    the sparse OMP lags fit on Burg's own residual): at every step, predict
    the short-term part from the rolling sample buffer via the Burg filter,
    predict the long-term/periodic correction from the rolling excitation
    buffer via the sparse lags, and combine them. Both buffers are then
    extended with the new values so later steps can recurse on this
    packet's own predictions exactly as they would on real audio.
    """
    p = len(burg_coeff)
    max_lag = 0
    for lag in lags:
        if lag > max_lag:
            max_lag = lag

    s_buf = np.zeros(p + steps)
    for i in range(p):
        src = len(s_tail) - p + i
        s_buf[i] = s_tail[src] if src >= 0 else 0.0

    e_buf = np.zeros(max_lag + steps)
    for i in range(max_lag):
        src = len(e_tail) - max_lag + i
        e_buf[i] = e_tail[src] if src >= 0 else 0.0

    pred = np.zeros(steps)
    for t in range(steps):
        s_pred = 0.0
        for i in range(p):
            s_pred += burg_coeff[i] * s_buf[p + t - 1 - i]
        e_pred = 0.0
        for k in range(len(lags)):
            e_pred += phis[k] * e_buf[max_lag + t - lags[k]]
        new_sample = s_pred + e_pred
        s_buf[p + t] = new_sample
        e_buf[max_lag + t] = e_pred
        pred[t] = new_sample
    return pred


@njit
def _condition_signal(pred: np.ndarray, hist_max: float, decay: float) -> np.ndarray:
    """Section V-B's signal conditioning: a soft-knee compressor between
    1.01x and 1.5x the historical context's peak amplitude (a C1-continuous
    cubic Hermite spline -- value- and slope-matched to the linear region at
    the inner edge, flat at the outer edge), hard limiting beyond that, and
    -- once the limiter first triggers within this packet -- a gradual
    exponential damping of every later sample, so a run of out-of-range
    predictions decays toward silence instead of staying pinned at the
    ceiling or oscillating at it.
    """
    if hist_max <= 0.0:
        hist_max = 1e-6
    linear_th = 1.01 * hist_max
    hard_th = 1.5 * hist_max
    span = hard_th - linear_th

    out = np.empty_like(pred)
    damp = 1.0
    triggered = False
    for i in range(len(pred)):
        v = pred[i] * damp
        av = abs(v)
        if av <= linear_th:
            out[i] = v
        elif av >= hard_th:
            out[i] = hard_th if v > 0 else -hard_th
            triggered = True
        else:
            t = (av - linear_th) / span
            h00 = 2 * t ** 3 - 3 * t ** 2 + 1
            h10 = t ** 3 - 2 * t ** 2 + t
            h01 = -2 * t ** 3 + 3 * t ** 2
            mag = h00 * linear_th + h10 * span + h01 * hard_th  # m0=span (slope 1 at x0), m1=0 (flat at x1)
            out[i] = mag if v > 0 else -mag
        if triggered:
            damp *= decay
    return out


class SparseARModel:
    """Sparse AR concealer: a Burg(p_b) short-term model plus OMP-selected
    non-contiguous long-term lags (Algorithm 1) fit on the Burg residual --
    see the module docstring for why the lag search runs on the residual
    rather than the raw signal, and why prediction is one continuous
    recursion rather than the paper's own two-phase splice. Same calling
    convention as ar_model.py's ARModel (`.predict(valid, steps)`), so it
    drops into the same per-packet recursive loop research/concealment.py
    already uses for the other two methods.
    """

    def __init__(self, max_k_lags: int = 3, max_lag_l: int = 1024, burg_order_pb: int = 8,
                 lambda_reg: float = 0.01):
        """
            :param max_k_lags: k, maximum number of sparse long-term lags (paper default 3)
            :param max_lag_l: L, maximum lag searched (paper default 1024)
            :param burg_order_pb: p_b, Burg (short-term) model order (paper: <=8)
            :param lambda_reg: Tikhonov regularization as a fraction of r0 (paper: 0.01)
        """
        self.max_k_lags = max_k_lags
        self.max_lag_l = max_lag_l
        self.burg_order_pb = burg_order_pb
        self.lambda_reg = lambda_reg

        # Pre-compile the numba-decorated functions to expedite future calls.
        _burg_coeffs(np.zeros(32), min(burg_order_pb, 4))
        _burg_residual(np.zeros(32), np.ones(4))
        _cascade_synthesis(np.zeros(16), np.zeros(16), np.ones(4), np.array([1, 2], dtype=np.int64),
                          np.array([0.1, 0.1]), 1)
        _condition_signal(np.zeros(4), 1.0, COMPRESSOR_DECAY)

    def predict(self, valid: np.ndarray, steps: int) -> np.ndarray:
        """Fits the model from an array of valid past samples and predicts
        `steps` samples ahead. Mirrors ARModel.predict's fail-safe: returns
        silence if the result is ever NaN (degenerate fit), since the
        compressor already keeps well-formed predictions bounded.

            :param valid: ndarray of past samples
            :param steps: the number of samples to be predicted
            :return: ndarray of predicted samples
        """
        valid = valid.astype(np.float64)
        burg_order = min(self.burg_order_pb, len(valid) - 1)
        burg_coeff = _burg_coeffs(valid, burg_order)
        residual = _burg_residual(valid, burg_coeff)

        max_lag_l = min(self.max_lag_l, len(residual) - 1)
        if max_lag_l < 1:
            lags, phis = np.empty(0, dtype=np.int64), np.empty(0)
        else:
            acf_full = scipy.signal.correlate(residual, residual, mode='full', method='auto')
            if np.any(np.isnan(acf_full)):
                acf_full = scipy.signal.correlate(residual, residual, mode='full', method='direct')
            zero_lag = len(acf_full) // 2
            acf = acf_full[zero_lag:zero_lag + max_lag_l + 1]
            lags, phis = _select_lags(acf, max_lag_l, self.max_k_lags, self.lambda_reg)

        pred = _cascade_synthesis(
            s_tail=valid,
            e_tail=residual,
            burg_coeff=np.ascontiguousarray(burg_coeff, dtype=np.float64),
            lags=lags,
            phis=phis,
            steps=steps,
        )

        hist_max = float(np.max(np.abs(valid))) if len(valid) else 0.0
        pred = _condition_signal(pred, hist_max, COMPRESSOR_DECAY)

        if np.any(np.isnan(pred)):
            warnings.warn('Sparse AR prediction contains NaN', RuntimeWarning)
            return np.zeros_like(pred)
        return pred
