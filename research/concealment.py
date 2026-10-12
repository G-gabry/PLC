"""
Three genuinely independent, fully-parameterized concealment implementations
-- not one class with a toggle:

  conceal_lpc       -- pure AR(order) recursive linear prediction. Zero
                        NN/torch dependency: no checkpoint load, no neural
                        network import, for this path at all. This is the
                        standalone LPC-recursive implementation (same
                        algorithm as the project's own earlier
                        LPCConcealer/lpc_conceal), used on its own merits,
                        not PARCnet with its NN switched off.

  conceal_nn        -- the real production PARCnet class
                        (parcnet-is2/parcnet.py), always with its neural
                        network enabled.

  conceal_sparse_ar -- a from-scratch implementation of Severi (2025)'s
                        OMP-based sparse AR concealer (parcnet-is2/
                        sparse_ar_model.py's SparseARModel): a short Burg
                        model plus up to a few non-contiguous, OMP-selected
                        long-term lags, in place of a single dense AR(order)
                        fit. Its own hyperparameters (max_k_lags, max_lag_l,
                        burg_order_pb, lambda_reg) are independent of the
                        other two methods' -- see sparse_ar_model.py's module
                        docstring for the algorithm and, importantly, the
                        two places this implementation had to depart from
                        the paper's literal text to behave sensibly on real
                        audio, with the empirical evidence for each.

conceal_lpc and conceal_nn were cross-validated against each other at the AR
level: given the same hyperparameters, conceal_lpc's output is bit-identical
(float32 noise floor) to PARCnet's own AR branch, because both implement the
identical autocorrelation + Levinson-Durbin fit (ar_model.py's ARModel) and
the identical crossfade logic. conceal_sparse_ar is a separate algorithm
entirely and does not participate in that check; it reuses the same external
crossfade loop (fade_in/fade_out over extra_dim) as conceal_lpc purely so its
output is mechanically comparable to the other two, not because it shares
any model-fitting logic with them.

Every hyperparameter that matters for the sweeps (AR order, diagonal loading,
context length, extra/crossfade dimension; or, for Sparse AR, its own four)
is a plain function argument with a sensible default (matching
parcnet-is2/config/config.yaml for the first two methods), never hardcoded
into logic.
"""
import sys
from pathlib import Path

import numpy as np

_PARCNET_DIR = Path(__file__).resolve().parent.parent / "parcnet-is2"
if str(_PARCNET_DIR) not in sys.path:
    sys.path.append(str(_PARCNET_DIR))

from ar_model import ARModel  # noqa: E402  (path must be set up first)

CHECKPOINT = _PARCNET_DIR / "pretrained_models" / "parcnet-is2_baseline_checkpoint.ckpt"
PACKET_SIZE = 512          # matches config.yaml's global.packet_dim
NN_FADE_DIM = 64           # matches config.yaml's neural_net.fade_dim


# --------------------------------------------------------------------------
# LPC recursive (AR-only) -- standalone, no NN dependency whatsoever
# --------------------------------------------------------------------------
def conceal_lpc(clean_or_lossy: np.ndarray, trace: np.ndarray, ar_order: int = 256,
                 diagonal_load: float = 0.001, context_dim_packets: int = 8,
                 extra_dim: int = 256, equal_power: bool = False,
                 packet_dim: int = PACKET_SIZE, **_other_methods_only_kwargs) -> np.ndarray:
    """Pure AR(ar_order) recursive packet-loss concealment: PARCnet's own
    per-packet loop and crossfade logic, with the neural network never
    imported or invoked (not just zeroed out -- this function has no torch
    dependency at all). `**_other_methods_only_kwargs` absorbs the other two
    methods' own keywords (nn_fade_dim/lite/device for conceal_nn;
    max_k_lags/max_lag_l/burg_order_pb/lambda_reg for conceal_sparse_ar) so
    all three can share one call signature in code that loops over them
    generically.
    """
    context_dim = context_dim_packets * packet_dim
    pred_dim = packet_dim + extra_dim
    ar_model = ARModel(ar_order, diagonal_load)
    output = np.pad(clean_or_lossy.copy(), (0, extra_dim))
    is_burst = False

    if extra_dim > 0:
        linear = np.linspace(0., 1., extra_dim)
        fade_in = np.sqrt(linear) if equal_power else linear
        fade_out = np.sqrt(1. - linear) if equal_power else 1. - linear

    for i, lost in enumerate(trace):
        if not lost:
            is_burst = False
            continue

        idx = i * packet_dim
        valid = output[max(0, idx - context_dim):idx]
        valid = np.pad(valid, (context_dim - len(valid), 0))
        prediction = ar_model.predict(valid, pred_dim)

        if extra_dim > 0:
            prediction[-extra_dim:] *= fade_out
            if is_burst:
                prediction[:extra_dim] *= fade_in
            output[idx + packet_dim:idx + pred_dim] *= fade_in

        output[idx:idx + pred_dim] += prediction
        is_burst = True

    return output[:len(clean_or_lossy)]


# --------------------------------------------------------------------------
# AR + NN -- the real production PARCnet class, NN always enabled
# --------------------------------------------------------------------------
_parcnet_cache: dict = {}  # keyed on (packet_dim, extra_dim, lite, device)


def get_nn_concealer(ar_order: int = 256, diagonal_load: float = 0.001,
                      context_dim_packets: int = 8, extra_dim: int = 256,
                      nn_fade_dim: int = NN_FADE_DIM, packet_dim: int = PACKET_SIZE,
                      lite: bool = True, device: str = "cpu", **_sparse_ar_only_kwargs):
    """Returns a real PARCnet instance (NN always enabled) configured with
    the given hyperparameters. Loading the NN checkpoint is the expensive
    part (~1-2s) and doesn't depend on ar_order/diagonal_load/
    context_dim_packets, so instances are cached by the parameters that
    actually require a reload, and the cheap AR/context attributes are
    overwritten on every call -- this is what makes sweeping ar_order a
    thousand times over fast instead of reloading the checkpoint each time.
    Imports PARCnet lazily so that conceal_lpc (and anything that only uses
    it) never triggers a torch import at all. `**_sparse_ar_only_kwargs`
    absorbs conceal_sparse_ar's own keywords (max_k_lags, max_lag_l,
    burg_order_pb, lambda_reg), same reason as conceal_lpc's catch-all.
    """
    from parcnet import PARCnet
    key = (packet_dim, extra_dim, lite, device)
    if key not in _parcnet_cache:
        _parcnet_cache[key] = PARCnet(
            model_checkpoint=CHECKPOINT,
            packet_dim=packet_dim,
            extra_pred_dim=extra_dim,
            ar_order=ar_order,
            ar_diagonal_load=diagonal_load,
            ar_context_dim=context_dim_packets,
            nn_context_dim=context_dim_packets,
            nn_fade_dim=nn_fade_dim,
            device=device,
            lite=lite,
            disable_nn=False,
        )
    concealer = _parcnet_cache[key]
    concealer.ar_model = ARModel(ar_order, diagonal_load)
    concealer.ar_context_dim = context_dim_packets * packet_dim
    concealer.nn_context_dim = context_dim_packets * packet_dim
    return concealer


def conceal_nn(clean_or_lossy: np.ndarray, trace: np.ndarray, **hparams) -> np.ndarray:
    """Runs the real PARCnet (AR+NN, NN always enabled) over a full signal.
    `**hparams` forwards to get_nn_concealer (ar_order, diagonal_load,
    context_dim_packets, extra_dim, nn_fade_dim, packet_dim, lite, device).
    """
    concealer = get_nn_concealer(**hparams)
    return concealer(clean_or_lossy, trace)


# --------------------------------------------------------------------------
# Sparse AR (Severi 2025) -- a separate algorithm, own hyperparameters
# --------------------------------------------------------------------------
from sparse_ar_model import SparseARModel  # noqa: E402  (path must be set up first)


def conceal_sparse_ar(clean_or_lossy: np.ndarray, trace: np.ndarray, max_k_lags: int = 3,
                      max_lag_l: int = 1024, burg_order_pb: int = 8, lambda_reg: float = 0.01,
                      context_dim_packets: int = 8, extra_dim: int = 256, equal_power: bool = False,
                      packet_dim: int = PACKET_SIZE, **_other_methods_only_kwargs) -> np.ndarray:
    """OMP-based sparse AR recursive packet-loss concealment (Severi 2025;
    see sparse_ar_model.py's module docstring for the algorithm and its two
    documented departures from the paper's literal description). Same
    per-packet recursive loop and external crossfade logic as conceal_lpc,
    so its output is directly comparable to the other two methods' --
    `context_dim_packets`/`extra_dim` are shared, structural parameters
    (not swept for this method, see run_hparam_sweeps.py), while
    max_k_lags/max_lag_l/burg_order_pb/lambda_reg are this method's own.
    `**_other_methods_only_kwargs` absorbs the other methods' hyperparameters
    (ar_order, diagonal_load, nn_fade_dim, lite, device) so all three methods
    can share one call signature in code that loops over them generically.
    """
    context_dim = context_dim_packets * packet_dim
    pred_dim = packet_dim + extra_dim
    model = SparseARModel(max_k_lags, max_lag_l, burg_order_pb, lambda_reg)
    output = np.pad(clean_or_lossy.copy(), (0, extra_dim))
    is_burst = False

    if extra_dim > 0:
        linear = np.linspace(0., 1., extra_dim)
        fade_in = np.sqrt(linear) if equal_power else linear
        fade_out = np.sqrt(1. - linear) if equal_power else 1. - linear

    for i, lost in enumerate(trace):
        if not lost:
            is_burst = False
            continue

        idx = i * packet_dim
        valid = output[max(0, idx - context_dim):idx]
        valid = np.pad(valid, (context_dim - len(valid), 0))
        prediction = model.predict(valid, pred_dim)

        if extra_dim > 0:
            prediction[-extra_dim:] *= fade_out
            if is_burst:
                prediction[:extra_dim] *= fade_in
            output[idx + packet_dim:idx + pred_dim] *= fade_in

        output[idx:idx + pred_dim] += prediction
        is_burst = True

    return output[:len(clean_or_lossy)]
