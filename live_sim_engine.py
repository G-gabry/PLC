"""
Core engine for the live PLC network simulation.

Everything a future UI needs to control lives in SimConfig -- nothing here is
hardcoded that a slider/button should be able to change. Three pieces:

  - ar_fit / ar_predict   -- unchanged, causal AR(order) predictor.
  - LiveConcealer         -- packet-at-a-time version of lpc_conceal. Call
                             .step(chunk_or_None) once per packet tick.
  - NetworkSimulator      -- decides lost/received per tick: Gilbert-Elliott
                             state -> a delay draw -> checked against a
                             deadline. A manual "force drop next N" always wins.
"""
from pathlib import Path
import numpy as np
import torch
import scipy.signal
import scipy.linalg
from dataclasses import dataclass
from numba import njit


@njit
def _ar_predict_loop(past, coeff, steps):
    prediction = np.zeros(steps)
    for i in range(steps):
        pred = np.dot(past, coeff)
        prediction[i] = pred
        past = np.roll(past, -1)
        past[-1] = pred
    return prediction


def ar_fit(past_audio, order, diagonal_load):
    acf = scipy.signal.correlate(past_audio, past_audio, mode='full')
    if np.any(np.isnan(acf)):
        acf = scipy.signal.correlate(past_audio, past_audio, mode='full', method='direct')
    zero_lag = len(acf) // 2
    c = acf[zero_lag:zero_lag + order].copy()
    c[0] += diagonal_load
    b = acf[zero_lag + 1:zero_lag + order + 1]
    return scipy.linalg.solve_toeplitz(c, b, check_finite=False)


def ar_predict(past_audio, order, diagonal_load, steps):
    coeff = ar_fit(past_audio, order, diagonal_load)
    past_view = np.ascontiguousarray(past_audio[-order:], dtype=np.float32)
    coeff_view = np.ascontiguousarray(coeff[::-1], dtype=np.float32)
    pred = _ar_predict_loop(past_view, coeff_view, steps)
    if np.any(np.isnan(pred)) or np.any(np.abs(pred) > 1.5):
        return np.zeros(steps)
    return pred


@dataclass
class SimConfig:
    # AR / concealment
    packet_size: int = 512
    order: int = 256
    diagonal_load: float = 0.001
    context_dim: int = 4096
    extra_dim: int = 256
    equal_power: bool = False

    # Neural network (PARCnet) concealment -- matches config.yaml's neural_net block
    nn_context_dim: int = 4096
    nn_fade_dim: int = 64

    # Timing
    sample_rate: int = 44100
    measured_compute_ms: float = 0.0     # AR compute time; set via calibrate()
    measured_nn_compute_ms: float = 0.0  # NN compute time; set via calibrate_nn()
    safety_margin_ms: float = 1.0
    buffer_depth_packets: int = 1        # how many packets of lookahead are held
    deadline_ms: float = None            # None -> auto = buffer_depth_packets * packet_duration_ms

    @property
    def packet_duration_ms(self):
        return 1000.0 * self.packet_size / self.sample_rate

    @property
    def pred_dim(self):
        return self.packet_size + self.extra_dim

    @property
    def effective_deadline_ms(self):
        total = self.deadline_ms if self.deadline_ms is not None else self.buffer_depth_packets * self.packet_duration_ms
        return total

    @property
    def wait_budget_ms(self):
        """How long we can wait for a real packet before the AR method must start generating."""
        return max(0.0, self.effective_deadline_ms - self.measured_compute_ms - self.safety_margin_ms)

    @property
    def wait_budget_nn_ms(self):
        """Same, but for the (typically much slower) NN method."""
        return max(0.0, self.effective_deadline_ms - self.measured_nn_compute_ms - self.safety_margin_ms)


def calibrate_compute_ms(config: SimConfig, sample_context: np.ndarray, trials: int = 20) -> float:
    """Measures how long ar_predict actually takes on this hardware, for wait_budget_ms."""
    import time
    # Warm up numba JIT before timing.
    ar_predict(sample_context, config.order, config.diagonal_load, config.pred_dim)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        ar_predict(sample_context, config.order, config.diagonal_load, config.pred_dim)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


class NetworkSimulator:
    """Packet loss is purely manual: force_drop(N) marks the next N packets lost. Nothing else drops a packet."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.manual_drop_remaining = 0

    def force_drop(self, count: int):
        """UI hook: the 'Drop N' button."""
        self.manual_drop_remaining += count

    def  next_packet_status(self):
        """Returns (lost, delay_ms, state) for the next packet tick."""
        if self.manual_drop_remaining > 0:
            self.manual_drop_remaining -= 1
            return True, float('inf'), "MANUAL"
        return False, 0.0, "OK"


class NoCrossfadeView:
    """
    Proxies a SimConfig but forces extra_dim (and therefore pred_dim) to 0 --
    used to run a second LiveConcealer alongside the real one, showing what a
    hard splice (no crossfade at all) would sound like for direct comparison.
    Everything else is read straight through, live, so it stays in sync with
    the real config even if parameters change mid-session.
    """

    def __init__(self, base):
        object.__setattr__(self, "_base", base)

    @property
    def extra_dim(self):
        return 0

    @property
    def pred_dim(self):
        return self._base.packet_size

    def __getattr__(self, name):
        return getattr(self._base, name)


class LiveConcealer:
    """
    Stateful, packet-at-a-time version of lpc_conceal. Call step(chunk_or_None)
    once per packet tick, in order -- pass the real packet_size samples if the
    packet arrived in time, or None if the network simulator declared it lost.
    Produces bit-identical output to the batch lpc_conceal given the same trace.
    """

    def __init__(self, config: SimConfig, initial_history: np.ndarray = None):
        self.config = config
        self.history = np.zeros(config.context_dim, dtype=np.float64)
        if initial_history is not None and len(initial_history) > 0:
            n = min(len(initial_history), config.context_dim)
            self.history[-n:] = initial_history[-n:]
        self.is_burst = False
        self.pending_tail = np.zeros(max(config.extra_dim, 1), dtype=np.float64)
        if config.extra_dim > 0:
            linear = np.linspace(0., 1., config.extra_dim)
            self.fade_in = np.sqrt(linear) if config.equal_power else linear
            self.fade_out = np.sqrt(1. - linear) if config.equal_power else 1. - linear

    def step(self, received_chunk: np.ndarray = None) -> np.ndarray:
        c = self.config

        if received_chunk is not None:
            core = np.asarray(received_chunk, dtype=np.float64).copy()
            if self.is_burst and c.extra_dim > 0:
                # A previous concealment's own tail is still pending -- blend it into
                # this real packet's start, exactly like the batch version's outbound
                # fade, just applied now because this is the first moment we actually
                # know what the real samples are.
                core[:c.extra_dim] = core[:c.extra_dim] * self.fade_in + self.pending_tail
            self.pending_tail = np.zeros(max(c.extra_dim, 1))
            self.is_burst = False
        else:
            prediction = ar_predict(self.history, c.order, c.diagonal_load, c.pred_dim)
            if c.extra_dim > 0:
                prediction[-c.extra_dim:] *= self.fade_out
                if self.is_burst:
                    prediction[:c.extra_dim] = prediction[:c.extra_dim] * self.fade_in + self.pending_tail
                self.pending_tail = prediction[-c.extra_dim:].copy()
            core = prediction[:c.packet_size].copy()
            self.is_burst = True

        self.history = np.concatenate([self.history[c.packet_size:], core])[-c.context_dim:]
        return core


class LiveConcealerNN:
    """
    Stepper version of PARCnet's full algorithm (AR + neural network) -- same
    per-packet loop as parcnet.py's PARCnet.__call__, restructured for one-packet-
    at-a-time streaming exactly like LiveConcealer (deferred pending_tail). The
    only combination difference from LiveConcealer: prediction = ar_pred + nn_pred,
    and the NN's own contribution gets its own inbound fade (nn_fade_dim), applied
    unconditionally, on top of the shared is_burst-gated fade the combined
    prediction still goes through.
    """

    def __init__(self, config: SimConfig, neural_net, device: str = "cpu"):
        self.config = config
        self.neural_net = neural_net
        self.device = device

        self.ar_history = np.zeros(config.context_dim, dtype=np.float64)
        self.nn_history = np.zeros(config.nn_context_dim, dtype=np.float64)
        self.is_burst = False
        self.pending_tail = np.zeros(max(config.extra_dim, 1), dtype=np.float64)
        if config.extra_dim > 0:
            linear = np.linspace(0., 1., config.extra_dim)
            self.fade_in = np.sqrt(linear) if config.equal_power else linear
            self.fade_out = np.sqrt(1. - linear) if config.equal_power else 1. - linear
        n_fade = min(config.nn_fade_dim, config.pred_dim)
        self.nn_fade = np.linspace(0., 1., n_fade)

    def _nn_predict(self):
        c = self.config
        nn_ctx = np.pad(self.nn_history, (0, c.pred_dim))
        nn_input = torch.tensor(nn_ctx[None, None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            nn_out = self.neural_net(nn_input)
        nn_pred = nn_out[..., -c.pred_dim:].squeeze().cpu().numpy().astype(np.float64)
        n_fade = len(self.nn_fade)
        nn_pred[:n_fade] *= self.nn_fade
        return nn_pred

    def step(self, received_chunk: np.ndarray = None) -> np.ndarray:
        c = self.config

        if received_chunk is not None:
            core = np.asarray(received_chunk, dtype=np.float64).copy()
            if self.is_burst and c.extra_dim > 0:
                core[:c.extra_dim] = core[:c.extra_dim] * self.fade_in + self.pending_tail
            self.pending_tail = np.zeros(max(c.extra_dim, 1))
            self.is_burst = False
        else:
            ar_pred = ar_predict(self.ar_history, c.order, c.diagonal_load, c.pred_dim)
            nn_pred = self._nn_predict()
            prediction = ar_pred + nn_pred

            if c.extra_dim > 0:
                prediction[-c.extra_dim:] *= self.fade_out
                if self.is_burst:
                    prediction[:c.extra_dim] = prediction[:c.extra_dim] * self.fade_in + self.pending_tail
                self.pending_tail = prediction[-c.extra_dim:].copy()
            core = prediction[:c.packet_size].copy()
            self.is_burst = True

        self.ar_history = np.concatenate([self.ar_history[c.packet_size:], core])[-c.context_dim:]
        self.nn_history = np.concatenate([self.nn_history[c.packet_size:], core])[-c.nn_context_dim:]
        return core


def load_nn_model(checkpoint_path: str, packet_dim: int, extra_pred_dim: int, lite: bool = True, device: str = "cpu"):
    """Loads PARCnet's pretrained HybridModel, same interface as parcnet.py."""
    import sys
    parcnet_dir = str(Path(checkpoint_path).resolve().parent.parent)
    if parcnet_dir not in sys.path:
        sys.path.insert(0, parcnet_dir)
    from nn_model import HybridModel

    model = HybridModel.load_from_checkpoint(
        checkpoint_path, packet_dim=packet_dim, extra_pred_dim=extra_pred_dim,
        channels=1, lite=lite, map_location=device,
    ).to(device)
    model.eval()
    return model


def calibrate_nn_compute_ms(concealer: "LiveConcealerNN", trials: int = 10) -> float:
    """Measures how long one NN concealment step actually takes on this hardware."""
    import time
    concealer._nn_predict()  # warm up (first call is often slower)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        concealer._nn_predict()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))
