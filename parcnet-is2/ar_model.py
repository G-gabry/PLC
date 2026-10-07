import scipy
import warnings
import numpy as np
from numba import njit


@njit
def _apply_prediction_filter(past: np.ndarray, coeff: np.ndarray, steps: int) -> np.ndarray:
    # Initialize output vector
    prediction = np.zeros(steps)

    # Predict future steps one at a time
    for i in range(steps):
        pred = np.dot(past, coeff)

        prediction[i] = pred

        past = np.roll(past, -1)
        past[-1] = pred

    return prediction


class ARModel:
    """AR model of order p."""

    def __init__(self, p: int, diagonal_load: float = 0.0):
        """
            :param p: the order of the AR(p) model
            :param y_true: diagonal loading term
        """
        self.p = p
        self.diagonal_load = diagonal_load

        # Pre-compile Numba decorated function to expedite future calls
        _apply_prediction_filter(past=np.zeros(self.p), coeff=np.ones(self.p), steps=1)

    def _autocorrelation_method(self, valid: np.ndarray) -> np.ndarray:
        """
        Finds the AR(p) model parameters via the autocorrelation method and Levinson-Durbin recursion.
        In doing so, applies a diagonal loading term to the autocorrelation matrix to combat ill-conditioning.
        """
        # Compute the sample autocorrelation function
        acf = scipy.signal.correlate(valid, valid, mode='full', method='auto')

        # In rare cases, method='fft' appears to produce NaN. If so, use method='direct' instead.
        if np.any(np.isnan(acf)):
            acf = scipy.signal.correlate(valid, valid, mode='full', method='direct')

        # Find the zeroth lag index
        zero_lag = len(acf) // 2

        # First column of the autocorrelation matrix
        c = acf[zero_lag:zero_lag + self.p]

        # Diagonal loading to improve conditioning
        c[0] += self.diagonal_load

        # Autocorrelation vector
        b = acf[zero_lag + 1:zero_lag + self.p + 1]

        # Solve the Toeplitz system of equations using Levinson-Durbin recursion
        ar_coeff = scipy.linalg.solve_toeplitz(c, b, check_finite=False)

        return ar_coeff

    def predict(self, valid: np.ndarray, steps: int) -> np.ndarray:
        """
        Fits the AR model from an array of valid samples before linearly predicting an arbitrary number of steps into
        the future. Uses Numba jit to accelerate sample-by-sample inference.

        As a fail-safe, returns an array of zeros if the output contains NaN or takes values outside of [-1.5, 1.5].
        This allows us to train PARCnet by sampling audio chunks at random without worrying about ill-conditioning.

            :param valid: ndarray of past samples
            :param steps: the number of samples to be predicted
            :return: ndarray of linearly predicted samples
        """
        # Find the AR model parameters
        ar_coeff = self._autocorrelation_method(valid)

        # Apply linear prediction
        pred = _apply_prediction_filter(
            past=valid[-self.p:],
            coeff=np.ascontiguousarray(ar_coeff[::-1], dtype=np.float32),  # needed for njit
            steps=steps
        )

        # Raise warnings; helpful in case the AR model becomes numerically unstable.
        if np.any(np.isnan(pred)):
            warnings.warn(f'AR prediction contains NaN', RuntimeWarning)
            return np.zeros_like(pred)

        elif np.any(np.abs(pred) > 1.5):
            warnings.warn(f'AR prediction exceeded the safety range: found [{pred.min()}, {pred.max()}]',
                          RuntimeWarning)
            return np.zeros_like(pred)

        return pred


def find_real_burst(trace, burst_length, context_packets):
    """
    Look through a real packet-loss trace (0 = received, 1 = lost) and find the
    first place where exactly `burst_length` packets in a row were really lost
    (not part of a longer burst), with enough room before it for `context_packets`
    of audio. Note: those preceding packets are NOT required to be clean - in a
    real trace they sometimes aren't, same as the real PARCnet pipeline.
    Returns the packet index where the burst starts, or None if not found.
    """
    for i in range(context_packets, len(trace) - burst_length):
        packet_before = trace[i - 1]
        packets_in_burst = trace[i:i + burst_length]
        packet_after = trace[i + burst_length] if i + burst_length < len(trace) else 1

        burst_is_fully_lost = packets_in_burst.all()
        burst_does_not_continue = packet_before == 0 and packet_after == 0

        if burst_is_fully_lost and burst_does_not_continue:
            return i

    return None


def ar_only_recursive_fill(base_signal, model, start_packet, burst_length, packet_size, ar_context_dim,
                            extra_pred_dim):
    """
    Fills one burst using the exact same per-packet, recursive procedure as PARCnet's own
    __call__ (parcnet.py), with the neural network's contribution fixed at zero: predicts
    `packet_size + extra_pred_dim` samples one packet at a time, feeding each packet's own
    prediction back in as context for the next, and applying the same outbound fade-out /
    inbound fade-in cross-fades. This is what "LPC (whole pipeline, no NN)" actually computes
    internally, so -- unlike a single one-shot AR call over the whole burst -- its output
    should match that pipeline's audio almost exactly for this burst.

    Note: if an earlier, unrelated packet loss sits within `ar_context_dim` samples before
    this burst, its cross-faded tail (real pipeline) vs. raw audio (this local copy) can
    cause a small residual difference -- `find_real_burst` only guarantees the immediately
    preceding packet was received, not that the whole context window is loss-free.
    """
    ar_context_samples = ar_context_dim * packet_size
    pred_dim = packet_size + extra_pred_dim
    fade_in = np.linspace(0., 1., extra_pred_dim)
    fade_out = np.linspace(1., 0., extra_pred_dim)

    # Local working copy, padded at the end just like PARCnet's own output_signal,
    # so a prediction tail near the end of the array never runs out of bounds.
    output = np.pad(base_signal.copy(), (0, extra_pred_dim))

    is_burst = False
    for offset in range(burst_length):
        idx = (start_packet + offset) * packet_size

        ar_context = output[max(0, idx - ar_context_samples):idx]
        ar_context = np.pad(ar_context, (ar_context_samples - len(ar_context), 0))
        prediction = model.predict(valid=ar_context, steps=pred_dim)

        # Cross-fade the compound prediction (outbound fade-out) -- NN contribution is zero
        prediction[-extra_pred_dim:] *= fade_out

        if is_burst:
            # Cross-fade the prediction in case of consecutive packet losses (inbound fade-in)
            prediction[:extra_pred_dim] *= fade_in

        # Cross-fade the output signal (outbound fade-in)
        output[idx + packet_size:idx + pred_dim] *= fade_in

        # Conceal lost packet
        output[idx: idx + pred_dim] += prediction

        is_burst = True

    return output[:len(base_signal)]


if __name__ == '__main__':
    import os
    import soundfile as sf
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    import seaborn as sns

    # ------------------------------------------------------------------
    # SETTINGS - change these to test a different file.
    # Keep them the same in Colab to compare results on the exact same test.
    # ------------------------------------------------------------------
    folder = os.path.dirname(os.path.abspath(__file__))
    file_id = '1UaHCp-394-87506-140741-186260-68344-3434-1214'  # has real bursts of length 2, 4, 16 (no 8 - no single file has all 4)
    
    audio_file = os.path.join(folder, 'example_test_set', 'lossy', file_id + '.wav')
    trace_file = os.path.join(folder, 'example_test_set', 'traces', file_id + '.txt')
    # Added path for the PARCnet enhanced file
    enhanced_file = os.path.join(folder, 'example_test_set', 'enhanced', file_id + '.wav')
    # Whole PARCnet pipeline with the neural network's contribution zeroed out (see parcnet.py)
    lpc_pipeline_file = os.path.join(folder, 'example_test_set', 'enhanced_ar_only', file_id + '.wav')

    # Every figure is saved here instead of shown, so the loop never waits on a closed window
    output_dir = os.path.join(folder, 'ar_model_output')
    os.makedirs(output_dir, exist_ok=True)

    ar_order = 256          # AR model order (same as config/config.yaml)
    diagonal_load = 0.001   # regularization term (same as config/config.yaml)
    packet_size = 512       # samples in one packet
    extra_pred_dim = 256    # extra lookahead per packet (same as config/config.yaml global.extra_pred_dim)
    context_packets = 8     # how many past packets the model is allowed to see (matches AR.ar_context_dim)
    burst_lengths = [2, 4, 8, 16]  # real burst lengths (in packets) to test
    
    # Plotting preferences
    context_ms = 40         # How many ms to show around the gap in the plots
    COLOR_LPC_PIPELINE = "#e8871e"
    COLOR_LPC = "#2a78d6"
    COLOR_NN = "#1baf7a"
    COLOR_DIFF = "#7d5ba6"
    COLOR_GAP_SHADE = "#0b0b0b"
    GRID_COLOR = "#e1e0d9"
    MUTED_INK = "#898781"
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": MUTED_INK, "grid.color": GRID_COLOR})
    # ------------------------------------------------------------------

    # 1. Load the broken audio, trace, and NN-enhanced audio
    signal, sr = sf.read(audio_file, dtype='float32')
    if signal.ndim > 1:
        signal = signal[:, 0]
        
    trace = np.loadtxt(trace_file, dtype=int)
    
    enhanced_signal, _ = sf.read(enhanced_file, dtype='float32')
    if enhanced_signal.ndim > 1:
        enhanced_signal = enhanced_signal[:, 0]

    lpc_pipeline_signal, _ = sf.read(lpc_pipeline_file, dtype='float32')
    if lpc_pipeline_signal.ndim > 1:
        lpc_pipeline_signal = lpc_pipeline_signal[:, 0]

    model = ARModel(p=ar_order, diagonal_load=diagonal_load)

    # 2. For each burst length, find a real occurrence in the trace and fill it in
    for example_num, burst_length in enumerate(burst_lengths, start=1):
        start_packet = find_real_burst(trace, burst_length, context_packets)

        if start_packet is None:
            print(f'burst length {burst_length}: no real occurrence found in this file, skipping')
            continue

        gap_start = start_packet * packet_size
        gap_size = burst_length * packet_size

        # Fill the burst the same recursive, per-packet way PARCnet's own pipeline does
        # (parcnet.py), with the neural network's contribution fixed at zero -- so this
        # should match "LPC (whole pipeline, no NN)" almost exactly, packet by packet.
        fixed_lpc = ar_only_recursive_fill(
            base_signal=signal,
            model=model,
            start_packet=start_packet,
            burst_length=burst_length,
            packet_size=packet_size,
            ar_context_dim=context_packets,
            extra_pred_dim=extra_pred_dim,
        )

        print(f'burst length {burst_length}: filled a real gap at sample {gap_start} '
              f'(packet {start_packet}), {gap_size} samples')

        # 3. 3x3 Grid Plotting Logic (Waveforms + Spectrograms + Pairwise Differences)
        context_samples_plot = int(sr * context_ms / 1000)
        win_start = max(0, gap_start - context_samples_plot)
        win_end = min(len(signal), gap_start + gap_size + context_samples_plot)

        signals = [
            ("1_lpc_whole_pipeline",  "LPC (whole pipeline, no NN)",   lpc_pipeline_signal, COLOR_LPC_PIPELINE),
            ("2_lpc_class_only",      "LPC (class only, recursive)",   fixed_lpc,           COLOR_LPC),
            ("3_nn_enhanced_parcnet", "NN Enhanced (PARCnet)",         enhanced_signal,     COLOR_NN),
        ]
        
        t_ms = (np.arange(win_start, win_end) - gap_start) / sr * 1000
        gap_end_ms = gap_size / sr * 1000

        # Shared y-limits for fair waveform comparison
        y_max = max(np.max(np.abs(sig[win_start:win_end])) for _, _, sig, _ in signals) * 1.1

        fig, axes = plt.subplots(3, 3, figsize=(15, 13),
                                  gridspec_kw={"height_ratios": [1, 1.15, 1], "hspace": 0.7})

        # Row 1: Waveforms
        for col, (_, label, sig, color) in enumerate(signals):
            ax = axes[0, col]
            ax.plot(t_ms, sig[win_start:win_end], color=color, linewidth=1.1)
            ax.axvspan(0, gap_end_ms, color=COLOR_GAP_SHADE, alpha=0.08)
            ax.axvline(0, color=MUTED_INK, linestyle="--", linewidth=1)
            ax.axvline(gap_end_ms, color=MUTED_INK, linestyle="--", linewidth=1)
            ax.set_title(label, fontsize=11, color=color)
            ax.set_ylim(-y_max, y_max)
            ax.set_xlabel("Time relative to gap start (ms)")
            if col == 0:
                ax.set_ylabel("Amplitude")

        # Row 2: Spectrograms
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
            ax.axvline((gap_start - win_start) / sr, color="white", linestyle="--", linewidth=1, alpha=0.7)
            ax.axvline((gap_start + gap_size - win_start) / sr, color="white", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_title(f"Spectrogram — {label}", fontsize=10, color=color)
            ax.set_ylim(0, 8000)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel("Frequency (Hz)")
            else:
                ax.set_ylabel("")

        fig.colorbar(img, ax=axes[1, :], format="%+2.0f dB", fraction=0.025, pad=0.02, label="Magnitude (dB)")

        # Row 3: pairwise differences between each two signals, with the MSE over the actual
        # lost-packet gap (not the surrounding context) printed under each subplot's title.
        pairs = [
            (signals[0], signals[1]),  # LPC whole pipeline vs. LPC class only
            (signals[0], signals[2]),  # LPC whole pipeline vs. NN enhanced
            (signals[1], signals[2]),  # LPC class only vs. NN enhanced
        ]

        diffs = [sig_a[win_start:win_end] - sig_b[win_start:win_end]
                 for (_, _, sig_a, _), (_, _, sig_b, _) in pairs]
        diff_y_max = max(np.max(np.abs(d)) for d in diffs) * 1.1

        for col, ((_, label_a, sig_a, _), (_, label_b, sig_b, _)) in enumerate(pairs):
            ax = axes[2, col]
            gap_mse = np.mean((sig_a[gap_start:gap_start + gap_size] -
                                sig_b[gap_start:gap_start + gap_size]) ** 2)

            ax.plot(t_ms, diffs[col], color=COLOR_DIFF, linewidth=1.0)
            ax.axvspan(0, gap_end_ms, color=COLOR_GAP_SHADE, alpha=0.08)
            ax.axvline(0, color=MUTED_INK, linestyle="--", linewidth=1)
            ax.axvline(gap_end_ms, color=MUTED_INK, linestyle="--", linewidth=1)
            ax.set_title(f"{label_a} minus {label_b}\nGap MSE: {gap_mse:.3e}",
                         fontsize=9.5, color=COLOR_DIFF)
            ax.set_ylim(-diff_y_max, diff_y_max)
            ax.set_xlabel("Time relative to gap start (ms)")
            if col == 0:
                ax.set_ylabel("Amplitude diff")

        fig.suptitle(f"{file_id} — {burst_length}-packet burst, example {example_num} "
                     f"({gap_size} samples)", fontsize=13, y=1.0)

        # Save the figure instead of showing it, so the loop keeps going on its own
        fig_path = os.path.join(output_dir, f'burst_{burst_length}pkt_{file_id}.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'saved: {fig_path}')
