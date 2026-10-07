# PLC Analysis — LPC Instability: Problem, Evidence, and Fix

This documents what `PLC_Analysis.py` found when testing the project's current
LPC-based packet-loss concealment (PLC) design against real audio from the
`Audio_Project_Data_VUT` dataset, and the fix that was applied.

## Setup

- Dataset: `Audio_Project_Data_VUT/{lossy,traces,enhanced}` (674 files, 44.1kHz, mono).
- Packet size: 512 samples (~11.6ms). LPC order: 256, fit from a 1024-sample
  (~23ms) context window immediately before each gap.
- Three signals are compared for every test case:
  1. **Broken (lossy)** — the zero-filled signal, i.e. no concealment at all.
  2. **LPC repaired** — this project's current design (`guess_missing_audio` /
     `fix_broken_file` in `PLC_Analysis.py`).
  3. **NN enhanced (PARCnet)** — the PARCnet-IS² neural-network baseline from
     this same repo, used as a reference point.
- Real network traces have burst lengths from 1–16 packets. The script locates
  actual occurrences of each target burst length in the traces and zooms in on
  the real gap, instead of an arbitrary slice of the file.

## The problem: LPC concealment can fail catastrophically on longer bursts

Testing across multiple real files at the 16-packet burst length (the worst
case in this dataset) surfaced **two distinct failure modes**, both confirmed
by inspecting the actual numbers, not just the plots.

### Failure mode 1 — amplitude blow-up

File `0XeqPp-350-89725-311379-207508-129636-384309-141428.wav`, packet 2165
(16-packet burst): the LPC repair diverged into a loud amplitude excursion
during the gap, with a spurious energy band around **7–8kHz** clearly visible
in its spectrogram — a band that exists in neither the broken signal nor the
NN-enhanced version. This is the classic symptom of an autoregressive filter
whose fitted poles sit outside (or right on) the unit circle: because each
predicted sample feeds back into predicting the next, small errors compound
exponentially over an 8192-sample (16-packet) gap.

### Failure mode 2 — NaN → total silence

File `1UaHCp-394-87506-140741-186260-68344-3434-1214.wav`, packet 544
(16-packet burst): the LPC-repaired waveform was **flat zero for the entire
gap** — not a decay, the very first predicted sample was already zero. Traced
directly:

```
pattern = librosa.lpc(clue, order=256)
# pattern[:5] = [1., nan, nan, nan, nan]   <-- the LPC fit itself returned NaN
```

The context window here (1024 samples, non-silent, max amplitude 0.14) was
perfectly normal audio — the fit failed anyway. The existing NaN-guard in
`guess_missing_audio` correctly caught the NaN and substituted zero, but that
means the entire repaired segment goes silent instead of continuing the note.

### Root cause

Both failures trace back to the same thing: `librosa.lpc()` solves the
autocorrelation-method equations (Levinson-Durbin) with **no regularization**.
Fitting a high order (256) from a short window (1024 samples) makes the
autocorrelation (Toeplitz) matrix ill-conditioned, especially over transients
or near-periodic tones. An ill-conditioned solve either:
- returns NaN coefficients outright (failure mode 2), or
- returns coefficients whose filter poles are unstable/near-unstable, causing
  runaway amplitude during extrapolation (failure mode 1).

This is a known, general weakness of plain LPC-based PLC, not a one-off bug —
and PARCnet's own AR module (`parcnet-is2/ar_model.py`, already in this repo)
was built with a fix for exactly this problem.

## The fix: diagonal loading (Tikhonov regularization)

`PLC_Analysis.py` now fits the AR coefficients itself instead of calling
`librosa.lpc()`, adding a small constant to the zeroth autocorrelation
coefficient before solving — the same regularization PARCnet's AR model uses
(`ar_model.py`: `c[0] += diagonal_load`, `ar_diagonal_load: 0.001` in
`config/config.yaml`):

```python
LPC_ORDER = 256
LPC_DIAGONAL_LOAD = 0.001

def fit_lpc_multipliers(past_audio, order=LPC_ORDER, diagonal_load=LPC_DIAGONAL_LOAD):
    acf = scipy.signal.correlate(past_audio, past_audio, mode="full")
    if np.any(np.isnan(acf)):
        acf = scipy.signal.correlate(past_audio, past_audio, mode="full", method="direct")

    zero_lag = len(acf) // 2
    c = acf[zero_lag:zero_lag + order].copy()
    c[0] += diagonal_load                       # <-- the fix
    b = acf[zero_lag + 1:zero_lag + order + 1]

    phi = scipy.linalg.solve_toeplitz(c, b, check_finite=False)
    if np.any(np.isnan(phi)) or np.any(np.isinf(phi)):
        return None
    return phi[::-1]
```

Adding a small positive value to the matrix diagonal keeps it well-conditioned
(equivalent to ridge/Tikhonov regularization on the least-squares problem),
which both avoids near-singular solves (fixing the NaN case) and pulls
marginal filter poles back toward stability (fixing the blow-up case).
`guess_missing_audio` falls back to silence only if the regularized fit is
*still* degenerate, which did not happen in any of the re-tested cases.

## Verification (before / after)

Both previously-broken cases were re-run after the fix, using the exact same
audio, gap position, and burst length:

| Case | Before | After |
|---|---|---|
| `burst_16pkt_ex1_0XeqPp...` | Amplitude blow-up + spurious 7-8kHz artifact | Smooth decay, closely tracking the NN version; artifact gone |
| `burst_16pkt_ex2_1UaHCp...` | NaN → flat silence for the whole gap | Real decaying tone, comparable in shape to the NN version |

No other case (1/4/8-packet bursts, or the other 16-packet examples) regressed.

## How to reproduce

```powershell
.\.venv\Scripts\python.exe .\parcnet-is2\pretrained_models\PLC_Analysis.py
```

Outputs land in `analysis_output/`:
- `figures/network_statistics.png` — global burst-length and packet-loss-rate distributions.
- `figures/burst_{length}pkt_ex{n}_{file}.png` — per-case waveform + spectrogram comparison (Broken / LPC / NN), zoomed on the real gap.
- `audio_snippets/burst_{length}pkt_ex{n}_{1_broken|2_lpc|3_nn}_{file}.wav` — ~2-4s listenable clips around each gap.

## A related, unrelated-looking observation

Some example plots show a flat/silent region right before the gap (most
visible on some 16-packet and 4-packet examples). This is **not a bug** —
verified directly against the raw samples, that silence is real (RMS
`0.00000` for stretches of some windows): the recordings contain genuine
rests/quiet passages between notes, and packet loss timing is independent of
musical content, so some randomly-located example bursts happen to land on
quiet moments and others don't (confirmed by comparing two bursts from the
*same* file, one silent going in and one not).
