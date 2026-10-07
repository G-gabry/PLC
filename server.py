"""
FastAPI backend for the live PLC network simulation.

One WebSocket connection = one simulation session. The frontend sends control
messages (init/start/pause/stop/reset/drop/set_speed/update_config); the backend
runs the packet loop as an asyncio task and pushes one "tick" message per
packet, paced by SimConfig.packet_duration_ms * slowdown. Each tick carries the
transmitted (as-arrived) signal plus both concealment methods (AR-only and
AR+NN), so the frontend can compare all three live.
"""
import asyncio
import json
import shutil
import time
from pathlib import Path

import numpy as np
import librosa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles

from live_sim_engine import (
    SimConfig, LiveConcealer, LiveConcealerNN, NetworkSimulator, NoCrossfadeView,
    calibrate_compute_ms, calibrate_nn_compute_ms, load_nn_model,
)

BASE = Path(__file__).parent
TEST_SET = BASE / "parcnet-is2" / "example_test_set"
LOSSY_DIR = TEST_SET / "lossy"
CHECKPOINT = BASE / "parcnet-is2" / "pretrained_models" / "parcnet-is2_baseline_checkpoint.ckpt"
UPLOAD_DIR = BASE / "live_sim_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()
_nn_model = None  # loaded once at startup, shared read-only across sessions


@app.on_event("startup")
def _startup():
    global _nn_model
    print("Loading PARCnet NN checkpoint...")
    _nn_model = load_nn_model(str(CHECKPOINT), packet_dim=512, extra_pred_dim=256, lite=True, device="cpu")
    print("NN model ready.")


def resolve_audio_path(name: str) -> Path:
    p = UPLOAD_DIR / name
    if p.exists():
        return p
    p2 = LOSSY_DIR / f"{name}.wav"
    if p2.exists():
        return p2
    p3 = LOSSY_DIR / name
    if p3.exists():
        return p3
    raise FileNotFoundError(name)


class SpecBuffer:
    """
    Sliding-window STFT, same shape as the matplotlib comparisons throughout this
    project (n_fft=1024, hop_length=64) -- fixed dB scale (like librosa's
    amplitude_to_db(ref=np.max)), so every column uses the same color mapping
    instead of being renormalized against its own packet's peak. Each packet's
    worth of new samples (512) yields ~8 STFT frames (512/64), giving smooth,
    fine-grained motion instead of one abrupt column per packet.
    """

    def __init__(self, n_fft=1024, hop=64, n_bins=128, db_floor=-80.0):
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = n_bins
        self.db_floor = db_floor
        self.window = np.hanning(n_fft)
        self.buf = np.zeros(n_fft)

    def push_packet(self, chunk):
        n_new = len(chunk)
        self.buf = np.concatenate([self.buf, chunk])
        total_len = len(self.buf)
        n_hops = max(1, n_new // self.hop)

        frames = []
        for h in range(n_hops):
            end = total_len - n_new + (h + 1) * self.hop
            start = end - self.n_fft
            seg = self.buf[max(0, start):end]
            if start < 0:
                seg = np.pad(seg, (-start, 0))
            mag = np.abs(np.fft.rfft(seg * self.window))
            db = np.clip(20 * np.log10(np.maximum(mag, 1e-6)), self.db_floor, 0.0)
            edges = np.linspace(0, len(db), self.n_bins + 1).astype(int)
            frames.append([
                float(np.mean(db[edges[j]:max(edges[j] + 1, edges[j + 1])]))
                for j in range(self.n_bins)
            ])

        self.buf = self.buf[-(self.n_fft + n_new):]
        return frames


class RunningSNR:
    def __init__(self):
        self.sq_err = 0.0
        self.sq_ref = 0.0

    def update(self, ref, out):
        if ref is None or len(ref) != len(out):
            return None
        err = ref - out
        self.sq_err += float(np.sum(err ** 2))
        self.sq_ref += float(np.sum(ref ** 2))
        if self.sq_err <= 1e-15:
            return None
        return 10 * np.log10(max(self.sq_ref, 1e-12) / self.sq_err)


@app.get("/api/files")
def list_files():
    return {"files": sorted(f.stem for f in LOSSY_DIR.glob("*.wav"))}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename}


class SimulationSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.config = SimConfig()
        self.audio = None
        self.clean = None
        self.sr = None
        self.packet_index = 0
        self.n_packets = 0
        self.concealer_ar = None
        self.concealer_ar_nocross = None
        self.concealer_nn = None
        self.net = None
        self.running = False
        self.slowdown = 50.0
        self.fast_mode = False  # skip pacing + heavy per-tick payload, just get to the end
        self.snr_ar = RunningSNR()
        self.snr_nn = RunningSNR()
        self.spec_tx = SpecBuffer()
        self.spec_ar = SpecBuffer()
        self.spec_nn = SpecBuffer()
        self.prev_lost = False
        self.prev_chunk_ar = None
        self.prev_chunk_nn = None
        self.prev_chunk_ar_nocross = None
        self.boundary_half_width = 256  # samples shown on each side of a crossfade boundary

    async def send(self, msg):
        await self.ws.send_text(json.dumps(msg))

    def load_file(self, file_id, clean_id=None):
        y, sr = librosa.load(resolve_audio_path(file_id), sr=None)
        self.audio = y.astype(np.float64)
        self.sr = sr
        self.config.sample_rate = sr
        self.n_packets = len(self.audio) // self.config.packet_size
        self.clean = None
        if clean_id:
            c, _ = librosa.load(resolve_audio_path(clean_id), sr=sr)
            self.clean = c.astype(np.float64)

    def apply_config(self, updates: dict):
        for k, v in updates.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

    def calibrate(self):
        n = self.config.context_dim
        sample_ctx = self.audio[:n] if len(self.audio) >= n else np.pad(self.audio, (n - len(self.audio), 0))
        self.config.measured_compute_ms = calibrate_compute_ms(self.config, sample_ctx, trials=10)

        probe = LiveConcealerNN(self.config, _nn_model, device="cpu")
        m = min(len(self.audio), self.config.nn_context_dim)
        probe.nn_history[-m:] = self.audio[:m]
        self.config.measured_nn_compute_ms = calibrate_nn_compute_ms(probe, trials=5)

    def calibration_payload(self):
        c = self.config
        return {
            "packet_duration_ms": round(c.packet_duration_ms, 3),
            "measured_compute_ms": round(c.measured_compute_ms, 3),
            "measured_nn_compute_ms": round(c.measured_nn_compute_ms, 3),
            "wait_budget_ar_ms": round(c.wait_budget_ms, 3),
            "wait_budget_nn_ms": round(c.wait_budget_nn_ms, 3),
            "deadline_ms": round(c.effective_deadline_ms, 3),
            "nn_overrun": c.measured_nn_compute_ms + c.safety_margin_ms > c.effective_deadline_ms,
        }

    def reset_engine(self):
        self.concealer_ar = LiveConcealer(self.config)
        self.concealer_ar_nocross = LiveConcealer(NoCrossfadeView(self.config))
        self.concealer_nn = LiveConcealerNN(self.config, _nn_model, device="cpu")
        self.net = NetworkSimulator(self.config)
        self.packet_index = 0
        self.fast_mode = False
        self.snr_ar = RunningSNR()
        self.snr_nn = RunningSNR()
        self.spec_tx = SpecBuffer()
        self.spec_ar = SpecBuffer()
        self.spec_nn = SpecBuffer()
        self.prev_lost = False
        self.prev_chunk_ar = None
        self.prev_chunk_nn = None
        self.prev_chunk_ar_nocross = None

    async def run_loop(self):
        self.running = True
        c = self.config

        while self.running and self.packet_index < self.n_packets:
            # Read slowdown/fast_mode fresh every iteration -- computing tick_s once
            # before the loop meant a mid-run slider change never actually took effect.
            fast = self.fast_mode
            tick_s = 0.0 if fast else (c.packet_duration_ms / 1000.0) * self.slowdown
            i = self.packet_index
            idx = i * c.packet_size

            lost, delay_ms, state = self.net.next_packet_status()
            real_chunk = self.audio[idx: idx + c.packet_size]
            transmitted = np.zeros_like(real_chunk) if lost else real_chunk

            t0 = time.perf_counter()
            chunk_ar = self.concealer_ar.step(None if lost else real_chunk)
            gen_ms_ar = (time.perf_counter() - t0) * 1000
            chunk_ar_nocross = self.concealer_ar_nocross.step(None if lost else real_chunk)
            t0 = time.perf_counter()
            chunk_nn = self.concealer_nn.step(None if lost else real_chunk)
            gen_ms_nn = (time.perf_counter() - t0) * 1000

            ref = self.clean[idx: idx + c.packet_size] if self.clean is not None else None
            snr_ar_db = self.snr_ar.update(ref, chunk_ar)
            snr_nn_db = self.snr_nn.update(ref, chunk_nn)

            if fast:
                # Fast-forward: skip pacing AND the expensive waveform/spectrogram
                # payload entirely -- at this point the only real bottleneck left
                # is the AR+NN compute itself, so this gets to the end (and the
                # final full-file SNR) as quickly as the hardware allows.
                self.prev_lost = lost
                self.prev_chunk_ar = chunk_ar
                self.prev_chunk_nn = chunk_nn
                self.prev_chunk_ar_nocross = chunk_ar_nocross
                await self.send({
                    "type": "tick_fast",
                    "packet_index": i,
                    "n_packets": self.n_packets,
                    "lost": bool(lost),
                    "snr_ar_db": None if snr_ar_db is None else round(float(snr_ar_db), 2),
                    "snr_nn_db": None if snr_nn_db is None else round(float(snr_nn_db), 2),
                    "gen_ms_ar": round(gen_ms_ar, 3),
                    "gen_ms_nn": round(gen_ms_nn, 3),
                })
                self.packet_index += 1
                await asyncio.sleep(0)
                continue

            # Crossfade boundary: the moment concealment hands back to real audio.
            # Captured once, right on the transition tick, so a dedicated zoomed
            # chart can show exactly how smooth (or not) that splice is -- including
            # a genuine hard-splice (no crossfade) counterfactual for comparison,
            # same shape as the earlier matplotlib "with vs without crossfade" plot.
            boundary_ar = boundary_nn = boundary_ar_nocross = None
            w = self.boundary_half_width
            if (not lost) and self.prev_lost and self.prev_chunk_ar is not None:
                boundary_ar = [round(float(x), 5) for x in np.concatenate([self.prev_chunk_ar[-w:], chunk_ar[:w]])]
                boundary_nn = [round(float(x), 5) for x in np.concatenate([self.prev_chunk_nn[-w:], chunk_nn[:w]])]
                boundary_ar_nocross = [round(float(x), 5) for x in np.concatenate(
                    [self.prev_chunk_ar_nocross[-w:], chunk_ar_nocross[:w]])]
            self.prev_lost = lost
            self.prev_chunk_ar = chunk_ar
            self.prev_chunk_nn = chunk_nn
            self.prev_chunk_ar_nocross = chunk_ar_nocross

            await self.send({
                "type": "tick",
                "packet_index": i,
                "n_packets": self.n_packets,
                "lost": bool(lost),
                "state": state,
                "delay_ms": None if delay_ms == float("inf") else round(delay_ms, 2),

                "transmitted": [round(float(x), 5) for x in transmitted[::4]],
                "this_lpc": [round(float(x), 5) for x in chunk_ar[::4]],
                "this_nn": [round(float(x), 5) for x in chunk_nn[::4]],
                "diff": [round(float(a - b), 5) for a, b in zip(chunk_ar[::4], chunk_nn[::4])],

                "spec_tx": self.spec_tx.push_packet(transmitted),
                "spec_lpc": self.spec_ar.push_packet(chunk_ar),
                "spec_nn": self.spec_nn.push_packet(chunk_nn),

                "snr_ar_db": None if snr_ar_db is None else round(float(snr_ar_db), 2),
                "snr_nn_db": None if snr_nn_db is None else round(float(snr_nn_db), 2),

                "wait_budget_ar_ms": round(c.wait_budget_ms, 2),
                "wait_budget_nn_ms": round(c.wait_budget_nn_ms, 2),
                "deadline_ms": round(c.effective_deadline_ms, 2),

                # Actual measured wall-clock time this specific packet took to
                # generate (not the one-time calibration estimate) -- lets you see
                # in real time how a parameter change (e.g. AR order) affects cost.
                "gen_ms_ar": round(gen_ms_ar, 3),
                "gen_ms_nn": round(gen_ms_nn, 3),

                "boundary_ar": boundary_ar,
                "boundary_nn": boundary_nn,
                "boundary_ar_nocross": boundary_ar_nocross,
                "boundary_half_width": self.boundary_half_width,
                "crossfade_samples": c.extra_dim,
            })

            self.packet_index += 1
            await asyncio.sleep(tick_s)

        if self.packet_index >= self.n_packets:
            await self.send({"type": "done"})
        self.running = False


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = SimulationSession(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "init":
                try:
                    session.load_file(msg["file"], msg.get("clean_file"))
                except FileNotFoundError as e:
                    await session.send({"type": "error", "message": f"File not found: {e}"})
                    continue
                session.apply_config(msg.get("config", {}))
                session.calibrate()
                session.reset_engine()
                await session.send({
                    "type": "ready",
                    "n_packets": session.n_packets,
                    "sample_rate": session.sr,
                    **session.calibration_payload(),
                })

            elif mtype == "start":
                if not session.running:
                    asyncio.create_task(session.run_loop())

            elif mtype == "pause":
                session.running = False

            elif mtype in ("reset", "stop"):
                session.running = False
                if session.audio is not None:
                    session.reset_engine()
                await session.send({"type": "reset_done"})

            elif mtype == "drop":
                if session.net is not None:
                    session.net.force_drop(int(msg.get("count", 1)))

            elif mtype == "set_speed":
                session.slowdown = float(msg.get("slowdown", session.slowdown))

            elif mtype == "fast_forward":
                session.fast_mode = True
                if not session.running:
                    asyncio.create_task(session.run_loop())

            elif mtype == "normal_mode":
                session.fast_mode = False

            elif mtype == "update_config":
                session.apply_config(msg.get("config", {}))
                # Re-measure generation cost against the new parameters (e.g. a
                # changed AR order) so the sidebar's timing numbers stay honest
                # without requiring a full reload.
                if session.audio is not None:
                    session.calibrate()
                    await session.send({"type": "calibration", **session.calibration_payload()})

    except WebSocketDisconnect:
        session.running = False


app.mount("/", StaticFiles(directory=str(BASE / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
