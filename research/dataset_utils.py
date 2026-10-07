"""
Loads (clean, lossy, trace) triplets from the real PARCnet-IS2 challenge
dataset at parcnet-is2/example_test_set. Each trace file's loss pattern was
actually applied to its matching lossy file from its matching clean file, so
there's no synthetic loss model and no arbitrary pairing involved here -- this
is the real, correctly-matched ground truth.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import librosa

EXAMPLE_TEST_SET_DIR = Path(__file__).resolve().parent.parent / "parcnet-is2" / "example_test_set"
PACKET_SIZE = 512  # samples -- matches config.yaml's global.packet_dim


@dataclass
class TestFile:
    stem: str
    clean: np.ndarray
    lossy: np.ndarray
    trace: np.ndarray    # 1 = lost, 0 = received; one entry per packet
    sample_rate: int

    @property
    def n_packets(self) -> int:
        return len(self.trace)

    @property
    def loss_rate(self) -> float:
        return float(self.trace.mean())


def find_bursts(trace: np.ndarray) -> list[tuple[int, int]]:
    """All (start_packet, length) bursts of consecutive lost packets in a trace."""
    bursts, i = [], 0
    while i < len(trace):
        if trace[i]:
            start = i
            while i < len(trace) and trace[i]:
                i += 1
            bursts.append((start, i - start))
        else:
            i += 1
    return bursts


def list_available_stems(test_set_dir: Path = EXAMPLE_TEST_SET_DIR) -> list[str]:
    """All file stems that have a clean + lossy + trace triplet available."""
    clean_dir, lossy_dir, traces_dir = test_set_dir / "clean", test_set_dir / "lossy", test_set_dir / "traces"
    clean_stems = {p.stem for p in clean_dir.glob("*.wav")}
    lossy_stems = {p.stem for p in lossy_dir.glob("*.wav")}
    trace_stems = {p.stem for p in traces_dir.glob("*.txt")}
    return sorted(clean_stems & lossy_stems & trace_stems)


def load_test_file(stem: str, test_set_dir: Path = EXAMPLE_TEST_SET_DIR, packet_size: int = PACKET_SIZE) -> TestFile:
    """Loads one (clean, lossy, trace) triplet, truncated to a whole number of
    packets (defensive -- in practice the dataset already divides evenly).
    """
    clean, sr = librosa.load(test_set_dir / "clean" / f"{stem}.wav", sr=None, mono=True)
    lossy, _ = librosa.load(test_set_dir / "lossy" / f"{stem}.wav", sr=None, mono=True)
    trace = np.loadtxt(test_set_dir / "traces" / f"{stem}.txt", dtype=int).flatten()

    n_packets = min(len(trace), len(clean) // packet_size, len(lossy) // packet_size)
    clean = clean[:n_packets * packet_size]
    lossy = lossy[:n_packets * packet_size]
    trace = trace[:n_packets]

    return TestFile(stem=stem, clean=clean, lossy=lossy, trace=trace, sample_rate=sr)


def sample_test_files(n: int, test_set_dir: Path = EXAMPLE_TEST_SET_DIR, seed: int = 0,
                       min_packets: int = 20) -> list[TestFile]:
    """Deterministically samples n files (seeded shuffle, not just the first n
    alphabetically, so a small n is still representative of the dataset).
    """
    stems = list_available_stems(test_set_dir)
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)

    files = []
    for stem in stems:
        if len(files) >= n:
            break
        tf = load_test_file(stem, test_set_dir)
        if tf.n_packets >= min_packets:
            files.append(tf)
    return files


def sample_diverse_test_files(n: int, test_set_dir: Path = EXAMPLE_TEST_SET_DIR, seed: int = 0,
                               pool_size: int = 150, n_bins: int = 5, min_packets: int = 20) -> list[TestFile]:
    """Stratified sample by signal complexity (spectral flatness), instead of
    pure random -- a plain random sample tends to clump wherever the dataset
    happens to be dense (this dataset is mostly tonal/simple content), so the
    "how does quality depend on complexity" analysis ends up mostly testing
    one kind of signal. This draws a candidate pool, measures each file's
    complexity, sorts into `n_bins` quantile buckets, and takes a roughly
    equal share from every bucket -- same dataset, genuinely varied sample.
    """
    stems = select_diverse_stems(n, test_set_dir, seed, pool_size, n_bins, min_packets)
    return [load_test_file(stem, test_set_dir) for stem in stems]


def _flatness_or_none(args) -> float | None:
    """Spectral flatness of one file's clean audio, or None if the file is too short."""
    from research.complexity import spectral_flatness
    stem, test_set_dir, min_packets = args
    tf = load_test_file(stem, test_set_dir)
    return spectral_flatness(tf.clean) if tf.n_packets >= min_packets else None


def parallel_map(fn, tasks, workers: int = 1, initializer=None, initargs=()):
    """Ordered map over tasks, in `workers` processes (in this process if
    workers <= 1). Yields results as they become available, in task order."""
    if workers <= 1:
        if initializer:
            initializer(*initargs)
        yield from map(fn, tasks)
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=initargs) as pool:
        yield from pool.map(fn, tasks)


def select_diverse_stems(n: int, test_set_dir: Path = EXAMPLE_TEST_SET_DIR, seed: int = 0,
                          pool_size: int = 150, n_bins: int = 5, min_packets: int = 20,
                          workers: int = 1) -> list[str]:
    """The file selection behind sample_diverse_test_files, returning stems
    only: each candidate is loaded just long enough to measure its flatness,
    so memory use doesn't grow with the pool size."""
    stems = list_available_stems(test_set_dir)
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)

    # The pool must hold at least n files, or asking for more than pool_size
    # would silently return only pool_size.
    pool_size = max(pool_size, n)
    pool, position = [], 0
    while len(pool) < pool_size and position < len(stems):
        chunk = stems[position:position + pool_size - len(pool)]
        position += len(chunk)
        tasks = [(stem, test_set_dir, min_packets) for stem in chunk]
        for stem, flatness in zip(chunk, parallel_map(_flatness_or_none, tasks, workers)):
            if flatness is not None:
                pool.append((flatness, stem))
    pool.sort(key=lambda item: item[0])

    per_bin = [n // n_bins + (1 if i < n % n_bins else 0) for i in range(n_bins)]
    bin_edges = np.linspace(0, len(pool), n_bins + 1, dtype=int)
    selected: list[str] = []
    for take, start, end in zip(per_bin, bin_edges[:-1], bin_edges[1:]):
        bucket = pool[start:end]
        rng.shuffle(bucket)
        selected.extend(stem for _, stem in bucket[:take])
    return selected[:n]
