"""
Atomic, retry-safe file writing. Windows has repeatedly held locks on this
project's output files -- antivirus/indexer scanning a freshly-written file,
or the file simply being open in a viewer or Excel -- which raises
OSError(22)/PermissionError(13) on the very next write to that exact path.

Writing to a fresh temp path first and then renaming over the destination
sidesteps this: Windows allows replacing a file that's merely open for
reading elsewhere, it just won't let a second process open that same path
for writing while something else holds it.
"""
import json
import os
import time
import uuid
from pathlib import Path


def _atomic_write(path, write_fn) -> None:
    path = Path(path)
    tmp_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp{path.suffix}")
    write_fn(tmp_path)
    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except OSError:
            if attempt == 4:
                tmp_path.unlink(missing_ok=True)
                raise
            time.sleep(0.5)


def save_fig(fig, path, **kwargs) -> None:
    _atomic_write(path, lambda tmp_path: fig.savefig(tmp_path, **kwargs))


def save_csv(df, path, **kwargs) -> None:
    _atomic_write(path, lambda tmp_path: df.to_csv(tmp_path, **kwargs))


def _save_with_fallback(save_fn, path, fallback_suffix: str) -> Path:
    """Tries `path` first; if it's held by a lock so persistent even the
    atomic rename can't get past it (seen repeatedly on this project -- the
    same exact file locked across multiple separate runs, presumably left
    open in something like Excel), falls back to a suffixed filename instead
    of raising. A script that just finished an expensive multi-hour
    computation should never lose that result to one stuck output file.
    Returns the path actually written to.
    """
    path = Path(path)
    try:
        save_fn(path)
        return path
    except OSError:
        fallback_path = path.with_name(f"{path.stem}{fallback_suffix}{path.suffix}")
        save_fn(fallback_path)
        return fallback_path


def save_fig_safe(fig, path, fallback_suffix: str = "_locked", **kwargs) -> Path:
    """save_fig that falls back to an alternate filename rather than raising
    -- see _save_with_fallback. Returns the path actually written to."""
    return _save_with_fallback(lambda p: save_fig(fig, p, **kwargs), path, fallback_suffix)


def save_csv_safe(df, path, fallback_suffix: str = "_locked", **kwargs) -> Path:
    """save_csv that falls back to an alternate filename rather than raising
    -- see _save_with_fallback. Returns the path actually written to."""
    return _save_with_fallback(lambda p: save_csv(df, p, **kwargs), path, fallback_suffix)


class Checkpoint:
    """Append-only JSON-lines store of finished results, keyed by a string.
    Each result is flushed to disk the moment it exists, so a crashed or
    killed run loses nothing that had finished, and re-running the same
    command resumes instead of starting over. With path=None it only keeps
    results in memory."""

    def __init__(self, path, fresh: bool = False):
        self.path = Path(path) if path is not None else None
        self.records = {}
        if self.path is None:
            return
        if fresh:
            self.path.unlink(missing_ok=True)
        elif self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue  # a line cut off mid-write by a crash
                self.records[entry["key"]] = entry["record"]

    def __contains__(self, key: str) -> bool:
        return key in self.records

    def add(self, key: str, record) -> None:
        self.records[key] = record
        if self.path is None:
            return
        line = json.dumps({"key": key, "record": record},
                          default=lambda o: o.item() if hasattr(o, "item") else str(o))
        with open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
