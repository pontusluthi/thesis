"""One-time conversion of the GazeBase CSV tree into a single float32 memmap.

The CSVs are ~38 GB of *text*. The numbers in them are far smaller: FXS+HSS+RAN
is ~375M samples, which as float32 x/y is ~3 GB. So the job here is to pay the
parsing cost once, offline, and leave behind something a Dataset can slice
without decoding anything.

Output layout (``out_dir``):
    data.f32        raw C-contiguous float32, shape (total_samples, n_channels)
    index.parquet   one row per recording: subject/round/session/task, offset
                    and length into data.f32
    manifest.json   the PrepConfig used, plus dtype/shape metadata

One recording in, one contiguous block out: blinks (NaN runs, plus the
unflagged ones that show up as y diving to about -20 deg) are linearly
interpolated over, however long they are. Recordings are never cut up, so the
Dataset just tiles each one with non-overlapping windows.
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pyarrow.csv as pa_csv
from scipy.ndimage import maximum_filter1d
from tqdm import tqdm

from scipy.ndimage import convolve1d, maximum_filter1d

RATE = 1000  # GazeBase is recorded at 1000 Hz throughout


@dataclass(frozen=True)
class PrepConfig:
    """Everything that is baked into data.f32 and therefore costs a re-run to change.

    Deliberately excluded: normalization, window length, augmentation. Those are
    cheap, get applied in the Dataset, and you will want to sweep them.
    """

    channels: tuple[str, ...] = ("x", "y", "xT", "yT")
    tasks: tuple[str, ...] = ("FXS",)
    blink_y_thresh: float | None = -15.0  # y below this (deg) is a blink; None to skip
    blink_margin_ms: int = 100  # also interpolate over this much either side
    smooth_window: int = 23  # odd; None smooth_type to skip smoothing
    smooth_type: str | None = None


def _samples(ms: int) -> int:
    return int(round(RATE * ms / 1000))

_WINDOWS = {
    "bartlett": np.bartlett, "hanning": np.hanning, "hamming": np.hamming,
    "blackman": np.blackman, "flat": np.ones,
}

def smooth(x: np.ndarray, window_length: int, window: str = "bartlett", axis: int = 0) -> np.ndarray:
    """Zero-phase FIR smoothing along `axis`.

    Use this exact function for real AND generated gaze (e.g. axis=-1 on a
    (B, 2, T) batch), so both go through an identical filter.
    """
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd (keeps the filter zero-phase)")
    w = _WINDOWS[window](window_length)
    return convolve1d(x, w / w.sum(), axis=axis, mode="nearest")

def _smooth(arr: np.ndarray, cfg: PrepConfig) -> np.ndarray:
    """Smooth gaze channels only; targets are step functions and stay untouched."""
    if cfg.smooth_type is None:
        return arr
    gaze = [i for i, c in enumerate(cfg.channels) if c in ("x", "y")]
    out = arr.copy()
    out[:, gaze] = smooth(arr[:, gaze], cfg.smooth_window, cfg.smooth_type, axis=0)
    return out

def parse_name(path: str) -> dict[str, str]:
    """`S_9180_S2_HSS.csv` -> round 9, subject 180, session 2, task HSS."""
    stem = os.path.basename(path).removesuffix(".csv")
    _, rs, session, task = stem.split("_")
    return {
        "round": rs[0],
        "subject": f"{int(rs[1:]):03d}",
        "session": session.lstrip("S"),
        "task": task,
    }

def read_channels(path: str, channels: tuple[str, ...]) -> np.ndarray:
    """Read the requested columns *by name* into a (T, C) float32 array.

    Column order is not consistent across the GazeBase tree -- both
    `n,x,y,val,dP,lab,xT,yT` and `n,x,y,val,xT,yT,dP,lab` occur -- so positional
    indexing silently permutes channels between files. Always select by name.
    """
    table = pa_csv.read_csv(path)
    missing = [c for c in channels if c not in table.column_names]
    if missing:
        raise KeyError(f"{path} is missing columns {missing}")
    arr = np.column_stack(
        [table.column(c).to_numpy(zero_copy_only=False) for c in channels]
    ).astype(np.float32, copy=False)
    arr[~np.isfinite(arr)] = np.nan  # fold +/-inf into the NaN handling below
    return arr


def _blink_mask(arr: np.ndarray, cfg: PrepConfig) -> np.ndarray:
    """Gaze samples to throw away: NaNs and unflagged blinks, widened by the margin.

    Not every blink is NaN in GazeBase. Many appear as y diving to about -20 deg
    and back within ~200 ms. No stimulus goes below about -10 deg (RAN's lowest
    target), so a plain threshold on y catches them. The threshold only sees the
    bottom of the dive, and NaN blinks have the same dive at their edges, hence
    the margin.
    """
    bad = ~np.isfinite(arr).all(axis=1)
    if cfg.blink_y_thresh is not None:
        bad |= arr[:, cfg.channels.index("y")] < cfg.blink_y_thresh
    margin = _samples(cfg.blink_margin_ms)
    return maximum_filter1d(bad, size=2 * margin + 1, mode="nearest")


def _interpolate(arr: np.ndarray, bad: np.ndarray, gaze: list[int]) -> bool:
    """Linearly fill the blinks in the gaze channels and any leftover NaNs, in place.

    Targets (xT, yT) are known during a blink, so only their own NaNs are
    filled. Returns False if a channel has no finite sample to interpolate from.
    """
    idx = np.arange(len(arr))
    for c in range(arr.shape[1]):
        hole = ~np.isfinite(arr[:, c])
        if c in gaze:
            hole |= bad
        if not hole.any():
            continue
        good = ~hole
        if not good.any():
            return False
        arr[hole, c] = np.interp(idx[hole], idx[good], arr[good, c])
    return True





def process_file(path: str, cfg: PrepConfig) -> np.ndarray | None:
    """CSV -> one clean, contiguous, (T, C) float32 recording, or None if unusable."""
    arr = read_channels(path, cfg.channels)
    if len(arr) <= cfg.smooth_window:
        return None

    gaze = [i for i, c in enumerate(cfg.channels) if c in ("x", "y")]
    if not _interpolate(arr, _blink_mask(arr, cfg), gaze):
        return None
    return np.ascontiguousarray(_smooth(arr, cfg), dtype=np.float32)


def _process_chunk(args) -> tuple[int, int, list[dict]]:
    chunk_id, files, cfg, out_dir = args
    shard = os.path.join(out_dir, f"shard_{chunk_id:05d}.f32")
    records: list[dict] = []
    offset = 0
    with open(shard, "wb") as fh:
        for path in files:
            arr = process_file(path, cfg)
            if arr is None:
                continue
            fh.write(arr.tobytes())
            records.append({**parse_name(path), "offset": offset, "length": len(arr)})
            offset += len(arr)
    return chunk_id, offset, records


def prepare(
    files: list[str],
    out_dir: str,
    cfg: PrepConfig = PrepConfig(),
    n_workers: int | None = None,
) -> str:
    """Convert ``files`` into ``out_dir``. Returns ``out_dir``.

    Runs once. Everything downstream reads the memmap, never the CSVs.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_workers = n_workers or max(1, (os.cpu_count() or 4) - 1)

    # More chunks than workers keeps the progress bar informative and evens out
    # the spread in file sizes (BLG files are ~7x smaller than HSS ones).
    n_chunks = min(len(files), n_workers * 8)
    bounds = np.linspace(0, len(files), n_chunks + 1).astype(int)
    tasks = [
        (i, files[bounds[i] : bounds[i + 1]], cfg, out_dir)
        for i in range(n_chunks)
        if bounds[i + 1] > bounds[i]
    ]

    results: dict[int, tuple[int, list[dict]]] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for chunk_id, n_samples, records in tqdm(
            pool.map(_process_chunk, tasks),
            total=len(tasks),
            desc=f"Preparing {len(files)} files",
        ):
            results[chunk_id] = (n_samples, records)

    # Concatenate shards in chunk order, shifting each shard's offsets by the
    # number of samples written before it.
    data_path = os.path.join(out_dir, "data.f32")
    all_records: list[dict] = []
    base = 0
    with open(data_path, "wb") as out:
        for chunk_id, *_ in tasks:
            n_samples, records = results[chunk_id]
            shard = os.path.join(out_dir, f"shard_{chunk_id:05d}.f32")
            with open(shard, "rb") as fh:
                shutil.copyfileobj(fh, out, length=16 << 20)
            os.remove(shard)
            all_records += [{**r, "offset": r["offset"] + base} for r in records]
            base += n_samples

    index = pd.DataFrame(all_records)
    index.to_parquet(os.path.join(out_dir, "index.parquet"), index=False)

    manifest = {
        "config": asdict(cfg),
        "dtype": "float32",
        "rate": RATE,
        "n_channels": len(cfg.channels),
        "total_samples": int(base),
        "n_segments": len(index),
        "n_files": len(files),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    hours = base / RATE / 3600
    print(
        f"Wrote {base:,} samples ({os.path.getsize(data_path) / 1e9:.1f} GB, "
        f"{hours:.1f} h of signal) in {len(index):,} recordings -> {out_dir}"
    )
    return out_dir
