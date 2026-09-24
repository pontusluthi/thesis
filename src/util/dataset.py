"""GazeBase dataset access.

Two stages, deliberately separated:

    GazeBaseDataset  -- finds the CSVs, splits subjects, and drives the one-time
                        conversion in `prepare.py`. Run once.
    GazeBaseWindows  -- a torch Dataset that slices windows out of the resulting
                        memmap. Run every epoch; touches no CSV.

Typical use:

    ds = GazeBaseDataset("data/GazeBase_v2_0")
    ds.prepare("data/prepared")                       # once, ~10 min

    train = ds.windows("data/prepared", "train", window=5000, stride=5000)
    stats = train.fit_stats()                         # train split only
    train.stats = stats
    val = ds.windows("data/prepared", "test", window=5000, stride=5000, stats=stats)

    loader = DataLoader(train, batch_size=64, shuffle=True,
                        num_workers=8, persistent_workers=True, pin_memory=True)
"""

from __future__ import annotations

import glob
import json
import os
import random
import zipfile
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:  # works both as `src.util.dataset` and as a plain script
    from .prepare import PrepConfig, parse_name, prepare
except ImportError:  # pragma: no cover
    from prepare import PrepConfig, parse_name, prepare

from .preprocessing import remove_blinks, na_replacement, smooth_data, detect_blinks_and_noises
from .microsaccade import vecvel

MEASURE_TYPES_ALL = ["FXS", "HSS", "RAN", "BLG", "TEX", "VD1", "VD2"]

# the once that are more likely to be used as there is less outside stimulus ig, easy to change
MEASURE_TYPES = ["FXS", "HSS", "RAN"]


class GazeBaseDataset:
    """Index over the raw GazeBase CSV tree: file discovery and subject splits."""

    def __init__(self, path: str, is_zip: bool = False, train_split: float = 0.8, seed: int = 1):
        """
        Args:
            path: Path to gazebase, either the original zip or an extracted tree.
            is_zip: Whether the path points to a zip file.
            train_split: Fraction of *subjects* (not files) used for training.
            seed: Seed for the subject split.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path '{path}' does not exist.")

        if zipfile.is_zipfile(path):
            if not is_zip:
                raise ValueError(f"The path '{path}' is a zip file, but 'is_zip' is set to False.")
            _unzip_gazebase(path)
            self.path = os.path.splitext(path)[0]
        else:
            self.path = path

        self.train_split = train_split
        self.seed = seed
        self.train_subjects, self.test_subjects = self._subject_split()

    def files(self, tasks: Sequence[str] | None = MEASURE_TYPES) -> list[str]:
        """All CSVs for the given task types, in a stable (sorted) order."""
        all_files = sorted(glob.glob(os.path.join(self.path, "**", "*.csv"), recursive=True))
        if tasks is None:
            return all_files
        wanted = set(tasks)
        return [f for f in all_files if parse_name(f)["task"] in wanted]

    def subjects(self) -> list[str]:
        return sorted({parse_name(f)["subject"] for f in self.files(tasks=None)})

    def _subject_split(self) -> tuple[list[str], list[str]]:
        """Split by subject to ensure no data leakage.

        The subject list is sorted before sampling: iteration order of a set of
        strings varies between interpreter runs (PYTHONHASHSEED), so seeding
        alone would not give you the same split twice.
        """
        subject_ids = self.subjects()
        rng = random.Random(self.seed)
        n_train = int(round(len(subject_ids) * self.train_split))
        train = set(rng.sample(subject_ids, n_train))
        return sorted(train), sorted(set(subject_ids) - train)

    def prepare(
        self,
        out_dir: str,
        cfg: PrepConfig | None = None,
        n_workers: int | None = None,
    ) -> str:
        """Convert the CSVs for `cfg.tasks` into a memmap under `out_dir`. Run once."""
        cfg = cfg or PrepConfig()
        return prepare(self.files(cfg.tasks), out_dir, cfg=cfg, n_workers=n_workers)

    def windows(self, prepared_dir: str, split: str, normalization: str, **kwargs) -> "GazeBaseWindows":
        """Build a windowed torch Dataset over one split of a prepared directory."""
        subjects = {"train": self.train_subjects, "test": self.test_subjects}[split]
        return GazeBaseWindows(prepared_dir, normalization=normalization, subjects=subjects, **kwargs)


class GazeBaseWindows(Dataset):
    """Fixed-length windows sliced out of the prepared memmap.

    Nothing is decoded at __getitem__ time: a window is a strided read from
    data.f32, so the OS page cache does the caching for you and a worker's job
    is a memcpy plus an optional normalize. Samples are laid out (T, C) on disk
    and returned (C, T), which is what 1D convs and most sequence models want.
    """

    def __init__(
        self,
        prepared_dir: str,
        window: int,
        normalization: str,
        stride: int | None = None,
        subjects: list[str] | None = None,
        tasks: list[str] | None = None,
        stats: dict | None = None,
        return_subject: bool = False,
    ):
        self.dir = prepared_dir
        self.window = window
        self.stride = stride or window
        self.stats = stats
        self.return_subject = return_subject
        self.normalization = normalization

        with open(os.path.join(prepared_dir, "manifest.json")) as fh:
            self.manifest = json.load(fh)
        self.n_channels = self.manifest["n_channels"]
        self.channels = list(self.manifest["config"]["channels"])
        self.rate = self.manifest["rate"]

        index = pd.read_parquet(os.path.join(prepared_dir, "index.parquet"))
        if subjects is not None:
            index = index[index["subject"].isin(set(subjects))]
        if tasks is not None:
            index = index[index["task"].isin(set(tasks))]
        self.index = index.reset_index(drop=True)

        self.starts, self.segment_of = self._build_window_index()

        # Subject ids as contiguous class labels, for identification tasks.
        self.subject_ids = sorted(self.index["subject"].unique())
        self.subject_to_label = {s: i for i, s in enumerate(self.subject_ids)}
        self._labels = self.index["subject"].map(self.subject_to_label).to_numpy(np.int64)

        self._mm: np.memmap | None = None  # opened lazily, per worker

    def _build_window_index(self) -> tuple[np.ndarray, np.ndarray]:
        """Global start offset of every window, plus which recording it came from.

        Windows never straddle a recording boundary. With the default
        stride == window they simply tile each recording, dropping only the
        remainder at its end.
        """
        offsets = self.index["offset"].to_numpy(np.int64)
        lengths = self.index["length"].to_numpy(np.int64)
        counts = np.maximum(0, (lengths - self.window) // self.stride + 1)

        segment_of = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        # position of each window within its own segment: 0, 1, 2, ... per segment
        within = np.arange(counts.sum(), dtype=np.int64) - np.repeat(
            np.concatenate(([0], np.cumsum(counts)[:-1])), counts
        )
        starts = offsets[segment_of] + within * self.stride
        return starts, segment_of

    @property
    def data(self) -> np.memmap:
        if self._mm is None:
            # Opened here rather than in __init__ so each DataLoader worker gets
            # its own mapping and nothing large has to be pickled at fork time.
            self._mm = np.memmap(
                os.path.join(self.dir, "data.f32"),
                dtype=np.float32,
                mode="r",
                shape=(self.manifest["total_samples"], self.n_channels),
            )
        return self._mm

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        start = self.starts[i]
        w = np.asarray(self.data[start : start + self.window], dtype=np.float32)

        pos = w[:, :2]
        target_pos = w[:, 2:]

        if self.normalization == 'sinusoidal':
            v = self.preprocess_sin(pos)

        

        x = torch.from_numpy(np.ascontiguousarray(v))  # (C, T)
        if self.return_subject:
            return x, int(self._labels[self.segment_of[i]])
        return v, pos.T, target_pos.T

    def preprocess_sin(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        v = vecvel(x[:, :2]) # dont send targets

        vx = np.clip(v[:, 0], -200, 200)
        vy = np.clip(v[:, 1], -200, 200)

        vx_sin = np.sin((np.pi / 2) * (vx / 200))
        vy_sin = np.sin((np.pi / 2) * (vy / 200))
        return np.stack([vx_sin, vy_sin]).astype(np.float32)

    def fit_stats(self, max_windows: int = 200_000, seed: int = 0) -> dict:
        """Per-channel mean/std, streamed over this split.

        Call this on the *train* split only and pass the result to the val/test
        split, otherwise test statistics leak into training.
        """
        rng = np.random.default_rng(seed)
        idx = np.arange(len(self))
        if len(idx) > max_windows:
            idx = rng.choice(idx, max_windows, replace=False)

        n = 0
        total = np.zeros(self.n_channels, dtype=np.float64)
        total_sq = np.zeros(self.n_channels, dtype=np.float64)
        for i in idx:
            w = np.asarray(self.data[self.starts[i] : self.starts[i] + self.window], np.float64)
            total += w.sum(axis=0)
            total_sq += (w**2).sum(axis=0)
            n += len(w)

        mean = total / n
        std = np.sqrt(np.maximum(total_sq / n - mean**2, 0)) + 1e-8
        return {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "channels": self.channels,
            "n_samples": int(n),
        }

    def save_stats(self, stats: dict, path: str) -> None:
        payload = {**stats, "mean": stats["mean"].tolist(), "std": stats["std"].tolist()}
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    @staticmethod
    def load_stats(path: str) -> dict:
        with open(path) as fh:
            payload = json.load(fh)
        payload["mean"] = np.asarray(payload["mean"], dtype=np.float32)
        payload["std"] = np.asarray(payload["std"], dtype=np.float32)
        return payload

    def __repr__(self) -> str:
        hours = self.index["length"].sum() / self.rate / 3600
        return (
            f"GazeBaseWindows({len(self):,} windows of {self.window} @ {self.rate} Hz, "
            f"{self.n_channels} ch {self.channels}, {len(self.index):,} segments, "
            f"{len(self.subject_ids)} subjects, {hours:.1f} h)"
        )


def _unzip_gazebase(path_zip_gazebase: str) -> None:
    # unzip gazebase file
    with zipfile.ZipFile(path_zip_gazebase, "r") as zip_ref:
        zip_ref.extractall(os.path.splitext(path_zip_gazebase)[0])

    _recursive_unzip(os.path.splitext(path_zip_gazebase)[0])


def _recursive_unzip(path: str) -> None:
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            if file.endswith(".zip"):
                zip_path = os.path.join(dirpath, file)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(dirpath)
                    os.remove(zip_path)  # remove the zip file after extraction
        for dir in dirnames:
            _recursive_unzip(os.path.join(dirpath, dir))




