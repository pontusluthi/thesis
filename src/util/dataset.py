"""GazeBase dataset access.

Two stages, deliberately separated:

    GazeBaseDataset  -- finds the CSVs, splits subjects, and drives the one-time
                        conversion in `prepare.py`. Run once.
    GazeBaseWindows  -- a torch Dataset that slices windows out of the resulting
                        memmap. Run every epoch; touches no CSV.

Typical use:

    ds = GazeBaseDataset("data/GazeBase_v2_0")
    ds.prepare("data/prepared")                       # once, ~10 min

    feats = "pos=robust,vel=sin"                      # or "pos", "vel=robust", ...
    train = ds.windows("data/prepared", "train", features=feats, window=5000)
    train.stats = train.fit_stats()                   # train split only; for "robust"
    val = ds.windows("data/prepared", "test", features=feats, window=5000, stats=train.stats)

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


# What a model sees: a subset of these features, each 2 channels (x, y), stacked in the
# order given, each with its own normalization:
#   none    physical units (deg, deg/s)
#   robust  (a - median) / 99.5th pct of |a - median|, clipped; needs fit_stats()
#   sin     sin(pi/2 * a / V_CLIP), saturating at V_CLIP deg/s; velocity only
NORMS = {"pos": ("none", "robust"), "vel": ("none", "robust", "sin")}
DEFAULT_FEATURES = "pos=robust"
V_CLIP = 200.0


def parse_features(spec: str | dict) -> dict[str, str]:
    """"pos=robust,vel=sin" (or the same as a dict) -> {"pos": "robust", "vel": "sin"}.

    A feature without "=..." is left unnormalized.
    """
    if isinstance(spec, str):
        spec = dict(f.split("=", 1) if "=" in f else (f, "none") for f in spec.replace(" ", "").split(","))
    for f, kind in spec.items():
        if kind not in NORMS.get(f, ()):
            raise ValueError(f"{f}={kind} is not one of {NORMS}")
    return dict(spec)


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

    def windows(self, prepared_dir: str, split: str, **kwargs) -> "GazeBaseWindows":
        """Build a windowed torch Dataset over one split of a prepared directory."""
        subjects = {"train": self.train_subjects, "test": self.test_subjects}[split]
        return GazeBaseWindows(prepared_dir, subjects=subjects, **kwargs)


class GazeBaseWindows(Dataset):
    """Fixed-length windows sliced out of the prepared memmap.

    Nothing is decoded at __getitem__ time: a window is a strided read from
    data.f32, so the OS page cache does the caching for you and a worker's job
    is a memcpy plus building the requested features (see NORMS). Samples are laid out (T, C) on disk
    and returned (C, T), which is what 1D convs and most sequence models want.
    """

    def __init__(
        self,
        prepared_dir: str,
        window: int,
        features: str | dict = DEFAULT_FEATURES,
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
        self.features = parse_features(features)

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
        pos, target_pos = w[:, :2].T, w[:, 2:].T  # (2, T) deg

        x = np.concatenate([self._norm(f, a) for f, a in self._raw(pos).items()])
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))  # (C, T)
        if self.return_subject:
            return x, int(self._labels[self.segment_of[i]])
        return x, torch.from_numpy(np.ascontiguousarray(pos)), torch.from_numpy(np.ascontiguousarray(target_pos))

    @property
    def in_channels(self) -> int:
        return 2 * len(self.features)

    @property
    def needs_stats(self) -> bool:
        return "robust" in self.features.values()

    def _raw(self, pos: np.ndarray) -> dict[str, np.ndarray]:
        """(2, T) position in deg -> the requested features in physical units, each (2, T)."""
        return {f: pos if f == "pos" else vecvel(pos.T).T for f in self.features}

    def _norm(self, f: str, a: np.ndarray, inverse: bool = False) -> np.ndarray:
        """One feature, (..., 2, T): physical units -> model space, or back with `inverse`."""
        kind = self.features[f]
        if kind == "sin":
            if inverse:
                return (2 * V_CLIP / np.pi) * np.arcsin(np.clip(a, -1, 1))
            return np.sin(np.pi / 2 * np.clip(a / V_CLIP, -1, 1))
        if kind == "robust":
            if self.stats is None:
                raise RuntimeError("fit_stats() on the train split and set .stats first")
            s = self.stats[f]
            c, sc = s["center"][:, None], s["scale"][:, None]
            if inverse:
                return a * sc + c
            return np.clip((a - c) / sc, -s["clip"], s["clip"])
        return a

    def denorm(self, x) -> dict[str, np.ndarray]:
        """(..., C, T) model space -> {feature: (..., 2, T) in deg or deg/s}."""
        x = np.asarray(x)
        return {f: self._norm(f, x[..., 2 * i : 2 * i + 2, :], inverse=True)
                for i, f in enumerate(self.features)}

    def fit_stats(self, max_windows: int = 2000, q: float = 99.5, clip: float = 1.5,
                  seed: int = 0) -> dict:
        """Per-channel center/scale for every "robust" feature. Fit on the train split only.

        center = median, scale = q-th percentile of |a - center|, so ~all data
        lands in [-1, 1] with headroom up to `clip` for rare large excursions.
        """
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self), size=min(max_windows, len(self)), replace=False))
        raws = [self._raw(np.asarray(self.data[s : s + self.window, :2]).T) for s in self.starts[idx]]
        stats = {}
        for f in (f for f, kind in self.features.items() if kind == "robust"):
            a = np.concatenate([r[f] for r in raws], axis=1)  # (2, N)
            center = np.nanmedian(a, axis=1)
            scale = np.nanpercentile(np.abs(a - center[:, None]), q, axis=1)
            stats[f] = {"center": center.astype(np.float32), "scale": scale.astype(np.float32),
                        "clip": clip}
        return stats

    @staticmethod
    def save_stats(stats: dict, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(stats, fh, indent=2, default=lambda a: a.tolist())

    @staticmethod
    def load_stats(path: str) -> dict:
        with open(path) as fh:
            return json.load(fh, object_hook=lambda d: {
                k: np.asarray(v, dtype=np.float32) if isinstance(v, list) else v for k, v in d.items()})

    def __repr__(self) -> str:
        hours = self.index["length"].sum() / self.rate / 3600
        return (
            f"GazeBaseWindows({len(self):,} windows of {self.window} @ {self.rate} Hz, "
            f"features {self.features}, {len(self.index):,} segments, "
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




