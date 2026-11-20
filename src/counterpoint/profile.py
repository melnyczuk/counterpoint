"""Pitch profiles: what a sample contributes harmonically, computed once and cached.

A profile is the sample's mean 12-bin chroma (normalised to sum to 1), whether it is
pitched at all, and how its energy splits across low, mid and high bands (for masking
checks when layering). The cache lives in Counterpoint's own directory, never in
winnower's storage, keyed by sample id plus a file fingerprint so a changed file is
re-analysed.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .catalogue import Catalogue, Sample

# Same floor winnower uses to decide a sound has no key: (max - min) / max of mean chroma.
MIN_CONTRAST = 0.35
BAND_EDGES_HZ = (250.0, 2000.0)
PROFILE_VERSION = 1


@dataclass(frozen=True)
class Profile:
    chroma: tuple[float, ...]  # 12 values, C..B, summing to 1
    contrast: float
    bands: tuple[float, float, float]  # low, mid, high energy share, summing to 1

    @property
    def pitched(self) -> bool:
        return self.contrast >= MIN_CONTRAST

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.chroma, dtype=float)

    @property
    def band_vector(self) -> np.ndarray:
        return np.asarray(self.bands, dtype=float)


def analyse(y: np.ndarray, sr: int) -> Profile:
    """Profile of a mono or (channels, samples) signal."""
    mono = y if y.ndim == 1 else y.mean(axis=0)
    mono = np.asarray(mono, dtype=np.float32)
    if mono.size < sr // 4 or np.abs(mono).max() < 1e-5:
        return Profile((1 / 12,) * 12, 0.0, (1 / 3,) * 3)
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=2048).mean(axis=1)
    contrast = 0.0 if chroma.max() <= 0 else float((chroma.max() - chroma.min()) / chroma.max())
    # Chroma floors every bin at some level; subtracting the minimum keeps a pure triad
    # from looking like it has energy on all twelve notes.
    lifted = chroma - chroma.min()
    total = lifted.sum()
    norm = lifted / total if total > 0 else np.full(12, 1 / 12)

    power = np.abs(librosa.stft(mono, n_fft=2048, hop_length=1024)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    per_bin = power.sum(axis=1)
    lo, hi = BAND_EDGES_HZ
    bands = np.array(
        [per_bin[freqs < lo].sum(), per_bin[(freqs >= lo) & (freqs < hi)].sum(),
         per_bin[freqs >= hi].sum()]
    )
    bands = bands / bands.sum() if bands.sum() > 0 else np.full(3, 1 / 3)
    return Profile(
        tuple(round(float(v), 5) for v in norm),
        round(contrast, 4),
        tuple(round(float(v), 5) for v in bands),
    )


def load_audio(data: bytes) -> tuple[np.ndarray, int]:
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return y.T, sr


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "counterpoint"


class ProfileCache:
    """A JSON file of profiles, keyed ``{id}:{fingerprint}``."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict] = {}
        self._dirty = False
        if path.exists():
            raw = json.loads(path.read_text())
            if raw.get("version") == PROFILE_VERSION:
                self._data = raw.get("profiles", {})

    def get(self, key: str) -> Profile | None:
        d = self._data.get(key)
        if d is None:
            return None
        return Profile(tuple(d["chroma"]), d["contrast"], tuple(d["bands"]))

    def put(self, key: str, profile: Profile) -> None:
        self._data[key] = asdict(profile)
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Several voices may save at once: keep what others wrote since we loaded, and
        # write through a temp file of our own so no writer renames another's away.
        try:
            raw = json.loads(self.path.read_text())
            if raw.get("version") == PROFILE_VERSION:
                self._data = {**raw.get("profiles", {}), **self._data}
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix="profiles.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"version": PROFILE_VERSION, "profiles": self._data}, f)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._dirty = False


def profiles_for(
    catalogue: Catalogue,
    samples: list[Sample],
    cache: ProfileCache,
    progress=None,
) -> dict[str, Profile]:
    """Profiles for every sample, analysing only those not cached. ``progress(done, total)``
    is called after each analysis so long first runs on a Pi show they're moving."""
    out: dict[str, Profile] = {}
    todo: list[tuple[Sample, str]] = []
    for s in samples:
        key = f"{s.id}:{catalogue.fingerprint(s)}"
        p = cache.get(key)
        if p is None:
            todo.append((s, key))
        else:
            out[s.id] = p
    for i, (s, key) in enumerate(todo, 1):
        with catalogue.open(s) as f:
            y, sr = load_audio(f.read())
        p = analyse(y, sr)
        cache.put(key, p)
        out[s.id] = p
        if progress:
            progress(i, len(todo))
        if i % 50 == 0:
            cache.save()
    cache.save()
    return out
