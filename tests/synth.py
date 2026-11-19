"""Deterministic test signals and a fake winnower catalogue."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050
NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8,
        "A": 9, "A#": 10, "B": 11}


def tones(pcs, dur=4.0, octave=4, seed=0):
    """Organ-like tones on the given pitch classes (a few harmonics each)."""
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    for pc in pcs:
        f = 440.0 * 2 ** ((pc - 9) / 12 + (octave - 4))
        for k, a in ((1, 1.0), (2, 0.4), (3, 0.2)):
            y += a * np.sin(2 * np.pi * f * k * t)
    y *= 0.2 / max(1, len(pcs))
    return y[None].astype(np.float32)


def noise(dur=4.0, seed=0):
    return (0.2 * np.random.default_rng(seed).standard_normal(int(dur * SR)))[None].astype(
        np.float32
    )


def write_sample(root: Path, sid: str, y, *, key=None, bpm=96, bars=4, tempo_source="fitted",
                 licence="allow", source=None):
    folder = "no key" if key is None else f"x {key}"
    rel = f"samples/{folder}/{sid}.flac"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y.T, SR, format="FLAC")
    sidecar = {
        "id": sid,
        "file": rel,
        "source": {"provider": "test", "provider_id": source or sid, "title": sid,
                   "attribution": f"{sid} by test"},
        "licence": {"id": "CC0-1.0" if licence == "allow" else "CC-BY-NC-4.0",
                    "decision": licence},
        "region": {"bpm": bpm, "bars": bars, "tempo_source": tempo_source},
        "key": None if key is None else {"short": key, "camelot": "0A"},
        "audio": {"duration_s": y.shape[1] / SR},
    }
    (root / f"samples/{folder}/{sid}.json").write_text(json.dumps(sidecar))
    return rel
