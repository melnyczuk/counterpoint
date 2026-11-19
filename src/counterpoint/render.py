"""Turning a pick into a loop buffer, in memory.

The stored FLAC is read, never written. It is resampled to the output rate, then shifted
and stretched with Rubber Band (through pedalboard, GPLv3), so the loop lasts exactly a
whole number of beats at the current tempo and stays on the bar lines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import librosa
import numpy as np

from .catalogue import Catalogue, Sample
from .clock import BEATS_PER_BAR
from .profile import load_audio
from .scoring import Match

SEAM_MS = 8.0


@dataclass(frozen=True)
class Loop:
    audio: np.ndarray  # (2, frames) float32, stereo
    sr: int
    beats: float  # loop length in beats at the tempo it was rendered for
    tempo: float

    @property
    def frames(self) -> int:
        return self.audio.shape[1]


def loop_beats(sample: Sample, match: Match, tempo: float, duration_s: float) -> float:
    """How many beats the loop lasts at ``tempo``.

    With a winnower beat grid, it's the sample's bars, scaled for half or double time.
    Without one, it's the duration rounded to whole bars at the current tempo.
    """
    st = match.stretch
    if sample.bars and sample.bpm and st is not None:
        scale = {"half": 0.5, "normal": 1.0, "double": 2.0}[st.time]
        return sample.bars * BEATS_PER_BAR * scale
    if sample.bars and sample.bpm:
        return sample.bars * BEATS_PER_BAR
    bars = max(1, round(duration_s * tempo / 60.0 / BEATS_PER_BAR))
    return float(bars * BEATS_PER_BAR)


def _stereo(y: np.ndarray) -> np.ndarray:
    if y.shape[0] == 1:
        return np.repeat(y, 2, axis=0)
    return y[:2]


def _seam(y: np.ndarray, sr: int) -> np.ndarray:
    """Blend the last few ms into the first so the loop point doesn't click after
    stretching moved it slightly."""
    n = min(int(sr * SEAM_MS / 1000), y.shape[1] // 4)
    if n < 2:
        return y
    fade = np.sin(np.linspace(0, math.pi / 2, n)) ** 2
    out = y.copy()
    out[:, :n] = y[:, :n] * fade + y[:, -n:] * (1 - fade)
    return out[:, :]


def render(audio: np.ndarray, src_sr: int, sample: Sample, match: Match, tempo: float,
           sr: int = 48000, beats: float | None = None) -> Loop:
    """Shift and stretch ``audio`` (channels, frames) into a loop of whole beats
    (``beats`` if given, else worked out from the sample's grid)."""
    from pedalboard import time_stretch

    y = _stereo(np.asarray(audio, dtype=np.float32))
    if src_sr != sr:
        y = librosa.resample(y, orig_sr=src_sr, target_sr=sr)
    if beats is None:
        beats = loop_beats(sample, match, tempo, y.shape[1] / sr)
    target = int(round(beats * 60.0 / tempo * sr))
    factor = y.shape[1] / target  # >1 speeds up
    if abs(factor - 1) > 1e-4 or match.shift:
        y = time_stretch(np.ascontiguousarray(y, dtype=np.float32), float(sr),
                         stretch_factor=float(factor),
                         pitch_shift_in_semitones=float(match.shift))
    if y.shape[1] >= target:
        y = y[:, :target]
    else:
        y = np.pad(y, ((0, 0), (0, target - y.shape[1])))
    return Loop(_seam(y, sr).astype(np.float32), sr, beats, tempo)


def render_from(catalogue: Catalogue, match: Match, tempo: float, sr: int = 48000) -> Loop:
    with catalogue.open(match.sample) as f:
        y, src_sr = load_audio(f.read())
    return render(y, src_sr, match.sample, match, tempo, sr)
