"""Turning a pick into a loop buffer, in memory.

The stored FLAC is read, never written. It is resampled to the output rate, then shifted
and stretched with Rubber Band (through pedalboard, GPLv3), so the loop lasts exactly a
whole number of beats at the current tempo and stays on the bar lines. Finally it is
brought to a common loudness, so a hot sample doesn't bury the other voices.
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
TARGET_LUFS = -20.0  # each loop; leaves room for several voices summing
MAX_BOOST_DB = 12.0  # a near-silent texture isn't dragged up into its noise floor
MAX_CUT_DB = 24.0
PEAK = 0.9


@dataclass(frozen=True)
class Loop:
    audio: np.ndarray  # (2, frames) float32, stereo
    sr: int
    beats: float  # loop length in beats at the tempo it was rendered for
    tempo: float
    gain_db: float = 0.0  # the loudness correction applied

    @property
    def frames(self) -> int:
        return self.audio.shape[1]


def loop_beats(sample: Sample, match: Match, tempo: float, duration_s: float) -> float:
    """How many beats the loop lasts at ``tempo``.

    With a winnower beat grid, it's the sample's bars, scaled for half, double or quad
    time.
    Without one, it's the duration rounded to whole bars at the current tempo.
    """
    st = match.stretch
    if sample.bars and sample.bpm and st is not None:
        return sample.bars * BEATS_PER_BAR * st.scale
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


def _k_weighting(sr: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The two BS.1770 pre-filters (a high shelf for the head, then a high pass), worked
    out for any sample rate."""
    # High shelf, +4 dB above ~1.7 kHz.
    g, q, fc = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    a = 10 ** (g / 40)
    w = 2 * math.pi * fc / sr
    alpha, c, ra = math.sin(w) / (2 * q), math.cos(w), math.sqrt(a)
    shelf = (
        np.array([a * ((a + 1) + (a - 1) * c + 2 * ra * alpha),
                  -2 * a * ((a - 1) + (a + 1) * c),
                  a * ((a + 1) + (a - 1) * c - 2 * ra * alpha)]),
        np.array([(a + 1) - (a - 1) * c + 2 * ra * alpha,
                  2 * ((a - 1) - (a + 1) * c),
                  (a + 1) - (a - 1) * c - 2 * ra * alpha]),
    )
    # High pass at ~38 Hz.
    q, fc = 0.5003270373253953, 38.13547087613982
    w = 2 * math.pi * fc / sr
    alpha, c = math.sin(w) / (2 * q), math.cos(w)
    hp = (np.array([1.0, -2.0, 1.0]), np.array([1 + alpha, -2 * c, 1 - alpha]))
    return [shelf, hp]


def loudness(y: np.ndarray, sr: int) -> float:
    """Integrated loudness of ``y`` (channels, frames) in LUFS, after BS.1770: K-weighted,
    in 400 ms blocks, gated at -70 LUFS and then 10 LU below the ungated level, so the
    silences in a sparse sample don't make it look quiet."""
    from scipy.signal import lfilter

    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    for b, a in _k_weighting(sr):
        y = lfilter(b, a, y, axis=1)
    sq = (y**2).sum(axis=0)  # channel weights are 1 for left and right
    block, hop = int(0.4 * sr), int(0.1 * sr)
    if sq.size < block:
        z = np.array([sq.mean()]) if sq.size else np.array([0.0])
    else:
        c = np.concatenate([[0.0], np.cumsum(sq)])
        starts = np.arange(0, sq.size - block + 1, hop)
        z = (c[starts + block] - c[starts]) / block

    def lufs(ms):
        return -0.691 + 10 * np.log10(np.maximum(ms, 1e-20))

    z = z[lufs(z) > -70]
    if not z.size:
        return -70.0
    z = z[lufs(z) > lufs(z.mean()) - 10]
    return float(lufs(z.mean()))


def normalise(y: np.ndarray, sr: int, target: float = TARGET_LUFS) -> tuple[np.ndarray, float]:
    """``y`` brought to ``target`` LUFS (within limits), and the gain used, in dB. The gain
    is limited so peaks stay under :data:`PEAK`."""
    gain_db = max(-MAX_CUT_DB, min(MAX_BOOST_DB, target - loudness(y, sr)))
    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak > 0:
        gain_db = min(gain_db, 20 * math.log10(PEAK / peak))
    return (y * 10 ** (gain_db / 20)).astype(np.float32), gain_db


def render(audio: np.ndarray, src_sr: int, sample: Sample, match: Match, tempo: float,
           sr: int = 48000, beats: float | None = None,
           target_lufs: float | None = TARGET_LUFS) -> Loop:
    """Shift and stretch ``audio`` (channels, frames) into a loop of whole beats
    (``beats`` if given, else worked out from the sample's grid), at ``target_lufs``
    (None leaves the level alone)."""
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
    y = _seam(y, sr).astype(np.float32)
    gain_db = 0.0
    if target_lufs is not None:
        y, gain_db = normalise(y, sr, target_lufs)
    return Loop(y, sr, beats, tempo, gain_db)


def render_from(catalogue: Catalogue, match: Match, tempo: float, sr: int = 48000,
                target_lufs: float | None = TARGET_LUFS) -> Loop:
    with catalogue.open(match.sample) as f:
        y, src_sr = load_audio(f.read())
    return render(y, src_sr, match.sample, match, tempo, sr, target_lufs=target_lufs)
