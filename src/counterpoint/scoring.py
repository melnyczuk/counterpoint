"""Scoring one sample against a target at every transposition, and tempo matching.

All of this is suggestion: the shift and stretch it reports are applied at use time (in
Ableton, or in memory by a voice), never to the stored file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .catalogue import Sample
from .harmony import Target
from .profile import Profile

SHIFTS = tuple(range(-6, 6))  # every pitch class reachable exactly once

# Roughness of each interval class (0 = unison .. 6 = tritone): a simple pitch-class
# version of Plomp-Levelt dissonance. Seconds and the tritone are rough, fifths and
# fourths smooth, thirds in between.
ROUGHNESS_BY_IC = np.array([0.0, 1.0, 0.55, 0.15, 0.1, 0.05, 0.45])
_IC = np.array([[min((i - j) % 12, (j - i) % 12) for j in range(12)] for i in range(12)])
ROUGHNESS = ROUGHNESS_BY_IC[_IC]  # 12 x 12, ROUGHNESS[i, j] for sample pc i vs chord pc j


@dataclass(frozen=True)
class Weights:
    fit: float = 1.0
    colour: float = 0.3
    colour_cap: float = 0.25  # colour beyond this share reharmonises rather than colours
    roughness: float = 1.0
    shift: float = 0.02  # per semitone
    stretch: float = 0.4  # per octave of stretch (|log2 ratio|), for detected grooves
    stretch_fitted: float = 0.1  # the same for fitted tempos (textures, drones)
    masking: float = 0.3
    same_source: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> Weights:
        known = {k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass(frozen=True)
class Limits:
    max_shift: int = 3
    max_stretch: float = 1.25  # ratio either way
    stretch: bool = True


@dataclass(frozen=True)
class Stretch:
    target_bpm: float
    sample_bpm: float
    time: str  # half | normal | double
    ratio: float  # playback speed: >1 is faster

    @property
    def octaves(self) -> float:
        return abs(math.log2(self.ratio))


@dataclass(frozen=True)
class Parts:
    fit: float
    colour: float
    roughness: float


def harmonic_parts(q: np.ndarray, target: Target) -> Parts:
    """Parts for an already-shifted, normalised profile ``q``."""
    fit = float(q @ target.scale)
    colour = float(q @ (target.scale * (1 - target.chord_mask)))
    h = target.weights / target.weights.sum()
    # Only energy off the chord tones adds roughness: a sample doubling the chord adds
    # nothing the chord doesn't already have.
    rough = float((q * (1 - target.chord_mask)) @ ROUGHNESS @ h)
    return Parts(fit, colour, rough)


def harmonic_score(parts: Parts, w: Weights) -> float:
    colour = min(parts.colour, w.colour_cap)
    return w.fit * parts.fit + w.colour * colour - w.roughness * parts.roughness


def shifted(profile: Profile, shift: int) -> np.ndarray:
    """The sample's profile after transposing by ``shift`` semitones."""
    return np.roll(profile.vector, shift)


def best_stretch(sample: Sample, target_bpm: float | None) -> Stretch | None:
    """The playback ratio to bring the sample to the target tempo, trying it at half,
    normal and double time and keeping whichever needs the least stretching."""
    if not target_bpm or not sample.bpm:
        return None
    options = [("half", sample.bpm / 2), ("normal", sample.bpm), ("double", sample.bpm * 2)]
    time, eff = min(options, key=lambda o: abs(math.log2(target_bpm / o[1])))
    return Stretch(target_bpm, sample.bpm, time, round(target_bpm / eff, 4))


@dataclass(frozen=True)
class Match:
    sample: Sample
    shift: int
    score: float
    parts: Parts
    stretch: Stretch | None
    profile: Profile = field(repr=False)

    @property
    def shifted(self) -> np.ndarray:
        return shifted(self.profile, self.shift)

    @property
    def repitch_speed(self) -> float:
        """Playback speed that gives this shift by resampling (it changes tempo too)."""
        return round(2 ** (self.shift / 12), 4)

    def to_dict(self) -> dict:
        s = self.sample
        return {
            "id": s.id,
            "file": s.file,
            "key": s.key,
            "camelot": s.camelot,
            "shift": self.shift,
            "score": round(self.score, 4),
            "fit": round(self.parts.fit, 4),
            "colour": round(self.parts.colour, 4),
            "roughness": round(self.parts.roughness, 4),
            "bpm": s.bpm,
            "bars": s.bars,
            "tempo_source": s.tempo_source,
            "stretch": None
            if self.stretch is None
            else {
                "target_bpm": self.stretch.target_bpm,
                "time": self.stretch.time,
                "ratio": self.stretch.ratio,
            },
            "repitch_speed": self.repitch_speed,
            "pitched": self.profile.pitched,
            "licence": s.licence,
            "licence_decision": s.licence_decision,
            "title": s.title,
        }


def stretch_cost(sample: Sample, stretch: Stretch | None, w: Weights) -> float:
    if stretch is None:
        return 0.0
    per = w.stretch if sample.tempo_source == "detected" else w.stretch_fitted
    return per * stretch.octaves


def score_sample(
    sample: Sample, profile: Profile, target: Target, w: Weights, limits: Limits
) -> Match | None:
    """The best match for a pitched sample within the limits, or None if it's excluded
    (unpitched, or needs more stretch than allowed)."""
    if not profile.pitched:
        return None
    stretch = best_stretch(sample, target.bpm) if limits.stretch else None
    if stretch is not None and stretch.octaves > math.log2(limits.max_stretch) + 1e-9:
        return None
    if not limits.stretch and target.bpm and sample.bpm:
        # Without stretching, the tempo is only a filter: half/double time is still fine.
        s = best_stretch(sample, target.bpm)
        if s and s.octaves > math.log2(1.02):
            return None
    best: Match | None = None
    t_cost = stretch_cost(sample, stretch, w)
    for t in SHIFTS:
        if abs(t) > limits.max_shift:
            continue
        parts = harmonic_parts(shifted(profile, t), target)
        score = harmonic_score(parts, w) - w.shift * abs(t) - t_cost
        if best is None or score > best.score + 1e-12:
            best = Match(sample, t, score, parts, stretch, profile)
    return best


def texture_match(sample: Sample, profile: Profile, target: Target, w: Weights, limits: Limits):
    """An unpitched sample offered as a texture layer: never shifted, scored on stretch
    only."""
    if profile.pitched:
        return None
    stretch = best_stretch(sample, target.bpm) if limits.stretch else None
    if stretch is not None and stretch.octaves > math.log2(limits.max_stretch) + 1e-9:
        return None
    cost = stretch_cost(sample, stretch, w)
    return Match(sample, 0, -cost, Parts(0.0, 0.0, 0.0), stretch, profile)
