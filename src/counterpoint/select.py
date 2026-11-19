"""Ranking matches and picking a set that layers well together."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalogue import Sample
from .harmony import Target
from .profile import Profile
from .scoring import (
    Limits,
    Match,
    Weights,
    harmonic_parts,
    harmonic_score,
    score_sample,
    texture_match,
)


@dataclass(frozen=True)
class Filters:
    commercial: bool = False
    bpm_range: tuple[float, float] | None = None  # used only when there's no target tempo


def _keep(s: Sample, f: Filters) -> bool:
    if f.commercial and not s.commercial:
        return False
    if f.bpm_range and s.bpm is not None:
        lo, hi = f.bpm_range
        if not lo <= s.bpm <= hi:
            return False
    return True


def rank(
    samples: list[Sample],
    profiles: dict[str, Profile],
    target: Target,
    weights: Weights,
    limits: Limits,
    filters: Filters = Filters(),
) -> tuple[list[Match], list[Match]]:
    """Pitched matches best first, and unpitched texture candidates best first."""
    pitched, textures = [], []
    for s in samples:
        p = profiles.get(s.id)
        if p is None or not _keep(s, filters):
            continue
        m = score_sample(s, p, target, weights, limits)
        if m is not None:
            pitched.append(m)
            continue
        t = texture_match(s, p, target, weights, limits)
        if t is not None:
            textures.append(t)
    pitched.sort(key=lambda m: -m.score)
    textures.sort(key=lambda m: -m.score)
    return pitched, textures


def _overlap(a: Profile, b: Profile) -> float:
    """How much two samples compete for the same band, 0..1."""
    return float(np.minimum(a.band_vector, b.band_vector).sum())


def layering_set(
    pitched: list[Match],
    target: Target,
    weights: Weights,
    size: int = 3,
    textures: list[Match] | None = None,
    texture: bool = False,
    pool: int = 50,
    max_drop: float = 0.25,
) -> list[Match]:
    """Greedy: start from the best match, then repeatedly add the candidate that makes the
    combined (summed, shifted) profile fit the target best, penalising band masking and a
    second sample from the same source recording. Candidates scoring more than
    ``max_drop`` below the best match are left out, so a set can come back smaller."""
    chosen: list[Match] = []
    # Weak matches never fill a set just to reach its size.
    floor = pitched[0].score - max_drop if pitched else 0.0
    candidates = [m for m in pitched[:pool] if m.score >= floor]
    while candidates and len(chosen) < size:
        best, best_val = None, -np.inf
        for c in candidates:
            if any(c.sample.id == m.sample.id for m in chosen):
                continue
            combined = sum((m.shifted for m in chosen), c.shifted)
            combined = combined / combined.sum()
            val = harmonic_score(harmonic_parts(combined, target), weights)
            # Keep each member's own costs (shift, stretch) in play.
            val += c.score - harmonic_score(c.parts, weights)
            if chosen:
                val -= weights.masking * max(_overlap(c.profile, m.profile) for m in chosen)
                if any(c.sample.source == m.sample.source for m in chosen):
                    val -= weights.same_source
            if val > best_val:
                best, best_val = c, val
        if best is None:
            break
        chosen.append(best)
        candidates = [c for c in candidates if c is not best]
    if texture and textures:
        sources = {m.sample.source for m in chosen}
        pick = next((t for t in textures if t.sample.source not in sources), textures[0])
        chosen.append(pick)
    return chosen
