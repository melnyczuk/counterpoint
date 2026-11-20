"""How a voice picks its next sample from what the other voices are playing.

The reference isn't a typed chord: it's the combined shifted profile of every other voice
(their planned next sample where announced), plus the input voice and the anchor when
present. Each candidate is scored against that with the same scoring as ``match``, then
adjusted for this voice's role (band), for masking and for novelty. The pick is drawn
from the best few, weighted by score, so an ensemble doesn't settle into a fixed cycle.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from .bus import Playing, VoiceState
from .catalogue import Sample
from .harmony import Target, parse_chord
from .profile import Profile
from .scoring import LAST_RESORT_SLOW, Limits, Match, Weights, score_sample, texture_match

BANDS = {"low": 0, "mid": 1, "high": 2}


@dataclass
class Library:
    samples: list[Sample]
    profiles: dict[str, Profile]

    def __post_init__(self):
        self.by_id = {s.id: s for s in self.samples if s.id in self.profiles}
        self.pitched = [s for s in self.by_id.values() if self.profiles[s.id].pitched]
        self.unpitched = [s for s in self.by_id.values() if not self.profiles[s.id].pitched]


@dataclass
class Role:
    band: str | None = None  # favour samples with most energy in this band
    texture: bool = False  # only unpitched samples
    anchor: str | None = None  # chord symbol pulling the harmony back
    anchor_weight: float = 0.5  # relative to all other voices together
    input_weight: float = 1.5  # an input voice counts this much more than a sample voice
    band_weight: float = 0.3
    novelty: int = 8  # this voice's own last N picks are excluded
    top_k: int = 5  # draw from this many best
    temperature: float = 0.05  # lower = more often the very best


@dataclass
class Choice:
    match: Match
    reference: str  # what it was scored against, for the log

    def playing(self, start_beat: float, loop_beats: float) -> Playing:
        m = self.match
        return Playing(
            id=m.sample.id,
            shift=m.shift,
            ratio=1.0 if m.stretch is None else m.stretch.ratio,
            profile=[round(float(v), 4) for v in m.shifted],
            bands=list(m.profile.bands),
            source=m.sample.source,
            start_beat=start_beat,
            loop_beats=loop_beats,
        )


@dataclass
class Chooser:
    library: Library
    role: Role = field(default_factory=Role)
    weights: Weights = field(default_factory=Weights)
    limits: Limits = field(default_factory=Limits)
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        self.history: deque[str] = deque(maxlen=max(1, self.role.novelty))

    def reference(self, others: list[VoiceState], tempo: float) -> Target | None:
        """The combined profile of the other voices (and anchor), or None when there's
        nothing to fit with yet."""
        total = np.zeros(12)
        names = []
        for s in others:
            p = s.heard()
            if p is None or s.texture:
                continue
            w = self.role.input_weight if s.is_input else 1.0
            total += w * np.asarray(p.profile)
            names.append(s.name)
        scale = None
        if self.role.anchor:
            chord = parse_chord(self.role.anchor)
            h = chord.weights / chord.weights.sum()
            mass = total.sum() if total.sum() > 0 else 1.0
            total += self.role.anchor_weight * mass * h
            names.append(self.role.anchor)
        if total.sum() <= 0:
            return None
        root = int(np.argmax(total))
        if self.role.anchor:
            root = parse_chord(self.role.anchor).root
        return Target.from_pitch_classes(total, " + ".join(names), scale, tempo, root=root)

    def choose(self, others: list[VoiceState], tempo: float) -> Choice:
        taken = {p.id for s in others for p in (s.now, s.next) if p}
        sources = {p.source for s in others for p in (s.now, s.next) if p}
        bands = [np.asarray(p.bands) for s in others for p in [s.heard()] if p]
        target = self.reference(others, tempo)
        label = target.label if target else "nothing yet"
        pool = self.library.unpitched if self.role.texture else self.library.pitched
        # Never go silent: relax the limits step by step if nothing fits.
        # Never go silent: with a small library, relax step by step until something fits.
        # Doubling another voice's sample is the last resort; repeating our own is
        # the one before it.
        # Slowing is relaxed far more than speeding: nothing ever plays near twice as fast.
        wide = Limits(6, max_stretch=1.5, stretch=True, max_slow=LAST_RESORT_SLOW)
        for limits, avoid_taken, avoid_own in (
            (self.limits, True, True),
            (replace(self.limits, max_stretch=max(self.limits.max_stretch, 1.5),
                     max_slow=max(self.limits.max_slow, 8.0)), True, True),
            (wide, True, True),
            (wide, True, False),
            (wide, False, False),
        ):
            cands = self._candidates(pool, target, tempo, limits,
                                     taken if avoid_taken else set(), use_history=avoid_own)
            if cands:
                break
        else:
            raise RuntimeError("the library has nothing this voice can play")
        scored = [(self._adjust(m, sources, bands), m) for m in cands]
        scored.sort(key=lambda x: -x[0])
        top = scored[: self.role.top_k]
        best = top[0][0]
        t = max(self.role.temperature, 1e-6)
        weights = [math.exp((v - best) / t) for v, _ in top]
        pick = self.rng.choices([m for _, m in top], weights=weights)[0]
        self.history.append(pick.sample.id)
        return Choice(pick, label)

    def _candidates(self, pool, target, tempo, limits, taken, use_history=True) -> list[Match]:
        if target is None:
            # Nothing to fit with yet: every shift is as good as none, so don't shift.
            target = Target(np.ones(12), np.zeros(12), np.ones(12), "free", "any", tempo)
            limits = replace(limits, max_shift=0)
        out = []
        for s in pool:
            if s.id in taken or (use_history and s.id in self.history):
                continue
            p = self.library.profiles[s.id]
            if self.role.texture or not p.pitched:
                m = texture_match(s, p, target, self.weights, limits)
            else:
                m = score_sample(s, p, target, self.weights, limits)
            if m is not None:
                out.append(m)
        return out

    def _adjust(self, m: Match, sources: set[str], bands: list[np.ndarray]) -> float:
        v = m.score
        mine = m.profile.band_vector
        if self.role.band:
            v += self.role.band_weight * float(mine[BANDS[self.role.band]])
        if bands:
            v -= self.weights.masking * max(float(np.minimum(mine, b).sum()) for b in bands)
        if m.sample.source in sources:
            v -= self.weights.same_source
        return v
