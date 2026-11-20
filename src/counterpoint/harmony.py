"""Chords and scales as 12-bin pitch-class weights.

A :class:`Target` is what samples are scored against: weighted chord tones, a scale mask
(which notes are "in"), and optionally a tempo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Weight of each chord member: the root and third define the chord, the fifth is often
# implied, extensions colour it.
W_ROOT, W_THIRD, W_FIFTH, W_SEVENTH, W_EXT, W_BASS = 1.0, 0.9, 0.6, 0.8, 0.6, 1.0

# Seven-note scales as semitone steps from the tonic, in order of preference when more
# than one contains every chord tone.
SCALES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "melodic minor": (0, 2, 3, 5, 7, 9, 11),
    "harmonic minor": (0, 2, 3, 5, 7, 8, 11),
    "lydian dominant": (0, 2, 4, 6, 7, 9, 10),
    "altered": (0, 1, 3, 4, 6, 8, 10),
    "whole tone": (0, 2, 4, 6, 8, 10),
    "diminished": (0, 2, 3, 5, 6, 8, 9, 11),
}
ALIASES = {"ionian": "major", "aeolian": "minor", "natural minor": "minor"}


def pitch_class(name: str) -> int:
    m = re.fullmatch(r"([A-Ga-g])([#b♯♭]*)", name.strip())
    if not m:
        raise ValueError(f"not a note name: {name!r}")
    pc = _LETTER[m.group(1).upper()]
    for acc in m.group(2):
        pc += 1 if acc in "#♯" else -1
    return pc % 12


@dataclass(frozen=True)
class Chord:
    name: str
    root: int
    tones: dict[int, float]  # pitch class -> weight
    bass: int

    @property
    def weights(self) -> np.ndarray:
        h = np.zeros(12)
        for pc, w in self.tones.items():
            h[pc] = max(h[pc], w)
        return h

    @property
    def mask(self) -> np.ndarray:
        m = np.zeros(12)
        m[list(self.tones)] = 1.0
        return m


_CHORD = re.compile(r"^([A-Ga-g][#b♯♭]?)(.*?)(?:/([A-Ga-g][#b♯♭]?))?$")


def parse_chord(text: str) -> Chord:
    """Parse chord symbols such as ``Am9``, ``Fmaj7#11``, ``C/G``, ``Bø``, ``Dsus4``,
    ``G7b9``, ``E5``, ``Cadd9``."""
    m = _CHORD.match(text.strip())
    if not m:
        raise ValueError(f"can't read chord {text!r}")
    root = pitch_class(m.group(1))
    q = m.group(2).replace("♭", "b").replace("♯", "#").replace("(", "").replace(")", "")
    bass = pitch_class(m.group(3)) if m.group(3) else root

    third: int | None = 4
    fifth = 7
    seventh: int | None = None
    ext: list[int] = []

    def take(prefix: str) -> bool:
        nonlocal q
        if q.startswith(prefix):
            q = q[len(prefix) :]
            return True
        return False

    # Quality.
    if take("maj") or take("Maj") or take("M") or take("Δ") or take("^"):
        major7 = True
    else:
        major7 = False
    if not major7:
        if take("min") or take("mi") or take("m") or take("-"):
            third = 3
        elif take("dim") or take("°") or take("o"):
            third, fifth = 3, 6
            if take("7"):
                seventh = 9
        elif take("ø"):
            third, fifth, seventh = 3, 6, 10
        elif take("aug") or take("+"):
            fifth = 8
        elif take("5"):
            third = None
        if take("maj") or take("M") or take("Δ"):  # mMaj7, m(maj7)
            major7 = True
    if take("6/9") or take("69"):
        ext += [9, 2]
    # Number: 6, 7, 9, 11, 13.
    num = re.match(r"(13|11|9|7|6)", q)
    if num:
        q = q[len(num.group(1)) :]
        n = int(num.group(1))
        if n == 6:
            ext.append(9)
        else:
            if seventh is None:
                seventh = 11 if major7 else 10
            if n >= 9:
                ext.append(2)
            if n >= 11:
                ext.append(5)
            if n >= 13:
                ext.append(9)
    elif major7:
        seventh = 11
    # Suspensions, additions and alterations, in any order.
    while q:
        q = q.lstrip(", ")
        if not q:
            break
        if take("sus2"):
            third = 2
        elif take("sus4") or take("sus"):
            third = 5
        elif m2 := re.match(r"add(b|#)?(\d+)", q):
            q = q[m2.end() :]
            ext.append(_degree(int(m2.group(2)), m2.group(1)))
        elif m2 := re.match(r"(b|#|-|\+)(5|9|11|13)", q):
            q = q[m2.end() :]
            acc = {"-": "b", "+": "#"}.get(m2.group(1), m2.group(1))
            deg = int(m2.group(2))
            if deg == 5:
                fifth = 6 if acc == "b" else 8
            else:
                plain = _degree(deg, None)
                ext = [e for e in ext if e != plain]
                ext.append(_degree(deg, acc))
        elif take("alt"):
            fifth = 6
            ext += [1, 3, 8]
        else:
            raise ValueError(f"can't read {q!r} in chord {text!r}")

    tones: dict[int, float] = {}

    def add(interval: int, w: float) -> None:
        pc = (root + interval) % 12
        tones[pc] = max(tones.get(pc, 0.0), w)

    add(0, W_ROOT)
    if third is not None:
        add(third, W_THIRD)
    add(fifth, W_FIFTH)
    if seventh is not None:
        add(seventh, W_SEVENTH)
    for e in ext:
        add(e, W_EXT)
    tones[bass] = max(tones.get(bass, 0.0), W_BASS)
    return Chord(text.strip(), root, tones, bass)


# Relative major/minor: a minor chord's suffix and the matching major one a minor third
# up, keeping the chord's colour where there is an obvious equivalent.
_RELATIVE_MAJOR = {"m": "", "min": "", "-": "", "m6": "6", "m7": "maj7", "min7": "maj7",
                   "-7": "maj7", "m9": "maj9", "min9": "maj9", "-9": "maj9", "m11": "maj9",
                   "madd9": "add9", "m69": "69"}
_RELATIVE_MINOR = {"": "m", "M": "m", "maj": "m", "6": "m7", "maj7": "m7", "M7": "m7",
                   "Δ7": "m7", "maj9": "m9", "M9": "m9", "add9": "madd9", "69": "m9",
                   "6/9": "m9", "maj7#11": "m9"}


def related(name: str) -> list[str]:
    """Chords close to ``name``: the same chord a fourth and a fifth up, and its relative
    major or minor where the suffix has an obvious counterpart. Only symbols that parse
    are returned."""
    m = _CHORD.match(name.strip())
    if not m:
        return []
    root, suffix, bass = pitch_class(m.group(1)), m.group(2), m.group(3)
    bass_pc = pitch_class(bass) if bass else None

    def spell(r: int, s: str, b: int | None) -> str:
        return NOTES[r % 12] + s + ("" if b is None else "/" + NOTES[b % 12])

    out = [spell(root + 5, suffix, None if bass_pc is None else bass_pc + 5),
           spell(root + 7, suffix, None if bass_pc is None else bass_pc + 7)]
    if suffix in _RELATIVE_MAJOR:
        out.append(spell(root + 3, _RELATIVE_MAJOR[suffix], None))
    elif suffix in _RELATIVE_MINOR:
        out.append(spell(root - 3, _RELATIVE_MINOR[suffix], None))
    good = []
    for c in out:
        try:
            parse_chord(c)
        except ValueError:
            continue
        if c not in good:
            good.append(c)
    return good


def _degree(n: int, acc: str | None) -> int:
    base = {2: 2, 4: 5, 6: 9, 9: 2, 11: 5, 13: 9, 5: 7, 3: 4, 7: 10}[n]
    return (base + (1 if acc == "#" else -1 if acc == "b" else 0)) % 12


def parse_scale(text: str) -> np.ndarray:
    """``A dorian``, ``F# minor``, ``Eb``: a 12-bin 0/1 mask."""
    parts = text.strip().split(None, 1)
    tonic = pitch_class(parts[0])
    mode = parts[1].lower() if len(parts) > 1 else "major"
    mode = ALIASES.get(mode, mode)
    if mode not in SCALES:
        raise ValueError(f"unknown scale {mode!r}; known: {', '.join(SCALES)}")
    m = np.zeros(12)
    m[[(tonic + s) % 12 for s in SCALES[mode]]] = 1.0
    return m


def infer_scale(chord: Chord) -> tuple[np.ndarray, str]:
    """The first scale on the chord's root that contains every chord tone. Falls back to
    the chord tones alone, so nothing outside the chord counts as in-scale."""
    tones = {(pc - chord.root) % 12 for pc in chord.tones}
    for name, steps in SCALES.items():
        if tones <= set(steps):
            m = np.zeros(12)
            m[[(chord.root + s) % 12 for s in steps]] = 1.0
            return m, f"{NOTES[chord.root]} {name}"
    return chord.mask, "chord tones"


@dataclass(frozen=True)
class Target:
    """What samples are scored against."""

    weights: np.ndarray  # 12 chord-tone weights
    chord_mask: np.ndarray  # 1 on chord tones
    scale: np.ndarray  # 1 on scale tones
    label: str
    scale_label: str
    bpm: float | None = None
    notes: dict = field(default_factory=dict)

    @classmethod
    def from_chord(cls, chord: Chord, scale: str | None = None, bpm: float | None = None):
        if scale:
            mask, label = parse_scale(scale), scale
        else:
            mask, label = infer_scale(chord)
        # Chord tones always count as in-scale, even against a scale that lacks them.
        mask = np.maximum(mask, chord.mask)
        return cls(chord.weights, chord.mask, mask, chord.name, label, bpm)

    @classmethod
    def from_pitch_classes(
        cls,
        weights: np.ndarray,
        label: str,
        scale: str | None = None,
        bpm: float | None = None,
        root: int | None = None,
    ):
        """From measured pitch-class weights (MIDI or audio). Notes carrying at least a
        quarter of the strongest note's weight count as chord tones. The root (used to
        infer the scale) is the given one, e.g. MIDI's lowest note, else the strongest."""
        w = np.asarray(weights, dtype=float)
        if w.max() <= 0:
            raise ValueError("no pitched notes in the input")
        w = w / w.max()
        chord_mask = (w >= 0.25).astype(float)
        if scale:
            mask, slabel = parse_scale(scale), scale
        else:
            if root is None or not chord_mask[root]:
                root = int(np.argmax(w))
            tones = {pc: float(w[pc]) for pc in np.flatnonzero(chord_mask)}
            mask, slabel = infer_scale(Chord(label, root, tones, root))
        mask = np.maximum(mask, chord_mask)
        return cls(w * chord_mask, chord_mask, mask, label, slabel, bpm)
