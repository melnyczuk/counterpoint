"""Rebuilding an ensemble run from its voices' logs, as a stereo mix.

Each log line says which sample a voice started on which beat, with what shift, at what
tempo and pan, and for how many loops. Replaying renders each one again (from the
untouched catalogue files), at the same matched loudness, and lays it on a timeline until
that voice's next entry, with the voice's swell drawn again from the same seeds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np

from .catalogue import Sample
from .clock import BEATS_PER_BAR
from .player import Swell, pan_gains
from .profile import load_audio
from .render import render
from .scoring import Match, Parts


@dataclass
class Entry:
    voice: str
    beat: float
    tempo: float
    path: str
    shift: int
    ratio: float | None
    loop_beats: float
    loops: int
    pan: float
    swell_db: float = 0.0  # absent in older logs: no swell
    swell_bars: float = 4.0
    swell_drift: float = 0.0
    swell_phase: float = 0.0


def read_logs(paths: list[Path]) -> list[Entry]:
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(Entry(d["voice"], d["beat"], d["tempo"], d["catalogue"], d["shift"],
                             d.get("ratio"), d["loop_beats"], d["loops"], d.get("pan", 0.0),
                             d.get("swell_db", 0.0), d.get("swell_bars", 4.0),
                             d.get("swell_drift", 0.0), d.get("swell_phase", 0.0)))
    return sorted(out, key=lambda e: (e.beat, e.voice))


def _match_for(e: Entry) -> Match:
    s = Sample(e.path, e.path, None, None, None, None, None, None, None, "local", None, None,
               "replay", None)
    return Match(s, e.shift, 0.0, Parts(0, 0, 0), None, None)


def swell_curve(e: Entry, frames: int, frames_per_beat: float) -> np.ndarray:
    """The voice's swell gain over one entry's span, as the voice drew it: the logged
    phase at the entry, and a new period at every loop point."""
    if e.swell_db <= 0 or frames <= 0:
        return np.ones(frames, dtype=np.float32)
    sw = Swell(e.swell_db, e.swell_bars * BEATS_PER_BAR, e.swell_drift)
    sw.phase = e.swell_phase
    lb = max(e.loop_beats, 1.0)
    out = np.empty(frames, dtype=np.float32)
    k, beat = 0, e.beat
    while True:
        a = int(round(k * lb * frames_per_beat))
        if a >= frames:
            break
        b = min(frames, int(round((k + 1) * lb * frames_per_beat)))
        sw.new_loop(f"{e.voice}:{beat:g}")
        t = np.arange(b - a) / frames_per_beat  # beats into this loop
        phase = sw.phase + t / sw.period
        out[a:b] = 10 ** (sw.depth_db / 2 * np.sin(2 * np.pi * phase) / 20)
        sw.advance(lb)
        k, beat = k + 1, beat + lb
    return out


def mixdown(entries: list[Entry], sr: int = 48000, open_file=None) -> np.ndarray:
    """A (2, frames) mix of the run. Tempo is taken per entry; beats are converted at
    the first entry's tempo so the timeline lines up with the logged beats."""
    if not entries:
        return np.zeros((2, 0), dtype=np.float32)
    open_file = open_file or (lambda p: fsspec.open(p, "rb"))
    t0, tempo0 = entries[0].beat, entries[0].tempo
    spb = 60.0 / tempo0 * sr

    def frame(beat: float) -> int:
        return int(round((beat - t0) * spb))

    by_voice: dict[str, list[Entry]] = {}
    for e in entries:
        by_voice.setdefault(e.voice, []).append(e)
    spans = []
    for es in by_voice.values():
        for i, e in enumerate(es):
            end = es[i + 1].beat if i + 1 < len(es) else e.beat + max(
                BEATS_PER_BAR, e.loops * e.loop_beats)
            spans.append((e, frame(e.beat), frame(end)))
    out = np.zeros((2, max(b for _, _, b in spans)), dtype=np.float32)
    for e, a, b in spans:
        with open_file(e.path) as f:
            y, src_sr = load_audio(f.read())
        m = _match_for(e)
        audio = render(y, src_sr, m.sample, m, e.tempo, sr, beats=e.loop_beats).audio
        n = b - a
        reps = int(np.ceil(n / audio.shape[1]))
        seg = np.tile(audio, (1, reps))[:, :n] * swell_curve(e, n, spb)
        lg, rg = pan_gains(e.pan)
        out[0, a:b] += seg[0] * lg
        out[1, a:b] += seg[1] * rg
    peak = np.abs(out).max()
    if peak > 0.99:
        out *= 0.99 / peak
    return out
