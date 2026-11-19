"""Turning MIDI files and audio clips into a :class:`Target` (pitch classes plus tempo)."""

from __future__ import annotations

import librosa
import mido
import numpy as np

from .harmony import NOTES, Target
from .profile import load_audio


def midi_pitch_classes(
    path: str, start_beat: float = 0.0, end_beat: float | None = None
) -> tuple[np.ndarray, float | None, int | None]:
    """Pitch-class weights (duration x velocity) of the notes sounding between two beats,
    the file's first tempo in BPM (None if it has no tempo event), and the pitch class of
    the lowest note, taken as the root."""
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    bpm: float | None = None
    weights = np.zeros(12)
    lowest: int | None = None
    lo = start_beat * tpb
    hi = None if end_beat is None else end_beat * tpb
    for track in mid.tracks:
        tick = 0
        on: dict[tuple[int, int], tuple[int, int]] = {}
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo" and bpm is None:
                bpm = round(mido.tempo2bpm(msg.tempo), 2)
            elif msg.type == "note_on" and msg.velocity > 0:
                on[(msg.channel, msg.note)] = (tick, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                started = on.pop((msg.channel, msg.note), None)
                if started is None:
                    continue
                t0, vel = started
                a, b = max(t0, lo), tick if hi is None else min(tick, hi)
                if b > a:
                    weights[msg.note % 12] += (b - a) / tpb * vel / 127
                    lowest = msg.note if lowest is None else min(lowest, msg.note)
    return weights, bpm, None if lowest is None else lowest % 12


def target_from_midi(
    path: str,
    scale: str | None = None,
    bpm: float | None = None,
    start_beat: float = 0.0,
    end_beat: float | None = None,
) -> Target:
    w, file_bpm, bass = midi_pitch_classes(path, start_beat, end_beat)
    if w.max() <= 0:
        raise ValueError(f"no notes in {path} in that range")
    return Target.from_pitch_classes(w, _label(w, path), scale, bpm or file_bpm, root=bass)


# A tempo only counts if beat tracking finds clear, regular beats.
MIN_BEAT_REGULARITY = 0.85


def audio_tempo(y: np.ndarray, sr: int) -> float | None:
    """Tempo of a steadily pulsed clip, or None for free time (drones, rain, rubato)."""
    mono = y.mean(axis=0) if y.ndim > 1 else y
    if mono.size < sr * 4:
        return None
    tempo, beats = librosa.beat.beat_track(y=mono, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    if len(beats) < 8 or tempo <= 0:
        return None
    ibi = np.diff(beats)
    regularity = 1.0 - float(np.std(ibi) / np.mean(ibi))
    onset = librosa.onset.onset_strength(y=mono, sr=sr)
    if regularity < MIN_BEAT_REGULARITY or onset.std() < 0.3 * onset.mean():
        return None
    return round(tempo, 2)


def target_from_audio(path: str, scale: str | None = None, bpm: float | None = None) -> Target:
    with open(path, "rb") as f:
        y, sr = load_audio(f.read())
    mono = y.mean(axis=0)
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=2048).mean(axis=1)
    chroma = chroma - chroma.min()
    return Target.from_pitch_classes(
        chroma, _label(chroma, path), scale, bpm if bpm else audio_tempo(y, sr)
    )


def _label(w: np.ndarray, path: str) -> str:
    top = [NOTES[i] for i in np.argsort(w)[::-1][:4] if w[i] >= 0.25 * w.max()]
    return f"{path} ({' '.join(top)})"
