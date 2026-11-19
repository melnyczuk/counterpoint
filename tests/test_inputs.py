import mido
import numpy as np
import soundfile as sf
from synth import SR, noise, tones

from counterpoint.harmony import NOTES
from counterpoint.inputs import audio_tempo, target_from_audio, target_from_midi


def write_midi(path, notes, bpm=100):
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    for n in notes:
        tr.append(mido.Message("note_on", note=n, velocity=100, time=0))
    for i, n in enumerate(notes):
        tr.append(mido.Message("note_off", note=n, velocity=0, time=1920 if i == 0 else 0))
    mid.save(path)


def test_midi_chord(tmp_path):
    p = tmp_path / "c.mid"
    write_midi(p, [57, 60, 64, 71], bpm=92)  # A C E B: Am(add9)
    t = target_from_midi(str(p))
    assert t.bpm == 92
    assert {NOTES[i] for i in np.flatnonzero(t.chord_mask)} == {"A", "C", "E", "B"}
    assert t.scale_label == "A minor"


def test_audio_chord_no_tempo(tmp_path):
    p = tmp_path / "drone.wav"
    sf.write(p, tones([9, 0, 4], dur=6).T, SR)
    t = target_from_audio(str(p))
    assert {NOTES[i] for i in np.flatnonzero(t.chord_mask)} >= {"A", "C", "E"}
    assert t.bpm is None


def test_audio_tempo_clicks():
    y = noise(dur=12) * 0.05
    period = int(SR * 60 / 120)
    for start in range(0, y.shape[1] - 400, period):
        y[0, start : start + 300] += np.hanning(300) * 0.9
    assert abs(audio_tempo(y, SR) - 120) < 3
