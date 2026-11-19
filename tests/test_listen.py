import mido
import numpy as np

from counterpoint.clock import FreeClock
from counterpoint.harmony import NOTES
from counterpoint.listen import MidiHearing, input_state


def test_midi_hearing_fades():
    t = [0.0]
    h = MidiHearing(memory_s=4, now=lambda: t[0])
    for n in (57, 60, 64):
        h.message(mido.Message("note_on", note=n, velocity=100))
    held = h.profile()
    assert {NOTES[i] for i in np.flatnonzero(held)} == {"A", "C", "E"}
    for n in (57, 60, 64):
        h.message(mido.Message("note_off", note=n))
    t[0] = 2.0
    assert 0 < h.profile().sum() < held.sum()
    t[0] = 5.0
    assert h.profile().sum() == 0


def test_input_state():
    s = input_state("input", FreeClock(), np.array([1.0] + [0] * 11))
    assert s.is_input and s.now.profile[0] == 1.0
    assert input_state("input", FreeClock(), np.zeros(12)).now is None
