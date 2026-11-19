import numpy as np
from synth import SR, tones

from counterpoint.catalogue import Sample
from counterpoint.profile import analyse
from counterpoint.render import render
from counterpoint.scoring import Match, Parts, Stretch


def sample(bpm=96, bars=1):
    return Sample("x", "x.flac", None, None, bpm, bars, "fitted", None, None, "allow", None,
                  None, "t:x", None)


def test_render_shifts_and_fits_the_bar():
    y = tones([9, 0, 4], dur=2.4)  # A minor, a little short of a bar at 96
    s = sample()
    m = Match(s, 2, 0.0, Parts(0, 0, 0), Stretch(100.0, 96.0, "normal", 100 / 96), None)
    loop = render(y, SR, s, m, tempo=100.0, sr=22050)
    assert loop.beats == 4
    assert loop.frames == round(4 * 60 / 100 * 22050)
    p = analyse(loop.audio, 22050)
    top = set(np.argsort(p.chroma)[::-1][:3])
    assert top == {11, 2, 6}  # B minor: A minor up a tone


def test_render_double_time():
    y = tones([9], dur=2.4)
    s = sample(bpm=50, bars=1)
    m = Match(s, 0, 0.0, Parts(0, 0, 0), Stretch(100.0, 50.0, "double", 1.0), None)
    loop = render(y, SR, s, m, tempo=100.0, sr=22050)
    assert loop.beats == 8  # one bar at 50 BPM is two bars at 100
