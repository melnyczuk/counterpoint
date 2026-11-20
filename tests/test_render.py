import numpy as np
import pytest
from synth import SR, noise, tones

from counterpoint.catalogue import Sample
from counterpoint.profile import analyse
from counterpoint.render import MAX_BOOST_DB, PEAK, loudness, normalise, render
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


def test_render_quad_time_plays_slower():
    y = tones([9], dur=2.4)
    s = sample(bpm=100, bars=1)
    m = Match(s, 0, 0.0, Parts(0, 0, 0), Stretch(100.0, 100.0, "quad", 0.25), None)
    loop = render(y, SR, s, m, tempo=100.0, sr=22050)
    assert loop.beats == 16  # one bar, a quarter of the speed
    assert loop.frames == round(16 * 60 / 100 * 22050)


def test_loudness_of_a_reference_sine():
    sr = 48000
    t = np.arange(3 * sr) / sr
    y = 0.1 * np.sin(2 * np.pi * 997 * t)
    assert loudness(y[None], sr) == pytest.approx(-23.0, abs=0.5)  # one channel
    assert loudness(np.zeros((2, sr)), sr) == -70.0


def test_loudness_ignores_silence():
    sr = 48000
    t = np.arange(sr) / sr
    tone = 0.1 * np.sin(2 * np.pi * 997 * t)
    sparse = np.concatenate([tone, np.zeros(3 * sr)])
    # Only the blocks straddling the edge of the tone count against it.
    assert loudness(sparse[None], sr) == pytest.approx(loudness(tone[None], sr), abs=1.0)


def test_render_matches_loudness():
    s = sample()
    m = Match(s, 0, 0.0, Parts(0, 0, 0), None, None)
    quiet = render(tones([9, 0, 4], dur=2.5) * 0.3, SR, s, m, 96.0, sr=22050)
    loud = render(tones([7, 11, 2], dur=2.5), SR, s, m, 96.0, sr=22050)
    assert loudness(quiet.audio, 22050) == pytest.approx(loudness(loud.audio, 22050), abs=0.5)
    assert quiet.gain_db > loud.gain_db + 8
    assert max(np.abs(quiet.audio).max(), np.abs(loud.audio).max()) <= PEAK + 1e-6


def test_normalise_limits():
    sr = 22050
    whisper = noise(dur=2.0)[:, :] * 1e-4
    y, gain = normalise(np.repeat(whisper, 2, axis=0), sr)
    assert gain == pytest.approx(MAX_BOOST_DB)
    spiky = np.zeros((2, sr), np.float32)
    spiky[:, ::1000] = 0.5  # quiet overall, but with peaks
    y, gain = normalise(spiky, sr)
    assert np.abs(y).max() <= PEAK + 1e-6


def test_render_can_leave_the_level_alone():
    s = sample()
    m = Match(s, 0, 0.0, Parts(0, 0, 0), None, None)
    loop = render(tones([9], dur=2.5), SR, s, m, 96.0, sr=22050, target_lufs=None)
    assert loop.gain_db == 0.0
