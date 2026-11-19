import numpy as np
import pytest

from counterpoint.catalogue import Sample
from counterpoint.harmony import NOTES, Target, parse_chord
from counterpoint.profile import Profile
from counterpoint.scoring import Limits, Weights, best_stretch, score_sample


def prof(*notes):
    v = np.zeros(12)
    for n in notes:
        v[NOTES.index(n)] = 1
    v /= v.sum()
    return Profile(tuple(v), 1.0, (0.2, 0.6, 0.2))


def sample(bpm=96, tempo_source="fitted"):
    return Sample("x", "x.flac", None, None, bpm, 4, tempo_source, 10.0, "CC0-1.0", "allow",
                  None, "x", "t:x", None)


def target(chord, bpm=None):
    return Target.from_chord(parse_chord(chord), bpm=bpm)


def test_shift_found():
    m = score_sample(sample(), prof("C", "E", "G"), target("D"), Weights(), Limits())
    assert m.shift == 2


def test_no_shift_limit():
    m = score_sample(sample(), prof("C", "E", "G"), target("D"), Weights(), Limits(max_shift=0))
    assert m.shift == 0
    assert m.parts.fit < 1


def test_shift_beyond_limit_not_chosen():
    # C major against F# needs a tritone; with max_shift 3 the best is something else.
    m = score_sample(sample(), prof("C", "E", "G"), target("F#"), Weights(), Limits())
    assert abs(m.shift) <= 3


def test_triad_beats_cluster():
    w, lim, t = Weights(), Limits(max_shift=0), target("Am")
    triad = score_sample(sample(), prof("A", "C", "E"), t, w, lim)
    cluster = score_sample(sample(), prof("A", "A#", "B", "C"), t, w, lim)
    assert triad.score > cluster.score
    assert cluster.parts.roughness > triad.parts.roughness


def test_colour_rewarded_up_to_cap():
    w, lim, t = Weights(), Limits(max_shift=0), target("Am")
    plain = score_sample(sample(), prof("A", "C", "E"), t, w, lim)
    ninth = score_sample(sample(), prof("A", "C", "E", "B"), t, w, lim)
    assert ninth.parts.colour > 0
    assert plain.parts.colour == 0


def test_unpitched_excluded():
    flat = Profile((1 / 12,) * 12, 0.0, (1 / 3,) * 3)
    assert score_sample(sample(), flat, target("Am"), Weights(), Limits()) is None


@pytest.mark.parametrize(
    "sample_bpm,target_bpm,time,ratio",
    [(96, 96, "normal", 1.0), (72, 140, "double", 0.9722), (140, 70, "half", 1.0),
     (100, 110, "normal", 1.1)],
)
def test_stretch(sample_bpm, target_bpm, time, ratio):
    s = best_stretch(sample(sample_bpm), target_bpm)
    assert s.time == time
    assert s.ratio == pytest.approx(ratio, abs=1e-4)


def test_stretch_limit_excludes():
    m = score_sample(sample(96), prof("A", "C", "E"), target("Am", bpm=130), Weights(), Limits())
    assert m is None  # 130/96 = 1.35 > 1.25


def test_detected_groove_costs_more_to_stretch():
    t, w, lim = target("Am", bpm=105), Weights(), Limits(max_shift=0)
    fitted = score_sample(sample(96, "fitted"), prof("A", "C", "E"), t, w, lim)
    detected = score_sample(sample(96, "detected"), prof("A", "C", "E"), t, w, lim)
    assert fitted.score > detected.score


def test_no_stretch_filters_by_tempo():
    t = target("Am", bpm=110)
    lim = Limits(stretch=False)
    assert score_sample(sample(96), prof("A", "C", "E"), t, Weights(), lim) is None
    assert score_sample(sample(55), prof("A", "C", "E"), t, Weights(), lim) is not None
