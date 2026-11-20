import numpy as np
import pytest

from counterpoint.harmony import NOTES, Target, infer_scale, parse_chord, parse_scale, related


def names(chord):
    return {NOTES[pc] for pc in chord.tones}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("C", {"C", "E", "G"}),
        ("Am", {"A", "C", "E"}),
        ("Am9", {"A", "C", "E", "G", "B"}),
        ("F#m7", {"F#", "A", "C#", "E"}),
        ("Bbmaj7", {"A#", "D", "F", "A"}),
        ("Fmaj7#11", {"F", "A", "C", "E", "B"}),
        ("G7b9", {"G", "B", "D", "F", "G#"}),
        ("Bø", {"B", "D", "F", "A"}),
        ("Bm7b5", {"B", "D", "F", "A"}),
        ("Cdim7", {"C", "D#", "F#", "A"}),
        ("Caug", {"C", "E", "G#"}),
        ("Dsus4", {"D", "G", "A"}),
        ("Esus2", {"E", "F#", "B"}),
        ("Cadd9", {"C", "E", "G", "D"}),
        ("C6/9", {"C", "E", "G", "A", "D"}),
        ("E5", {"E", "B"}),
        ("C/G", {"C", "E", "G"}),
        ("D/F#", {"D", "F#", "A"}),
        ("Dm(maj7)", {"D", "F", "A", "C#"}),
        ("C13", {"C", "E", "G", "A#", "D", "F", "A"}),
    ],
)
def test_chords(text, expected):
    assert names(parse_chord(text)) == expected


def test_slash_bass_is_weighted():
    c = parse_chord("C/G")
    assert c.bass == NOTES.index("G")
    assert c.tones[NOTES.index("G")] == 1.0


def test_bad_chord():
    with pytest.raises(ValueError):
        parse_chord("Hm7")


def test_scale_parse():
    m = parse_scale("A dorian")
    assert {NOTES[i] for i in np.flatnonzero(m)} == {"A", "B", "C", "D", "E", "F#", "G"}


@pytest.mark.parametrize(
    "chord,scale",
    [("C", "C major"), ("Am", "A minor"), ("G7", "G mixolydian"), ("Fmaj7#11", "F lydian"),
     ("Bm7b5", "B locrian")],
)
def test_inferred_scale(chord, scale):
    assert infer_scale(parse_chord(chord))[1] == scale


def test_chord_tones_always_in_scale():
    t = Target.from_chord(parse_chord("C7"), scale="C major")
    assert t.scale[NOTES.index("A#")] == 1.0


def test_related_chords():
    assert set(related("Am9")) == {"Dm9", "Em9", "Cmaj9"}
    assert set(related("C")) == {"F", "G", "Am"}
    assert set(related("Cmaj7")) == {"Fmaj7", "Gmaj7", "Am7"}
    for text in ("Fmaj7#11", "C/G", "G7b9", "Bø", "Dsus4", "Bbmaj7"):
        out = related(text)
        assert len(out) >= 2
        for c in out:
            parse_chord(c)
    assert "F/C" in related("C/G")
