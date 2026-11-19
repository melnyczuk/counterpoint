import pytest
from synth import NOTE, noise, tones, write_sample


@pytest.fixture
def catalogue(tmp_path):
    """A small winnower-shaped catalogue."""
    root = tmp_path / "winnower"
    n = NOTE
    write_sample(root, "a_minor", tones([n["A"], n["C"], n["E"]]), key="Am")
    write_sample(root, "c_major", tones([n["C"], n["E"], n["G"]]), key="C", bpm=110)
    write_sample(root, "g_minor", tones([n["G"], n["A#"], n["D"]]), key="Gm", bpm=72,
                 tempo_source="detected")
    write_sample(root, "cluster", tones([n["A"], n["A#"], n["B"], n["C"]]), key="A")
    write_sample(root, "a_fifth", tones([n["A"], n["E"]]), key="Am", source="a_minor",
                 licence="flag")
    write_sample(root, "rain", noise(), key=None, bpm=100)
    return root


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"
