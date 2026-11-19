import json
import os

from counterpoint.catalogue import Catalogue
from counterpoint.cli import main
from counterpoint.harmony import Target, parse_chord
from counterpoint.profile import ProfileCache, profiles_for
from counterpoint.scoring import Limits, Weights
from counterpoint.select import Filters, layering_set, rank


def snapshot(root):
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*")}


def load(root, cache_dir):
    cat = Catalogue(str(root))
    samples = list(cat.samples())
    profiles = profiles_for(cat, samples, ProfileCache(cache_dir / "p.json"))
    return cat, samples, profiles


def test_reads_catalogue(catalogue, cache_dir):
    cat, samples, profiles = load(catalogue, cache_dir)
    assert {s.id for s in samples} == {"a_minor", "c_major", "g_minor", "cluster", "a_fifth",
                                       "rain"}
    assert profiles["a_minor"].pitched
    assert not profiles["rain"].pitched


def test_never_writes_to_catalogue(catalogue, cache_dir):
    before = snapshot(catalogue)
    load(catalogue, cache_dir)
    main(["--catalogue", str(catalogue), "match", "--chord", "Am", "--json"])
    assert snapshot(catalogue) == before


def test_cache_reused_and_invalidated(catalogue, cache_dir):
    cat, samples, _ = load(catalogue, cache_dir)
    calls = []
    profiles_for(cat, samples, ProfileCache(cache_dir / "p.json"), lambda d, t: calls.append(d))
    assert calls == []
    f = catalogue / samples[0].file
    os.utime(f, ns=(1, 1))
    profiles_for(cat, samples, ProfileCache(cache_dir / "p.json"), lambda d, t: calls.append(d))
    assert calls == [1]


def test_profile_finds_notes(catalogue, cache_dir):
    _, _, profiles = load(catalogue, cache_dir)
    top = sorted(range(12), key=lambda i: -profiles["a_minor"].chroma[i])[:3]
    assert set(top) == {9, 0, 4}  # A, C, E


def test_rank_and_set(catalogue, cache_dir):
    _, samples, profiles = load(catalogue, cache_dir)
    t = Target.from_chord(parse_chord("Am"))
    pitched, textures = rank(samples, profiles, t, Weights(), Limits())
    assert pitched[0].sample.id in {"a_minor", "a_fifth"}
    assert pitched[0].shift == 0
    assert pitched[-1].sample.id == "cluster" or pitched[-1].parts.roughness > 0.1
    assert [m.sample.id for m in textures] == ["rain"]
    g = next(m for m in pitched if m.sample.id == "g_minor")
    assert g.shift == 2  # G minor up a tone is A minor
    chosen = layering_set(pitched, t, Weights(), size=3, textures=textures, texture=True)
    ids = [m.sample.id for m in chosen]
    assert len(set(ids)) == len(ids)
    assert ids[-1] == "rain"
    # a_minor and a_fifth share a source recording: both only if nothing else fits.
    assert not {"a_minor", "a_fifth"} <= set(ids[:2])


def test_commercial_filter(catalogue, cache_dir):
    _, samples, profiles = load(catalogue, cache_dir)
    t = Target.from_chord(parse_chord("Am"))
    pitched, _ = rank(samples, profiles, t, Weights(), Limits(), Filters(commercial=True))
    assert "a_fifth" not in {m.sample.id for m in pitched}


def test_cli_json_and_export(catalogue, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    out = tmp_path / "set"
    main(["--catalogue", str(catalogue), "match", "--chord", "Am9", "--bpm", "96", "--json",
          "--export", str(out)])
    data = json.loads(capsys.readouterr().out)
    assert data["scale"] == "A minor"
    assert data["matches"] and data["set"]
    links = sorted(p.name for p in out.iterdir() if p.name != "set.json")
    assert len(links) == len(data["set"])
    for p in out.iterdir():
        if p.is_symlink():
            assert p.resolve().is_file()
    assert json.loads((out / "set.json").read_text())["bpm"] == 96


def test_cli_table(catalogue, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    main(["--catalogue", str(catalogue), "match", "--chord", "Am", "--texture"])
    out = capsys.readouterr().out
    assert "Am in A minor" in out and "layering set" in out
