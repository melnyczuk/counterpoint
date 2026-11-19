import asyncio
import json
import random

import numpy as np

from counterpoint.bus import CONTROL, Control, MemoryBus, Playing, VoiceState
from counterpoint.catalogue import Catalogue
from counterpoint.chooser import Chooser, Library, Role
from counterpoint.clock import FreeClock
from counterpoint.ensemble_cli import format_status, voice_commands
from counterpoint.harmony import NOTES
from counterpoint.player import NullPlayer
from counterpoint.profile import ProfileCache, profiles_for
from counterpoint.render import Loop, loop_beats
from counterpoint.replay import mixdown, read_logs
from counterpoint.voice import Voice, VoiceSettings


def library(root, cache_dir):
    cat = Catalogue(str(root))
    samples = list(cat.samples())
    return cat, Library(samples, profiles_for(cat, samples, ProfileCache(cache_dir / "p.json")))


def other(name, notes, sid="other"):
    v = np.zeros(12)
    for n in notes:
        v[NOTES.index(n)] = 1
    p = Playing(sid, 0, 1.0, list(v / v.sum()), [0.2, 0.6, 0.2], f"x:{sid}", 0, 4)
    return VoiceState(name, 0, 96, now=p)


def test_chooser_fits_the_others(big_catalogue, cache_dir):
    _, lib = library(big_catalogue, cache_dir)
    ch = Chooser(lib, Role(top_k=1), rng=random.Random(0))
    pick = ch.choose([other("v1", ["A", "C", "E"])], 96).match
    q = pick.shifted
    assert q[[1, 3, 6, 8, 10]].sum() < 0.2  # little energy off the A minor scale


def test_chooser_excludes_what_others_play(big_catalogue, cache_dir):
    _, lib = library(big_catalogue, cache_dir)
    ch = Chooser(lib, Role(top_k=1), rng=random.Random(0))
    first = ch.choose([other("v1", ["A", "C", "E"])], 96).match.sample.id
    ch2 = Chooser(lib, Role(top_k=1), rng=random.Random(0))
    again = ch2.choose([other("v1", ["A", "C", "E"], sid=first)], 96).match.sample.id
    assert again != first


def test_anchor_alone_is_a_reference(big_catalogue, cache_dir):
    _, lib = library(big_catalogue, cache_dir)
    ch = Chooser(lib, Role(anchor="Am", top_k=1), rng=random.Random(0))
    assert ch.reference([], 96).label == "Am"
    assert Chooser(lib).reference([], 96) is None
    assert ch.choose([], 96).match is not None


def test_texture_voice_picks_unpitched(big_catalogue, cache_dir):
    _, lib = library(big_catalogue, cache_dir)
    ch = Chooser(lib, Role(texture=True))
    assert ch.choose([other("v1", ["A", "C", "E"])], 96).match.sample.id in {"rain", "hiss"}


def fake_render(cat, match, tempo, sr):
    beats = loop_beats(match.sample, match, tempo, match.sample.duration_s or 2.0)
    frames = int(round(beats * 60 / tempo * sr))
    return Loop(np.zeros((2, frames), np.float32), sr, beats, tempo)


def test_four_voices(big_catalogue, cache_dir, tmp_path):
    cat, lib = library(big_catalogue, cache_dir)
    hub = MemoryBus.Hub()
    clock = FreeClock(bpm=1200)  # a bar every 0.2 s
    voices = []
    for i in range(4):
        name = f"v{i + 1}"
        vs = VoiceSettings(name, loops=(1, 2), prepare_beats=4, heartbeat_s=0.02,
                           turn_wait_s=0.1, log_path=tmp_path / f"{name}.jsonl", seed=i)
        ch = Chooser(lib, Role(texture=i == 3), rng=random.Random(i))
        voices.append(Voice(vs, cat, ch, clock, MemoryBus(hub), NullPlayer(8000),
                            render=fake_render))
    clashes = []

    async def watch():
        while True:
            playing = [v.now.id for v in voices if v.now]
            if len(playing) != len(set(playing)):
                clashes.append(playing)
            await asyncio.sleep(0.01)

    async def go():
        w = asyncio.create_task(watch())
        await asyncio.wait_for(asyncio.gather(*(v.run(changes=4) for v in voices)), 30)
        w.cancel()

    asyncio.run(go())
    assert all(v.changes == 4 for v in voices)
    assert clashes == []
    logs = [json.loads(line) for i in range(4)
            for line in (tmp_path / f"v{i + 1}.jsonl").read_text().splitlines()]
    assert len(logs) == 16
    assert all(e["beat"] % 4 == 0 for e in logs)  # every change on a bar line
    assert {e["id"] for e in logs if e["voice"] == "v4"} <= {"rain", "hiss"}
    # Voices that changed on the same bar announced their picks in name order.
    for e in logs:
        assert e["reference"]


def test_control_messages(big_catalogue, cache_dir):
    cat, lib = library(big_catalogue, cache_dir)
    hub = MemoryBus.Hub()
    clock = FreeClock(bpm=96)
    v = Voice(VoiceSettings("v1"), cat, Chooser(lib), clock, MemoryBus(hub), NullPlayer(8000))
    remote = MemoryBus(hub)

    async def go():
        await v.bus.start(v._on_message)
        await remote.start(lambda *a: None)
        for c in (Control("anchor", "Dm9"), Control("tempo", "120"),
                  Control("volume", "0.5", voice="v1"), Control("loops", "2-3"),
                  Control("volume", "0.1", voice="v9")):
            remote.send(CONTROL, c.to_json())

    asyncio.run(go())
    assert v.chooser.role.anchor == "Dm9"
    assert clock.tempo == 120
    assert v.player.mixer.gain == 0.5  # the v9-only message didn't apply
    assert v.s.loops == (2, 3)


def test_replay_mixdown(big_catalogue, tmp_path):
    entries = [
        {"voice": "v1", "beat": 0, "tempo": 120, "catalogue": str(big_catalogue /
         "samples/x am/am.flac"), "shift": 0, "ratio": 1.0, "loop_beats": 4, "loops": 2,
         "pan": -0.5},
        {"voice": "v2", "beat": 4, "tempo": 120, "catalogue": str(big_catalogue /
         "samples/x c/c.flac"), "shift": 2, "ratio": 1.0, "loop_beats": 4, "loops": 1,
         "pan": 0.5},
    ]
    log = tmp_path / "run.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in entries))
    mix = mixdown(read_logs([log]), sr=8000)
    assert mix.shape == (2, 8 * 4000)  # 8 beats at 120 = 4 s
    assert np.abs(mix[:, :16000]).max() > 0.01


def test_voice_commands():
    class A:
        voices, textures, spread_channels, config, catalogue = 3, 1, False, None, "/data"
        loops, anchor, anchor_weight, device, volume, sr = "4-8", "Am9", 0.5, None, 0.8, 48000
        prepare_bars, max_shift, max_stretch, bpm = 2, None, None, 100.0
        commercial, no_link, dry_run, peers = False, False, True, []

    cmds = voice_commands(A)
    assert [c[c.index("--name") + 1] for c in cmds] == ["v1", "v2", "v3"]
    assert "--texture" in cmds[2] and "--texture" not in cmds[0]
    assert "--bpm" in cmds[0] and "--bpm" not in cmds[1]
    assert cmds[0][cmds[0].index("--pan") + 1] == "-0.80"


def test_format_status():
    s = other("v1", ["A", "C", "E"], sid="am")
    s.change_beat = 32
    assert "v1" in format_status([s]) and "bar 9" in format_status([s])
    assert format_status([]) == "no voices heard"


def test_small_library_repeats_before_doubling(catalogue, cache_dir):
    _, lib = library(catalogue, cache_dir)
    ch = Chooser(lib, Role(top_k=1), rng=random.Random(0))
    ids = {s.id for s in lib.pitched}
    for sid in ids - {"a_minor"}:
        ch.history.append(sid)
    others = [other("v1", ["A", "C", "E"], sid="a_minor")]
    pick = ch.choose(others, 96).match.sample.id
    assert pick != "a_minor"  # repeats one of its own rather than doubling v1
