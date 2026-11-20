import asyncio
import json
import math
import random

import numpy as np
import pytest

from counterpoint.bus import CONTROL, Control, MemoryBus, Playing, VoiceState
from counterpoint.catalogue import Catalogue
from counterpoint.chooser import Chooser, Library, Role
from counterpoint.clock import FreeClock
from counterpoint.ensemble_cli import format_status, voice_commands
from counterpoint.harmony import NOTES
from counterpoint.player import NullPlayer
from counterpoint.profile import ProfileCache, profiles_for
from counterpoint.render import Loop, loop_beats
from counterpoint.replay import Entry, mixdown, read_logs, swell_curve
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
    assert all(v.player.mixer.swell != 1.0 for v in voices)  # every voice is swelling
    logs = [json.loads(line) for i in range(4)
            for line in (tmp_path / f"v{i + 1}.jsonl").read_text().splitlines()]
    assert len(logs) == 16
    assert all(e["beat"] % 4 == 0 for e in logs)  # every change on a bar line
    assert {e["id"] for e in logs if e["voice"] == "v4"} <= {"rain", "hiss"}
    # Voices that changed on the same bar announced their picks in name order.
    for e in logs:
        assert e["reference"]
        assert e["swell_db"] == 6.0 and 0 <= e["swell_phase"] < 1
    # No two voices start a sample on the same bar.
    starts = [e["beat"] for e in logs]
    assert len(starts) == len(set(starts))
    # Each voice fades out by a bar line and rests at least a bar before the next.
    for i in range(4):
        mine = [e for e in logs if e["voice"] == f"v{i + 1}"]
        for e, nxt in zip(mine, mine[1:], strict=False):
            assert e["end_beat"] % 4 == 0 and e["end_beat"] > e["beat"]
            assert nxt["beat"] >= e["end_beat"] + 4
            assert e["fade_ms"] == pytest.approx(200, abs=0.2)


def test_a_lone_voice_does_not_rest(big_catalogue, cache_dir, tmp_path):
    cat, lib = library(big_catalogue, cache_dir)
    vs = VoiceSettings("v1", loops=(1, 2), prepare_beats=4, heartbeat_s=0.02,
                       turn_wait_s=0.05, log_path=tmp_path / "v1.jsonl", seed=0)
    v = Voice(vs, cat, Chooser(lib, rng=random.Random(0)), FreeClock(bpm=1200),
              MemoryBus(MemoryBus.Hub()), NullPlayer(8000), render=fake_render)
    asyncio.run(asyncio.wait_for(v.run(changes=3), 30))
    logs = [json.loads(line) for line in (tmp_path / "v1.jsonl").read_text().splitlines()]
    assert all(e["end_beat"] is None for e in logs)  # crossfades, no rest
    for e, nxt in zip(logs, logs[1:], strict=False):
        assert nxt["beat"] == e["beat"] + math.ceil(e["loops"] * e["loop_beats"] / 4) * 4


def claim(name, bar, locked=False):
    return VoiceState(name, 0, 96, change_beat=bar, change_locked=locked)


def test_two_voices_claiming_one_bar(big_catalogue, cache_dir):
    cat, lib = library(big_catalogue, cache_dir)

    def voice(name):
        v = Voice(VoiceSettings(name), cat, Chooser(lib), FreeClock(96),
                  MemoryBus(MemoryBus.Hub()), NullPlayer(8000))
        v.change_beat = 32.0
        return v

    v1, v2 = voice("v1"), voice("v2")
    v1.peers.update(claim("v2", 32.0))
    v2.peers.update(claim("v1", 32.0))
    v1._settle()
    v2._settle()
    assert v1.change_beat == 32.0  # the earlier name keeps it
    assert v2.change_beat == 36.0
    # A voice that has locked its bar keeps it, whatever its name.
    v1.peers.update(claim("v2", 32.0, locked=True))
    v1._settle()
    assert v1.change_beat == 36.0
    # Moving skips bars other voices have claimed, and so does a fresh claim.
    v3 = voice("v3")
    for name, bar in (("v1", 32.0), ("v2", 36.0), ("v4", 40.0)):
        v3.peers.update(claim(name, bar))
    v3._settle()
    assert v3.change_beat == 44.0
    assert v3._free_bar(33.0) == 44.0 and v3._free_bar(45.0) == 48.0


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
                  Control("volume", "0.1", voice="v9"), Control("swell", "0"),
                  Control("swell-bars", "6", voice="v1"), Control("swell-drift", "0.3")):
            remote.send(CONTROL, c.to_json())

    asyncio.run(go())
    assert v.chooser.role.anchor == "Dm9"
    assert clock.tempo == 120
    assert v.player.mixer.gain == 0.5  # the v9-only message didn't apply
    assert v.s.loops == (2, 3)
    assert v.swell.depth_db == 0 and v.swell.gain() == 1.0
    assert v.swell.base_beats == 24 and v.swell.drift == 0.3


def test_swell_redraws_at_loop_points(big_catalogue, cache_dir):
    cat, lib = library(big_catalogue, cache_dir)
    v = Voice(VoiceSettings("v1", swell_drift=0.3), cat, Chooser(lib), FreeClock(96),
              MemoryBus(MemoryBus.Hub()), NullPlayer(8000))
    v._start_loop(8.0, 4.0)
    first = v.swell.period
    phase = v.swell.phase
    v._swell_to(11.0)
    assert v.swell.period == first and v.swell.phase != phase
    periods = {first}
    for beat in (12.5, 16.5, 20.5, 24.5):
        v._swell_to(beat)
        periods.add(v.swell.period)
    assert len(periods) == 5  # a new period every loop
    assert v.player.mixer.swell == pytest.approx(v.swell.gain())


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
    entries[0].update(swell_db=6.0, swell_bars=1.0, swell_drift=0.2, swell_phase=0.25)
    log.write_text("\n".join(json.dumps(e) for e in entries))
    swelled = mixdown(read_logs([log]), sr=8000)
    assert not np.allclose(swelled, mix)
    # A voice that fades out by its end_beat rests in silence until its next entry.
    v1 = dict(entries[0], swell_db=0.0, pan=-1.0, end_beat=4, fade_ms=100)
    log.write_text("\n".join(json.dumps(e) for e in (v1, dict(v1, beat=12, end_beat=None))))
    left = mixdown(read_logs([log]), sr=8000)[0]  # 4000 frames a beat
    body = np.abs(left[4000:12000]).max()
    assert body > 0.01
    assert np.abs(left[:10]).max() < 0.05 * body  # fading in
    assert np.abs(left[16000 - 10:16000]).max() < 0.05 * body  # faded out by beat 4
    assert np.all(left[16000:48000] == 0)  # resting
    assert np.abs(left[48000:48010]).max() < 0.05 * body  # fading in again
    assert np.abs(left[52000:56000]).max() > 0.5 * body

def test_replay_swell_curve():
    e = Entry("v1", 0, 120, "x", 0, 1.0, 4, 2, 0.0, 6.0, 1.0, 0.2, 0.25)
    c = swell_curve(e, 16000, 2000.0)  # 8 beats at 2000 frames a beat
    db = 20 * np.log10(c)
    assert db[0] == pytest.approx(3.0, abs=0.01)  # phase 0.25: the top of the swell
    assert db.max() <= 3.0 + 1e-4 and db.min() >= -3.0 - 1e-4
    assert np.abs(np.diff(db)).max() < 0.05  # smooth across the loop point
    flat = Entry("v1", 0, 120, "x", 0, 1.0, 4, 2, 0.0)
    assert np.all(swell_curve(flat, 100, 2000.0) == 1)


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
    bars = [c[c.index("--swell-bars") + 1] for c in cmds]
    assert len(set(bars)) == 3  # each voice swells on its own period
    A.swell_bars = [3.0, 5.0]
    assert [c[c.index("--swell-bars") + 1] for c in voice_commands(A)] == ["3", "5", "3"]


def test_catalogue_flag_either_side_of_subcommand():
    from counterpoint.cli import build_parser

    parse = build_parser().parse_args
    args = parse(["ensemble", "--catalogue", "/data", "--voices", "2"])
    assert args.catalogue == "/data"
    for cmd in voice_commands(args):
        assert cmd[cmd.index("--catalogue") + 1] == "/data"
    assert parse(["--catalogue", "/data", "ensemble"]).catalogue == "/data"
    assert parse(["ensemble"]).catalogue is None
    assert parse(["ensemble", "--config", "c.toml"]).config == "c.toml"
    assert parse(["voice", "--catalogue", "/data", "--dry-run"]).catalogue == "/data"


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
