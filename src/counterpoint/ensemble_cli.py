"""Command line for live use: ``voice``, ``ensemble``, ``listen`` and ``replay``.

    counterpoint voice --loops 4-8 --anchor Am9          # one voice (one per Pi)
    counterpoint ensemble --voices 4 --anchor Am9        # four voices on this machine
    counterpoint ensemble status                         # who's playing what
    counterpoint ensemble set anchor Dm9                 # steer every voice
    counterpoint ensemble set volume 0.5 --voice v2
    counterpoint ensemble set swell-bars 6 --voice v3     # a slower swell for one voice
    counterpoint ensemble set rest 2                     # two silent bars between samples
    counterpoint listen --midi                           # a MIDI player as a voice
    counterpoint replay ~/.local/state/counterpoint/*.jsonl --out run.wav
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .bus import CONTROL, STATE, Control, MemoryBus, UdpBus, VoiceState
from .chooser import BANDS, Chooser, Library, Role
from .clock import BEATS_PER_BAR, FreeClock, LinkClock
from .render import render_from
from .scoring import Limits
from .voice import Voice, VoiceSettings, parse_loops


def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(base) / "counterpoint"


def _channels(text: str) -> tuple[int, ...]:
    ch = tuple(int(c) for c in text.split(","))
    if not 1 <= len(ch) <= 2 or min(ch) < 1:
        raise argparse.ArgumentTypeError("channels: one or two outputs, 1-based, e.g. 3,4")
    return ch


def _swell_bars(text: str) -> list[float]:
    try:
        bars = [float(b) for b in str(text).split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("swell bars: a number, or a list like 3,4,5") from None
    if not bars or min(bars) <= 0:
        raise argparse.ArgumentTypeError("swell bars must be positive")
    return bars


# How a single --swell-bars is spread across ensemble voices, so they start out of step.
SWELL_SPREAD = (1.0, 1.13, 0.89, 1.27, 0.79, 1.41, 0.71, 1.19)


def add_voice_args(p: argparse.ArgumentParser, single: bool = True) -> None:
    if single:
        p.add_argument("--name", default=socket.gethostname().split(".")[0],
                       help="voice name (default: hostname)")
        p.add_argument("--pan", type=float, default=0.0, help="-1 left .. 1 right")
        p.add_argument("--channels", type=_channels, default=(1, 2),
                       help="output channels, 1-based, e.g. 3,4")
        p.add_argument("--band", choices=list(BANDS), help="favour this band")
        p.add_argument("--texture", action="store_true", help="only unpitched samples")
        p.add_argument("--log", type=Path, help="play log (default: state dir/NAME.jsonl)")
        p.add_argument("--seed", type=int)
    p.add_argument("--loops", default="4-8", help="loops before changing: 6 or 4-8")
    p.add_argument("--anchor", help="chord pulling the harmony back, e.g. Am9")
    p.add_argument("--anchor-weight", type=float, default=0.5)
    p.add_argument("--bpm", type=float, help="set the session tempo")
    p.add_argument("--device", help="output device name or number")
    p.add_argument("--volume", type=float, default=0.8)
    p.add_argument("--swell", type=float, default=6.0,
                   help="how far each voice's level swells, dB peak to trough (0: off)")
    p.add_argument("--swell-bars", type=(float if single else _swell_bars),
                   default=(4.0 if single else [4.0]),
                   help="bars per swell" + ("" if single else
                        "; a list like 3,4,5,7 gives one per voice"))
    p.add_argument("--swell-drift", type=float, default=0.15,
                   help="how far the swell period wanders each loop (fraction)")
    p.add_argument("--rest-bars", type=int, default=1,
                   help="silent bars between one sample and the next (with other voices)")
    p.add_argument("--fade-ms", type=float, default=200.0,
                   help="how long each sample fades in and out")
    p.add_argument("--no-normalise", action="store_true",
                   help="play loops at their own level instead of matching loudness")
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--prepare-bars", type=int, default=2,
                   help="bars before a change to pick and render")
    p.add_argument("--max-shift", type=int)
    p.add_argument("--max-stretch", type=float)
    p.add_argument("--commercial", action="store_true", help="only allow-listed licences")
    p.add_argument("--peers", nargs="*", default=[], help="host[:port] where multicast fails")
    p.add_argument("--no-link", action="store_true", help="keep time locally, no Link")
    p.add_argument("--dry-run", action="store_true", help="no sound card: schedule silently")


def _library(args) -> tuple:
    from .cli import _open

    settings, cat, samples, profiles = _open(args)
    if args.commercial:
        samples = [s for s in samples if s.commercial]
    return settings, cat, Library(samples, profiles)


def _make_voice(args, bus=None, clock=None, player=None) -> Voice:
    settings, cat, library = _library(args)
    if args.texture and not library.unpitched:
        sys.exit("--texture needs unpitched samples, and the catalogue has none")
    if not args.texture and not library.pitched:
        sys.exit("the catalogue has no pitched samples")
    limits = settings.limits
    if args.max_shift is not None or args.max_stretch is not None:
        limits = Limits(
            limits.max_shift if args.max_shift is None else args.max_shift,
            limits.max_stretch if args.max_stretch is None else args.max_stretch,
        )
    role = Role(band=args.band, texture=args.texture, anchor=args.anchor,
                anchor_weight=args.anchor_weight)
    import random

    chooser = Chooser(library, role, settings.weights, limits, random.Random(args.seed))
    if clock is None:
        clock = FreeClock(args.bpm or 96) if args.no_link else LinkClock(args.bpm or 96)
    if bus is None:
        bus = UdpBus(peers=args.peers)
    if player is None:
        from .player import DevicePlayer, NullPlayer

        if args.dry_run:
            player = NullPlayer(args.sr, args.pan, args.volume, fade_ms=args.fade_ms)
        else:
            device = int(args.device) if args.device and args.device.isdigit() else args.device
            player = DevicePlayer(args.sr, device, args.channels, args.pan, args.volume,
                                  fade_ms=args.fade_ms)
    vs = VoiceSettings(
        name=args.name, loops=parse_loops(args.loops),
        prepare_beats=args.prepare_bars * BEATS_PER_BAR,
        log_path=args.log or state_dir() / f"{args.name}.jsonl", pan=args.pan, seed=args.seed,
        swell_db=args.swell, swell_bars=args.swell_bars, swell_drift=args.swell_drift,
        rest_bars=args.rest_bars, fade_ms=args.fade_ms,
    )
    render = render_from
    if args.no_normalise:
        render = functools.partial(render_from, target_lufs=None)
    return Voice(vs, cat, chooser, clock, bus, player, render=render)


async def _run_voice(args) -> None:
    # Link needs a running event loop, so the voice is built inside it.
    voice = _make_voice(args)
    print(f"{voice.s.name}: {len(voice.chooser.library.by_id)} samples, "
          f"tempo {voice.clock.tempo:g}, Link peers {voice.clock.peers}", file=sys.stderr)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, voice.stop)
        except NotImplementedError:
            pass
    try:
        if args.bpm:
            await asyncio.sleep(0.5)  # join the session first, then set its tempo
            voice.clock.tempo = args.bpm
        await voice.run()
    finally:
        voice.player.close()
        voice.bus.close()


def cmd_voice(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    asyncio.run(_run_voice(args))


# -- ensemble --------------------------------------------------------------------------


def _passthrough(args) -> list[str]:
    out = []
    for flag in ("loops", "anchor", "anchor_weight", "device", "volume", "sr", "prepare_bars",
                 "max_shift", "max_stretch", "swell", "swell_drift", "rest_bars", "fade_ms"):
        v = getattr(args, flag, None)
        if v is not None:
            out += [f"--{flag.replace('_', '-')}", str(v)]
    for flag in ("commercial", "no_link", "dry_run", "no_normalise"):
        if getattr(args, flag, False):
            out.append(f"--{flag.replace('_', '-')}")
    if args.peers:
        out += ["--peers", *args.peers]
    return out


def voice_commands(args) -> list[list[str]]:
    """One ``counterpoint voice`` command line per voice: names v1..vN, panned across
    the stereo field, or each on its own channel pair with ``--spread-channels``. Each
    gets its own swell period: from the ``--swell-bars`` list, or one value spread out."""
    n = args.voices
    bars = list(getattr(args, "swell_bars", None) or [4.0])
    if len(bars) == 1:
        bars = [bars[0] * SWELL_SPREAD[i % len(SWELL_SPREAD)] for i in range(n)]
    base = [sys.executable, "-m", "counterpoint"]
    if args.config:
        base += ["--config", args.config]
    if args.catalogue:
        base += ["--catalogue", args.catalogue]
    cmds = []
    for i in range(n):
        name = f"v{i + 1}"
        cmd = [*base, "voice", "--name", name, *_passthrough(args),
               "--swell-bars", f"{bars[i % len(bars)]:g}"]
        if args.spread_channels:
            cmd += ["--channels", f"{2 * i + 1},{2 * i + 2}"]
        else:
            pan = 0.0 if n == 1 else -0.8 + 1.6 * i / (n - 1)
            cmd += ["--pan", f"{pan:.2f}"]
        if args.textures and i >= n - args.textures:
            cmd.append("--texture")
        if args.bpm and i == 0:
            cmd += ["--bpm", str(args.bpm)]
        cmds.append(cmd)
    return cmds


def cmd_ensemble(args) -> None:
    if args.action == "status":
        return asyncio.run(_status(args.seconds, args.peers))
    if args.action == "set":
        if len(args.rest) != 2:
            sys.exit("usage: counterpoint ensemble set "
                     "anchor|tempo|volume|swell|swell-bars|swell-drift|rest|fade|loops|stop VALUE")
        return _set(args.rest[0], args.rest[1], args.voice, args.peers)
    procs = [subprocess.Popen(c) for c in voice_commands(args)]
    print(f"started {len(procs)} voices; Ctrl-C stops them", file=sys.stderr)
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    # The voices got the same Ctrl-C; give them a moment to stop cleanly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def _set(key: str, value: str, voice: str | None, peers: list[str]) -> None:
    if key not in ("anchor", "tempo", "volume", "swell", "swell-bars", "swell-drift", "rest",
                   "fade", "loops", "stop"):
        sys.exit(f"can't set {key!r}")
    bus = UdpBus(peers=peers)
    c = Control(key, value, voice)
    for _ in range(3):  # UDP can drop; voices ignore repeats
        bus.send(CONTROL, c.to_json())
        time.sleep(0.05)
    bus.close()
    print(f"sent {key} = {value} to {voice or 'every voice'}")


def format_status(states: list[VoiceState]) -> str:
    if not states:
        return "no voices heard"
    rows = [f"{'voice':<10} {'bar':>5} {'tempo':>6}  {'playing':<36} {'shift':>5}  next change"]
    for s in states:
        p = s.now
        bar = int(s.beat // BEATS_PER_BAR) + 1
        what = "(input)" if s.is_input else (p.id if p else "(resting)")
        shift = f"{p.shift:+d}" if p and p.shift else "0"
        nxt = "" if s.change_beat is None else f"bar {int(s.change_beat // BEATS_PER_BAR) + 1}"
        if s.next:
            nxt += f" -> {s.next.id} ({s.next.shift:+d})"
        rows.append(f"{s.name:<10} {bar:>5} {s.tempo:>6.1f}  {what[:36]:<36} {shift:>5}  {nxt}")
    return "\n".join(rows)


async def _status(seconds: float, peers: list[str]) -> None:
    from .bus import Peers

    seen = Peers("", timeout_s=seconds + 5)
    bus = UdpBus(peers=peers)

    def handle(address, payload, addr):
        if address == STATE:
            seen.update(VoiceState.from_json(payload))

    await bus.start(handle)
    await asyncio.sleep(seconds)
    bus.close()
    print(format_status(seen.live()))


# -- listen and replay -----------------------------------------------------------------


def cmd_listen(args) -> None:
    asyncio.run(_listen(args))


async def _listen(args) -> None:
    from .listen import AudioHearing, MidiHearing, publish

    clock = FreeClock(96) if args.no_link else LinkClock(96)
    bus = UdpBus(peers=args.peers)
    stream = port = None
    if args.midi is not None:
        import mido

        hearing = MidiHearing(args.memory)
        names = mido.get_input_names()
        name = args.midi or (names[0] if names else None)
        if name is None:
            sys.exit("no MIDI inputs found")
        port = mido.open_input(name, callback=hearing.message)
        print(f"listening to MIDI {name}", file=sys.stderr)
    else:
        import sounddevice as sd

        hearing = AudioHearing(window_s=args.memory)
        dev = args.audio_in
        dev = int(dev) if dev and dev.isdigit() else dev
        stream = sd.InputStream(samplerate=hearing.sr, channels=1, device=dev,
                                callback=lambda d, f, t, s: hearing.audio(d.copy()))
        stream.start()
        print("listening to audio input", file=sys.stderr)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    try:
        await publish(args.name, hearing, clock, bus, stop=stop)
    finally:
        if port:
            port.close()
        if stream:
            stream.stop()
        bus.close()


def cmd_replay(args) -> None:
    import soundfile as sf

    from .replay import mixdown, read_logs

    entries = read_logs(args.logs)
    if not entries:
        sys.exit("no log entries")
    mix = mixdown(entries, args.sr)
    sf.write(args.out, mix.T, args.sr)
    voices = len({e.voice for e in entries})
    print(f"wrote {mix.shape[1] / args.sr:.1f} s from {len(entries)} picks by {voices} voices "
          f"to {args.out}")


def add_parsers(sub) -> None:
    v = sub.add_parser("voice", help="run one ensemble voice")
    add_voice_args(v)
    v.set_defaults(func=cmd_voice)

    e = sub.add_parser("ensemble", help="start voices here, or check or steer an ensemble")
    e.add_argument("action", nargs="?", default="start", choices=["start", "status", "set"])
    e.add_argument("rest", nargs="*", help="for set: KEY VALUE")
    e.add_argument("--voices", type=int, default=4)
    e.add_argument("--textures", type=int, default=0, help="how many voices play textures")
    e.add_argument("--spread-channels", action="store_true",
                   help="voice N on outputs 2N-1,2N of the interface instead of panning")
    e.add_argument("--voice", help="for set: only this voice")
    e.add_argument("--seconds", type=float, default=3.0, help="for status: listen this long")
    add_voice_args(e, single=False)
    e.set_defaults(func=cmd_ensemble)

    li = sub.add_parser("listen", help="publish a live MIDI or audio input as a voice")
    li.add_argument("--name", default="input")
    li.add_argument("--midi", nargs="?", const="", help="MIDI input port (default: first)")
    li.add_argument("--audio-in", help="audio input device (default: system input)")
    li.add_argument("--memory", type=float, default=4.0, help="seconds a note is remembered")
    li.add_argument("--peers", nargs="*", default=[])
    li.add_argument("--no-link", action="store_true")
    li.set_defaults(func=cmd_listen)

    r = sub.add_parser("replay", help="mix an ensemble run back down from its logs")
    r.add_argument("logs", nargs="+", type=Path)
    r.add_argument("--out", default="replay.wav")
    r.add_argument("--sr", type=int, default=48000)
    r.set_defaults(func=cmd_replay)


__all__ = ["add_parsers", "MemoryBus"]
