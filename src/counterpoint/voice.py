"""One voice of an ensemble: plays a loop, and every few loops picks a new one that fits
with what the other voices are playing.

Timeline for one sample starting at beat ``B`` (always a bar line):

1. until ``B - prepare`` the start bar may still move: no two voices start on the same
   bar, so if another voice claims it first (or at the same moment, with an earlier
   name), this voice moves to the next free bar. Then it locks the bar and announces;
2. it picks (waiting its turn if another voice somehow starts on the same bar) and
   announces the pick so later pickers fit with it;
3. renders the pick in memory (a worker thread) and schedules it for ``B``;
4. at ``B`` the new loop fades in. The voice draws how many loops to play, schedules a
   fade out that ends on the bar line ``E`` after them, and claims the first free bar at
   least ``rest_bars`` after ``E`` for its next sample. Between them it is silent.

A voice alone in the ensemble doesn't rest: its next sample crossfades in at ``E``.

All the while, the voice's level swells slowly on its own period (see
:class:`~counterpoint.player.Swell`), redrawn a little differently every loop.

The voice never writes audio to disk and never touches the catalogue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from .bus import CONTROL, STATE, Control, Peers, Playing, VoiceState
from .catalogue import Catalogue
from .chooser import Choice, Chooser
from .clock import BEATS_PER_BAR, Clock, next_bar, seconds_until, wait_until
from .player import Swell
from .render import Loop, loop_beats, render_from

log = logging.getLogger("counterpoint.voice")


def parse_loops(text: str) -> tuple[int, int]:
    """``6`` or ``4-8``."""
    lo, _, hi = str(text).partition("-")
    a, b = int(lo), int(hi or lo)
    if a < 1 or b < a:
        raise ValueError(f"bad loop count {text!r}")
    return a, b


@dataclass
class VoiceSettings:
    name: str
    loops: tuple[int, int] = (4, 8)
    prepare_beats: float = 2 * BEATS_PER_BAR  # how long before a change to pick and render
    heartbeat_s: float = 1.0
    turn_wait_s: float = 1.0  # how long to wait for an earlier-named voice to pick
    log_path: Path | None = None
    pan: float = 0.0
    seed: int | None = None
    swell_db: float = 6.0  # peak to trough; 0 holds the level steady
    swell_bars: float = 4.0  # this voice's swell period, before drift
    swell_drift: float = 0.15  # how far the period wanders each loop, as a fraction
    swell_tick_s: float = 0.05
    rest_bars: int = 1  # silent bars between samples when other voices are playing
    fade_ms: float = 200.0  # every sample fades in and out over this
    extra: dict = field(default_factory=dict)


class Voice:
    def __init__(self, settings: VoiceSettings, catalogue: Catalogue, chooser: Chooser,
                 clock: Clock, bus, player, render=render_from):
        self.s = settings
        self.catalogue = catalogue
        self.chooser = chooser
        self.clock = clock
        self.bus = bus
        self.player = player
        self._render = render
        self.rng = random.Random(settings.seed)
        timeout = max(5.0, 4 * BEATS_PER_BAR * 60.0 / clock.tempo)
        self.peers = Peers(settings.name, timeout)
        self.now: Playing | None = None
        self.next: Playing | None = None
        self.change_beat: float | None = None
        self.change_locked = False
        self.end_beat: float | None = None
        self._controls_seen: set[str] = set()
        self._stop = asyncio.Event()
        self.changes = 0
        self.swell = Swell(settings.swell_db, settings.swell_bars * BEATS_PER_BAR,
                           settings.swell_drift, seed=settings.name)
        self._swell_beat: float | None = None  # where the swell has been advanced to
        self._next_wrap: float | None = None  # the current loop's next loop point
        self._loop_beats: float | None = None

    # -- messages ---------------------------------------------------------------------

    def state(self) -> VoiceState:
        return VoiceState(
            name=self.s.name, beat=round(self.clock.beat, 3), tempo=self.clock.tempo,
            now=self.now, next=self.next, change_beat=self.change_beat,
            change_locked=self.change_locked, end_beat=self.end_beat,
            texture=self.chooser.role.texture, band=self.chooser.role.band,
        )

    def announce(self) -> None:
        self.bus.send(STATE, self.state().to_json())

    def _on_message(self, address: str, payload: str, addr) -> None:
        try:
            if address == STATE:
                self.peers.update(VoiceState.from_json(payload))
            elif address == CONTROL:
                self._control(Control.from_json(payload))
        except Exception as e:  # a malformed message never stops a voice
            log.warning("ignored message: %s", e)

    def _control(self, c: Control) -> None:
        if c.id in self._controls_seen or (c.voice and c.voice != self.s.name):
            return
        self._controls_seen.add(c.id)
        if c.set == "anchor":
            self.chooser.role.anchor = None if c.value.lower() in ("", "none") else c.value
        elif c.set == "tempo":
            self.clock.tempo = float(c.value)
        elif c.set == "volume":
            self.player.mixer.gain = max(0.0, float(c.value))
        elif c.set == "swell":
            self.swell.depth_db = self.s.swell_db = max(0.0, float(c.value))
        elif c.set == "swell-bars":
            self.s.swell_bars = max(0.25, float(c.value))
            self.swell.base_beats = self.s.swell_bars * BEATS_PER_BAR
        elif c.set == "swell-drift":
            self.swell.drift = self.s.swell_drift = max(0.0, min(0.9, float(c.value)))
        elif c.set == "rest":
            self.s.rest_bars = max(0, int(c.value))
        elif c.set == "fade":
            self.player.mixer.fade_ms = self.s.fade_ms = max(0.0, float(c.value))
        elif c.set == "loops":
            self.s.loops = parse_loops(c.value)
        elif c.set == "stop":
            self._stop.set()
        log.info("%s: set %s = %s", self.s.name, c.set, c.value)

    async def _heartbeat(self) -> None:
        while not self._stop.is_set():
            self.announce()
            await asyncio.sleep(self.s.heartbeat_s)

    # -- swelling ---------------------------------------------------------------------

    def _swell_to(self, beat: float) -> None:
        """Advance the swell to ``beat``, redrawing its period at every loop point passed."""
        if self._swell_beat is None:
            self._swell_beat = beat
            return
        while self._next_wrap is not None and self._next_wrap <= beat:
            wrap = self._next_wrap
            if wrap > self._swell_beat:
                self.swell.advance(wrap - self._swell_beat)
                self._swell_beat = wrap
            self.swell.new_loop(f"{self.s.name}:{wrap:g}")
            self._next_wrap = wrap + self._loop_beats
        if beat > self._swell_beat:
            self.swell.advance(beat - self._swell_beat)
            self._swell_beat = beat
        self.player.mixer.swell = self.swell.gain()

    def _start_loop(self, beat: float, loop_beats: float) -> None:
        """A new loop starts at ``beat``: its loop points drive the swell from here."""
        self._next_wrap = None
        self._swell_to(beat)
        self._loop_beats = max(loop_beats, 1.0)
        self._next_wrap = beat
        self._swell_to(beat)

    async def _swell_loop(self) -> None:
        while not self._stop.is_set():
            self._swell_to(self.clock.beat)
            await asyncio.sleep(self.s.swell_tick_s)

    # -- claiming start bars ------------------------------------------------------------

    def _others(self):
        return [p for p in self.peers.live() if not p.is_input]

    def _claimed(self) -> list[float]:
        return [p.change_beat for p in self._others() if p.change_beat is not None]

    def _free_bar(self, earliest: float) -> float:
        """The first bar line at or after ``earliest`` no other voice starts on."""
        bar = math.ceil(earliest / BEATS_PER_BAR - 1e-6) * BEATS_PER_BAR
        claimed = self._claimed()
        while any(abs(c - bar) < 0.01 for c in claimed):
            bar += BEATS_PER_BAR
        return bar

    def _settle(self) -> None:
        """If another voice starts on our bar too, the one that has locked it keeps it;
        if neither has, the earlier name does. The other moves on to a free bar."""
        if self.change_locked or self.change_beat is None:
            return
        change = self.change_beat
        rivals = [p for p in self._others() if p.change_beat is not None
                  and abs(p.change_beat - change) < 0.01
                  and (p.change_locked or p.name < self.s.name)]
        if rivals:
            self.change_beat = self._free_bar(change + BEATS_PER_BAR)
            log.debug("%s: bar %d is taken by %s; moving to bar %d", self.s.name,
                      int(change // BEATS_PER_BAR) + 1, rivals[0].name,
                      int(self.change_beat // BEATS_PER_BAR) + 1)
            self.announce()

    async def _settle_until_locked(self) -> float:
        """Keep the start bar clear of other voices' until it's time to prepare, then
        lock it."""
        while True:
            self._settle()
            left = seconds_until(self.clock, self.change_beat - self.s.prepare_beats)
            if left <= 0.002:
                break
            await asyncio.sleep(min(left, self.s.heartbeat_s, 0.05))
        self._settle()
        self.change_locked = True
        self.announce()
        return self.change_beat

    async def _fall_silent(self, end: float) -> None:
        await wait_until(self.clock, end)
        if self.end_beat == end:
            self.now = None
            self.announce()

    # -- choosing ---------------------------------------------------------------------

    async def _wait_turn(self, change: float) -> None:
        """Let voices with earlier names that change on the same bar pick first."""
        deadline = time.monotonic() + self.s.turn_wait_s
        while time.monotonic() < deadline:
            waiting = [
                p for p in self.peers.live()
                if p.name < self.s.name and p.change_beat is not None
                and abs(p.change_beat - change) < 0.01
                and not (p.next and abs(p.next.start_beat - change) < 0.01)
            ]
            if not waiting:
                return
            await asyncio.sleep(0.01)

    async def prepare(self, change: float) -> tuple[Choice, Loop]:
        await self._wait_turn(change)
        tempo = self.clock.tempo
        choice = self.chooser.choose(self.peers.live(), tempo)
        m = choice.match
        beats = loop_beats(m.sample, m, tempo, m.sample.duration_s or 0.0)
        self.next = choice.playing(change, beats)
        self.announce()
        loop = await asyncio.get_running_loop().run_in_executor(
            None, self._render, self.catalogue, m, tempo, self.player.mixer.sr
        )
        return choice, loop

    # -- running ----------------------------------------------------------------------

    async def run(self, changes: int | None = None) -> None:
        """Play until stopped (or for ``changes`` changes, for tests)."""
        await self.bus.start(self._on_message)
        hb = asyncio.create_task(self._heartbeat())
        sw = asyncio.create_task(self._swell_loop())
        silent: asyncio.Task | None = None
        mixer = self.player.mixer
        mixer.fade_ms = self.s.fade_ms
        try:
            self.change_beat = self._free_bar(next_bar(self.clock.beat + self.s.prepare_beats))
            self.announce()
            while not self._stop.is_set() and (changes is None or self.changes < changes):
                change = await self._settle_until_locked()
                choice, loop = await self.prepare(change)
                frame = self.player.frame_in(seconds_until(self.clock, change))
                mixer.schedule(loop, frame)
                await wait_until(self.clock, change)
                self._start_loop(change, loop.beats)
                n = self.rng.randint(*self.s.loops)
                self.now, self.next = self.next, None
                self.now.loop_beats = loop.beats
                end = change + math.ceil(n * loop.beats / BEATS_PER_BAR) * BEATS_PER_BAR
                rests = bool(self._others()) and self.s.rest_bars > 0
                if rests:
                    # Fade out so it's silent by the bar line, then rest.
                    fade_s = mixer.fade_frames / mixer.sr
                    mixer.stop(self.player.frame_in(seconds_until(self.clock, end) - fade_s))
                    nxt = self._free_bar(end + self.s.rest_bars * BEATS_PER_BAR)
                else:
                    nxt = self._free_bar(end)  # alone: crossfade straight into the next
                self.end_beat, self.change_beat, self.change_locked = end, nxt, False
                self.announce()
                if silent:
                    silent.cancel()
                if rests:
                    silent = asyncio.create_task(self._fall_silent(end))
                self._log(change, choice, loop, n, end if rests else None)
                self.changes += 1
        finally:
            self._stop.set()
            hb.cancel()
            sw.cancel()
            if silent:
                silent.cancel()

    def stop(self) -> None:
        self._stop.set()

    def _log(self, beat: float, choice: Choice, loop: Loop, loops: int,
             end: float | None) -> None:
        m = choice.match
        entry = {
            "t": round(time.time(), 3), "voice": self.s.name, "beat": beat,
            "tempo": loop.tempo, "id": m.sample.id, "file": m.sample.file,
            "catalogue": self.catalogue.path(m.sample), "shift": m.shift,
            "ratio": None if m.stretch is None else m.stretch.ratio,
            "loop_beats": loop.beats, "loops": loops, "end_beat": end,
            "fade_ms": round(self.player.mixer.fade_ms, 1), "reference": choice.reference,
            "anchor": self.chooser.role.anchor, "pan": self.s.pan,
            "score": round(m.score, 4), "gain_db": round(loop.gain_db, 2),
            "swell_db": self.swell.depth_db, "swell_bars": self.s.swell_bars,
            "swell_drift": self.swell.drift, "swell_phase": round(self.swell.phase, 6),
        }
        log.info("%s: bar %d plays %s (shift %+d, %s loops) against %s", self.s.name,
                 int(beat // BEATS_PER_BAR) + 1, Path(m.sample.file).name, m.shift, loops,
                 choice.reference)
        if self.s.log_path:
            self.s.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.s.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
