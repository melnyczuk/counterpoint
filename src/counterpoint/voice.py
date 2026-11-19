"""One voice of an ensemble: plays a loop, and every few loops picks a new one that fits
with what the other voices are playing.

Timeline for one change at beat ``B`` (always a bar line):

1. at ``B - prepare`` it waits its turn (voices changing on the same bar pick one at a
   time, in name order), picks, and announces the pick so later pickers fit with it;
2. renders the pick in memory (a worker thread) and schedules it for ``B``;
3. at ``B`` the new loop takes over with a short crossfade; the voice draws how many
   loops to play this time and announces its next change.

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
        self._controls_seen: set[str] = set()
        self._stop = asyncio.Event()
        self.changes = 0

    # -- messages ---------------------------------------------------------------------

    def state(self) -> VoiceState:
        return VoiceState(
            name=self.s.name, beat=round(self.clock.beat, 3), tempo=self.clock.tempo,
            now=self.now, next=self.next, change_beat=self.change_beat,
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
        elif c.set == "loops":
            self.s.loops = parse_loops(c.value)
        elif c.set == "stop":
            self._stop.set()
        log.info("%s: set %s = %s", self.s.name, c.set, c.value)

    async def _heartbeat(self) -> None:
        while not self._stop.is_set():
            self.announce()
            await asyncio.sleep(self.s.heartbeat_s)

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
        try:
            change = next_bar(self.clock.beat + self.s.prepare_beats)
            self.change_beat = change
            while not self._stop.is_set() and (changes is None or self.changes < changes):
                await wait_until(self.clock, change - self.s.prepare_beats)
                choice, loop = await self.prepare(change)
                frame = self.player.frame_in(seconds_until(self.clock, change))
                self.player.mixer.schedule(loop, frame)
                await wait_until(self.clock, change)
                n = self.rng.randint(*self.s.loops)
                self.now, self.next = self.next, None
                self.now.loop_beats = loop.beats
                nxt = change + math.ceil(n * loop.beats / BEATS_PER_BAR) * BEATS_PER_BAR
                self.change_beat = nxt
                self.announce()
                self._log(change, choice, loop, n)
                self.changes += 1
                change = nxt
        finally:
            self._stop.set()
            hb.cancel()

    def stop(self) -> None:
        self._stop.set()

    def _log(self, beat: float, choice: Choice, loop: Loop, loops: int) -> None:
        m = choice.match
        entry = {
            "t": round(time.time(), 3), "voice": self.s.name, "beat": beat,
            "tempo": loop.tempo, "id": m.sample.id, "file": m.sample.file,
            "catalogue": self.catalogue.path(m.sample), "shift": m.shift,
            "ratio": None if m.stretch is None else m.stretch.ratio,
            "loop_beats": loop.beats, "loops": loops, "reference": choice.reference,
            "anchor": self.chooser.role.anchor, "pan": self.s.pan,
            "score": round(m.score, 4),
        }
        log.info("%s: bar %d plays %s (shift %+d, %s loops) against %s", self.s.name,
                 int(beat // BEATS_PER_BAR) + 1, Path(m.sample.file).name, m.shift, loops,
                 choice.reference)
        if self.s.log_path:
            self.s.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.s.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
