"""Shared musical time for voices.

:class:`LinkClock` joins an Ableton Link session, so every voice (and Ableton, if it's
open) agrees on tempo and on where the bar lines are. :class:`FreeClock` keeps time on
its own, for tests and for running without a network.

Times are in beats; a bar is :data:`BEATS_PER_BAR` beats (4/4 throughout).
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

BEATS_PER_BAR = 4


class Clock(Protocol):
    @property
    def beat(self) -> float: ...

    @property
    def tempo(self) -> float: ...

    @tempo.setter
    def tempo(self, bpm: float) -> None: ...

    @property
    def peers(self) -> int: ...


def seconds_until(clock: Clock, beat: float) -> float:
    return (beat - clock.beat) * 60.0 / clock.tempo


def next_bar(beat: float) -> float:
    """The first bar line strictly after ``beat`` (with a little slack for jitter)."""
    return (int((beat + 1e-6) // BEATS_PER_BAR) + 1) * BEATS_PER_BAR


async def wait_until(clock: Clock, beat: float, early: float = 0.0) -> None:
    """Sleep until ``early`` seconds before ``beat``. Re-reads the clock as it goes, so a
    tempo change on the Link session while waiting is followed."""
    while True:
        left = seconds_until(clock, beat) - early
        if left <= 0.002:
            return
        await asyncio.sleep(min(left, 0.25) if left > 0.02 else left)


class FreeClock:
    """A local clock: beats from ``time.monotonic``. Changing the tempo keeps the current
    beat where it is."""

    def __init__(self, bpm: float = 96.0, now=time.monotonic):
        self._now = now
        self._bpm = float(bpm)
        self._origin = now()
        self._origin_beat = 0.0

    @property
    def beat(self) -> float:
        return self._origin_beat + (self._now() - self._origin) * self._bpm / 60.0

    @property
    def tempo(self) -> float:
        return self._bpm

    @tempo.setter
    def tempo(self, bpm: float) -> None:
        b = self.beat
        self._origin, self._origin_beat, self._bpm = self._now(), b, float(bpm)

    @property
    def peers(self) -> int:
        return 0


class LinkClock:
    """An Ableton Link session member (via aalink). Link itself is GPLv2+."""

    def __init__(self, bpm: float = 96.0):
        import aalink

        self._link = aalink.Link(float(bpm))
        self._link.quantum = BEATS_PER_BAR
        self._link.enabled = True

    @property
    def beat(self) -> float:
        return float(self._link.beat)

    @property
    def tempo(self) -> float:
        return float(self._link.tempo)

    @tempo.setter
    def tempo(self, bpm: float) -> None:
        self._link.tempo = float(bpm)

    @property
    def peers(self) -> int:
        return int(self._link.num_peers)

    def close(self) -> None:
        self._link.enabled = False
