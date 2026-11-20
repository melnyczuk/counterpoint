"""Looping a buffer through an audio output and switching to the next one on a bar line.

The mixing is a pure function (:meth:`Mixer.block`) so it can be tested without a sound
card; :class:`DevicePlayer` drives it from a sounddevice (PortAudio) callback.

:class:`Swell` is the slow rise and fall of a voice's level. Each voice swells on its own
period, which drifts a little every loop, so the voices phase against each other and
each one comes to the fore now and then.
"""

from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass

import numpy as np

from .render import Loop

FADE_MS = 200.0  # every sample fades in and out over this


def pan_gains(pan: float) -> tuple[float, float]:
    """Equal-power pan, -1 (left) to 1 (right)."""
    a = (max(-1.0, min(1.0, pan)) + 1) * math.pi / 4
    return math.cos(a), math.sin(a)


class Swell:
    """A slow sine on a voice's level, ``depth_db`` peak to trough, about ``base_beats``
    long. Every loop, :meth:`new_loop` redraws the period within ``drift`` (a fraction)
    of the base. The phase is accumulated rather than worked out from the beat, so a new
    period never makes the level jump. Draws are seeded, so a replay can repeat them."""

    def __init__(self, depth_db: float = 6.0, base_beats: float = 16.0, drift: float = 0.15,
                 seed: str = ""):
        self.depth_db = depth_db
        self.base_beats = base_beats
        self.drift = drift
        self.period = base_beats
        self.phase = random.Random(f"{seed}:phase").random()

    def new_loop(self, key: str) -> None:
        u = random.Random(key).uniform(-1.0, 1.0)
        self.period = max(1.0, self.base_beats * (1 + self.drift * u))

    def advance(self, beats: float) -> None:
        self.phase = (self.phase + max(0.0, beats) / self.period) % 1.0

    def db(self) -> float:
        return self.depth_db / 2 * math.sin(2 * math.pi * self.phase)

    def gain(self) -> float:
        return 10 ** (self.db() / 20)


@dataclass
class _Voice:
    loop: Loop
    pos: int = 0


class Mixer:
    """Plays one loop at a time. Starts and stops are queued at exact frames; each one
    fades over ``fade_ms``: a start fades in (from silence, or crossfading from the loop
    before), a stop fades out to silence. Thread-safe between the audio callback and the
    scheduler."""

    def __init__(self, sr: int, pan: float = 0.0, gain: float = 1.0,
                 fade_ms: float = FADE_MS):
        self.sr = sr
        self.pan = pan
        self.gain = gain
        self.swell = 1.0  # set from outside; ramped to across each block
        self._swell_now = 1.0
        self._cur: _Voice | None = None
        self._out: _Voice | None = None  # fading out
        self._fade_left = 0
        self._events: list[tuple[int, Loop | None, int]] = []  # frame, loop, offset
        self._lock = threading.Lock()
        self.frame = 0  # frames output so far
        self.fade_ms = fade_ms
        self.started: list[tuple[int, Loop]] = []

    @property
    def fade_ms(self) -> float:
        return self.fade_frames * 1000 / self.sr

    @fade_ms.setter
    def fade_ms(self, ms: float) -> None:
        self.fade_frames = max(1, int(self.sr * max(0.0, ms) / 1000))

    def _queue(self, at_frame: int, loop: Loop | None, offset: int) -> None:
        self._events.append((at_frame, loop, offset))
        self._events.sort(key=lambda e: e[0])

    def schedule(self, loop: Loop, at_frame: int) -> None:
        """Start ``loop`` at ``at_frame``. If that's already past (a late render), start
        now, part-way in, so the loop stays in phase with the bar lines."""
        with self._lock:
            late = max(0, self.frame - at_frame)
            self._queue(max(at_frame, self.frame), loop, late)

    def stop(self, at_frame: int | None = None) -> None:
        """Fade out to silence starting at ``at_frame`` (now if None or already past)."""
        with self._lock:
            self._queue(self.frame if at_frame is None else max(at_frame, self.frame),
                        None, 0)

    def _read(self, v: _Voice, n: int) -> np.ndarray:
        L = v.loop.frames
        idx = (v.pos + np.arange(n)) % L
        v.pos = (v.pos + n) % L
        return v.loop.audio[:, idx]

    def block(self, n: int) -> np.ndarray:
        """The next ``n`` frames, stereo (2, n)."""
        out = np.zeros((2, n), dtype=np.float32)
        with self._lock:
            start = 0
            while self._events and self._events[0][0] < self.frame + n:
                at, loop, offset = self._events.pop(0)
                split = max(start, at - self.frame)
                self._render(out, start, split)
                if loop is None and self._cur is None:
                    start = split
                    continue  # already silent
                self._out, self._fade_left = self._cur, self.fade_frames
                self._cur = None if loop is None else _Voice(loop, offset % loop.frames)
                if loop is not None:
                    self.started.append((at, loop))
                start = split
            self._render(out, start, n)
            self.frame += n
        lg, rg = pan_gains(self.pan)
        target = self.swell
        env = self._swell_now + (target - self._swell_now) * np.arange(1, n + 1) / n
        self._swell_now = target
        out[0] *= lg * self.gain * env
        out[1] *= rg * self.gain * env
        return out

    def _render(self, out: np.ndarray, a: int, b: int) -> None:
        n = b - a
        if n <= 0:
            return
        k = min(n, self._fade_left)
        up = (self.fade_frames - self._fade_left + np.arange(k)) / self.fade_frames
        if self._cur is not None:
            seg = self._read(self._cur, n)
            if k:
                seg = seg.copy()
                seg[:, :k] *= up
            out[:, a:b] += seg
        if self._out is not None and k:
            out[:, a : a + k] += self._read(self._out, k) * (1 - up)
        self._fade_left -= k
        if self._fade_left <= 0:
            self._out = None


class DevicePlayer:
    """A Mixer on a real output. ``channels`` picks which interface outputs (1-based) the
    left and right go to, e.g. (3, 4); a single channel gets a mono sum."""

    def __init__(self, sr: int = 48000, device=None, channels: tuple[int, ...] = (1, 2),
                 pan: float = 0.0, gain: float = 1.0, blocksize: int = 512,
                 fade_ms: float = FADE_MS):
        import sounddevice as sd

        self.mixer = Mixer(sr, pan, gain, fade_ms)
        self.channels = channels
        self._stream = sd.OutputStream(
            samplerate=sr, device=device, channels=max(channels), dtype="float32",
            blocksize=blocksize, callback=self._callback, latency="high",
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status):
        st = self.mixer.block(frames)
        outdata[:] = 0
        if len(self.channels) == 1:
            outdata[:, self.channels[0] - 1] = st.mean(axis=0)
        else:
            outdata[:, self.channels[0] - 1] = st[0]
            outdata[:, self.channels[1] - 1] = st[1]

    @property
    def latency(self) -> float:
        return float(self._stream.latency)

    def frame_in(self, seconds: float) -> int:
        """The mixer frame that will be heard ``seconds`` from now."""
        return self.mixer.frame + int(round((seconds - self.latency) * self.mixer.sr))

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class NullPlayer:
    """No sound card: the mixer runs on a timer so scheduling behaves the same. Used for
    tests, dry runs and headless checks."""

    def __init__(self, sr: int = 48000, pan: float = 0.0, gain: float = 1.0,
                 fade_ms: float = FADE_MS):
        import time

        self.mixer = Mixer(sr, pan, gain, fade_ms)
        self._t0 = time.monotonic()
        self._time = time.monotonic

    @property
    def latency(self) -> float:
        return 0.0

    def _advance(self) -> None:
        target = int((self._time() - self._t0) * self.mixer.sr)
        while self.mixer.frame < target:
            self.mixer.block(min(4096, target - self.mixer.frame))

    def frame_in(self, seconds: float) -> int:
        self._advance()
        return self.mixer.frame + int(round(seconds * self.mixer.sr))

    def close(self) -> None:
        pass
