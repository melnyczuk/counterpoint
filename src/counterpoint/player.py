"""Looping a buffer through an audio output and switching to the next one on a bar line.

The mixing is a pure function (:meth:`Mixer.block`) so it can be tested without a sound
card; :class:`DevicePlayer` drives it from a sounddevice (PortAudio) callback.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from .render import Loop

FADE_MS = 10.0


def pan_gains(pan: float) -> tuple[float, float]:
    """Equal-power pan, -1 (left) to 1 (right)."""
    a = (max(-1.0, min(1.0, pan)) + 1) * math.pi / 4
    return math.cos(a), math.sin(a)


@dataclass
class _Voice:
    loop: Loop
    pos: int = 0


class Mixer:
    """Plays one loop; a scheduled loop takes over at an exact frame with a short
    crossfade. Thread-safe between the audio callback and the scheduler."""

    def __init__(self, sr: int, pan: float = 0.0, gain: float = 1.0):
        self.sr = sr
        self.pan = pan
        self.gain = gain
        self._cur: _Voice | None = None
        self._out: _Voice | None = None  # fading out
        self._fade_left = 0
        self._pending: tuple[Loop | None, int, int] | None = None  # loop, start, offset
        self._lock = threading.Lock()
        self.frame = 0  # frames output so far
        self.fade_frames = max(1, int(sr * FADE_MS / 1000))
        self.started: list[tuple[int, Loop]] = []

    def schedule(self, loop: Loop, at_frame: int) -> None:
        """Start ``loop`` at ``at_frame``. If that's already past (a late render), start
        now, part-way in, so the loop stays in phase with the bar lines."""
        with self._lock:
            late = max(0, self.frame - at_frame)
            self._pending = (loop, max(at_frame, self.frame), late)

    def stop(self, at_frame: int | None = None) -> None:
        with self._lock:
            self._pending = (None, self.frame if at_frame is None else at_frame, 0)

    def _read(self, v: _Voice, n: int) -> np.ndarray:
        L = v.loop.frames
        idx = (v.pos + np.arange(n)) % L
        v.pos = (v.pos + n) % L
        return v.loop.audio[:, idx]

    def block(self, n: int) -> np.ndarray:
        """The next ``n`` frames, stereo (2, n)."""
        out = np.zeros((2, n), dtype=np.float32)
        with self._lock:
            pending = self._pending
            start = 0
            if pending is not None and pending[1] < self.frame + n:
                split = pending[1] - self.frame
                self._render(out, 0, split)
                self._out, self._fade_left = self._cur, self.fade_frames
                loop, _, offset = pending
                self._cur = None if loop is None else _Voice(loop, offset % loop.frames)
                if pending[0] is not None:
                    self.started.append((pending[1], pending[0]))
                self._pending = None
                start = split
            self._render(out, start, n)
            self.frame += n
        lg, rg = pan_gains(self.pan)
        out[0] *= lg * self.gain
        out[1] *= rg * self.gain
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
                 pan: float = 0.0, gain: float = 1.0, blocksize: int = 512):
        import sounddevice as sd

        self.mixer = Mixer(sr, pan, gain)
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

    def __init__(self, sr: int = 48000, pan: float = 0.0, gain: float = 1.0):
        import time

        self.mixer = Mixer(sr, pan, gain)
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
