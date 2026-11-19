"""A live input as a voice: a player on MIDI or a microphone/line in.

It never plays anything. It publishes what it hears as a voice that never changes
sample, flagged ``is_input``, and the other voices fit their picks to it (weighted more
than a sample voice; see ``Role.input_weight``).
"""

from __future__ import annotations

import asyncio
import collections
import threading
import time

import numpy as np

from .bus import STATE, Playing, VoiceState
from .clock import Clock


class MidiHearing:
    """Pitch classes of the notes held or recently released, fading over ``memory_s``."""

    def __init__(self, memory_s: float = 4.0, now=time.monotonic):
        self.memory_s = memory_s
        self._now = now
        self._held: dict[int, tuple[float, float]] = {}  # note -> (start, velocity)
        self._released: collections.deque = collections.deque(maxlen=256)
        self._lock = threading.Lock()

    def message(self, msg) -> None:
        with self._lock:
            if msg.type == "note_on" and msg.velocity > 0:
                self._held[msg.note] = (self._now(), msg.velocity / 127)
            elif msg.type in ("note_off", "note_on"):
                start = self._held.pop(msg.note, None)
                if start:
                    self._released.append((msg.note, self._now(), start[1]))

    def profile(self) -> np.ndarray:
        t = self._now()
        w = np.zeros(12)
        with self._lock:
            for note, (_, vel) in self._held.items():
                w[note % 12] += vel
            for note, end, vel in self._released:
                age = t - end
                if age < self.memory_s:
                    w[note % 12] += vel * (1 - age / self.memory_s)
        return w


class AudioHearing:
    """Chroma of the last ``window_s`` of an input device."""

    def __init__(self, sr: int = 22050, window_s: float = 4.0):
        self.sr = sr
        self._buf = np.zeros(int(sr * window_s), dtype=np.float32)
        self._lock = threading.Lock()

    def audio(self, block: np.ndarray) -> None:
        mono = block.mean(axis=1) if block.ndim > 1 else block
        with self._lock:
            n = len(mono)
            self._buf = np.concatenate([self._buf[n:], mono.astype(np.float32)])

    def profile(self) -> np.ndarray:
        import librosa

        with self._lock:
            y = self._buf.copy()
        if np.abs(y).max() < 1e-3:
            return np.zeros(12)
        c = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=2048).mean(axis=1)
        c = c - c.min()
        contrast = (c.max() / (c.max() + c.min() + 1e-9)) if c.max() > 0 else 0
        return c if contrast > 0 else np.zeros(12)


def input_state(name: str, clock: Clock, w: np.ndarray) -> VoiceState:
    total = w.sum()
    now = None
    if total > 0:
        now = Playing(id=name, shift=0, ratio=1.0, profile=[float(v) for v in w / total],
                      bands=[1 / 3] * 3, source=name, start_beat=clock.beat, loop_beats=0)
    return VoiceState(name=name, beat=round(clock.beat, 3), tempo=clock.tempo, now=now,
                      is_input=True)


async def publish(name: str, hearing, clock: Clock, bus, every_s: float = 0.5,
                  stop: asyncio.Event | None = None) -> None:
    await bus.start(lambda *a: None)
    stop = stop or asyncio.Event()
    while not stop.is_set():
        bus.send(STATE, input_state(name, clock, hearing.profile()).to_json())
        await asyncio.sleep(every_s)
