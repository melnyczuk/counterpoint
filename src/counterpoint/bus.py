"""How voices hear each other: small OSC messages over UDP multicast.

Each voice multicasts its state whenever it changes and once a second as a heartbeat:

    /counterpoint/state    one JSON string argument (see :class:`VoiceState`)
    /counterpoint/control  one JSON string argument: {"id", "voice", "set", "value"}

Multicast reaches every voice on the local network with no configuration, including
other processes on the same machine. Where a network blocks multicast (some Wi-Fi), list
the other voices' addresses as ``peers`` and every message is also sent to each one
directly. Messages are idempotent, so receiving one twice is harmless.
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder

GROUP = "239.255.43.21"
PORT = 47300
STATE = "/counterpoint/state"
CONTROL = "/counterpoint/control"


@dataclass
class Playing:
    """A sample as a voice plays it: what it is and how it's shifted and stretched."""

    id: str
    shift: int
    ratio: float
    profile: list[float]  # 12-bin chroma after the shift
    bands: list[float]  # low, mid, high
    source: str
    start_beat: float
    loop_beats: float


@dataclass
class VoiceState:
    name: str
    beat: float  # the sender's Link beat when it sent this
    tempo: float
    now: Playing | None = None
    next: Playing | None = None  # the planned pick, once chosen
    change_beat: float | None = None
    is_input: bool = False
    texture: bool = False
    band: str | None = None
    sent_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> VoiceState:
        d = json.loads(text)
        for k in ("now", "next"):
            if d.get(k):
                d[k] = Playing(**d[k])
        return cls(**d)

    def heard(self) -> Playing | None:
        """What this voice will be playing next, if planned, else what it plays now."""
        return self.next or self.now


@dataclass
class Control:
    set: str  # anchor | tempo | volume | swell | swell-bars | swell-drift | loops | stop
    value: str
    voice: str | None = None  # None means every voice
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> Control:
        return cls(**json.loads(text))


def encode(address: str, payload: str) -> bytes:
    b = OscMessageBuilder(address)
    b.add_arg(payload)
    return b.build().dgram


def decode(data: bytes) -> tuple[str, str] | None:
    try:
        msg = OscMessage(data)
    except Exception:
        return None
    if msg.address not in (STATE, CONTROL) or not msg.params:
        return None
    return msg.address, str(msg.params[0])


Handler = Callable[[str, str, tuple], None]


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, handler: Handler):
        self.handler = handler

    def datagram_received(self, data, addr):
        m = decode(data)
        if m:
            self.handler(m[0], m[1], addr)


class UdpBus:
    """Multicast (plus optional unicast peers) transport."""

    def __init__(
        self,
        group: str = GROUP,
        port: int = PORT,
        peers: list[str] | None = None,
        interface: str = "0.0.0.0",
    ):
        self.group, self.port, self.interface = group, port, interface
        self.peers = [self._addr(p) for p in peers or []]
        self._send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._send.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        self._send.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self._transport = None

    def _addr(self, p: str) -> tuple[str, int]:
        host, _, port = p.partition(":")
        return host, int(port or self.port)

    async def start(self, handler: Handler) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind(("", self.port))
        mreq = struct.pack("4s4s", socket.inet_aton(self.group), socket.inet_aton(self.interface))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.setblocking(False)
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _Protocol(handler), sock=sock
        )

    def send(self, address: str, payload: str) -> None:
        data = encode(address, payload)
        for dest in [(self.group, self.port), *self.peers]:
            try:
                self._send.sendto(data, dest)
            except OSError:
                pass  # a dropped network never stops a voice

    def close(self) -> None:
        if self._transport:
            self._transport.close()
        self._send.close()


class MemoryBus:
    """In-process bus for tests and for several voices in one process."""

    class Hub:
        def __init__(self):
            self.members: list[MemoryBus] = []

    def __init__(self, hub: MemoryBus.Hub):
        self.hub = hub
        self.handler: Handler | None = None
        hub.members.append(self)

    async def start(self, handler: Handler) -> None:
        self.handler = handler

    def send(self, address: str, payload: str) -> None:
        for m in self.hub.members:
            if m.handler:
                m.handler(address, payload, ("memory", 0))

    def close(self) -> None:
        if self in self.hub.members:
            self.hub.members.remove(self)


class Peers:
    """The latest state heard from every other voice, forgetting voices gone quiet."""

    def __init__(self, me: str, timeout_s: float = 10.0, now=time.monotonic):
        self.me, self.timeout_s, self._now = me, timeout_s, now
        self._seen: dict[str, tuple[float, VoiceState]] = {}

    def update(self, state: VoiceState) -> None:
        if state.name != self.me:
            self._seen[state.name] = (self._now(), state)

    def drop(self, name: str) -> None:
        self._seen.pop(name, None)

    def live(self) -> list[VoiceState]:
        cutoff = self._now() - self.timeout_s
        for name in [n for n, (t, _) in self._seen.items() if t < cutoff]:
            del self._seen[name]
        return [s for _, s in sorted(self._seen.values(), key=lambda x: x[1].name)]
