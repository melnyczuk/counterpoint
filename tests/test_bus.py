import asyncio
import socket

import pytest

from counterpoint.bus import (
    CONTROL,
    STATE,
    Control,
    MemoryBus,
    Peers,
    Playing,
    UdpBus,
    VoiceState,
    decode,
    encode,
)


def state(name="v1"):
    p = Playing("a", 2, 0.97, [1 / 12] * 12, [0.2, 0.6, 0.2], "t:a", 16.0, 16.0)
    return VoiceState(name, 12.5, 96.0, now=p, change_beat=32.0)


def test_state_roundtrip():
    s = state()
    back = VoiceState.from_json(s.to_json())
    assert back == s
    assert back.heard().id == "a"


def test_osc_roundtrip():
    assert decode(encode(STATE, "hello")) == (STATE, "hello")
    assert decode(b"not osc") is None


def test_control_roundtrip():
    c = Control("anchor", "Dm9", "v2")
    assert Control.from_json(c.to_json()) == c


def test_peers_forget_quiet_voices():
    t = [0.0]
    peers = Peers("me", timeout_s=5, now=lambda: t[0])
    peers.update(state("v1"))
    peers.update(state("me"))  # its own messages are ignored
    assert [s.name for s in peers.live()] == ["v1"]
    t[0] = 6.0
    assert peers.live() == []


def test_memory_bus():
    hub = MemoryBus.Hub()
    a, b = MemoryBus(hub), MemoryBus(hub)
    got = []

    async def go():
        await a.start(lambda *m: None)
        await b.start(lambda addr, payload, src: got.append((addr, payload)))
        a.send(CONTROL, "x")

    asyncio.run(go())
    assert got == [(CONTROL, "x")]


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_udp_multicast():
    port = _free_port()
    got = []

    async def go():
        rx, tx = UdpBus(port=port), UdpBus(port=port, peers=[f"127.0.0.1:{port}"])
        try:
            await rx.start(lambda addr, payload, src: got.append(payload))
        except OSError as e:
            pytest.skip(f"no multicast here: {e}")
        tx.send(STATE, state().to_json())
        for _ in range(50):
            if got:
                break
            await asyncio.sleep(0.02)
        rx.close()
        tx.close()

    asyncio.run(go())
    assert got and VoiceState.from_json(got[0]).name == "v1"
