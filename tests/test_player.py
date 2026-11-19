import numpy as np
import pytest

from counterpoint.player import Mixer, pan_gains
from counterpoint.render import Loop


def ramp_loop(n=100, value=1.0):
    a = np.full((2, n), value, dtype=np.float32)
    return Loop(a, 1000, 4.0, 96.0)


def test_switch_at_exact_frame():
    m = Mixer(1000)
    m.fade_frames = 1
    m.schedule(ramp_loop(value=1.0), 37)
    out = m.block(64)
    lg, _ = pan_gains(0)
    assert np.all(out[0, :37] == 0)
    assert out[0, 37] == pytest.approx(0.0, abs=1e-6) or out[0, 38] == pytest.approx(lg)
    assert np.allclose(out[0, 40:], lg)


def test_crossfade_is_smooth():
    m = Mixer(1000)
    m.fade_frames = 10
    m.schedule(ramp_loop(value=1.0), 0)
    m.block(50)
    m.schedule(ramp_loop(value=-1.0), 60)
    out = m.block(50)[0] / pan_gains(0)[0]
    assert np.allclose(out[:10], 1.0)
    assert np.all(np.diff(out[10:20]) <= 1e-6)  # falling through the fade
    assert np.allclose(out[20:], -1.0)


def test_late_schedule_starts_in_phase():
    m = Mixer(1000)
    loop = Loop(np.tile(np.arange(100, dtype=np.float32), (2, 1)), 1000, 4.0, 96.0)
    m.block(30)
    m.fade_frames = 1
    m.schedule(loop, 10)  # 20 frames late
    out = m.block(10)[0] / pan_gains(0)[0]
    assert out[5] == pytest.approx(25.0)


def test_pan():
    left, right = pan_gains(-1)
    assert left == pytest.approx(1) and right == pytest.approx(0, abs=1e-9)
    assert sum(g * g for g in pan_gains(0.3)) == pytest.approx(1)
