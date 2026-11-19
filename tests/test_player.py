import numpy as np
import pytest

from counterpoint.player import Mixer, Swell, pan_gains
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


def test_swell_stays_within_depth():
    sw = Swell(6.0, 16.0, 0.15, seed="v1")
    dbs = []
    for i in range(400):
        if i % 10 == 0:
            sw.new_loop(f"v1:{i}")
        sw.advance(0.25)
        dbs.append(sw.db())
    assert max(dbs) <= 3.0 + 1e-9 and min(dbs) >= -3.0 - 1e-9
    assert max(dbs) > 2.5 and min(dbs) < -2.5  # it really does swell


def test_swell_period_drifts_per_loop_and_repeats():
    a, b = Swell(6, 16, 0.2), Swell(6, 16, 0.2)
    periods = set()
    for i in range(20):
        a.new_loop(f"v1:{i * 4}")
        b.new_loop(f"v1:{i * 4}")
        assert a.period == b.period  # seeded, so a replay draws the same
        assert 16 * 0.8 <= a.period <= 16 * 1.2
        periods.add(a.period)
    assert len(periods) > 10


def test_swell_is_continuous_across_a_new_period():
    sw = Swell(6, 16, 0.5)
    sw.advance(5)
    before = sw.db()
    sw.new_loop("x:5")
    assert sw.db() == before


def test_voices_start_out_of_step():
    assert Swell(seed="v1").phase != Swell(seed="v2").phase


def test_flat_swell():
    sw = Swell(0.0, 16, 0.2)
    sw.advance(3)
    assert sw.gain() == 1.0


def test_mixer_ramps_the_swell():
    m = Mixer(1000)
    m.fade_frames = 1
    m.schedule(ramp_loop(value=1.0), 0)
    m.block(10)
    m.swell = 0.5
    out = m.block(100)[0] / pan_gains(0)[0]
    assert out[-1] == pytest.approx(0.5)
    assert np.all(np.diff(out) <= 0) and np.abs(np.diff(out)).max() < 0.01
    assert np.allclose(m.block(10)[0] / pan_gains(0)[0], 0.5)
