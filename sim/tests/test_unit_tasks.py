"""Unit checks without SITL for the task library: trajectories, feedforward, gusts, pushes, event metrics.

    python -m pytest sim/tests/test_unit_tasks.py        or        python sim/tests/test_unit_tasks.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import numpy as np  # noqa: E402

from analysis.metrics import event_metrics, trajectory_metrics  # noqa: E402
from client.controller import G, PDGains, PDPositionController, Setpoint, State  # noqa: E402
from client.trajectory import Trajectory  # noqa: E402
from runner.disturbance import GustProfile, air_velocity, push_params, wind_params  # noqa: E402
from runner.scenario import load_scenario  # noqa: E402


# ---------------------------------------------------------------- trajectories

def _fd_check(tr: Trajectory):
    """Analytic velocity / acceleration equal finite differences of position and velocity."""
    h = 1e-4
    for t in np.linspace(0, tr.end_s + 2, 801):
        p0, v0, a0 = tr.offset(t)
        pp, vp, _ = tr.offset(t + h)
        pm, vm, _ = tr.offset(t - h)
        v_fd = (np.subtract(pp, pm)) / (2 * h)
        a_fd = (np.subtract(vp, vm)) / (2 * h)
        assert np.allclose(v0, v_fd, atol=1e-5), (t, v0, v_fd)
        assert np.allclose(a0, a_fd, atol=2e-3), (t, a0, a_fd)


def test_circle_derivatives_match_finite_differences():
    _fd_check(Trajectory("circle", 1.5, 8.0, 2, 4.0, 3.0, 0.5))


def test_figure8_derivatives_match_finite_differences():
    _fd_check(Trajectory("figure8", 2.0, 12.0, 1.5, 2.0, 4.0))


def test_trajectory_starts_and_ends_at_rest_at_the_handover_point():
    for shape in ("circle", "figure8"):
        tr = Trajectory(shape, 1.5, 8.0, 2, 4.0, 3.0)
        for t in (0.0, tr.start_s, tr.end_s, tr.end_s + 5):
            p, v, a = tr.offset(t)
            assert np.allclose(p, 0, atol=1e-9) and np.allclose(v, 0) and np.allclose(a, 0, atol=1e-9), (shape, t)


def test_circle_full_speed_is_omega_r_and_velocity_continuous_at_ramp_ends():
    tr = Trajectory("circle", 1.5, 8.0, 2, 4.0, 3.0)
    assert math.isclose(tr.max_speed(), 2 * math.pi / 8 * 1.5, rel_tol=1e-3)
    for tb in (tr.start_s + tr.ramp_s, tr.end_s - tr.ramp_s):
        assert np.allclose(tr.offset(tb - 1e-7)[1], tr.offset(tb + 1e-7)[1], atol=1e-5)


def test_trajectory_rejects_bad_values():
    for kw in ({"shape": "square"}, {"size_m": 0}, {"ramp_s": 100}):
        try:
            Trajectory(**kw)
        except ValueError:
            continue
        raise AssertionError(kw)


# ---------------------------------------------------------------- feedforward in the PD law

def _old_pd_step(law: PDPositionController, s: State, sp: Setpoint, dt: float):
    """The PD step before feedforward existed (copied), for a step task."""
    g = law.g
    e_n, e_e = sp.pos[0] - s.pos[0], sp.pos[1] - s.pos[1]
    e_up = s.pos[2] - sp.pos[2]
    a_n = g.kp_xy * e_n - g.kd_xy * s.vel[0]
    a_e = g.kp_xy * e_e - g.kd_xy * s.vel[1]
    a_up = g.kp_z * e_up + g.kd_z * s.vel[2] + g.ki_z * law.i_z
    f_up = max(G + a_up, 0.3 * G)
    f_mag = math.sqrt(a_n ** 2 + a_e ** 2 + f_up ** 2)
    return f_mag


def test_pd_unchanged_for_steps_and_uses_feedforward_when_given():
    s = State(0.0, (0.3, -0.2, -5.1), (0.4, 0.1, -0.2), 0.0, 0.0, 0.0)
    a = PDPositionController(0.58, PDGains())
    b = PDPositionController(0.58, PDGains())
    c1 = a.step(s, Setpoint((0.0, 0.0, -5.0), 0.0), 0.02)
    c2 = b.step(s, Setpoint((0.0, 0.0, -5.0), 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 0.02)
    assert (c1.roll, c1.pitch, c1.thrust) == (c2.roll, c2.pitch, c2.thrust)
    law = PDPositionController(0.58, PDGains())
    f = _old_pd_step(law, s, Setpoint((0.0, 0.0, -5.0), 0.0), 0.02)
    assert math.isclose(c1.thrust, min(max(0.58 * f / G, 0.1), 0.9), rel_tol=1e-12)
    # pure feedforward: no error, reference accelerating north at 2 m/s^2 -> pitch forward (negative)
    s0 = State(0.0, (0.0, 0.0, -5.0), (1.0, 0.0, 0.0), 0.0, 0.0, 0.0)
    c = PDPositionController(0.58).step(s0, Setpoint((0.0, 0.0, -5.0), 0.0, (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)), 0.02)
    assert math.isclose(c.pitch, -math.atan2(2.0, G), rel_tol=1e-9)


# ---------------------------------------------------------------- gusts and pushes

def test_wind_params_round_trip_and_direction_convention():
    for spd, d in ((3.0, 0.0), (2.0, 45.0), (1.0, 270.0)):
        s2, d2 = wind_params(air_velocity(spd, d))
        assert math.isclose(s2, spd) and math.isclose(d2 % 360, d % 360, abs_tol=1e-9)
    # wind from the north (dir 0) moves the air south
    assert np.allclose(air_velocity(2.0, 0.0), (-2.0, 0.0))


def test_gust_profile_statistics_start_and_determinism():
    gp = GustProfile(0.0, 0.0, sigma_ms=1.0, tau_s=2.0, rate_hz=10.0, duration_s=4000.0, seed=7)
    assert np.allclose(gp.gust[0], 0)
    g = gp.gust[200:, 0]                              # past the start transient
    assert abs(g.std() - 1.0) < 0.1
    lag = 20                                          # tau = 2 s at 10 Hz
    r = np.corrcoef(g[:-lag], g[lag:])[0, 1]
    assert abs(r - math.exp(-1)) < 0.08
    again = GustProfile(0.0, 0.0, 1.0, 2.0, 10.0, 4000.0, 7)
    other = GustProfile(0.0, 0.0, 1.0, 2.0, 10.0, 4000.0, 8)
    assert np.array_equal(gp.gust, again.gust) and not np.array_equal(gp.gust, other.gust)


def test_push_rotated_into_body_frame():
    p = push_params({"n": 1.0, "e": 0.0, "d": 0.5, "ms": 300}, yaw=math.pi / 2)   # facing east
    assert math.isclose(p["SIM_SHOVE_X"], 0.0, abs_tol=1e-12) and math.isclose(p["SIM_SHOVE_Y"], -1.0)
    assert p["SIM_SHOVE_Z"] == 0.5 and p["SIM_SHOVE_TIME"] == 300
    assert list(p)[-1] == "SIM_SHOVE_TIME"            # the time starts the push, so it goes last


# ---------------------------------------------------------------- metrics

def test_event_metrics_measure_the_response_not_the_standing_offset():
    t = np.arange(0, 20, 0.02)
    e = np.zeros((len(t), 3))
    e[:, 0] = 0.3                                     # standing offset before and after
    bump = (t >= 10) & (t < 13)
    e[bump, 0] += 0.5 * np.sin(np.pi * (t[bump] - 10) / 3)
    m = event_metrics(t, e, [10.0], band_m=0.1)
    assert math.isclose(m["ev0_max_dev_m"], 0.5, rel_tol=1e-3)
    assert 2.7 < m["ev0_recover_s"] < 3.0
    e2 = e.copy()
    e2[t >= 10, 0] += 0.5                             # never comes back
    assert math.isnan(event_metrics(t, e2, [10.0], 0.1)["ev0_recover_s"])


def test_trajectory_metrics_window():
    t = np.arange(0, 10, 0.02)
    e = np.zeros((len(t), 3))
    e[(t >= 2) & (t <= 6), 1] = 0.2
    e[t > 8, 1] = 5.0                                 # outside the window, ignored
    m = trajectory_metrics(t, e, 2.0, 6.0)
    assert math.isclose(m["traj_rms_h_m"], 0.2) and math.isclose(m["traj_max_m"], 0.2)


# ---------------------------------------------------------------- scenario validation

def test_scenario_validation_of_tasks_and_events():
    ok = load_scenario({"task": {"type": "trajectory", "duration_s": 40.0,
                                 "events": [{"t_s": 5, "push": {"n": 3, "ms": 200}}]},
                        "environment": {"gusts": {"sigma_ms": 1.0}}})
    assert ok["task"]["trajectory"]["shape"] == "circle" and ok["environment"]["gusts"]["sigma_ms"] == 1.0
    bad = [{"task": {"type": "trajectory", "duration_s": 20.0}},          # 4 + 2*10 + 3 = 27 s > 20 s
           {"task": {"events": [{"t_s": 5, "push": {"n": 3}}]}},           # push without ms
           {"task": {"events": [{"t_s": 5, "params": {}, "push": {"ms": 1}}]}},
           {"environment": {"gusts": {"tau_s": 0}}}]
    for over in bad:
        try:
            load_scenario(over)
        except (AssertionError, ValueError):
            continue
        raise AssertionError(over)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} passed")
