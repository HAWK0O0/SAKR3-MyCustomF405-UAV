"""Per-run metrics on simulator truth, one flat dict per run (a row of the batch table).

Inputs are the outer-loop rows with truth attached (analysis/truth.attach: t_n, t_e, t_d, t_roll,
t_pitch at each control step) and the FC's GUIA log. Position errors are true position minus setpoint.

Task "steps": segment i runs from steps[i].t_s to the next step (or the end). For each segment:
  seg{i}_final_err_m   |mean true position - setpoint| over the last final_window_s
  seg{i}_final_std_m   3D spread of the true position over that window
and if the setpoint moved at its start (a step of length L along unit vector u):
  seg{i}_rise_s        10 % -> 90 % of the step, progress = (p - p_start).u / L
  seg{i}_overshoot_pct max progress beyond 100 %
  seg{i}_settle_s      time after the step of the last sample farther than band_m from the setpoint
                       (NaN if still outside at the end of the segment; seg{i}_settled = 0 then)
  seg{i}_cross_max_m   max distance from the straight line of the step
Task "trajectory" (moving reference with feedforward, client/trajectory.py): one segment (seg0, its
final window is the hold after the motion), and over the motion [start_s, end_s]:
  traj_rms_m, traj_rms_h_m, traj_rms_v_m, traj_max_m   true position minus reference
Events (task.events, pushes / parameter changes) at t_e, each up to the next event or the end:
  ev{i}_max_dev_m      max |e(t) - e0|, e = true position - setpoint, e0 = mean e over the second before
                       t_e: the response to the disturbance, without the offset that was already there
  ev{i}_recover_s      time after t_e of the last sample with |e - e0| > band_m (0 if never; NaN if still
                       outside at the end of the window)
Whole run: IAE (3D, horizontal, vertical) of the true error, hold RMS over all final windows,
EKF-vs-truth error (RMS, max, and its drift from the first to the last second of the run), max true
tilt, clamp fraction, and the interface (latency, FC command timing).
"""
from __future__ import annotations

import fnmatch
import math

import numpy as np
import pandas as pd

from .outer_loop import interval_stats, match_commands


def _trapz(y, x) -> float:
    return float(np.trapz(y, x)) if len(x) > 1 else 0.0


def _rms(x) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def segments(steps, duration: float):
    """[(i, t0, t1, offset_before, offset_after)] for task.steps = [[t_s, dN, dE, dD], ...]."""
    out = []
    for i, s in enumerate(steps):
        t1 = steps[i + 1][0] if i + 1 < len(steps) else duration
        before = np.asarray(steps[i - 1][1:] if i else s[1:], float)
        out.append((i, float(s[0]), float(t1), before, np.asarray(s[1:], float)))
    return out


def trajectory_metrics(t, e, start_s: float, end_s: float) -> dict:
    w = (t >= start_s) & (t <= end_s)
    if not w.any():
        return {}
    ew = e[w]
    return {"traj_rms_m": _rms(np.linalg.norm(ew, axis=1)), "traj_rms_h_m": _rms(np.hypot(ew[:, 0], ew[:, 1])),
            "traj_rms_v_m": _rms(ew[:, 2]), "traj_max_m": float(np.linalg.norm(ew, axis=1).max())}


def event_metrics(t, e, event_times, band_m: float) -> dict:
    out = {}
    ts = sorted(event_times)
    for i, te in enumerate(ts):
        t1 = ts[i + 1] if i + 1 < len(ts) else t[-1] + 1e-9
        pre, win = (t >= te - 1.0) & (t < te), (t >= te) & (t < t1)
        if not pre.any() or not win.any():
            continue
        dev = np.linalg.norm(e[win] - e[pre].mean(axis=0), axis=1)
        out[f"ev{i}_max_dev_m"] = float(dev.max())
        outside = dev > band_m
        out[f"ev{i}_recover_s"] = (float("nan") if outside[-1] else
                                   float(t[win][np.where(outside)[0][-1]] - te) if outside.any() else 0.0)
    return out


def run_metrics(rows: pd.DataFrame, guia: pd.DataFrame, steps, origin, duration: float,
                final_window_s: float = 2.0, band_m: float = 0.1, t0_ms: int | None = None,
                trajectory: tuple[float, float] | None = None, event_times=()) -> dict:
    """trajectory: (start_s, end_s) of the motion for a trajectory task (steps is then [[0, 0, 0, 0]]);
    event_times: outer-loop times of task.events."""
    m: dict = {}
    if rows is None or rows.empty:
        return m
    t = rows.t.values
    sp = rows[["sp_x", "sp_y", "sp_z"]].to_numpy()
    p = rows[["t_n", "t_e", "t_d"]].to_numpy()
    est = rows[["x", "y", "z"]].to_numpy()
    e = p - sp
    eh, ev = np.hypot(e[:, 0], e[:, 1]), np.abs(e[:, 2])
    m["t_end_s"] = float(t[-1])
    m["iae_m_s"] = _trapz(np.linalg.norm(e, axis=1), t)
    m["iae_h_m_s"] = _trapz(eh, t)
    m["iae_v_m_s"] = _trapz(ev, t)
    de = est - p                                   # EKF minus truth
    m["est_err_h_rms_m"] = _rms(np.hypot(*de[:, :2].T))
    m["est_err_v_rms_m"] = _rms(de[:, 2])
    m["est_err_h_max_m"] = float(np.hypot(*de[:, :2].T).max())
    # drift of the EKF error over the outer-loop run (1 s means at start and end): optical flow
    # integrates velocity, so its horizontal position error grows; with GPS it should not
    first, last = t <= t[0] + 1.0, t >= t[-1] - 1.0
    m["est_drift_h_m"] = float(np.hypot(*(de[last, :2].mean(0) - de[first, :2].mean(0))))
    m["est_drift_v_m"] = float(abs(de[last, 2].mean() - de[first, 2].mean()))
    tilt = np.arccos(np.clip(np.cos(rows.t_roll.values) * np.cos(rows.t_pitch.values), -1, 1))
    m["max_tilt_deg"] = float(np.degrees(tilt.max()))
    m["clamped_frac"] = float(rows.clamped.mean())
    sent = rows.cmd_thrust[rows.sent]           # thrust headroom, also when the law saturates internally
    if len(sent):
        m["thrust_p50"], m["thrust_p99"] = float(sent.median()), float(sent.quantile(0.99))
    if "client_us" in rows:
        m["client_ms_p50"] = float(rows.client_us.median() / 1e3)
        m["client_ms_p99"] = float(rows.client_us.quantile(0.99) / 1e3)
        gaps = np.diff(rows.wall_s.values) * 1e3
        m["state_gap_wall_max_ms"] = float(gaps.max()) if len(gaps) else float("nan")

    origin = np.asarray(origin, float)
    hold = np.zeros(len(t), bool)
    dt_nom = 0.02
    for i, t0, t1, before, after in segments(steps, duration):
        seg = (t >= t0) & (t < t1)
        if not seg.any():
            continue
        target = origin + after
        ts, ps = t[seg], p[seg]
        if ts[-1] >= t1 - 2 * dt_nom:                 # segment flown to its end
            fin = seg & (t >= t1 - final_window_s)
            hold |= fin
            mean = p[fin].mean(axis=0)
            m[f"seg{i}_final_err_m"] = float(np.linalg.norm(mean - target))
            m[f"seg{i}_final_std_m"] = float(np.sqrt(((p[fin] - mean) ** 2).sum(axis=1).mean()))
        d = after - before
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        u = d / L
        rel = ps - (origin + before)
        prog = rel @ u / L
        cross = np.linalg.norm(rel - np.outer(prog * L, u), axis=1)
        reached = prog.max() >= 0.9
        m[f"seg{i}_rise_s"] = (float(ts[np.argmax(prog >= 0.9)] - ts[np.argmax(prog >= 0.1)])
                               if reached else float("nan"))
        m[f"seg{i}_overshoot_pct"] = float(max(0.0, prog.max() - 1.0) * 100)
        outside = np.linalg.norm(ps - target, axis=1) > band_m
        m[f"seg{i}_settled"] = float(not outside[-1])
        if outside[-1]:
            m[f"seg{i}_settle_s"] = float("nan")
        else:
            m[f"seg{i}_settle_s"] = float(ts[np.where(outside)[0][-1]] - t0) if outside.any() else 0.0
        m[f"seg{i}_cross_max_m"] = float(cross.max())
    m["hold_rms_h_m"] = _rms(eh[hold])
    m["hold_rms_v_m"] = _rms(ev[hold])
    if trajectory is not None:
        m.update(trajectory_metrics(t, e, *trajectory))
    m.update(event_metrics(t, e, event_times, band_m))

    if guia is not None and not guia.empty and t0_ms is not None:
        g = guia[(guia.TimeUS >= t0_ms * 1e3) & (guia.TimeUS <= (rows.t_ms.iloc[-1] + 100) * 1e3)]
        if len(g) > 2:
            fr = interval_stats(g.TimeUS.values / 1e6)
            m["cmd_rate_hz"], m["cmd_gap_p99_ms"], m["cmd_gap_max_ms"] = (
                fr["rate_hz"], fr["interval_p99_ms"], fr["interval_max_ms"])
            lat = match_commands(rows, g)
            if len(lat):
                m["lat_p50_ms"] = float(lat.latency_ms.median())
                m["lat_p95_ms"] = float(lat.latency_ms.quantile(0.95))
                m["lat_max_ms"] = float(lat.latency_ms.max())
                m["lat_matched_frac"] = len(lat) / max(int(rows.sent.sum()), 1)
    return m


def evaluate_pass(metrics: dict, bounds: dict, outcome: str) -> tuple[bool, list[str]]:
    """Pass = outcome ok and every bound holds; patterns (fnmatch) apply to all matching metrics."""
    fails = [] if outcome == "ok" else [f"outcome {outcome}"]
    for pat, ub in (bounds or {}).items():
        keys = [k for k in metrics if fnmatch.fnmatchcase(k, pat)]
        if not keys:
            fails.append(f"{pat}: no such metric")
        for k in keys:
            v = metrics[k]
            if v is None or not math.isfinite(v) or v > ub:
                fails.append(f"{k}={v:.4g} > {ub}" if v is not None else f"{k} missing")
    return not fails, fails
