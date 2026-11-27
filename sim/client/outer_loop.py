"""Event-driven outer loop over MAVLink: one control step per LOCAL_POSITION_NED sample.

The FC streams LOCAL_POSITION_NED and ATTITUDE at STATE_HZ; each new position sample triggers
    state -> supervisor.check -> law.step -> supervisor.command_fault / clamp -> SET_ATTITUDE_TARGET
A law that raises or returns a non-finite command is a controller fault: nothing is sent for that step
and the loop trips into its fallback mode.

Who owns the vehicle: while the FC is in GUIDED the loop does. When the FC reports any other mode that
the loop did not command itself (pilot, GCS on another link, FC failsafe), the FC owns it: the loop
stops, commands no mode, and sets LoopResult.fc_owned so that the caller sends nothing either (no LAND).
so the loop is paced by FC time, not by a client timer: it runs unchanged at any SITL speedup
and on the real vehicle (companion computer on UART4 / SERIAL4). The few timers it has are FC time
too (1 Hz heartbeat, monitor durations). Missing state is checked twice: a gap between consecutive
samples longer than state_timeout_s of FC time trips the loop; and if nothing arrives for
state_timeout_s of wall time the link is taken as dead. On the real vehicle (or SITL at 1x) both are
the same check. At speedup S the wall check is deliberately not divided by S: SITL pauses itself when
the client falls behind reading, so a wall gap is not a gap in simulated time (the first T5 version
scaled it and tripped on scheduling hiccups under load).

Python's cyclic garbage collector is off while the loop runs: a full collection over the recorded
rows took ~25-30 ms of wall time (one missed control period at 1x, a 'no state' trip at 10x in T5).
Reference counting still frees everything that is not in a cycle; gc is restored afterwards.

Everything the loop sees and sends is recorded (rows) together with the FC's own echo of the
active target (ATTITUDE_TARGET), for latency / dropout analysis against the FC's GUIA log.
"""
from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import mav
from .controller import Command, Setpoint, State
from .supervisor import Supervisor

M = mav.M
STATE_HZ = 50


def request_streams(m, hz: float = STATE_HZ):
    for mid in (M.MAVLINK_MSG_ID_LOCAL_POSITION_NED, M.MAVLINK_MSG_ID_ATTITUDE,
                M.MAVLINK_MSG_ID_ATTITUDE_TARGET):
        mav.set_message_rate(m, mid, hz)
    mav.set_message_rate(m, M.MAVLINK_MSG_ID_SYS_STATUS, 2)


def _is_fc_heartbeat(m, msg) -> bool:
    # pymavlink leaves target_component at 0 (broadcast), so identify the autopilot by sysid + type
    return (msg.get_srcSystem() == m.target_system and msg.autopilot != M.MAV_AUTOPILOT_INVALID
            and msg.type != M.MAV_TYPE_GCS)


def step_schedule(origin_ned, yaw: float, steps: list[tuple[float, tuple[float, float, float]]]):
    """Piecewise-constant setpoint: steps = [(t_start, (dN, dE, dD)), ...] relative to origin."""
    origin = np.asarray(origin_ned, float)

    def fn(t: float) -> Setpoint:
        off = steps[0][1]
        for ts, o in steps:
            if t >= ts:
                off = o
        return Setpoint(tuple(origin + np.asarray(off, float)), yaw)
    return fn


@dataclass
class LoopResult:
    rows: list[dict] = field(default_factory=list)   # one per state sample
    echo: list[dict] = field(default_factory=list)   # FC ATTITUDE_TARGET (active target incl. thrust)
    text: list[str] = field(default_factory=list)
    t0_ms: int | None = None
    trip: str | None = None
    trip_t: float | None = None
    fault: str | None = None     # controller fault (law raised / non-finite command), also the trip reason
    fc_owned: bool = False       # the FC left GUIDED on its own: pilot / failsafe owns the vehicle,
    fc_mode: str | None = None   # in this mode; the client must not command anything from here on
    takeover_wall: float | None = None   # time.monotonic() when the takeover was detected
    end_reason: str = ""
    wall_s: float = 0.0          # wall time of the loop
    fc_s: float = 0.0            # FC time covered by the loop (fc_s / wall_s = achieved speedup)


class OuterLoop:
    def __init__(self, m, law, supervisor: Supervisor, setpoint_fn: Callable[[float], Setpoint], *,
                 drop_fn: Callable[[float], bool] | None = None, fallback_mode: str | None = "BRAKE",
                 step_hook: Callable[[float], None] | None = None, time_scale: float = 1.0, log=print):
        self.m, self.law, self.sup, self.setpoint_fn = m, law, supervisor, setpoint_fn
        self.time_scale = time_scale             # SITL speedup; 1 on the real vehicle
        self.drop_fn = drop_fn or (lambda t: False)
        self.step_hook = step_hook               # test-only event injection, called once per step
        self.fallback_mode = fallback_mode
        self.log = log

    def run(self, duration_s: float) -> LoopResult:
        gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        try:
            return self._run(duration_s)
        finally:
            if gc_was_enabled:
                gc.enable()

    def _run(self, duration_s: float) -> LoopResult:
        m, sup, res = self.m, self.sup, LoopResult()
        mav.drain(m)
        att = None
        last_t = None
        t = 0.0
        wall0 = last_state_wall = time.monotonic()
        state_timeout_wall = sup.lim.state_timeout_s
        last_hb_t = None
        while True:
            msg = m.recv_match(type=["LOCAL_POSITION_NED", "ATTITUDE", "ATTITUDE_TARGET", "HEARTBEAT",
                                     "STATUSTEXT"], blocking=True, timeout=0.05)
            wall = time.monotonic()
            dropping = self.drop_fn(t)
            if not dropping and (last_hb_t is None or t - last_hb_t >= 1.0):   # 1 Hz FC time;
                mav.send_heartbeat(m)                                          # dropped with the link
                last_hb_t = t
            if wall - last_state_wall > state_timeout_wall:
                res.trip = f"no state for {wall - last_state_wall:.3f} s wall (link dead?)"
                res.trip_t = t
                break
            if msg is None:
                continue
            typ = msg.get_type()
            if typ == "ATTITUDE":
                att = msg
                continue
            if typ == "ATTITUDE_TARGET":
                if res.t0_ms is not None:
                    r, p, y = mav.quat_to_euler(msg.q)
                    res.echo.append({"t_ms": msg.time_boot_ms, "t": (msg.time_boot_ms - res.t0_ms) / 1e3,
                                     "roll": r, "pitch": p, "yaw": y, "thrust": msg.thrust})
                continue
            if typ == "STATUSTEXT":
                res.text.append(f"{t:7.2f} {msg.text}")
                self.log(f"FC: {msg.text}")
                continue
            if typ == "HEARTBEAT":
                if _is_fc_heartbeat(m, msg) and mav.mavutil.mode_string_v10(msg) != "GUIDED":
                    res.fc_mode = mav.mavutil.mode_string_v10(msg)
                    res.trip, res.trip_t = f"FC left GUIDED ({res.fc_mode})", t
                    res.fc_owned, res.takeover_wall = True, wall
                    self.fallback_mode = None       # the FC (pilot / failsafe) owns the vehicle now
                    break
                continue
            if att is None:                          # need one attitude sample first
                continue

            # --- one control step per position sample ---
            t_ms = msg.time_boot_ms
            if res.t0_ms is None:
                res.t0_ms = t_ms
            t = (t_ms - res.t0_ms) / 1e3
            if last_t is not None and t - last_t > sup.lim.state_timeout_s:   # before the duration check:
                res.trip, res.trip_t = f"state gap {t - last_t:.3f} s (FC time)", t   # a gap that runs past
                break                                                       # the end is not a completion
            if t > duration_s:
                res.end_reason = "duration"
                break
            dt = 1.0 / STATE_HZ if last_t is None else min(max(t - last_t, 0.005), 0.1)
            last_t, last_state_wall = t, wall
            s = State(t, (msg.x, msg.y, msg.z), (msg.vx, msg.vy, msg.vz), att.roll, att.pitch, att.yaw)
            sp = self.setpoint_fn(t)
            reason = sup.check(s)
            c0 = time.perf_counter()
            try:
                cmd = self.law.step(s, sp, dt)
                fault = sup.command_fault(cmd)
            except Exception as e:                   # the law itself failed: same handling
                cmd, fault = Command(math.nan, math.nan, math.nan, math.nan), f"controller fault: {e!r}"
            if fault:                                # never sent, not even clamped: trip in this step
                clamped, res.fault = False, fault
                reason = reason or fault
                cmd = Command(math.nan, math.nan, math.nan, math.nan)   # logged as NaN; the bad values
                                                                        # are in the fault message
            else:
                cmd, clamped = sup.clamp(cmd)
            thrust32 = float(np.float32(cmd.thrust))   # exactly what the FC will log in GUIA.Thrust
            dropping = self.drop_fn(t)
            sent = reason is None and not dropping
            if sent:
                mav.send_attitude_target(m, cmd.roll, cmd.pitch, cmd.yaw, thrust32, t_ms)
            wall_out = time.monotonic()
            res.rows.append({
                "t": t, "t_ms": t_ms, "att_age_ms": t_ms - att.time_boot_ms,
                "x": s.pos[0], "y": s.pos[1], "z": s.pos[2], "vx": s.vel[0], "vy": s.vel[1], "vz": s.vel[2],
                "roll": s.roll, "pitch": s.pitch, "yaw": s.yaw,
                "sp_x": sp.pos[0], "sp_y": sp.pos[1], "sp_z": sp.pos[2], "sp_yaw": sp.yaw,
                "sp_vx": sp.vel[0], "sp_vy": sp.vel[1], "sp_vz": sp.vel[2],
                "sp_ax": sp.acc[0], "sp_ay": sp.acc[1], "sp_az": sp.acc[2],
                "cmd_roll": cmd.roll, "cmd_pitch": cmd.pitch, "cmd_yaw": cmd.yaw, "cmd_thrust": thrust32,
                "clamped": clamped, "sent": sent, "dropped": dropping, "fault": fault is not None,
                "compute_us": (time.perf_counter() - c0) * 1e6,
                "wall_s": wall - wall0,                      # state sample parsed (wall clock)
                "client_us": (wall_out - wall) * 1e6,        # parsed -> command sent (wall clock)
            })
            if reason:
                res.trip, res.trip_t = reason, t
                break
            if self.step_hook:
                self.step_hook(t)

        res.wall_s, res.fc_s = time.monotonic() - wall0, t
        if res.trip:
            self.log(f"outer loop stopped at t={res.trip_t:.2f}: {res.trip}")
            if self.fallback_mode:
                m.set_mode(m.mode_mapping()[self.fallback_mode])   # confirmed by monitor()
        return res


def monitor(m, seconds: float, t0_ms: int, stop: Callable[[dict], bool] | None = None,
            time_scale: float = 1.0, hook: Callable[[float], None] | None = None) -> list[dict]:
    """Record position, FC mode and armed state (e.g. after a trip, or while landing). Sends nothing.

    Runs for `seconds` of FC time (wall-clock guard seconds/time_scale + 10 s) or until stop(row).
    hook(t) is called once per position sample with the outer-loop time (test-only event injection).
    """
    rows, mode, first_ms = [], None, None
    end = time.monotonic() + seconds / time_scale + 10
    while time.monotonic() < end:
        msg = m.recv_match(type=["LOCAL_POSITION_NED", "HEARTBEAT"], blocking=True, timeout=0.2)
        if msg is None:
            continue
        if msg.get_type() == "HEARTBEAT":
            if _is_fc_heartbeat(m, msg):
                mode = mav.mavutil.mode_string_v10(msg)
            continue
        row = {"t": (msg.time_boot_ms - t0_ms) / 1e3, "x": msg.x, "y": msg.y, "z": msg.z,
               "vx": msg.vx, "vy": msg.vy, "vz": msg.vz, "mode": mode, "armed": bool(m.motors_armed())}
        rows.append(row)
        if hook:
            hook(row["t"])
        first_ms = msg.time_boot_ms if first_ms is None else first_ms
        if (stop and stop(row)) or (msg.time_boot_ms - first_ms) / 1e3 >= seconds:
            break
    return rows


def fc_took_over(rows: list[dict], commanded: str | None) -> str | None:
    """After monitor(): the FC mode if it is one this client did not command, else None.

    `commanded` is the mode the client set last (BRAKE after a trip). GUIDED means the command has not
    taken effect (yet); any other mode was set by someone else (pilot / GCS / FC failsafe), who owns the
    vehicle from then on.
    """
    mode = next((r["mode"] for r in reversed(rows) if r["mode"]), None)
    return None if mode in (None, "GUIDED", commanded) else mode
