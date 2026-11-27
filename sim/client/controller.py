"""Outer-loop interface types and the PD test controller.

The interface every outer-loop law implements (later including RuleForge laws):

    cmd = law.step(state, setpoint, dt)

    state    : State     position/velocity NED [m, m/s] from the FC's EKF, attitude [rad], FC time [s]
    setpoint : Setpoint  position NED [m], yaw [rad], and the reference's velocity / acceleration NED
                         [m/s, m/s^2] (zero for step tasks; a law may use them as feedforward or not)
    cmd      : Command   attitude target roll/pitch/yaw [rad] + normalized collective thrust [0..1]

Command.thrust has the units of CTUN.ThO: with ATC_ANGLE_BOOST=0 it is the collective thrust
demand, ~linear in physical thrust (ArduCopter linearizes the prop curve, MOT_THST_EXPO).

PDPositionController is a *test* controller for the interface (T4/T6), not a research controller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.80665


@dataclass
class State:
    t: float                         # FC time of the position sample [s]
    pos: tuple[float, float, float]  # NED [m], relative to EKF origin
    vel: tuple[float, float, float]  # NED [m/s]
    roll: float
    pitch: float
    yaw: float

    @property
    def alt(self) -> float:
        return -self.pos[2]


@dataclass
class Setpoint:
    pos: tuple[float, float, float]  # NED [m]
    yaw: float
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0)   # NED [m/s], reference velocity
    acc: tuple[float, float, float] = (0.0, 0.0, 0.0)   # NED [m/s^2], reference acceleration


@dataclass
class Command:
    roll: float
    pitch: float
    yaw: float
    thrust: float


@dataclass
class PDGains:
    kp_xy: float = 1.0    # [1/s^2]
    kd_xy: float = 1.6    # [1/s]
    kp_z: float = 4.0
    kd_z: float = 3.0
    ki_z: float = 1.0     # [1/s^3], removes hover-thrust mismatch
    i_lim: float = 2.0    # max integrator contribution [m/s^2]
    ki_xy: float = 0.0    # [1/s^3] horizontal integrator (steady wind); off by default, T4/T6 ran without
    i_zone_xy: float = 1.0  # [m] integrate only this close to the setpoint, so large steps do not wind it up
    i_lim_xy: float = 2.0   # max horizontal integrator contribution [m/s^2]


class PDPositionController:
    """Position PD (+ slow altitude integrator) -> desired specific force -> attitude + thrust."""

    def __init__(self, hover_thrust: float, gains: PDGains | None = None, max_tilt_deg: float = 20.0,
                 thrust_limits: tuple[float, float] = (0.1, 0.9), feedforward: bool = True):
        self.hover_thrust = hover_thrust
        self.g = gains or PDGains()
        self.ff = 1.0 if feedforward else 0.0     # use the reference's velocity / acceleration (trajectories)
        self.max_tilt = math.radians(max_tilt_deg)
        self.thr_min, self.thr_max = thrust_limits
        self.i_z = 0.0
        self.i_n = self.i_e = 0.0

    @classmethod
    def from_config(cls, hover_thrust: float, gains: dict | None = None, **kw) -> "PDPositionController":
        """Build from a scenario's controller.args (plain YAML values)."""
        if "thrust_limits" in kw:
            kw["thrust_limits"] = tuple(kw["thrust_limits"])
        return cls(hover_thrust, PDGains(**(gains or {})), **kw)

    def reset(self):
        self.i_z = 0.0
        self.i_n = self.i_e = 0.0

    def step(self, s: State, sp: Setpoint, dt: float) -> Command:
        g = self.g
        e_n, e_e = sp.pos[0] - s.pos[0], sp.pos[1] - s.pos[1]
        e_up = s.pos[2] - sp.pos[2]
        if g.ki_xy > 0 and math.hypot(e_n, e_e) < g.i_zone_xy:
            lim = g.i_lim_xy / g.ki_xy
            self.i_n = min(max(self.i_n + e_n * dt, -lim), lim)
            self.i_e = min(max(self.i_e + e_e * dt, -lim), lim)
        # velocity error and acceleration feedforward from the reference (both zero for step tasks, so
        # steps fly exactly as before)
        ff = self.ff
        a_n = g.kp_xy * e_n + g.kd_xy * (ff * sp.vel[0] - s.vel[0]) + g.ki_xy * self.i_n + ff * sp.acc[0]
        a_e = g.kp_xy * e_e + g.kd_xy * (ff * sp.vel[1] - s.vel[1]) + g.ki_xy * self.i_e + ff * sp.acc[1]
        a_up = g.kp_z * e_up + g.kd_z * (s.vel[2] - ff * sp.vel[2]) + g.ki_z * self.i_z - ff * sp.acc[2]

        # required specific force (thrust/mass), NED horizontal + up component
        f_up = max(G + a_up, 0.3 * G)
        h = math.hypot(a_n, a_e)
        h_max = f_up * math.tan(self.max_tilt)
        if h > h_max:                  # keep altitude authority, scale back the horizontal demand
            a_n, a_e = a_n * h_max / h, a_e * h_max / h
        c, sn = math.cos(sp.yaw), math.sin(sp.yaw)
        f_fwd = c * a_n + sn * a_e
        f_right = -sn * a_n + c * a_e
        f_mag = math.sqrt(f_fwd ** 2 + f_right ** 2 + f_up ** 2)
        roll = math.asin(f_right / f_mag)
        pitch = math.atan2(-f_fwd, f_up)

        thrust_raw = self.hover_thrust * f_mag / G
        thrust = min(max(thrust_raw, self.thr_min), self.thr_max)
        saturated = (thrust_raw > self.thr_max and e_up > 0) or (thrust_raw < self.thr_min and e_up < 0)
        if not saturated:
            lim = g.i_lim / g.ki_z if g.ki_z > 0 else 0.0
            self.i_z = min(max(self.i_z + e_up * dt, -lim), lim)
        return Command(roll, pitch, sp.yaw, thrust)
