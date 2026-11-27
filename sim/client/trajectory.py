"""Smooth reference trajectories for tracking tasks: position, velocity and acceleration at any time.

A trajectory starts and ends at the handover point at rest. It holds for `start_s`, then runs `cycles`
times around a closed horizontal curve and holds again. The phase speed rises from 0 to full along a
smoothstep over `ramp_s` and falls the same way at the end, so the reference velocity is continuous
and the acceleration bounded (no step in the commanded tilt at the start or end).

Shapes (offsets from the handover point, N/E in metres, phase phi in rad):
  circle   radius R = size_m:          N = R (cos phi - 1),  E = R sin phi
  figure8  half-width A = size_m:      N = A sin phi,        E = (A/2) sin 2 phi    (lemniscate of Gerono)
Optional `vertical_m` adds an altitude excursion  up = vertical_m (1 - cos phi) / 2  on each cycle.
One cycle takes `period_s` at full speed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .controller import Setpoint

SHAPES = ("circle", "figure8")


@dataclass(frozen=True)
class Trajectory:
    shape: str = "circle"
    size_m: float = 1.5
    period_s: float = 10.0
    cycles: float = 2.0
    start_s: float = 4.0
    ramp_s: float = 3.0
    vertical_m: float = 0.0

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"trajectory shape {self.shape!r} not in {SHAPES}")
        if not (self.size_m > 0 and self.period_s > 0 and self.cycles > 0 and self.start_s >= 0
                and 0 <= self.ramp_s <= self.cycles * self.period_s):
            raise ValueError("trajectory: size_m, period_s, cycles > 0; start_s >= 0; "
                             "0 <= ramp_s <= cycles * period_s")

    @classmethod
    def from_config(cls, cfg: dict) -> "Trajectory":
        return cls(**cfg)

    @property
    def omega(self) -> float:
        return 2 * math.pi / self.period_s

    @property
    def move_s(self) -> float:
        """Time from the start of motion to rest again: full-speed time + two half ramps."""
        return self.cycles * self.period_s + self.ramp_s

    @property
    def end_s(self) -> float:
        return self.start_s + self.move_s

    # --- phase: phi(t) = omega * S(t), S' = speed fraction s(t) in [0, 1]
    def _phase(self, t: float) -> tuple[float, float, float]:
        """(phi, phi_dot, phi_ddot) at time t."""
        w, tr = self.omega, self.ramp_s
        tm = t - self.start_s
        if tm <= 0:
            return 0.0, 0.0, 0.0
        T = self.move_s
        if tm >= T:
            return w * (T - tr), 0.0, 0.0           # = 2 pi cycles: back at the start point
        if tr > 0 and tm < tr:                      # ramp up: s = 3u^2 - 2u^3, S = tr (u^3 - u^4/2)
            u = tm / tr
            return w * tr * (u ** 3 - u ** 4 / 2), w * (3 * u ** 2 - 2 * u ** 3), w * 6 * u * (1 - u) / tr
        if tr > 0 and tm > T - tr:                  # ramp down, mirror image
            u = (T - tm) / tr
            S_end = T - tr
            return (w * (S_end - tr * (u ** 3 - u ** 4 / 2)), w * (3 * u ** 2 - 2 * u ** 3),
                    -w * 6 * u * (1 - u) / tr)
        return w * (tm - tr / 2), w, 0.0

    def _shape(self, phi: float):
        """Offset p(phi) and its first two derivatives dp/dphi, d2p/dphi2 (N, E, D)."""
        a, v = self.size_m, self.vertical_m
        c, s = math.cos(phi), math.sin(phi)
        if self.shape == "circle":
            p = (a * (c - 1), a * s)
            d1 = (-a * s, a * c)
            d2 = (-a * c, -a * s)
        else:                                       # figure8
            c2, s2 = math.cos(2 * phi), math.sin(2 * phi)
            p = (a * s, a / 2 * s2)
            d1 = (a * c, a * c2)
            d2 = (-a * s, -2 * a * s2)
        # altitude excursion (down is positive): D = -v (1 - cos phi) / 2
        pd, d1d, d2d = -v * (1 - c) / 2, -v * s / 2, -v * c / 2
        return (p[0], p[1], pd), (d1[0], d1[1], d1d), (d2[0], d2[1], d2d)

    def offset(self, t: float):
        """Reference offset from the handover point at time t: (pos, vel, acc), each NED."""
        phi, phid, phidd = self._phase(t)
        p, d1, d2 = self._shape(phi)
        vel = tuple(d * phid for d in d1)
        acc = tuple(dd * phid ** 2 + d * phidd for d, dd in zip(d1, d2))
        return p, vel, acc

    def max_speed(self) -> float:
        ts = np.linspace(self.start_s, self.end_s, 2000)
        return max(math.sqrt(sum(v * v for v in self.offset(t)[1])) for t in ts)

    def max_accel(self) -> float:
        ts = np.linspace(self.start_s, self.end_s, 2000)
        return max(math.sqrt(sum(a * a for a in self.offset(t)[2])) for t in ts)


def trajectory_schedule(origin_ned, yaw: float, traj: Trajectory):
    """Setpoint function for the outer loop: position, velocity and acceleration of the reference."""
    o = np.asarray(origin_ned, float)

    def fn(t: float) -> Setpoint:
        p, v, a = traj.offset(t)
        return Setpoint(tuple(o + np.asarray(p)), yaw, v, a)
    return fn
