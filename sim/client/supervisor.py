"""Client-side safety supervisor for the outer loop.

Three kinds of protection, all independent of the control law:
  fault : a command that is not a finite number in every field is a controller fault. It is never
          sent (not even in a "safe" substitute form: the law's state is unknown) and trips the loop
          in the same step
  clamp : every command is limited before it is sent: total tilt (angle between the thrust axis and
          the vertical, acos(cos roll * cos pitch), the same measure the trip uses) and thrust range
  trip  : the outer loop stops and hands the vehicle back to an FC mode (default BRAKE, which
          holds position without pilot input; then the test lands) when the state leaves the
          safe box, the measured tilt is excessive, state messages stop, the controller faults, or
          the FC itself has left GUIDED (pilot or FC failsafe took over; then no mode is commanded)

The horizontal fence trips on the *predicted stopping point* of the fallback mode, not only on
the current position. ArduCopter 4.7 BRAKE (mode_brake.cpp -> AC_PosControl shape_vel_accel)
drives the velocity to zero with accel_target = -(jerk/accel_max) * v, i.e. time constant
BRAKE_MODE_DECEL_RATE / PSC_NE_JERK = 7.5 / 5 = 1.5 s, and changes the acceleration at most at
PSC_NE_JERK, starting from zero desired acceleration (NE_init_controller zeroes it).
brake_stop_point() integrates exactly that, plus a short delay for the switch and attitude lag.
Checked against two SITL BRAKE stops (T6, SAKR3 1.40 kg): 1.63 m/s -> predicted 6.72 / actual
6.54 m from centre, 3.01 m/s -> 6.84 / 6.72 m (both slightly conservative). Stopping distance is
~1.6 m at 1 m/s, ~3.4 m at 2 m/s, ~5.3 m at 3 m/s.

The FC-side protections stay active underneath (GUID_TIMEOUT, ANGLE_MAX, failsafes).
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from .controller import Command, State


def brake_stop_point(pos, vel, accel_max: float = 7.5, jerk_max: float = 5.0, delay_s: float = 0.1,
                     dt: float = 0.01, t_max: float = 10.0) -> tuple[float, float]:
    """Where BRAKE stops, horizontal NE [m]: jerk-limited first-order velocity shaping (see module doc)."""
    pn, pe = pos[0] + vel[0] * delay_s, pos[1] + vel[1] * delay_s
    vn, ve = vel
    an, ae = 0.0, 0.0
    kpa = jerk_max / accel_max
    t = 0.0
    while t < t_max and math.hypot(vn, ve) > 0.02:
        tn, te = -kpa * vn, -kpa * ve
        h = math.hypot(tn, te)
        if h > accel_max:
            tn, te = tn * accel_max / h, te * accel_max / h
        dn, de = tn - an, te - ae
        d = math.hypot(dn, de)
        if d > jerk_max * dt:
            dn, de = dn * jerk_max * dt / d, de * jerk_max * dt / d
        an, ae = an + dn, ae + de
        vn, ve = vn + an * dt, ve + ae * dt
        pn, pe = pn + vn * dt, pe + ve * dt
        t += dt
    return pn, pe


@dataclass
class SafetyLimits:
    max_tilt_deg: float = 25.0       # command clamp on the total tilt (not per axis)
    thrust_min: float = 0.05         # command clamp
    thrust_max: float = 0.95
    fence_radius_m: float = 15.0     # trip: horizontal distance of position or predicted stop point
    brake_accel: float = 7.5         # BRAKE_MODE_DECEL_RATE_MSS (ArduCopter config.h)
    brake_jerk: float = 5.0          # PSC_NE_JERK
    brake_delay_s: float = 0.1       # mode switch + attitude lag
    alt_min_m: float = 1.0           # trip: altitude above EKF origin
    alt_max_m: float = 20.0
    max_att_deg: float = 45.0        # trip: measured tilt
    state_timeout_s: float = 0.3     # trip: no new position sample for this long (wall clock)


class Supervisor:
    def __init__(self, limits: SafetyLimits, centre_ne: tuple[float, float]):
        self.lim = limits
        self.centre = centre_ne

    @staticmethod
    def command_fault(cmd: Command) -> str | None:
        """Reason if the command must not be sent at all (a non-finite field), else None."""
        bad = [f"{k}={v}" for k, v in (("roll", cmd.roll), ("pitch", cmd.pitch), ("yaw", cmd.yaw),
                                       ("thrust", cmd.thrust))
               if not isinstance(v, numbers.Real) or not math.isfinite(v)]   # Real: numpy scalars too
        return f"controller fault: non-finite command ({', '.join(bad)})" if bad else None

    def clamp(self, cmd: Command) -> tuple[Command, bool]:
        """Limit total tilt and thrust; returns (command, changed). Call command_fault() first."""
        fault = self.command_fault(cmd)
        if fault:
            raise ValueError(fault)
        lim = self.lim
        roll, pitch = cmd.roll, cmd.pitch
        # thrust axis in the yaw-aligned frame: forward sin(p)cos(r), right -sin(r), up cos(p)cos(r).
        # Over the limit, the axis is brought back onto the cone of max_tilt_deg in the same direction.
        fwd, right = math.sin(pitch) * math.cos(roll), -math.sin(roll)
        up = math.cos(pitch) * math.cos(roll)
        t_max = math.radians(lim.max_tilt_deg)
        if math.acos(max(-1.0, min(1.0, up))) > t_max:
            h = math.hypot(fwd, right)
            k = math.sin(t_max) / h if h > 1e-9 else 0.0      # inverted exactly: no direction, level
            roll = math.asin(max(-1.0, min(1.0, -right * k)))
            pitch = math.atan2(fwd * k, math.cos(t_max))
        thrust = min(max(cmd.thrust, lim.thrust_min), lim.thrust_max)
        out = Command(roll, pitch, cmd.yaw, thrust)
        return out, (roll, pitch, thrust) != (cmd.roll, cmd.pitch, cmd.thrust)

    def check(self, s: State) -> str | None:
        lim = self.lim
        r = math.hypot(s.pos[0] - self.centre[0], s.pos[1] - self.centre[1])
        sn, se = brake_stop_point(s.pos[:2], s.vel[:2], lim.brake_accel, lim.brake_jerk, lim.brake_delay_s)
        r_stop = math.hypot(sn - self.centre[0], se - self.centre[1])
        if max(r, r_stop) > lim.fence_radius_m:
            return (f"fence: at {r:.2f} m, predicted stop {r_stop:.2f} m from centre "
                    f"> {lim.fence_radius_m} m")
        if not lim.alt_min_m <= s.alt <= lim.alt_max_m:
            return f"altitude {s.alt:.2f} m outside [{lim.alt_min_m}, {lim.alt_max_m}]"
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(s.roll) * math.cos(s.pitch)))))
        if tilt > lim.max_att_deg:
            return f"tilt {tilt:.1f} deg > {lim.max_att_deg}"
        return None
