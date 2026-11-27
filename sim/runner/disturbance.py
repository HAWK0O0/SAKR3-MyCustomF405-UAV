"""Disturbances the client injects into SITL during the outer loop: low-frequency gusts and pushes.

Gusts. SITL's own turbulence (SIM_WIND_TURB) is close to white noise and barely moves the vehicle (M3),
and SIM_WIND_TC is unused in Copter-4.7.3. Here each horizontal gust component is a first-order
Gauss-Markov process (std sigma, correlation time tau), sampled at rate_hz and added to the mean wind;
the client writes the total to SIM_WIND_SPD / SIM_WIND_DIR whenever the outer-loop time (FC clock)
crosses a sample. The whole profile is drawn up front from its own generator seeded with run.seed, so it
depends only on the scenario: not on SITL's random() stream and not on when the samples are applied.
It starts from zero at handover (no wind step at t = 0) and becomes stationary after ~tau.

Pushes. SITL's SIM_SHOVE_X/Y/Z add an acceleration in the body frame for SIM_SHOVE_TIME ms; a push given
in NED is rotated into the body frame with the handover yaw (the outer loop holds that yaw).
"""
from __future__ import annotations

import math

import numpy as np

GUST_SEED_TAG = 0x6057   # second word of the gust generator's seed: independent of other per-run draws


def air_velocity(speed_ms: float, dir_from_deg: float) -> np.ndarray:
    """SITL wind (speed, direction it comes from) -> velocity of the air, (N, E) [m/s]."""
    d = math.radians(dir_from_deg)
    return -speed_ms * np.array([math.cos(d), math.sin(d)])


def wind_params(v_air: np.ndarray) -> tuple[float, float]:
    """Velocity of the air (N, E) -> (SIM_WIND_SPD, SIM_WIND_DIR) with DIR = where the wind comes from."""
    spd = float(math.hypot(v_air[0], v_air[1]))
    return spd, (math.degrees(math.atan2(-v_air[1], -v_air[0])) % 360.0) if spd > 0 else 0.0


class GustProfile:
    def __init__(self, mean_speed_ms: float, mean_dir_deg: float, sigma_ms: float, tau_s: float,
                 rate_hz: float, duration_s: float, seed: int):
        self.sigma, self.tau, self.rate = sigma_ms, tau_s, rate_hz
        n = int(math.ceil(duration_s * rate_hz)) + 2
        rng = np.random.default_rng([int(seed), GUST_SEED_TAG])
        a = math.exp(-1.0 / (rate_hz * tau_s))
        b = sigma_ms * math.sqrt(1 - a * a)
        g = np.zeros((n, 2))
        for k in range(1, n):
            g[k] = a * g[k - 1] + b * rng.standard_normal(2)
        self.t = np.arange(n) / rate_hz
        self.gust = g                                            # (N, E) gust velocity of the air
        self.air = air_velocity(mean_speed_ms, mean_dir_deg) + g
        self.params = [wind_params(v) for v in self.air]         # (SIM_WIND_SPD, SIM_WIND_DIR) per sample

    def table(self):
        return {"t": self.t, "gust_n": self.gust[:, 0], "gust_e": self.gust[:, 1],
                "air_n": self.air[:, 0], "air_e": self.air[:, 1],
                "wind_spd": [p[0] for p in self.params], "wind_dir": [p[1] for p in self.params]}


def gust_hook(profile: GustProfile, set_param, applied: list):
    """Step hook: on each new sample index, write that sample's wind (only the newest if several are due)."""
    state = {"k": -1}

    def hook(t: float):
        k = min(int(t * profile.rate), len(profile.t) - 1)
        if k > state["k"]:
            spd, d = profile.params[k]
            set_param("SIM_WIND_SPD", spd)
            set_param("SIM_WIND_DIR", d)
            applied.append((t, k))
            state["k"] = k
    return hook


def push_params(push: dict, yaw: float) -> dict:
    """NED push {n, e, d [m/s^2], ms} -> SIM_SHOVE_* (body frame at yaw, level), SHOVE_TIME last (it starts the push)."""
    n, e, d = float(push.get("n", 0)), float(push.get("e", 0)), float(push.get("d", 0))
    c, s = math.cos(yaw), math.sin(yaw)
    return {"SIM_SHOVE_X": c * n + s * e, "SIM_SHOVE_Y": -s * n + c * e, "SIM_SHOVE_Z": d,
            "SIM_SHOVE_TIME": int(push["ms"])}
