"""Simulator ground truth from the .BIN log, in the frame the outer loop works in (EKF origin, NED).

SITL logs its true state (Copter-4.7.3):
  SIM2 : PN, PE, PD position relative to SITL home [m, double], VN, VE, VD [m/s], ASpdU achieved speedup
  SIM  : Roll, Pitch, Yaw [deg] (and Lat/Lng in 1e-7 deg, too coarse for mm-level hover statistics)
The outer loop's setpoints are in the EKF frame (LOCAL_POSITION_NED, relative to the EKF origin), so
the truth is shifted by the EKF origin's offset from SITL home, from ORGN (Type 0 = EKF origin);
vertically by the landed position instead when the origin was set by the client (profile B, no GPS).
Metrics computed on this truth therefore include the EKF's estimation error, like a real flight
judged by an external reference (motion capture / RTK).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LATLON_TO_M = 0.011131884502145034     # ArduPilot LOCATION_SCALING_FACTOR: m per 1e-7 deg latitude
ATTACHED = ("n", "e", "d", "vn", "ve", "vd", "roll", "pitch")   # attach() adds these as t_<name>


def parse_home(home: str) -> tuple[float, float, float]:
    lat, lon, alt, *_ = (float(x) for x in home.split(","))
    return lat, lon, alt


def origin_offset_ned(orgn: pd.DataFrame, home: str) -> np.ndarray:
    """EKF origin relative to SITL home, NED [m]."""
    o = orgn[orgn.Type == 0].iloc[-1]
    lat_h, lon_h, alt_h = parse_home(home)
    dn = (o.Lat - lat_h) * 1e7 * LATLON_TO_M
    de = (o.Lng - lon_h) * 1e7 * LATLON_TO_M * math.cos(math.radians((o.Lat + lat_h) / 2))
    return np.array([dn, de, -(o.Alt - alt_h)])


def rest_down(s2: pd.DataFrame, window_s: float = 0.5) -> float:
    """True down position (rel. to SITL home) of the landed vehicle: median of SIM2.PD over the first
    window_s of the log. Logging starts at arming, before the takeoff command."""
    w = s2[s2.TimeUS <= s2.TimeUS.iloc[0] + window_s * 1e6]
    if w.PD.max() - w.PD.min() > 0.01 or (w.VD.abs() > 0.05).any():
        raise ValueError("vehicle not at rest at the start of the log")
    return float(w.PD.median())


def truth_frame(msgs: dict[str, pd.DataFrame], home: str, vertical_ref: str = "origin") -> pd.DataFrame:
    """Full-rate truth: TimeUS, n, e, d, vn, ve, vd (EKF-origin frame), roll, pitch, yaw [rad], aspd.

    vertical_ref "origin": shift by the EKF origin's altitude (ORGN), right when the origin comes from
    GPS at rest. "rest": the vehicle's true landed position is d = 0; use it when the client set the
    origin (no GPS). The EKF's height zero is then where it initialised on the ground, not the origin
    altitude, which is SITL home, and the SITL ground sits 0.1 m above home at CMAC (terrain data).
    """
    s2, sim = msgs["SIM2"], msgs["SIM"]
    off = origin_offset_ned(msgs["ORGN"], home)
    if vertical_ref == "rest":
        off[2] = rest_down(s2)
    elif vertical_ref != "origin":
        raise ValueError(vertical_ref)
    t = s2.TimeUS.values
    out = pd.DataFrame({"TimeUS": t, "n": s2.PN.values - off[0], "e": s2.PE.values - off[1],
                        "d": s2.PD.values - off[2], "vn": s2.VN.values, "ve": s2.VE.values,
                        "vd": s2.VD.values, "aspd": s2.ASpdU.values})
    for k in ("Roll", "Pitch"):
        out[k.lower()] = np.radians(np.interp(t, sim.TimeUS.values, sim[k].values))
    yaw = np.unwrap(np.radians(sim.Yaw.values))
    out["yaw"] = np.interp(t, sim.TimeUS.values, yaw)
    out.attrs["origin_offset_ned"] = off.tolist()
    return out


def attach(rows: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Truth interpolated at each control step's state time (rows.t_ms, FC clock): columns t_<name>."""
    out = rows.drop(columns=[f"t_{k}" for k in ATTACHED], errors="ignore")   # re-analysis: attach again
    tq = rows.t_ms.values * 1e3
    for k in ATTACHED:
        out[f"t_{k}"] = np.interp(tq, truth.TimeUS.values, truth[k].values)
    return out
