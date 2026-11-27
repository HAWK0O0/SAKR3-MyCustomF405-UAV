"""Analysis of outer-loop runs: command timing at the FC, latency, step responses.

Latency is measured entirely on the FC clock, so it needs no clock sync and works at any speedup
and on the real vehicle:
    latency = GUIA.TimeUS (FC processed the SET_ATTITUDE_TARGET, mode_guided.cpp set_angle)
              - time_boot_ms of the LOCAL_POSITION_NED sample the command was computed from
It covers FC->client transport, client compute, client->FC transport and the wait for the FC's
next MAVLink receive slot. Commands are paired by their float32 thrust value, which the FC logs
unchanged; only values that occur exactly once on both sides are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def match_commands(rows: pd.DataFrame, guia: pd.DataFrame) -> pd.DataFrame:
    sent = rows[rows.sent].copy()
    sent["key"] = sent.cmd_thrust.astype(np.float32)
    g = guia.copy()
    g["key"] = g.Thrust.astype(np.float32)
    cs, cg = sent.key.value_counts(), g.key.value_counts()
    keys = set(cs[cs == 1].index) & set(cg[cg == 1].index)
    a = sent[sent.key.isin(keys)].set_index("key")
    b = g[g.key.isin(keys)].set_index("key")[["TimeUS"]]
    j = a.join(b, how="inner").reset_index()
    j["latency_ms"] = j.TimeUS / 1e3 - j.t_ms
    return j.sort_values("t").reset_index(drop=True)


def interval_stats(t_s: np.ndarray) -> dict:
    d = np.diff(np.asarray(t_s, float)) * 1e3
    return {"rate_hz": float(1e3 / d.mean()), "interval_p50_ms": float(np.percentile(d, 50)),
            "interval_p99_ms": float(np.percentile(d, 99)), "interval_max_ms": float(d.max()),
            "n": int(len(d) + 1)}


def step_metrics(t: np.ndarray, y: np.ndarray, t_step: float, t_end: float, y0: float, y1: float,
                 band: float = 0.05) -> dict:
    seg = (t >= t_step) & (t < t_end)
    ts, ys = t[seg], y[seg]
    frac = (ys - y0) / (y1 - y0)
    out = {"step": y1 - y0}
    i10, i90 = np.argmax(frac >= 0.1), np.argmax(frac >= 0.9)
    out["rise_s"] = float(ts[i90] - ts[i10]) if frac.max() >= 0.9 else float("nan")
    out["overshoot_pct"] = float(max(0.0, frac.max() - 1.0) * 100)
    out["final_err"] = float(abs(ys[ts > t_end - 2.0].mean() - y1))
    outside = np.abs(ys - y1) > band
    out["settle_s"] = float(ts[np.where(outside)[0][-1]] - t_step) if outside.any() else 0.0
    return out
