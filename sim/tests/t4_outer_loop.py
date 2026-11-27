"""T4: external PD outer loop through GUIDED + SET_ATTITUDE_TARGET (thrust mode) at 50 Hz.

SAKR3 1.40 kg. ArduCopter takes off to 5 m, then the Python outer loop takes over:
hold 8 s, altitude +1 m, back, 2 m north, back (12 s each). Checks:
  interface : FC params applied (GUID_OPTIONS=8, ATC_ANGLE_BOOST=0); every FC-logged target (GUIA)
              is in thrust mode; FC stays in GUIDED; supervisor never trips
  timing    : commands reach the FC at 50 Hz (GUIA intervals), latency state->FC <= one period (20 ms)
  tracking  : hover hold, step responses settle without large overshoot

Usage (WSL, venv active):
    python sim/tests/t4_outer_loop.py [--instance N] [--from-data]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_common import RESULTS, fc_time, fly_outer_loop, load, save  # noqa: E402

import numpy as np  # noqa: E402

from analysis.outer_loop import interval_stats, match_commands, step_metrics  # noqa: E402
from client.supervisor import SafetyLimits  # noqa: E402

PREFIX = "m2_t4"
STEPS = [(0.0, (0, 0, 0)), (8.0, (0, 0, -1.0)), (20.0, (0, 0, 0)), (32.0, (2.0, 0, 0)), (44.0, (0, 0, 0))]
DURATION = 56.0
LIMITS = SafetyLimits(fence_radius_m=10.0, alt_min_m=2.0, alt_max_m=10.0)


def analyze(data: dict) -> dict:
    rows, guia, meta = data["rows"], data["guia"], data["meta"]
    t_end_ms = rows.t_ms.iloc[-1]
    guia = guia[(guia.TimeUS >= meta["t0_ms"] * 1e3) & (guia.TimeUS <= (t_end_ms + 100) * 1e3)]
    lat = match_commands(rows, guia)
    t = rows.t.values
    x0, z0 = meta["origin"][0], meta["origin"][2]
    alt, north = -rows.z.values, rows.x.values
    hold = (t >= 3) & (t < 8)
    r = {
        "params": meta["params"],
        "trip": meta["trip"],
        "fc_rate": interval_stats(fc_time(data, guia.TimeUS).values),
        "client_rate": interval_stats(t[rows.sent.values]),
        "latency_ms": {"p50": float(lat.latency_ms.median()), "p95": float(lat.latency_ms.quantile(0.95)),
                       "max": float(lat.latency_ms.max()), "matched_frac": len(lat) / int(rows.sent.sum())},
        "compute_us_p99": float(rows.compute_us.quantile(0.99)),
        "clamped_frac": float(rows.clamped.mean()),
        "hold": {"alt_std": float(alt[hold].std()),
                 "horiz_std": float(np.sqrt(rows.x.values[hold].var() + rows.y.values[hold].var())),
                 "alt_err_mean": float(alt[hold].mean() + z0)},
        "alt_up": step_metrics(t, alt, 8, 20, -z0, -z0 + 1),
        "alt_down": step_metrics(t, alt, 20, 32, -z0 + 1, -z0),
        "north_out": step_metrics(t, north, 32, 44, x0, x0 + 2, band=0.1),
        "north_back": step_metrics(t, north, 44, 56, x0 + 2, x0, band=0.1),
    }
    mode = data["mode"]
    in_loop = mode[(mode.TimeUS > meta["t0_ms"] * 1e3) & (mode.TimeUS < t_end_ms * 1e3)]
    p = meta["params"]
    steps = [r[k] for k in ("alt_up", "alt_down", "north_out", "north_back")]
    c = {
        "params_applied": p["GUID_OPTIONS"] == 8 and p["ATC_ANGLE_BOOST"] == 0,
        "fc_in_thrust_mode": bool((guia.ClimbRt == 0).all() and (guia.Thrust > 0).all()),
        "fc_stayed_guided": len(in_loop) == 0,
        "no_trip": meta["trip"] is None and meta["end_reason"] == "duration",
        "fc_rate_50hz": abs(r["fc_rate"]["rate_hz"] - 50) < 1.0,
        "fc_interval_p99_lt_30ms": r["fc_rate"]["interval_p99_ms"] < 30,
        "fc_max_gap_lt_100ms": r["fc_rate"]["interval_max_ms"] < 100,
        "latency_p95_le_20ms": r["latency_ms"]["p95"] <= 20,
        "latency_matched_gt_90pct": r["latency_ms"]["matched_frac"] > 0.9,
        "hold_std_lt_5cm": r["hold"]["alt_std"] < 0.05 and r["hold"]["horiz_std"] < 0.05,
        "steps_overshoot_lt_25pct": all(s["overshoot_pct"] < 25 for s in steps),
        "steps_final_err_lt_5cm": all(s["final_err"] < 0.05 for s in steps),
        "steps_rise_lt_4s": all(s["rise_s"] < 4 for s in steps),
    }
    c["T4_pass"] = all(c.values())
    return {"metrics": r, "checks": c, "latency": lat}


def plot(data: dict, out: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, echo, lat = data["rows"], data["echo"], out["latency"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    a = ax[0, 0]
    a.plot(rows.t, -rows.sp_z, color="#999", lw=1, drawstyle="steps-post", label="setpoint")
    a.plot(rows.t, -rows.z, color="#2a7ab0", lw=1.2, label="altitude (EKF)")
    a.set_ylabel("altitude [m]")
    a.legend(fontsize=8, frameon=False)
    a = ax[1, 0]
    a.plot(rows.t, rows.sp_x - rows.sp_x.iloc[0], color="#999", lw=1, drawstyle="steps-post", label="setpoint")
    a.plot(rows.t, rows.x - rows.sp_x.iloc[0], color="#2a7ab0", lw=1.2, label="north (EKF)")
    a.set_ylabel("north offset [m]")
    a.set_xlabel("outer-loop time [s]")
    a.legend(fontsize=8, frameon=False)
    a = ax[0, 1]
    a.plot(rows.t, rows.cmd_thrust, color="#2a7ab0", lw=1, label="thrust sent")
    a.plot(echo.t, echo.thrust, color="#c0602a", lw=0.8, ls="--", label="FC active thrust (ATTITUDE_TARGET)")
    a.plot(rows.t, np.degrees(rows.cmd_pitch) / 100 + 0.45, color="#5a5", lw=0.8,
           label="pitch cmd [deg/100 + 0.45]")
    a.set_ylabel("thrust [-]")
    a.legend(fontsize=8, frameon=False)
    a = ax[1, 1]
    a.hist(lat.latency_ms, bins=np.arange(0, max(25, lat.latency_ms.max() + 1), 0.5), color="#2a7ab0")
    m = out["metrics"]["latency_ms"]
    a.axvline(20, color="#c0602a", lw=1, ls="--")
    a.set_xlabel("state sample -> FC applies command [ms, FC clock]")
    a.set_ylabel("commands")
    a.set_title(f"latency p50 {m['p50']:.1f} ms, p95 {m['p95']:.1f} ms, max {m['max']:.1f} ms", fontsize=9)
    for a in ax.flat:
        a.spines[["top", "right"]].set_visible(False)
    fr = out["metrics"]["fc_rate"]
    fig.suptitle(f"T4: PD outer loop via SET_ATTITUDE_TARGET, SAKR3 1.40 kg — commands at FC "
                 f"{fr['rate_hz']:.2f} Hz (p99 interval {fr['interval_p99_ms']:.1f} ms)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    instance = int(sys.argv[sys.argv.index("--instance") + 1]) if "--instance" in sys.argv else 0
    if "--from-data" in sys.argv:
        data = load(PREFIX)
    else:
        data = fly_outer_loop("M2_T4_outer_loop", instance, STEPS, DURATION, LIMITS,
                              log=lambda s: print(f"[T4] {s}", flush=True))
        save(data, PREFIX)
    out = analyze(data)
    out["latency"].to_csv(RESULTS / f"{PREFIX}_latency.csv", index=False)
    (RESULTS / f"{PREFIX}_checks.json").write_text(json.dumps(
        {"metrics": out["metrics"], "checks": out["checks"]}, indent=2))
    plot(data, out, RESULTS / f"{PREFIX}_outer_loop.png")
    print(json.dumps({"metrics": out["metrics"], "checks": out["checks"]}, indent=2))
    return 0 if out["checks"]["T4_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
