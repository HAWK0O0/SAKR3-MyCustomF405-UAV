"""T6: command dropout and safety reactions of the GUIDED + SET_ATTITUDE_TARGET interface.

SAKR3 1.40 kg under the PD outer loop (as T4). Link-loss model: during a dropout the client keeps
computing but sends nothing (no commands, no heartbeats). Events:
  A  1 s dropout in hover               expect: FC holds the last attitude + thrust, no FC reaction
  B  1 s dropout 0.4 s after a 2 m step  expect: FC holds the last (tilted) target -> overshoot
  C  5 s dropout in hover               expect: hold for GUID_TIMEOUT (3 s), then FC levels and
                                                switches to its own climb-rate-0 altitude hold
  D  setpoint 9 m north, fence 6 m       expect: supervisor trips on the predicted stopping point,
                                                FC to BRAKE, vehicle stops within 0.5 m of the fence
After every dropout the outer loop must resume and return to its setpoint.

The takeover case (a pilot / GCS on a second link changes the FC mode) is tests/t6e_takeover.py. It
used to be event E here, with the GCS switching to LAND while this procedure sent LAND itself after
every loop: the check could not tell a client that respects the takeover from one that lands anyway.

Usage (WSL, venv active):
    python sim/tests/t6_dropout.py [--instance N] [--from-data]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_common import RESULTS, fc_time, fly_outer_loop, load, save  # noqa: E402

import numpy as np  # noqa: E402

from analysis.outer_loop import step_metrics  # noqa: E402
from client.supervisor import SafetyLimits  # noqa: E402

PREFIX = "m2_t6"
STEPS = [(0.0, (0, 0, 0)), (12.0, (2.0, 0, 0)), (22.0, (0, 0, 0)), (42.0, (9.0, 0, 0))]
DROPOUTS = {"A_hover_1s": (5.0, 1.0), "B_moving_1s": (12.4, 1.0), "C_hover_5s": (28.0, 5.0)}
FENCE = 6.0
DURATION = 60.0
LIMITS = SafetyLimits(fence_radius_m=FENCE, alt_min_m=2.0, alt_max_m=10.0)


def drop_fn(t: float) -> bool:
    return any(t0 <= t < t0 + L for t0, L in DROPOUTS.values())


def analyze_dropout(data: dict, t_d: float, L: float) -> dict:
    rows, echo = data["rows"], data["echo"]
    tg = fc_time(data, data["guia"].TimeUS).values
    last = rows[(rows.t < t_d) & rows.sent].iloc[-1]
    first = rows[(rows.t >= t_d + L) & rows.sent].iloc[0]
    t_last_fc = tg[tg >= last.t][0]           # FC applied the last command before the dropout
    t_resume_fc = tg[tg >= first.t][0]        # ... and the first one after it
    during = echo[(echo.t > t_last_fc + 0.03) & (echo.t < t_resume_fc)]
    changed = during[np.abs(during.thrust - last.cmd_thrust) > 1e-6]
    end = during[during.t > t_resume_fc - 0.5]
    win = rows[(rows.t >= t_d) & (rows.t < t_d + L + 3)]
    rec = rows[(rows.t >= t_d + L + 4) & (rows.t < t_d + L + 5)]
    e_alt = (win.z - win.sp_z).abs()
    e_hor = np.hypot(win.x - win.sp_x, win.y - win.sp_y)
    return {
        "fc_gap_s": float(t_resume_fc - t_last_fc),
        "fc_reaction_s": float(changed.t.iloc[0] - t_last_fc) if len(changed) else None,
        "last_cmd_tilt_deg": float(np.degrees(max(abs(last.cmd_roll), abs(last.cmd_pitch)))),
        "fc_target_tilt_end_deg": float(np.degrees(np.maximum(end.roll.abs(), end.pitch.abs()).max())),
        "alt_dev_max_m": float(e_alt.max()),
        "horiz_dev_max_m": float(e_hor.max()),
        "speed_at_resume_ms": float(np.hypot(first.vx, first.vy)),
        "recovered_alt_err_m": float((rec.z - rec.sp_z).abs().mean()),
        "recovered_horiz_err_m": float(np.hypot(rec.x - rec.sp_x, rec.y - rec.sp_y).mean()),
    }


def analyze(data: dict) -> dict:
    rows, after, meta = data["rows"], data["after"], data["meta"]
    timeout = meta["params"]["GUID_TIMEOUT"]
    ev = {k: analyze_dropout(data, *v) for k, v in DROPOUTS.items()}
    x0 = meta["origin"][0]
    ev["B_moving_1s"]["north_step"] = step_metrics(rows.t.values, rows.x.values, 12, 22, x0, x0 + 2, band=0.1)
    trip_t = meta["trip_t"]
    brake = after[after["mode"] == "BRAKE"]
    r_after = np.hypot(after.x - meta["origin"][0], after.y - meta["origin"][1])
    ev["D_fence"] = {
        "trip": meta["trip"], "trip_t": trip_t,
        "brake_confirmed_after_s": float(brake.t.iloc[0] - trip_t) if len(brake) else None,
        "max_radius_m": float(r_after.max()), "overrun_m": float(r_after.max() - FENCE),
        "landed": bool(len(after) and not after.armed.iloc[-1]),
    }
    mode = data["mode"]
    before_trip = mode[(mode.TimeUS > meta["t0_ms"] * 1e3) & (fc_time(data, mode.TimeUS) < (trip_t or 1e9))]
    A, B, C, D = ev["A_hover_1s"], ev["B_moving_1s"], ev["C_hover_5s"], ev["D_fence"]
    c = {
        "fc_stayed_guided_until_trip": len(before_trip) == 0,
        "A_fc_held_last_command": A["fc_reaction_s"] is None,
        "A_gap_1s": abs(A["fc_gap_s"] - 1.0) < 0.1,
        "A_alt_dev_lt_20cm": A["alt_dev_max_m"] < 0.2,
        "B_fc_held_last_command": B["fc_reaction_s"] is None,
        "C_fc_reacts_at_GUID_TIMEOUT": C["fc_reaction_s"] is not None and abs(C["fc_reaction_s"] - timeout) < 0.1,
        "C_fc_levels_attitude": C["fc_target_tilt_end_deg"] < 1.0,
        "C_alt_dev_lt_30cm": C["alt_dev_max_m"] < 0.3,
        "all_recovered": all(e["recovered_alt_err_m"] < 0.05 and e["recovered_horiz_err_m"] < 0.1
                             for e in (A, B, C)),
        "D_fence_trip": bool(D["trip"] and D["trip"].startswith("fence") and trip_t > 42),
        "D_brake_within_1s": D["brake_confirmed_after_s"] is not None and D["brake_confirmed_after_s"] < 1.0,
        "D_overrun_lt_0.5m": D["overrun_m"] < 0.5,
        "D_landed": D["landed"],
    }
    c["T6_pass"] = all(c.values())
    return {"events": ev, "checks": c, "params": meta["params"]}


def plot(data: dict, out: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, echo, after, meta = data["rows"], data["echo"], data["after"], data["meta"]
    x0 = meta["origin"][0]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    sent = rows.where(rows.sent)
    for a in ax:
        for name, (t0, L) in DROPOUTS.items():
            a.axvspan(t0, t0 + L, color="#f2c94c", alpha=0.35, lw=0)
        if meta["trip_t"]:
            a.axvline(meta["trip_t"], color="#c0392b", lw=1, ls="--")
        a.spines[["top", "right"]].set_visible(False)
    for name, (t0, L) in DROPOUTS.items():
        ax[0].annotate(name.split("_")[0], (t0 + L / 2, 8.6), ha="center", fontsize=8)
    if meta["trip_t"]:
        ax[0].annotate("D trip", (meta["trip_t"] + 0.3, 8.6), fontsize=8, color="#c0392b")
    ax[0].plot(rows.t, rows.sp_x - x0, color="#999", lw=1, drawstyle="steps-post", label="setpoint")
    ax[0].plot(rows.t, rows.x - x0, color="#2a7ab0", lw=1.2, label="north, outer loop")
    ax[0].plot(after.t, after.x - x0, color="#8e44ad", lw=1.2, label="north, FC BRAKE/LAND")
    ax[0].axhline(FENCE, color="#c0392b", lw=0.8, ls=":", label="supervisor fence")
    ax[0].set_ylabel("north offset [m]")
    ax[0].set_ylim(-1, 9.5)
    ax[0].legend(fontsize=8, frameon=False, loc="center left")
    ax[1].plot(rows.t, -rows.sp_z, color="#999", lw=1, drawstyle="steps-post")
    ax[1].plot(rows.t, -rows.z, color="#2a7ab0", lw=1.2)
    ax[1].plot(after.t, -after.z, color="#8e44ad", lw=1.2)
    ax[1].set_ylabel("altitude [m]")
    ax[2].plot(sent.t, sent.cmd_thrust, color="#2a7ab0", lw=1, label="thrust sent")
    ax[2].plot(echo.t, echo.thrust, color="#c0602a", lw=0.8, ls="--", label="FC active thrust")
    ax[2].plot(echo.t, np.degrees(echo.pitch) / 100 + 0.45, color="#27ae60", lw=0.8,
               label="FC pitch target [deg/100 + 0.45]")
    ax[2].set_ylabel("thrust [-]")
    ax[2].set_xlabel("outer-loop time [s]")
    ax[2].legend(fontsize=8, frameon=False, loc="lower left")
    C = out["events"]["C_hover_5s"]
    held = f"{C['fc_reaction_s']:.2f} s" if C["fc_reaction_s"] is not None else "the whole dropout"
    fig.suptitle(f"T6: command dropouts (shaded) and supervisor trip — in C the FC held the last command for "
                 f"{held}, then levelled (GUID_TIMEOUT {meta['params']['GUID_TIMEOUT']:.0f} s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    instance = int(sys.argv[sys.argv.index("--instance") + 1]) if "--instance" in sys.argv else 1
    if "--from-data" in sys.argv:
        data = load(PREFIX)
    else:
        data = fly_outer_loop("M2_T6_dropout", instance, STEPS, DURATION, LIMITS, drop_fn=drop_fn,
                              log=lambda s: print(f"[T6] {s}", flush=True))
        save(data, PREFIX)
    out = analyze(data)
    (RESULTS / f"{PREFIX}_checks.json").write_text(json.dumps(out, indent=2))
    plot(data, out, RESULTS / f"{PREFIX}_dropout.png")
    print(json.dumps(out, indent=2))
    return 0 if out["checks"]["T6_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
