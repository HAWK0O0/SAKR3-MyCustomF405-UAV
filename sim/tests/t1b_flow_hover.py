"""T1-B (M4): T1 under sensor profile B (optical flow + rangefinder, no GPS), SAKR3 1.40 kg.

The FC alone flies: EKF origin set by the client, GUIDED, arm, takeoff to 5 m, hover 10 s, land.
Checks that profile B is what actually flew and that the FC can fly on it:
  params_applied     every parameter of our files reached the FC (SITL ignores unknown names silently)
  profile_b_active   GPS1_TYPE 0, FLOW_TYPE 10, RNGFND1_TYPE 100, EK3 velocity from flow, height from
                     the rangefinder; EKF reported "fusing optical flow"; no GPS fix on the link
  flight             armed, reached 5 m, hovered 10 s (FC time), landed and disarmed, .BIN written
  hover_alt          EKF hover altitude within 0.5 m of 5 m, and true height too
  ekf_height         EKF height vs true height RMS < 3 cm during the hover (rangefinder height)
Reported, not judged: the true horizontal drift during the hover and the EKF horizontal error.

Usage (WSL, venv active):  python sim/tests/t1b_flow_hover.py [--instance N] [--seed S]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from airframes.make_airframe import Variant  # noqa: E402
from analysis.binlog import read_messages  # noqa: E402
from analysis.truth import parse_home, truth_frame  # noqa: E402
from client import mav  # noqa: E402
from client.outer_loop import request_streams  # noqa: E402
from runner.seed import build as build_seed, sitl_env  # noqa: E402
from sakr3 import param_check, sakr3_sitl  # noqa: E402

RESULTS = SIM / "results"
SENSORS = SIM / "params/sensors_flow.parm"
TARGET_ALT, HOVER_S = 5.0, 10.0
PROFILE = {"GPS1_TYPE": 0, "FLOW_TYPE": 10, "RNGFND1_TYPE": 100, "EK3_SRC1_POSXY": 0, "EK3_SRC1_VELXY": 5,
           "EK3_SRC1_POSZ": 2}


def fly(instance: int, seed: int) -> tuple[dict, pd.DataFrame, list[str], Path, str]:
    build_seed()
    sitl = sakr3_sitl("T1B_flow_hover", Variant("m1.40", 1.40), instance, extra_parms=[SENSORS])
    sitl.env = sitl_env(seed)
    res, rows, texts = {}, [], []
    phase = "boot"
    try:
        sitl.start()
        m = mav.connect(sitl.connection)
        request_streams(m)
        mav.set_message_rate(m, mav.M.MAVLINK_MSG_ID_GPS_RAW_INT, 2)
        res["origin_wait_s"] = round(mav.set_ekf_origin(m, *parse_home(sitl.home), log=texts.append), 1)
        res["ekf_wait_s"] = round(mav.wait_ekf_position(m, log=texts.append, relative=True), 1)
        mav.set_mode(m, "GUIDED")
        res["arm_wait_s"] = round(mav.arm_with_retry(m), 1)
        mav.takeoff(m, TARGET_ALT, log=texts.append)
        phase, t_phase, t0 = "climb", None, time.time()
        gps_fix = []
        while time.time() - t0 < 180:
            msg = m.recv_match(type=["LOCAL_POSITION_NED", "STATUSTEXT", "GPS_RAW_INT"], blocking=True, timeout=1)
            if msg is None:
                continue
            typ = msg.get_type()
            if typ == "STATUSTEXT":
                texts.append(msg.text)
                continue
            if typ == "GPS_RAW_INT":
                gps_fix.append(msg.fix_type)
                continue
            rows.append({"t_ms": msg.time_boot_ms, "phase": phase, "x": msg.x, "y": msg.y, "z": msg.z,
                         "vx": msg.vx, "vy": msg.vy, "vz": msg.vz})
            if phase in ("climb", "hover") and not m.motors_armed():
                texts.append(f"unexpected disarm during {phase}")
                break
            if phase == "climb" and -msg.z > 0.95 * TARGET_ALT:
                phase, t_phase = "hover", msg.time_boot_ms
            elif phase == "hover" and (msg.time_boot_ms - t_phase) / 1e3 > HOVER_S:
                mav.set_mode(m, "LAND")
                phase = "land"
            elif phase == "land" and not m.motors_armed():
                phase = "done"
                break
        res["gps_fix_types_seen"] = sorted(set(gps_fix))
        m.close()
        time.sleep(1.0)
    finally:
        sitl.stop()
    res["final_phase"] = phase
    return res, pd.DataFrame(rows), texts, sitl.run_dir, sitl


def analyze(res: dict, rows: pd.DataFrame, texts: list[str], run_dir: Path, sitl) -> dict:
    out = dict(res)
    bins = sorted((run_dir / "logs").glob("*.BIN"))
    out["bin_logs"] = [b.name for b in bins]
    msgs = read_messages(bins[-1], ["SIM", "SIM2", "ORGN", "PARM", "RFND"])
    first = msgs["PARM"].drop_duplicates("Name", keep="first")
    fc = dict(zip(first.Name, first.Value))
    out["param_problems"] = param_check(fc, sitl.extra_defaults)
    out["profile_params"] = {k: fc.get(k) for k in PROFILE}
    truth = truth_frame(msgs, sitl.home, vertical_ref="rest")
    h = rows[rows.phase == "hover"].copy()
    tq = h.t_ms.values * 1e3
    for k in ("n", "e", "d"):
        h[f"t_{k}"] = np.interp(tq, truth.TimeUS.values, truth[k].values)
    out["hover_samples"] = len(h)
    out["hover_alt_ekf_mean_m"] = float(-h.z.mean())
    out["hover_alt_true_mean_m"] = float(-h.t_d.mean())
    out["ekf_height_err_rms_m"] = float(np.sqrt(np.mean((h.z - h.t_d) ** 2)))
    eh = np.hypot(h.x - h.t_n, h.y - h.t_e)
    out["ekf_horiz_err_hover_start_m"], out["ekf_horiz_err_hover_end_m"] = float(eh.iloc[0]), float(eh.iloc[-1])
    out["true_horiz_drift_hover_m"] = float(np.hypot(h.t_n.iloc[-1] - h.t_n.iloc[0], h.t_e.iloc[-1] - h.t_e.iloc[0]))
    out["ekf_horiz_drift_hover_m"] = float(np.hypot(h.x.iloc[-1] - h.x.iloc[0], h.y.iloc[-1] - h.y.iloc[0]))
    r = msgs["RFND"]
    out["rangefinder_good_frac_in_flight"] = float((r.Stat[r.Dist > 0.5] == 4).mean())
    flow_txt = [t for t in texts if "optical flow" in t or "relative aiding" in t]
    out["fc_text_flow"] = sorted(set(flow_txt))
    checks = {
        "params_applied": not out["param_problems"],
        "profile_b_active": (all(fc.get(k) == v for k, v in PROFILE.items()) and bool(flow_txt)
                             and all(f < 2 for f in out["gps_fix_types_seen"])),
        "flight": out["final_phase"] == "done" and bool(bins) and len(h) > 0,
        "hover_alt": (abs(out["hover_alt_ekf_mean_m"] - TARGET_ALT) < 0.5
                      and abs(out["hover_alt_true_mean_m"] - TARGET_ALT) < 0.5),
        "ekf_height": out["ekf_height_err_rms_m"] < 0.03,
    }
    checks["T1B_pass"] = all(checks.values())
    out["checks"] = checks
    out["run_dir"] = str(run_dir)
    return out


def main() -> int:
    a = sys.argv
    instance = int(a[a.index("--instance") + 1]) if "--instance" in a else 0
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 1
    res, rows, texts, run_dir, sitl = fly(instance, seed)
    out = analyze(res, rows, texts, run_dir, sitl)
    out["seed"] = seed
    RESULTS.mkdir(exist_ok=True)
    rows.to_csv(run_dir / "t1b_local_position.csv", index=False)
    (RESULTS / "m4_t1b.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    return 0 if out["checks"]["T1B_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
