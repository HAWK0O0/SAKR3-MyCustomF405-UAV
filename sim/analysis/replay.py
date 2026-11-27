"""Replay export: finished run directories -> data files for the replay viewer (analysis/replay_viewer.html).

The viewer is one self-contained HTML page (no internet, no server: open it by double-click). It shows a
run in 3D (true path, reference, EKF estimate, the vehicle at its true attitude, wind, pushes), with a
timeline and time-series plots, and can overlay a second run. Everything it shows comes from the files
the metrics were computed from: rows.csv.gz (outer loop, truth attached), the .BIN (whole-flight truth
and modes), after.csv.gz, gusts.csv.gz, manifest.json, metrics.json.

Frame: positions relative to the handover point, east / north / up [m]; time relative to the start of
the outer loop [s] (takeoff is at negative times).

Usage (WSL, venv active):
    python sim/analysis/replay.py results/suite_core             # a batch -> results/suite_core/replay/
    python sim/analysis/replay.py --run ~/uavsim_runs/adhoc/<id>  # single runs -> results/replay/
then open <out>/index.html in a browser.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pymavlink import mavutil  # noqa: E402

from analysis.binlog import read_messages  # noqa: E402
from analysis.truth import truth_frame  # noqa: E402
from sitl_runner import DEFAULT_HOME  # noqa: E402

VIEWER = SIM / "analysis/replay_viewer.html"
RESULTS = SIM / "results"
TRUTH_HZ = 25.0
KEY_METRICS = ["iae_m_s", "hold_rms_h_m", "hold_rms_v_m", "traj_rms_m", "traj_max_m", "ev0_max_dev_m",
               "ev0_recover_s", "ev1_max_dev_m", "ev2_max_dev_m", "seg1_overshoot_pct", "seg3_overshoot_pct",
               "max_tilt_deg", "thrust_p99", "est_err_h_rms_m", "est_err_v_rms_m", "est_drift_h_m",
               "lat_p95_ms", "cmd_rate_hz"]
MODE_NAMES = dict(mavutil.mode_mapping_acm)          # ArduCopter mode number -> name


def _r(a, nd=4):
    """Array -> list of rounded floats (None for NaN), compact JSON."""
    a = np.asarray(a, float)
    return [None if not math.isfinite(v) else round(float(v), nd) for v in a]


def _num(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def export_run(rd: str | Path, group: str | None = None) -> dict:
    rd = Path(rd)
    man = json.loads((rd / "manifest.json").read_text())
    met = json.loads((rd / "metrics.json").read_text())
    summ, metrics = met["summary"], met["metrics"]
    sc, loop, hand = man["scenario"], man.get("loop") or {}, man["handover"]
    t0_ms = loop.get("t0_ms")
    o = np.asarray(hand["origin_ned"] or (0.0, 0.0, 0.0), float)   # handover point, EKF frame (NED)

    def enu(n, e, d):                    # EKF-frame NED -> relative to handover, east / north / up
        return np.asarray(e) - o[1], np.asarray(n) - o[0], -(np.asarray(d) - o[2])

    out = {"id": man["run_id"], "batch": man.get("batch"), "group": group,
           "meta": {"name": sc.get("name"), "description": (sc.get("description") or "").strip(),
                    "task_type": sc["task"].get("type", "steps"), "mass_kg": sc["vehicle"]["mass_kg"],
                    "derate": sc["vehicle"]["derate"], "sensors": sc["environment"]["sensors"],
                    "wind": sc["environment"]["wind"], "gusts": sc["environment"].get("gusts"),
                    "law": sc["controller"]["law"], "law_args": sc["controller"].get("args"),
                    "seed": man.get("seed"), "speedup": man.get("speedup"),
                    "hover_thrust": hand.get("hover_thrust"), "alt_m": sc["task"]["alt_m"],
                    "duration_s": sc["task"]["duration_s"], "band_m": sc["metrics"]["band_m"],
                    "outcome": summ.get("outcome"), "reason": summ.get("reason"),
                    "completed": summ.get("completed", summ.get("outcome") == "ok"),   # older runs: from outcome
                    "pass": summ.get("pass"), "landed": summ.get("landed"),
                    "pass_fails": summ.get("pass_fails"), "pass_bounds": sc.get("pass") or {},
                    "trip": loop.get("trip"), "trip_t": loop.get("trip_t"),
                    "takeover": (man.get("takeover") or {}).get("mode"), "wall_s": man.get("wall_s")},
           "metrics": {k: _num(v) for k, v in metrics.items() if not isinstance(v, (list, dict))},
           "key_metrics": [k for k in KEY_METRICS if k in metrics]}

    # outer loop at 50 Hz: truth (attached), reference, EKF, command
    rows_f = rd / "rows.csv.gz"
    if rows_f.exists():
        r = pd.read_csv(rows_f)
        x, y, z = enu(r.t_n, r.t_e, r.t_d) if "t_n" in r else (None, None, None)
        sx, sy, sz = enu(r.sp_x, r.sp_y, r.sp_z)
        ex, ey, ez = enu(r.x, r.y, r.z)
        out["loop"] = {"t": _r(r.t, 3), "x": _r(x) if x is not None else None, "y": _r(y) if y is not None else None,
                       "z": _r(z) if z is not None else None, "sx": _r(sx), "sy": _r(sy), "sz": _r(sz),
                       "ex": _r(ex), "ey": _r(ey), "ez": _r(ez),
                       "roll": _r(np.degrees(r.t_roll if "t_roll" in r else r.roll), 2),
                       "pitch": _r(np.degrees(r.t_pitch if "t_pitch" in r else r.pitch), 2),
                       "yaw": _r(np.degrees(r.yaw), 2),
                       "croll": _r(np.degrees(r.cmd_roll), 2), "cpitch": _r(np.degrees(r.cmd_pitch), 2),
                       "thrust": _r(r.cmd_thrust, 4), "clamped": [int(v) for v in r.clamped.astype(bool)]}

    # whole flight (takeoff, loop, landing): truth from the .BIN, modes
    bins = sorted((rd / "logs").glob("*.BIN"))
    if bins and t0_ms is not None:
        msgs = read_messages(bins[-1], ["SIM", "SIM2", "ORGN", "MODE"])
        home = (man.get("sitl") or {}).get("home", DEFAULT_HOME)   # manifests before 2026-11-26 lack "sitl"
        truth = truth_frame(msgs, home, vertical_ref="rest" if hand.get("ekf_origin_set") else "origin")
        t = truth.TimeUS.values / 1e6 - t0_ms / 1e3
        keep = np.r_[0, np.where(np.diff(np.floor(t * TRUTH_HZ)) > 0)[0] + 1]
        tr = truth.iloc[keep]
        fx, fy, fz = enu(tr.n, tr.e, tr.d)
        out["flight"] = {"t": _r(t[keep], 3), "x": _r(fx), "y": _r(fy), "z": _r(fz),
                         "roll": _r(np.degrees(tr.roll), 2), "pitch": _r(np.degrees(tr.pitch), 2),
                         "yaw": _r(np.degrees(tr.yaw), 2)}
        d_rest = float(truth.d.iloc[0])                     # landed position = the floor under the vehicle
        out["ground_z"] = round(-(d_rest - o[2]), 3)
        mo, modes = msgs["MODE"], []
        for tm, n in zip(mo.TimeUS, mo.Mode):           # the log repeats the mode at its start: keep changes
            name = MODE_NAMES.get(int(n), str(int(n)))
            if not modes or modes[-1]["mode"] != name:
                modes.append({"t": round(float(tm) / 1e6 - t0_ms / 1e3, 3), "mode": name})
        out["modes"] = modes
    else:
        out["ground_z"] = round(-float(sc["task"]["alt_m"]), 3)

    # landing / BRAKE hold after the loop: EKF only
    af = rd / "after.csv.gz"
    try:
        a = pd.read_csv(af)
        if len(a):
            ax_, ay_, az_ = enu(a.x, a.y, a.z)
            out["after"] = {"t": _r(a.t, 3), "ex": _r(ax_), "ey": _r(ay_), "ez": _r(az_)}
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass

    # events: steps / trajectory phases, disturbances, pilot modes, the trip
    ev = []
    task = sc["task"]
    if task.get("type") == "trajectory":
        tj = task["trajectory"]
        T = tj["cycles"] * tj["period_s"] + tj["ramp_s"]
        ev += [{"t": tj["start_s"], "kind": "phase", "label": f"{tj['shape']} starts"},
               {"t": tj["start_s"] + T, "kind": "phase", "label": f"{tj['shape']} ends"}]
    else:
        for s in task["steps"][1:]:
            comp = [(k, v) for k, v in (("N", s[1]), ("E", s[2]), ("up", -s[3])) if v]
            ev.append({"t": s[0], "kind": "step", "label": ("step to " + " ".join(f"{k}{v:+g}" for k, v in comp) + " m")
                       if comp else "step back to the handover point"})
    for e in man.get("events_applied") or []:
        if "push" in e:
            p = e["push"]
            comp = [(k, v) for k, v in (("N", p.get("n", 0)), ("E", p.get("e", 0)), ("up", -p.get("d", 0))) if v]
            lab = "push " + " ".join(f"{k}{v:+g}" for k, v in comp) + f" m/s² for {p['ms']} ms"
            ev.append({"t": e["t_applied"], "kind": "push", "label": lab,
                       "vec": [p.get("e", 0), p.get("n", 0), -p.get("d", 0)], "ms": p["ms"]})
        elif "derate" in e:
            ev.append({"t": e["t_applied"], "kind": "theta", "label": f"thrust to {e['derate']:g} × available"})
        else:
            ev.append({"t": e["t_applied"], "kind": "param",
                       "label": ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                                          for k, v in (e.get("params") or e.get("params_sent") or {}).items())})
    for p in man.get("pilot_modes_applied") or []:
        ev.append({"t": p.get("t_applied", p["t_s"]), "kind": "pilot", "label": f"pilot sets {p['mode']}"})
    if loop.get("trip"):
        ev.append({"t": loop["trip_t"], "kind": "trip", "label": f"loop stopped: {loop['trip']}"})
    out["events"] = sorted(ev, key=lambda e: e["t"])

    gf = rd / "gusts.csv.gz"
    if gf.exists():
        g = pd.read_csv(gf)
        out["gusts"] = {"t": _r(g.t, 3), "wx": _r(g.air_e, 3), "wy": _r(g.air_n, 3)}   # air velocity, E/N
    return out


def _write_run(args):
    rd, data_dir, group = args
    try:
        d = export_run(rd, group)
    except Exception as e:                              # one bad run must not stop the export
        return {"id": Path(rd).name, "error": repr(e)}
    (data_dir / f"{d['id']}.js").write_text(
        "window.REPLAY_LOADED(" + json.dumps(d, separators=(",", ":")) + ");\n", encoding="utf-8")
    m = d["meta"]
    return {"id": d["id"], "group": group or m["name"], "name": m["name"], "mass_kg": m["mass_kg"],
            "derate": m["derate"], "sensors": m["sensors"], "seed": m["seed"], "outcome": m["outcome"],
            "pass": m["pass"], "landed": m["landed"], "law": m["law"], "law_args": m["law_args"]}


def export(targets: list[tuple[Path, str | None]], out: Path, title: str, workers: int = 8) -> dict:
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    idx_f = out / "index.js"
    old = {}
    if idx_f.exists():                                   # adding runs keeps the ones exported before
        txt = idx_f.read_text(encoding="utf-8")
        old = {r["id"]: r for r in json.loads(txt[txt.index("=") + 1:].rstrip().rstrip(";"))["runs"]}
    with Pool(min(workers, max(len(targets), 1))) as pool:
        res = pool.map(_write_run, [(str(rd), data_dir, g) for rd, g in targets])
    errors = [r for r in res if "error" in r]
    for r in res:
        if "error" not in r:
            old[r["id"]] = r
    index = {"title": title, "runs": sorted(old.values(), key=lambda r: (str(r["group"]), r["mass_kg"], -r["derate"],
                                                                          r["seed"] or 0, r["id"]))}
    idx_f.write_text("window.REPLAY_INDEX = " + json.dumps(index, separators=(",", ":")) + ";\n", encoding="utf-8")
    shutil.copyfile(VIEWER, out / "index.html")
    return {"out": str(out), "runs": len(res) - len(errors), "errors": errors}


def main() -> int:
    ap = argparse.ArgumentParser(description="export runs for the replay viewer")
    ap.add_argument("batch", nargs="?", help="results/<batch> folder (its runs.csv names the runs)")
    ap.add_argument("--run", action="append", default=[], help="a run directory (repeatable)")
    ap.add_argument("--out", help="output folder (default: <batch>/replay or results/replay)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.batch:
        bd = Path(a.batch)
        df = pd.read_csv(bd / "runs.csv")
        targets = [(Path(rd), g) for rd, g in zip(df.run_dir, df.group) if isinstance(rd, str)]
        out, title = Path(a.out or bd / "replay"), bd.name
    elif a.run:
        targets = [(Path(r), None) for r in a.run]
        out, title = Path(a.out or RESULTS / "replay"), "runs"
    else:
        ap.error("give a results/<batch> folder or --run <run_dir>")
    r = export(targets, out, title, a.workers)
    print(json.dumps(r, indent=2))
    print(f"open {out / 'index.html'}")
    return 0 if not r["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
