"""One scenario run: SITL -> FC takeoff -> outer loop -> land -> truth, metrics, manifest.

Three separate verdicts per run, none implies another:
  completed  the script ran to its end and the outer loop flew the whole planned duration
             (outcome "ok"; "trip" = the loop stopped early, "error" = setup / SITL / analysis failed)
  pass       completed and every bound of the scenario's `pass` section holds (task criteria)
  landed     the vehicle was seen disarmed at the end (landing confirmed)
The CLI exit code is 0 only if all three hold: 1 error, 2 trip, 3 task bounds not met, 4 landing not
confirmed.

Who lands: the client commands LAND itself, unless the FC left GUIDED on its own (pilot, GCS on
another link, FC failsafe). Then the FC owns the vehicle and the client only watches: no mode command,
no attitude target, no parameter. What the client sent after the takeover is counted from its own link
(client.mav.TxLog) and stored in the manifest (takeover.tx_after).

Everything belonging to a run goes into its own directory ~/uavsim_runs/<batch>/<run_id>/:
  cmdline.txt, sitl_stdout.log, eeprom.bin, logs/*.BIN   SITL (BIN removed if run.keep_bin is false)
  model/                frame model JSON (+ .parm with MOT_SPIN_MAX for a derated variant)
  scenario.yaml         the fully resolved scenario
  scenario.parm         wind + extra parameters generated from it (last in the defaults chain)
  rows.csv.gz           one row per control step (state, setpoint, command) with truth attached
  after.csv.gz          BRAKE hold / landing, from the monitor
  metrics.json          metrics + pass/fail
  manifest.json         firmware, parameter files, model, code and law hashes, seed, timing; the flight
                        version and the analysis version (runner/manifest.py)
reanalyze() computes truth, metrics and verdicts again from the stored logs of a finished run.

Usage (WSL, venv active):  python sim/runner/run.py scenarios/t4_steps.yaml [--instance N] [--speedup X] [--seed S]
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
import time
import traceback
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from airframes.make_airframe import Variant, spin_max_for_derate  # noqa: E402
from analysis.binlog import read_messages  # noqa: E402
from analysis.metrics import evaluate_pass, run_metrics  # noqa: E402
from analysis.truth import attach, truth_frame  # noqa: E402
from client import mav  # noqa: E402
from client.laws import make_law  # noqa: E402
from client.outer_loop import OuterLoop, fc_took_over, monitor, step_schedule  # noqa: E402
from client.supervisor import SafetyLimits, Supervisor  # noqa: E402
from client.trajectory import Trajectory, trajectory_schedule  # noqa: E402
from runner import manifest as mf  # noqa: E402
from runner.disturbance import GustProfile, gust_hook, push_params  # noqa: E402
from runner.pilot import Pilot  # noqa: E402
from runner.scenario import load_scenario, scenario_hash  # noqa: E402
from runner.seed import sitl_env  # noqa: E402
from sakr3 import OUTER_LOOP_PARM, fly_to_hover, needs_ekf_origin, param_check, sakr3_sitl  # noqa: E402

LOG_TYPES = ["GUIA", "MODE", "SIM", "SIM2", "ORGN", "PARM"]
FC_PARAMS = ("GUID_OPTIONS", "GUID_TIMEOUT", "ATC_ANGLE_BOOST", "MOT_SPIN_MAX", "MOT_THST_HOVER",
             "SIM_WIND_T", "SIM_WIND_SPD", "SIM_WIND_DIR", "SIM_WIND_TURB", "SIM_BARO_RND", "SIM_SPEEDUP",
             "GPS1_TYPE", "FLOW_TYPE", "RNGFND1_TYPE", "EK3_SRC1_POSXY", "EK3_SRC1_VELXY", "EK3_SRC1_POSZ",
             "EK3_SRC1_VELZ", "EK3_SRC1_YAW", "SIM_FLOW_RATE", "SIM_FLOW_RND", "SIM_FLOW_DELAY", "SIM_SONAR_RND")


def scenario_parm(sc: dict) -> str:
    w = sc["environment"]["wind"]
    lines = ["# generated from scenario.yaml",
             "SIM_WIND_T 1                 # wind constant with altitude",
             f"SIM_WIND_SPD {w['speed_ms']}", f"SIM_WIND_DIR {w['dir_deg']}", f"SIM_WIND_TURB {w['turbulence_ms']}"]
    lines += [f"{k} {v}" for k, v in sc["environment"]["params"].items()]
    return "\n".join(lines) + "\n"


def event_params(ev: dict, yaw: float, derate: float) -> dict:
    """One task.event as parameters: params as given; push (NED) -> SIM_SHOVE_* with the handover yaw;
    derate k -> MOT_SPIN_MAX for k times the variant's available thrust (same mapping as the variants)."""
    if "params" in ev:
        return dict(ev["params"])
    if "push" in ev:
        return push_params(ev["push"], yaw)
    return {"MOT_SPIN_MAX": round(spin_max_for_derate(derate * float(ev["derate"])), 4)}


def _event_hook(m, events, applied: list, log, yaw: float = 0.0, derate: float = 1.0):
    pending = sorted(events, key=lambda e: e["t_s"])

    def hook(t: float):
        while pending and t >= pending[0]["t_s"]:
            ev = pending.pop(0)
            params = event_params(ev, yaw, derate)
            for k, v in params.items():
                mav.set_param_nowait(m, k, v)
            applied.append({**ev, "t_applied": t, "params_sent": params})
            log(f"event at t={t:.2f}: {ev}")
    return hook


def setpoint_fn(task: dict, origin, yaw: float):
    if task["type"] == "trajectory":
        return trajectory_schedule(origin, yaw, Trajectory.from_config(task["trajectory"]))
    return step_schedule(origin, yaw, [(s[0], tuple(s[1:])) for s in task["steps"]])


def _num(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def _flight_outcome(error: str | None, loop: dict | None) -> tuple[str, str]:
    if error:
        return "error", error.strip().splitlines()[-1]
    if loop is None or loop["trip"]:
        return "trip", (loop or {}).get("trip") or "no loop result"
    return "ok", ""


def run_scenario(sc: dict, run_id: str, batch: str = "adhoc", instance: int = 0, log=print) -> dict:
    """Fly one resolved scenario; returns the summary row for the batch table."""
    task, run = sc["task"], sc["run"]
    speedup, seed = float(run["speedup"]), int(run["seed"])
    setup_speedup = float(run["setup_speedup"] or speedup)   # boot / takeoff / landing
    variant = Variant("model", sc["vehicle"]["mass_kg"], sc["vehicle"]["derate"])
    sensors = SIM / "params" / f"sensors_{sc['environment']['sensors']}.parm"
    sitl = sakr3_sitl(f"{batch}/{run_id}", variant, instance, extra_parms=[OUTER_LOOP_PARM, sensors],
                      speedup=setup_speedup)
    rd = sitl.run_dir
    (rd / "scenario.yaml").write_text(yaml.safe_dump(sc, sort_keys=False))
    sparm = rd / "scenario.parm"
    sparm.write_text(scenario_parm(sc))
    sitl.extra_defaults.append(str(sparm))
    sitl.env = sitl_env(seed)
    # no GPS (profile B): the EKF origin is set to SITL home, so the truth frame needs no shift
    ekf_origin = tuple(float(x) for x in sitl.home.split(",")[:3]) if needs_ekf_origin(sitl) else None

    gu, wind = sc["environment"]["gusts"], sc["environment"]["wind"]
    gust_prof = (GustProfile(wind["speed_ms"], wind["dir_deg"], gu["sigma_ms"], gu["tau_s"], gu["rate_hz"],
                             task["duration_s"], seed) if gu["sigma_ms"] > 0 else None)
    if gust_prof is not None:
        pd.DataFrame(gust_prof.table()).to_csv(rd / "gusts.csv.gz", index=False)

    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    wall0 = time.monotonic()
    res = loop = m = pilot = tx = None
    after, applied, gust_applied = [], [], []
    hover = hover_fc = law_info = error = takeover = None
    origin = yaw0 = None
    try:
        sitl.start()
        m = mav.connect(sitl.connection)
        tx = mav.TxLog(m)
        lpn, yaw0 = fly_to_hover(m, task["alt_m"], log, task["settle_s"], ekf_origin=ekf_origin)
        hover_fc = mav.get_param(m, "MOT_THST_HOVER")
        ht = sc["controller"]["hover_thrust"]
        hover = hover_fc if ht == "fc" else float(ht)
        law, law_info = make_law(sc["controller"], hover)
        origin = (lpn.x, lpn.y, lpn.z)
        sup = Supervisor(SafetyLimits(**sc["safety"]), origin[:2])
        events = _event_hook(m, task["events"], applied, log, yaw0, sc["vehicle"]["derate"])
        if gust_prof is not None:
            gusts = gust_hook(gust_prof, lambda k, v: mav.set_param_nowait(m, k, v), gust_applied)
        if task["pilot_modes"]:
            pilot = Pilot(sitl, task["pilot_modes"], log)

        def hook(t: float):
            events(t)
            if gust_prof is not None:
                gusts(t)
            if pilot:
                pilot.poll(t)

        loop = OuterLoop(m, law, sup, setpoint_fn(task, origin, yaw0), step_hook=hook,
                         time_scale=speedup, log=log)
        if setup_speedup != speedup:                   # SITL applies SIM_SPEEDUP on the next frame
            mav.set_param(m, "SIM_SPEEDUP", speedup)
        log(f"handover at alt {-lpn.z:.2f} m, hover thrust {hover:.4f}; outer loop {task['duration_s']} s "
            f"at x{speedup:g}")
        res = loop.run(task["duration_s"])
        watch = pilot.poll if pilot else None          # the pilot keeps acting after the loop has stopped
        fc_mode, takeover_wall = res.fc_mode, res.takeover_wall
        if res.trip and loop.fallback_mode:            # the loop commanded its fallback (BRAKE): hold
            hold = monitor(m, 4.0, res.t0_ms, time_scale=speedup, hook=watch)
            after += hold
            fc_mode = fc_took_over(hold, loop.fallback_mode)    # ... unless someone else took over meanwhile
            takeover_wall = time.monotonic() if fc_mode else None
        if fc_mode:
            # The FC (pilot / GCS / failsafe) owns the vehicle: watch only. No LAND, no parameter, nothing.
            log(f"FC is in {fc_mode}, not commanded by this client: sending nothing, watching until disarmed")
            after += monitor(m, 60.0, res.t0_ms, stop=lambda r: not r["armed"], time_scale=speedup, hook=watch)
            takeover = {"mode": fc_mode, "detected_t": res.trip_t if res.fc_owned else None,
                        "tx_after": tx.counts(since=takeover_wall)}
        else:
            if setup_speedup != speedup:
                mav.set_param_nowait(m, "SIM_SPEEDUP", setup_speedup)
            m.set_mode(m.mode_mapping()["LAND"])
            after += monitor(m, 60.0, res.t0_ms, stop=lambda r: not r["armed"], time_scale=setup_speedup,
                             hook=watch)
    except Exception:
        error = traceback.format_exc()
        log(f"ERROR {error.strip().splitlines()[-1]}")
    finally:
        if pilot is not None:
            pilot.close()
        if m is not None:
            m.close()
        sitl.stop()
    wall = time.monotonic() - wall0

    # raw flight data first, so a failed analysis loses nothing and a later one starts from the same files
    rows = pd.DataFrame(res.rows) if res is not None else pd.DataFrame()
    after = pd.DataFrame(after)
    if len(rows):
        rows.to_csv(rd / "rows.csv.gz", index=False)
    after.to_csv(rd / "after.csv.gz", index=False)
    manifest = {
        "run_id": run_id, "batch": batch, "instance": instance, "started_utc": started, "wall_s": wall,
        "seed": seed, "speedup": speedup, "setup_speedup": setup_speedup, "scenario_sha256": scenario_hash(sc),
        "scenario": sc,
        "ardupilot": mf.ardupilot_info(), "defaults_chain": mf.param_files(sitl.defaults_chain()),
        "frame_model": mf.param_files(sorted((rd / "model").glob("*"))),
        "seed_shim": mf.param_files([sitl.env["LD_PRELOAD"]]), "law": law_info, "code": mf.code_info(),
        "flight": mf.flight_info(), "host": mf.host_info(),
        "sitl": {"home": sitl.home, "extra_defaults": list(sitl.extra_defaults)},
        "handover": {"origin_ned": origin, "yaw": yaw0, "hover_thrust": hover, "mot_thst_hover_fc": hover_fc,
                     "ekf_origin_set": ekf_origin},
        "loop": None if res is None else {"t0_ms": res.t0_ms, "trip": res.trip, "trip_t": res.trip_t,
                                          "fault": res.fault, "end_reason": res.end_reason,
                                          "wall_s": res.wall_s, "fc_s": res.fc_s,
                                          "fallback_mode": loop.fallback_mode, "fc_text": res.text},
        "events_applied": applied, "pilot_modes_applied": pilot.applied if pilot else [],
        "gusts": None if gust_prof is None else {
            "samples_applied": len(gust_applied), "last_t": gust_applied[-1][0] if gust_applied else None,
            "max_wind_ms": float(max(p[0] for p in gust_prof.params)), "file": "gusts.csv.gz"},
        "takeover": takeover, "client_tx": tx.counts() if tx else None, "flight_error": error,
    }
    return _analyze(rd, manifest, rows, after, log)


def _analyze(rd: Path, manifest: dict, rows: pd.DataFrame, after: pd.DataFrame, log=print) -> dict:
    """Truth, metrics and the three verdicts from the raw data of a flight; writes manifest and metrics."""
    sc, loop, handover = manifest["scenario"], manifest["loop"], manifest["handover"]
    task, error = sc["task"], manifest["flight_error"]
    outcome, reason = _flight_outcome(error, loop)

    metrics, params, param_problems = {}, {}, None
    try:
        bins = sorted((rd / "logs").glob("*.BIN"))
        if len(rows) and bins:
            msgs = read_messages(bins[-1], LOG_TYPES)
            params = dict(zip(msgs["PARM"].Name, msgs["PARM"].Value))
            first = msgs["PARM"].drop_duplicates("Name", keep="first")
            # our own files (not ArduPilot's copter*.parm): every value must have reached the FC
            param_problems = param_check(dict(zip(first.Name, first.Value)), manifest["sitl"]["extra_defaults"])
            if param_problems:
                log(f"PARAM PROBLEMS {param_problems}")
                if outcome == "ok":
                    outcome, reason = "error", "params: " + "; ".join(param_problems)
            truth = truth_frame(msgs, manifest["sitl"]["home"],
                                vertical_ref="rest" if handover["ekf_origin_set"] else "origin")
            rows = attach(rows, truth)
            rows.to_csv(rd / "rows.csv.gz", index=False)
            traj = (Trajectory.from_config(task["trajectory"]) if task.get("type") == "trajectory" else None)
            metrics = run_metrics(rows, msgs["GUIA"], [[0.0, 0.0, 0.0, 0.0]] if traj else task["steps"],
                                  handover["origin_ned"], task["duration_s"], t0_ms=loop["t0_ms"],
                                  trajectory=(traj.start_s, traj.end_s) if traj else None,
                                  event_times=[ev["t_s"] for ev in task["events"]], **sc["metrics"])
            win = truth[(truth.TimeUS >= loop["t0_ms"] * 1e3) & (truth.TimeUS <= rows.t_ms.iloc[-1] * 1e3)]
            metrics["speedup_fc_aspd"] = float(win.aspd.mean()) if len(win) else float("nan")
            metrics["truth_origin_offset_m"] = truth.attrs["origin_offset_ned"]
        if bins and not sc["run"].get("keep_bin", True):
            for b in bins:
                b.unlink()
    except Exception:
        tb = traceback.format_exc()
        error = (error or "") + "\n[analysis]\n" + tb
        if outcome == "ok":
            outcome, reason = "error", "analysis: " + tb.strip().splitlines()[-1]
        log(f"ANALYSIS ERROR {tb.strip().splitlines()[-1]}")

    if loop is not None and loop["wall_s"] > 0:
        metrics["speedup_achieved"] = loop["fc_s"] / loop["wall_s"]
    completed = outcome == "ok"
    landed = bool(len(after)) and not bool(after.armed.iloc[-1])
    ok, fails = evaluate_pass(metrics, sc["pass"], outcome)
    takeover = manifest.get("takeover")

    manifest = {**manifest, "analysis": {**mf.analysis_info(),
                                         "analysed_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
                "fc_params": {k: params.get(k) for k in FC_PARAMS}, "param_problems": param_problems,
                "completed": completed, "pass": ok, "landed": landed, "outcome": outcome, "reason": reason,
                "error": error}
    (rd / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    summary = {"run_id": manifest["run_id"], "outcome": outcome, "reason": reason, "completed": completed,
               "pass": ok, "pass_fails": "; ".join(fails), "landed": landed,
               "fc_takeover": takeover["mode"] if takeover else "",
               "seed": manifest["seed"], "speedup": manifest["speedup"],
               "setup_speedup": manifest["setup_speedup"], "sensors": sc["environment"]["sensors"],
               "mass_kg": sc["vehicle"]["mass_kg"], "derate": sc["vehicle"]["derate"],
               "hover_thrust": handover["hover_thrust"], "wall_s": manifest["wall_s"], "run_dir": str(rd),
               "flight_sha": manifest["flight"]["sha256"][:12], "analysis_sha": manifest["analysis"]["sha256"][:12],
               **{k: _num(v) for k, v in metrics.items() if not isinstance(v, list)}}
    (rd / "metrics.json").write_text(json.dumps({"summary": summary, "metrics": metrics}, indent=2, default=str))
    log(f"{outcome} {reason} | completed={completed} pass={ok} landed={landed} | wall {manifest['wall_s']:.0f} s")
    return summary


def reanalyze(rd: str | Path, log=print) -> dict:
    """Analyse a finished run again from its stored logs (rows, after, .BIN) with the current analysis code.

    The flight is untouched; the manifest keeps the analysis versions it was analysed with before
    (analysis_history). Needs the .BIN log (run.keep_bin) and a manifest written by this version of the
    runner; raises otherwise.
    """
    rd = Path(rd)
    manifest = json.loads((rd / "manifest.json").read_text())
    for k in ("flight", "sitl", "loop", "handover"):
        if k not in manifest:
            raise KeyError(f"manifest of {rd.name} has no {k!r} (flown by an older runner)")
    if not sorted((rd / "logs").glob("*.BIN")):
        raise FileNotFoundError(f"{rd.name}: no .BIN log to analyse again")
    rows_file = rd / "rows.csv.gz"
    rows = pd.read_csv(rows_file, float_precision="round_trip") if rows_file.exists() else pd.DataFrame()
    try:
        after = pd.read_csv(rd / "after.csv.gz", float_precision="round_trip")
    except pd.errors.EmptyDataError:
        after = pd.DataFrame()
    old = manifest.get("analysis") or {}
    manifest["analysis_history"] = manifest.get("analysis_history", []) + [
        {"sha256": old.get("sha256"), "analysed_utc": old.get("analysed_utc")}]
    return _analyze(rd, manifest, rows, after, log)


EXIT = {"ok": 0, "error": 1, "trip": 2, "task bounds not met": 3, "landing not confirmed": 4}


def verdict(summary: dict) -> str:
    """ok only if the run completed, met its task bounds and the landing was confirmed (see EXIT)."""
    if summary["outcome"] != "ok":
        return summary["outcome"]
    if not summary["pass"]:
        return "task bounds not met"
    return "ok" if summary["landed"] else "landing not confirmed"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="fly one scenario",
                                 epilog="exit code: " + ", ".join(f"{v} {k}" for k, v in EXIT.items()))
    ap.add_argument("scenario")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--speedup", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--batch", default="adhoc")
    a = ap.parse_args()
    sc = load_scenario(a.scenario)
    if a.speedup:
        sc["run"]["speedup"] = a.speedup
    if a.seed is not None:
        sc["run"]["seed"] = a.seed
    rid = f"{sc['name']}__s{sc['run']['seed']}_x{sc['run']['speedup']:g}"
    s = run_scenario(sc, rid, a.batch, a.instance, log=lambda x: print(f"[{rid}] {x}", flush=True))
    print(json.dumps(s, indent=2, default=str))
    v = verdict(s)
    print(f"[{rid}] verdict: {v} (completed={s['completed']} pass={s['pass']} landed={s['landed']})", flush=True)
    return EXIT[v]


if __name__ == "__main__":
    sys.exit(main())
