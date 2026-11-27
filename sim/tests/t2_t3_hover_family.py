"""T2 + T3: SAKR3 airframe model and its mass / thrust-derate family.

For each variant: generate the SITL frame model, take off to 5 m in GUIDED (ArduCopter's own
controllers), hover, land; then read CTUN.ThO (hover throttle) and BAT.Volt from the .BIN log
over a 15 s steady-hover window and compare with the model prediction.

T2 pass: base 1.40 kg variant hovers steadily and |ThO - predicted| <= 0.03.
T3 pass: ThO rises monotonically with mass, ThO-vs-mass slope within 10 % of prediction,
         derated variants hover higher than the nominal one and match their prediction,
         body drag identical across masses (SITL's suggested EK3_DRAG_BCOEF / mass constant),
         MOT_SPIN_MAX overrides actually applied in the firmware.

Variants run in parallel as separate SITL instances (-I 0..N-1).

Usage (inside WSL, uavsim venv active):
    python tests/t2_t3_hover_family.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from airframes.make_airframe import (G, Variant, load_spec,  # noqa: E402
                                     predicted_hover_throttle, write_variant)
from analysis.binlog import read_messages, read_params, window  # noqa: E402
from client import mav  # noqa: E402
from sitl_runner import SitlInstance  # noqa: E402

TARGET_ALT = 5.0
SETTLE_S = 5.0
HOVER_S = 15.0
THO_TOL = 0.03
BASE_PARM = SIM / "params/sakr3_base.parm"
RESULTS = SIM / "results"

VARIANTS = [
    Variant("m1.20", 1.20),
    Variant("m1.30", 1.30),
    Variant("m1.40", 1.40),           # T2 base point
    Variant("m1.50", 1.50),
    Variant("m1.60", 1.60),
    Variant("m1.40_d0.90", 1.40, 0.90),
    Variant("m1.40_d0.85", 1.40, 0.85),
    Variant("m1.55_d0.85", 1.55, 0.85),   # P4-like extrapolation point (low thrust/weight)
]


def fly_hover(v: Variant, instance: int, spec: dict) -> dict:
    tag = f"[{v.name}]"
    log = lambda s: print(f"{tag} {s}", flush=True)  # noqa: E731
    sitl = SitlInstance(run_name=f"M1_hover_{v.name}", instance=instance)
    run_dir = sitl.prepare_dir()
    jpath, ppath = write_variant(v, run_dir / "model", spec)
    # SITL's AP_Filesystem strips a leading '/', so the model path must be relative to run_dir
    sitl.model = f"x:{jpath.relative_to(run_dir)}"
    sitl.extra_defaults = [str(BASE_PARM)] + ([str(ppath)] if ppath else [])
    out = {"variant": v.name, "mass": v.mass, "derate": v.derate, "run_dir": str(run_dir)}
    try:
        sitl.start()
        m = mav.connect(sitl.connection)
        mav.request_streams(m, 20)
        out["ekf_wait_s"] = round(mav.wait_ekf_position(m, log=lambda s: None), 1)
        mav.set_mode(m, "GUIDED")
        mav.arm_with_retry(m)
        mav.takeoff(m, TARGET_ALT, log=log)

        phase, t_phase, t0 = "climb", time.time(), time.time()
        alts, win = [], {}
        while time.time() - t0 < 150:
            msg = m.recv_match(type=["LOCAL_POSITION_NED", "HEARTBEAT"], blocking=True, timeout=1)
            if msg is None:
                continue
            if phase in ("climb", "settle", "hover") and not m.motors_armed():
                log(f"unexpected disarm during {phase}")
                break
            if msg.get_type() != "LOCAL_POSITION_NED":
                if phase == "land" and not m.motors_armed():
                    phase = "done"
                    break
                continue
            alt = -msg.z
            if phase == "climb" and alt > 0.95 * TARGET_ALT:
                phase, t_phase = "settle", time.time()
            elif phase == "settle" and time.time() - t_phase > SETTLE_S:
                phase, t_phase = "hover", time.time()
                win["t0_us"] = msg.time_boot_ms * 1000
            elif phase == "hover":
                alts.append(alt)
                if time.time() - t_phase > HOVER_S:
                    win["t1_us"] = msg.time_boot_ms * 1000
                    mav.set_mode(m, "LAND")
                    phase = "land"
        out["final_phase"] = phase
        m.close()
        time.sleep(2)
    finally:
        sitl.stop()

    if alts:
        out["hover_alt_mean"] = float(np.mean(alts))
        out["hover_alt_std"] = float(np.std(alts))
    stdout = (run_dir / "sitl_stdout.log").read_text(errors="replace")
    out["model_loaded"] = "Loaded model params from" in stdout
    mb = re.search(r"Suggested EK3_DRAG_BCOEF_\* = ([0-9.]+)", stdout)
    out["drag_bcoef"] = float(mb.group(1)) if mb else None

    bins = sorted((run_dir / "logs").glob("*.BIN"))
    if bins and "t1_us" in win:
        msgs = read_messages(bins[-1], ["CTUN", "BAT"])
        ctun = window(msgs["CTUN"], win["t0_us"], win["t1_us"])
        bat = window(msgs["BAT"], win["t0_us"], win["t1_us"])
        params = read_params(bins[-1])
        out["ThO_mean"] = float(ctun.ThO.mean())
        out["ThO_std"] = float(ctun.ThO.std())
        out["volt_mean"] = float(bat.Volt.mean())
        out["curr_mean"] = float(bat.Curr.mean())
        out["MOT_SPIN_MAX"] = params.get("MOT_SPIN_MAX")
        out["FRAME_TYPE"] = params.get("FRAME_TYPE")
        out["ThO_pred"] = predicted_hover_throttle(v.mass, out["volt_mean"], out["MOT_SPIN_MAX"],
                                                   spec, alt_amsl=spec["refAlt"] + TARGET_ALT)
        out["ThO_err"] = out["ThO_mean"] - out["ThO_pred"]
    out["flight_ok"] = (out.get("final_phase") == "done" and out.get("hover_alt_std", 1) < 0.1
                        and out["model_loaded"])
    out["tho_ok"] = abs(out.get("ThO_err", 1)) <= THO_TOL
    log(f"done: ThO={out.get('ThO_mean', float('nan')):.4f} "
        f"pred={out.get('ThO_pred', float('nan')):.4f} V={out.get('volt_mean', float('nan')):.2f}")
    return out


def evaluate(df: pd.DataFrame, spec: dict) -> dict:
    checks = {}
    base = df.set_index("variant")
    checks["T2_base_flight_ok"] = bool(base.loc["m1.40", "flight_ok"])
    checks["T2_base_ThO_within_tol"] = bool(base.loc["m1.40", "tho_ok"])

    nom = df[df.derate == 1.0].sort_values("mass")
    checks["T3_ThO_monotonic_in_mass"] = bool(np.all(np.diff(nom.ThO_mean.values) > 0))
    slope_meas = float(np.polyfit(nom.mass, nom.ThO_mean, 1)[0])
    # predictions use each point's own measured hover voltage (heavier -> more current -> more sag)
    slope_pred = float(np.polyfit(nom.mass, nom.ThO_pred, 1)[0])
    checks["T3_slope_meas"] = slope_meas
    checks["T3_slope_pred"] = slope_pred
    checks["T3_slope_within_10pct"] = bool(abs(slope_meas / slope_pred - 1) < 0.10)
    checks["T3_all_flights_ok"] = bool(df.flight_ok.all())
    checks["T3_all_ThO_within_tol"] = bool(df.tho_ok.all())
    der = df[df.derate < 1.0]
    checks["T3_derate_raises_ThO"] = bool(all(
        r.ThO_mean > base.loc["m1.40", "ThO_mean"] for r in der.itertuples() if r.mass == 1.40))
    checks["T3_derate_param_applied"] = bool(all(
        r.MOT_SPIN_MAX is not None and r.MOT_SPIN_MAX < 0.95 for r in der.itertuples()))
    ratio = (nom.drag_bcoef / nom.mass).values
    checks["T3_drag_bcoef_per_kg"] = [round(x, 3) for x in ratio]
    checks["T3_drag_constant"] = bool(np.ptp(ratio) / ratio.mean() < 0.01)
    checks["T2_pass"] = checks["T2_base_flight_ok"] and checks["T2_base_ThO_within_tol"]
    checks["T3_pass"] = all(checks[k] for k in (
        "T3_ThO_monotonic_in_mass", "T3_slope_within_10pct", "T3_all_flights_ok",
        "T3_all_ThO_within_tol", "T3_derate_raises_ThO", "T3_derate_param_applied",
        "T3_drag_constant"))
    return checks


def plot(df: pd.DataFrame, spec: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    nom = df[df.derate == 1.0].sort_values("mass")
    mm = np.linspace(1.15, 1.65, 50)
    v_ref = float(nom.volt_mean.mean())
    alt = spec["refAlt"] + TARGET_ALT
    ax.plot(mm, [predicted_hover_throttle(x, v_ref, spec=spec, alt_amsl=alt) for x in mm], "-", color="#888",
            lw=1, label=f"model prediction (full thrust, {v_ref:.2f} V)")
    ax.plot(nom.mass, nom.ThO_mean, "o", color="#2a7ab0", label="SITL, full thrust")
    ax.plot(nom.mass, nom.ThO_pred, "x", color="#2a7ab0", label="prediction at each point's voltage")
    der = df[df.derate < 1.0]
    for r in der.itertuples():
        ax.plot(r.mass, r.ThO_mean, "s", color="#c0602a")
        ax.plot(r.mass, r.ThO_pred, "x", color="#c0602a")
        ax.annotate(f"{int(round((1 - r.derate) * 100))}% derate", (r.mass, r.ThO_mean),
                    textcoords="offset points", xytext=(6, -3), fontsize=8, color="#c0602a")
    ax.plot([], [], "s", color="#c0602a", label="SITL, thrust derated (MOT_SPIN_MAX)")
    ax.plot([], [], "x", color="#c0602a", label="prediction for derated point")
    ax.set_xlabel("take-off mass [kg]")
    ax.set_ylabel("hover throttle CTUN.ThO [-]")
    ax.set_title("T2/T3: SAKR3 SITL model, hover throttle across the parameter family")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> int:
    spec = load_spec()
    if "--from-csv" in sys.argv:   # re-evaluate the last run without flying again
        df = pd.read_csv(RESULTS / "m1_hover_family.csv")
        checks = evaluate(df, spec)
        (RESULTS / "m1_hover_family_checks.json").write_text(json.dumps(checks, indent=2))
        plot(df, spec, RESULTS / "m1_hover_family.png")
        print(json.dumps(checks, indent=2))
        return 0 if checks["T2_pass"] and checks["T3_pass"] else 1
    with ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
        futs = [ex.submit(fly_hover, v, i, spec) for i, v in enumerate(VARIANTS)]
        rows = []
        for f in futs:
            try:
                rows.append(f.result())
            except Exception as e:  # keep the other variants' results
                rows.append({"error": repr(e)})
                print("variant failed:", repr(e), flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "m1_hover_family.csv", index=False)
    if "error" in df and df["error"].notna().any():
        print(df.to_string())
        return 1
    checks = evaluate(df, spec)
    (RESULTS / "m1_hover_family_checks.json").write_text(json.dumps(checks, indent=2))
    plot(df, spec, RESULTS / "m1_hover_family.png")
    cols = ["variant", "mass", "derate", "MOT_SPIN_MAX", "hover_alt_std", "volt_mean", "curr_mean",
            "ThO_mean", "ThO_pred", "ThO_err", "drag_bcoef", "flight_ok", "tho_ok"]
    print(df[cols].to_string(float_format=lambda x: f"{x:.4f}"))
    print(json.dumps(checks, indent=2))
    return 0 if checks["T2_pass"] and checks["T3_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
