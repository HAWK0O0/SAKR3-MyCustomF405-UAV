"""Shared M2 flight procedure: FC takeoff -> hand over to the external outer loop -> land.

The flight data are saved as CSV/JSON so the analysis can be re-run without flying again.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import pandas as pd  # noqa: E402

from airframes.make_airframe import Variant  # noqa: E402
from analysis.binlog import read_messages, read_params  # noqa: E402
from client import mav  # noqa: E402
from client.controller import PDPositionController  # noqa: E402
from client.outer_loop import OuterLoop, fc_took_over, monitor, step_schedule  # noqa: E402
from client.supervisor import SafetyLimits, Supervisor  # noqa: E402
from sakr3 import OUTER_LOOP_PARM, fly_to_hover, sakr3_sitl  # noqa: E402

RESULTS = SIM / "results"
FRAMES = ("rows", "echo", "after", "guia", "mode")
PARAMS = ("GUID_OPTIONS", "GUID_TIMEOUT", "ATC_ANGLE_BOOST", "MOT_THST_HOVER", "FS_GCS_ENABLE")


def fly_outer_loop(run_name: str, instance: int, steps, duration: float, limits: SafetyLimits,
                   drop_fn=None, brake_hold_s: float = 4.0, alt: float = 5.0,
                   variant: Variant = Variant("m1.40", 1.40), log=print) -> dict:
    """The client lands the vehicle itself, unless the FC left GUIDED on its own (pilot / failsafe):
    then it only watches. Takeover tests go through the formal entry: tests/t6e_takeover.py."""
    sitl = sakr3_sitl(run_name, variant, instance, extra_parms=[OUTER_LOOP_PARM])
    try:
        sitl.start()
        m = mav.connect(sitl.connection)
        lpn, yaw0 = fly_to_hover(m, alt, log)
        hover = mav.get_param(m, "MOT_THST_HOVER")
        log(f"handover at alt {-lpn.z:.2f} m, MOT_THST_HOVER={hover:.4f}")
        law = PDPositionController(hover)
        sup = Supervisor(limits, (lpn.x, lpn.y))
        origin = (lpn.x, lpn.y, lpn.z)
        loop = OuterLoop(m, law, sup, step_schedule(origin, yaw0, steps), drop_fn=drop_fn,
                         log=log)
        res = loop.run(duration)
        after, fc_mode = [], res.fc_mode
        if res.trip and loop.fallback_mode:          # loop requested its fallback mode (BRAKE)
            after += monitor(m, brake_hold_s, res.t0_ms)
            fc_mode = fc_took_over(after, loop.fallback_mode)
        if fc_mode:                                  # pilot / failsafe owns the vehicle: send nothing
            log(f"FC is in {fc_mode}, not commanded by this client: watching only")
        else:
            m.set_mode(m.mode_mapping()["LAND"])
        after += monitor(m, 60, res.t0_ms, stop=lambda r: not r["armed"])
        m.close()
    finally:
        sitl.stop()

    bins = sorted((sitl.run_dir / "logs").glob("*.BIN"))
    msgs = read_messages(bins[-1], ["GUIA", "MODE"])
    params = read_params(bins[-1])
    meta = {"run_dir": str(sitl.run_dir), "t0_ms": res.t0_ms, "trip": res.trip, "trip_t": res.trip_t,
            "end_reason": res.end_reason, "hover_thrust": hover, "origin": origin, "yaw0": yaw0,
            "fallback_mode": loop.fallback_mode, "fc_text": res.text, "fc_takeover": fc_mode,
            "params": {k: params.get(k) for k in PARAMS}}
    return {"rows": pd.DataFrame(res.rows), "echo": pd.DataFrame(res.echo), "after": pd.DataFrame(after),
            "guia": msgs["GUIA"], "mode": msgs["MODE"], "meta": meta}


def save(data: dict, prefix: str):
    RESULTS.mkdir(exist_ok=True)
    for k in FRAMES:
        data[k].to_csv(RESULTS / f"{prefix}_{k}.csv", index=False)
    (RESULTS / f"{prefix}_meta.json").write_text(json.dumps(data["meta"], indent=2))


def load(prefix: str) -> dict:
    data = {k: pd.read_csv(RESULTS / f"{prefix}_{k}.csv", float_precision="round_trip") for k in FRAMES}
    data["meta"] = json.loads((RESULTS / f"{prefix}_meta.json").read_text())
    return data


def fc_time(data: dict, time_us) -> pd.Series:
    """FC log TimeUS -> seconds since outer-loop start (same axis as rows.t)."""
    return (pd.Series(time_us) / 1e3 - data["meta"]["t0_ms"]) / 1e3
