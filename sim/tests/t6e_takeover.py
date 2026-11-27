"""T6-E: a pilot takes over on a second link; the client must hand over completely.

Flown through the formal entry (runner/run.py run_scenario, scenarios/t6e_takeover.yaml): PD outer
loop in hover, at t = 8 s a pilot / GCS on SITL SERIAL1 sets BRAKE, at t = 14 s the same pilot sets
LAND. BRAKE is not LAND and does not land, so the two readings of "the vehicle landed" differ:
  client respects the takeover : FC modes after handover are BRAKE (8 s), LAND (14 s, the pilot's)
  client lands anyway          : LAND appears right after 8 s, and the client's link shows a mode command
Checks, on two independent records:
  client link (client.mav.TxLog, every message the client sends, by kind)
      after the takeover was detected: 0 mode commands, 0 attitude targets, 0 other commands, 0 params
      the counter itself works: it saw the GUIDED mode command of the takeoff, and as many attitude
      targets as the loop recorded as sent
  FC log (.BIN)
      MODE after handover is exactly [BRAKE, LAND] at the pilot's times; no attitude target accepted
      (GUIA) after the mode change; attitude targets sent between the FC's mode change and its
      detection by the loop: 0
The run itself must report outcome "trip" (FC left GUIDED), not completed, and landed (by the pilot).

Usage (WSL, venv active):  python sim/tests/t6e_takeover.py [--instance N]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import pandas as pd  # noqa: E402

from analysis.binlog import read_messages  # noqa: E402
from runner.run import run_scenario  # noqa: E402
from runner.scenario import load_scenario  # noqa: E402

RESULTS = SIM / "results"
COPTER_MODE = {4: "GUIDED", 9: "LAND", 17: "BRAKE"}


def analyze(summary: dict) -> dict:
    rd = Path(summary["run_dir"])
    man = json.loads((rd / "manifest.json").read_text())
    rows = pd.read_csv(rd / "rows.csv.gz")
    msgs = read_messages(sorted((rd / "logs").glob("*.BIN"))[-1], ["MODE", "GUIA"])
    t0_us = man["loop"]["t0_ms"] * 1e3
    pilot = man["scenario"]["task"]["pilot_modes"]
    mode = msgs["MODE"][msgs["MODE"].TimeUS > t0_us]
    fc_modes = [{"t": float((r.TimeUS - t0_us) / 1e6), "mode": COPTER_MODE.get(int(r.Mode), int(r.Mode))}
                for r in mode.itertuples()]
    t_change = fc_modes[0]["t"] if fc_modes else None
    guia_t = (msgs["GUIA"].TimeUS - t0_us) / 1e6
    tk, tx = man["takeover"] or {}, man["client_tx"]
    after = tk.get("tx_after") or {}
    ev = {
        "outcome": summary["outcome"], "reason": summary["reason"], "completed": summary["completed"],
        "landed": summary["landed"], "takeover_mode": tk.get("mode"), "detected_t": tk.get("detected_t"),
        "client_tx_after_takeover": after, "client_tx_total": tx,
        "attitude_targets_recorded_by_loop": int(rows.sent.sum()),
        "fc_modes_after_handover": fc_modes, "pilot_modes": pilot,
        "t_fc_mode_change": t_change,
        "detect_delay_s": tk["detected_t"] - t_change if t_change is not None and tk.get("detected_t") else None,
        "attitude_targets_sent_after_fc_change": int((rows.sent & (rows.t > t_change)).sum())
        if t_change is not None else None,
        "fc_accepted_after_change": int((guia_t > t_change).sum()) if t_change is not None else None,
    }
    c = {
        "loop_detects_fc_left_guided": summary["outcome"] == "trip"
        and summary["reason"] == f"FC left GUIDED ({pilot[0]['mode']})",
        "not_reported_as_completed": summary["completed"] is False and summary["pass"] is False,
        "no_mode_command_after_takeover": after.get("mode") == 0,
        "no_attitude_target_after_takeover": after.get("attitude_target") == 0,
        "no_other_command_or_param_after_takeover": after.get("command") == 0 and after.get("param_set") == 0
        and after.get("position_target") == 0,
        "tx_counter_saw_mode_command_before": tx["mode"] >= 1,
        "tx_counter_matches_loop_record": tx["attitude_target"] == ev["attitude_targets_recorded_by_loop"] > 0,
        "fc_modes_are_the_pilots_only": [m["mode"] for m in fc_modes] == [p["mode"] for p in pilot]
        and all(abs(m["t"] - p["t_s"]) < 0.5 for m, p in zip(fc_modes, pilot)),
        "no_attitude_target_after_fc_change": ev["attitude_targets_sent_after_fc_change"] == 0,
        "fc_accepted_none_after_change": ev["fc_accepted_after_change"] == 0,
        "landed_by_pilot": summary["landed"] is True,
    }
    c["T6E_pass"] = all(c.values())
    return {"events": ev, "checks": c}


def main() -> int:
    instance = int(sys.argv[sys.argv.index("--instance") + 1]) if "--instance" in sys.argv else 1
    sc = load_scenario(SIM / "scenarios/t6e_takeover.yaml")
    summary = run_scenario(sc, "t6e_takeover", "tests", instance, log=lambda s: print(f"[T6E] {s}", flush=True))
    out = analyze(summary)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "m2_t6e_takeover_checks.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    return 0 if out["checks"]["T6E_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
