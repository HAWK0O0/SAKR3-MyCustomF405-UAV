"""Unit checks that need no SITL: supervisor, outer loop on a fake link, batch table, resume rules, verdicts.

    python -m pytest sim/tests/test_unit.py        or        python sim/tests/test_unit.py

Needs pymavlink (the uavsim venv). The SITL acceptance of the takeover is tests/t6e_takeover.py.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import pandas as pd  # noqa: E402

from analysis.batch_table import early_runs, summarize, to_markdown  # noqa: E402
from client import mav  # noqa: E402
from client.controller import Command, Setpoint  # noqa: E402
from client.outer_loop import OuterLoop, fc_took_over  # noqa: E402
from client.supervisor import SafetyLimits, Supervisor  # noqa: E402
from runner import batch  # noqa: E402
from runner.run import EXIT, verdict  # noqa: E402
from runner.scenario import RunSpec, scenario_hash  # noqa: E402

M = mav.M
NAN = float("nan")


def tilt_deg(c: Command) -> float:
    return math.degrees(math.acos(math.cos(c.roll) * math.cos(c.pitch)))


# ---------------------------------------------------------------- supervisor

def test_clamp_limits_total_tilt_with_both_axes_at_the_limit():
    sup = Supervisor(SafetyLimits(max_tilt_deg=25.0), (0.0, 0.0))
    for roll, pitch in ((1.0, 1.0), (math.radians(25), math.radians(25)), (-0.6, 0.9), (0.5, -0.5), (2.0, 0.3)):
        out, changed = sup.clamp(Command(roll, pitch, 0.3, 0.5))
        assert changed and abs(tilt_deg(out) - 25.0) < 1e-9, (roll, pitch, tilt_deg(out))
        # same horizontal direction of the thrust axis as commanded
        before = math.atan2(-math.sin(roll), math.sin(pitch) * math.cos(roll))
        after = math.atan2(-math.sin(out.roll), math.sin(out.pitch) * math.cos(out.roll))
        assert abs(math.remainder(before - after, math.tau)) < 1e-9
        assert out.yaw == 0.3 and out.thrust == 0.5


def test_clamp_leaves_commands_inside_the_cone_alone():
    sup = Supervisor(SafetyLimits(max_tilt_deg=25.0), (0.0, 0.0))
    for roll, pitch in ((0.0, 0.0), (math.radians(17), math.radians(17)), (math.radians(24.9), 0.0)):
        cmd = Command(roll, pitch, 1.0, 0.6)
        out, changed = sup.clamp(cmd)
        assert out == cmd and not changed
    out, changed = sup.clamp(Command(math.radians(40), 0.0, 0.0, 0.6))      # one axis: plain limit
    assert changed and abs(math.degrees(out.roll) - 25.0) < 1e-9 and abs(out.pitch) < 1e-12
    out, changed = sup.clamp(Command(0.0, 0.0, 0.0, 1.5))
    assert changed and out.thrust == 0.95


def test_non_finite_command_is_a_fault_and_is_never_clamped_into_a_command():
    sup = Supervisor(SafetyLimits(), (0.0, 0.0))
    assert sup.command_fault(Command(0.1, 0.1, 0.0, 0.5)) is None
    for bad in (Command(0.1, 0.1, 0.3, NAN), Command(NAN, 0, 0, 0.5), Command(0, math.inf, 0, 0.5),
                Command(0, 0, -math.inf, 0.5), Command(0, 0, 0, None)):
        assert sup.command_fault(bad).startswith("controller fault: non-finite command")
        try:
            sup.clamp(bad)
        except ValueError:
            continue
        raise AssertionError("clamp() turned a non-finite command into one that can be sent")


# ---------------------------------------------------------------- outer loop on a fake link

class FakeLink:
    """Feeds prepared messages to the loop and records what the loop sends."""
    target_system, target_component = 1, 0

    def __init__(self, msgs):
        self.q, self.sent = list(msgs), []
        self.mav = SimpleNamespace(
            heartbeat_send=lambda *a: self.sent.append(("heartbeat",)),
            set_attitude_target_send=lambda *a: self.sent.append(("attitude_target", a[-1])))

    def recv_match(self, type=None, blocking=False, timeout=None):
        return self.q.pop(0) if blocking and self.q else None      # drain() (non-blocking) sees nothing

    def mode_mapping(self):
        return {"GUIDED": 4, "LAND": 9, "BRAKE": 17}

    def set_mode(self, n):
        self.sent.append(("mode", n))

    def kinds(self, kind):
        return [s for s in self.sent if s[0] == kind]


def _msg(typ, **kw):
    return SimpleNamespace(get_type=lambda: typ, get_srcSystem=lambda: 1, **kw)


def _att(ms):
    return _msg("ATTITUDE", time_boot_ms=ms, roll=0.0, pitch=0.0, yaw=0.0)


def _pos(ms):
    return _msg("LOCAL_POSITION_NED", time_boot_ms=ms, x=0.0, y=0.0, z=-5.0, vx=0.0, vy=0.0, vz=0.0)


def _fc_heartbeat(custom_mode):
    return _msg("HEARTBEAT", type=M.MAV_TYPE_QUADROTOR, autopilot=M.MAV_AUTOPILOT_ARDUPILOTMEGA,
                base_mode=M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | M.MAV_MODE_FLAG_SAFETY_ARMED,
                custom_mode=custom_mode)


class ScriptedLaw:
    def __init__(self, bad_step: int, bad):
        self.n, self.bad_step, self.bad = 0, bad_step, bad

    def step(self, s, sp, dt):
        self.n += 1
        if self.n == self.bad_step:
            if isinstance(self.bad, Exception):
                raise self.bad
            return self.bad
        return Command(0.01, -0.01, 0.0, 0.5)


def _loop(link, law):
    sup = Supervisor(SafetyLimits(), (0.0, 0.0))
    return OuterLoop(link, law, sup, lambda t: Setpoint((0.0, 0.0, -5.0), 0.0), log=lambda s: None)


def _samples(n, start_ms=1000):
    return [_att(start_ms)] + [_pos(start_ms + 20 * i) for i in range(n)]


def test_nan_command_is_not_sent_and_trips_in_the_same_step():
    for bad in (Command(0.0, 0.0, 0.0, NAN), Command(math.inf, 0.0, 0.0, 0.5), ZeroDivisionError("law broke")):
        link = FakeLink(_samples(10))
        loop = _loop(link, ScriptedLaw(4, bad))
        res = loop.run(60.0)
        assert res.trip.startswith("controller fault") and res.fault == res.trip, res.trip
        assert len(res.rows) == 4 and res.trip_t == res.rows[-1]["t"]            # stopped in the faulty step
        assert [r["sent"] for r in res.rows] == [True, True, True, False]
        assert res.rows[-1]["fault"] and not res.rows[-1]["clamped"]
        assert len(link.kinds("attitude_target")) == 3                           # nothing sent for step 4
        assert all(math.isfinite(s[1]) for s in link.kinds("attitude_target"))
        assert link.kinds("mode") == [("mode", 17)] and not res.fc_owned         # fallback BRAKE, client still owns


def test_takeover_stops_the_loop_and_commands_no_mode():
    link = FakeLink(_samples(5) + [_fc_heartbeat(17)] + [_pos(1100 + 20 * i) for i in range(5)])
    loop = _loop(link, ScriptedLaw(-1, None))
    res = loop.run(60.0)
    assert res.fc_owned and res.fc_mode == "BRAKE" and res.trip == "FC left GUIDED (BRAKE)"
    assert res.takeover_wall is not None and loop.fallback_mode is None
    assert link.kinds("mode") == []                                              # no BRAKE, no LAND, nothing
    assert len(link.kinds("attitude_target")) == 5 == len(res.rows)              # none after the heartbeat


def test_guided_heartbeat_does_not_stop_the_loop():
    link = FakeLink(_samples(5) + [_fc_heartbeat(4)] + [_pos(1100 + 20 * i) for i in range(3)] + [_pos(99000)])
    res = _loop(link, ScriptedLaw(-1, None)).run(60.0)
    assert res.trip is None and res.end_reason == "duration" and len(res.rows) == 8


def test_fc_took_over_after_the_fallback():
    rows = lambda *modes: [{"mode": m} for m in modes]  # noqa: E731
    assert fc_took_over(rows("GUIDED", "BRAKE", "BRAKE"), "BRAKE") is None       # our own fallback
    assert fc_took_over(rows(None, "GUIDED"), "BRAKE") is None                   # not confirmed yet: still ours
    assert fc_took_over(rows("BRAKE", "LOITER"), "BRAKE") == "LOITER"            # the pilot changed it
    assert fc_took_over(rows("BRAKE", "LAND"), "BRAKE") == "LAND"
    assert fc_took_over([], "BRAKE") is None


def test_txlog_kinds():
    k = mav.TxLog.kind
    assert k(M.MAVLink_set_mode_message(1, 1, 9)) == "mode"
    assert k(M.MAVLink_command_long_message(1, 0, M.MAV_CMD_DO_SET_MODE, 0, 1, 9, 0, 0, 0, 0, 0)) == "mode"
    assert k(M.MAVLink_command_long_message(1, 0, M.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)) == "command"
    assert k(M.MAVLink_set_attitude_target_message(0, 1, 0, 7, [1, 0, 0, 0], 0, 0, 0, 0.5)) == "attitude_target"
    assert k(M.MAVLink_param_set_message(1, 0, b"SIM_SPEEDUP", 1.0, 9)) == "param_set"
    assert k(M.MAVLink_heartbeat_message(18, 8, 0, 0, 0, 3)) == "other"


# ---------------------------------------------------------------- batch table

def test_run_without_a_verdict_is_planned_and_not_passed():
    df = pd.DataFrame([
        {"group": "g", "run_id": "a", "outcome": "ok", "pass": True, "pass_fails": "", "landed": True, "iae_m_s": 10.0},
        {"group": "g", "run_id": "b", "outcome": "error", "reason": "BrokenProcessPool()"},     # worker crashed
    ])
    s = summarize(df, by=["group"]).iloc[0]
    assert (s.n, s.n_completed, s.n_pass, s.n_landed) == (2, 1, 1, 1)
    assert s.pass_rate == 0.5 and s.outcomes == "ok 1, error 1"
    after_csv = pd.read_csv(pd.io.common.StringIO(df.to_csv(index=False)))        # as read back from runs.csv
    assert summarize(after_csv, by=["group"]).iloc[0].pass_rate == 0.5


def test_early_runs_stay_out_of_the_metric_statistics():
    base = {"group": "g", "pass_fails": "", "landed": True}
    df = pd.DataFrame([
        {**base, "run_id": "a", "outcome": "ok", "pass": True, "t_end_s": 56.0, "iae_m_s": 15.0},
        {**base, "run_id": "b", "outcome": "ok", "pass": False, "pass_fails": "seg1_overshoot_pct=31 > 30",
         "t_end_s": 56.0, "iae_m_s": 17.0},
        {**base, "run_id": "c", "outcome": "trip", "pass": False, "pass_fails": "outcome trip", "t_end_s": 11.86,
         "iae_m_s": 11.46, "reason": "altitude 1.92 m outside [2.0, 10.0]"},
    ])
    s = summarize(df, by=["group"]).iloc[0]
    assert (s.n, s.n_completed, s.n_pass, s.n_landed, s.n_early) == (3, 2, 1, 3, 1)
    assert abs(s.pass_rate - 1 / 3) < 1e-12
    assert s.iae_m_s_mean == 16.0                       # 15 and 17; the 11.86 s run is not averaged in
    assert s.early_t_end_s == "11.86" and s.early_reasons == "altitude 1"
    assert s.typical_fail == "seg1_overshoot_pct"       # of completed runs, not "outcome trip"
    e = early_runs(df)
    assert list(e.run_id) == ["c"] and e.t_end_s.iloc[0] == 11.86 and "iae_until_end_m_s" in e
    md = to_markdown(summarize(df, by=["group"]), early=e)
    assert "ended early" in md and "altitude 1.92 m" in md and "33%" in md


# ---------------------------------------------------------------- resume

def test_resume_reuses_only_the_same_experiment():
    spec = RunSpec("r1", "main", 0, {}, {"controller": {"law": "pd"}, "x": 1})
    flight = {"sha256": "F1", "files": {"client/supervisor.py": "a"}, "firmware": {"binary_sha256": "b"}}
    analysis = {"sha256": "A1", "files": {"analysis/metrics.py": "m"}}
    saved = batch.RUNS_ROOT, batch.mf.flight_info, batch.mf.analysis_info, batch.source_sha256
    with tempfile.TemporaryDirectory() as tmp:
        rd = Path(tmp) / "b" / "r1"
        (rd / "logs").mkdir(parents=True)
        batch.RUNS_ROOT = Path(tmp)
        batch.mf.flight_info, batch.mf.analysis_info = lambda: flight, lambda: analysis
        batch.source_sha256 = lambda name: "L1"

        def stored(**over):
            man = {"scenario_sha256": scenario_hash(spec.scenario), "flight": flight, "analysis": analysis,
                   "law": {"source_sha256": "L1"}, **over}
            (rd / "manifest.json").write_text(json.dumps({k: v for k, v in man.items() if v is not None}))
            (rd / "metrics.json").write_text(json.dumps({"summary": {"run_id": "r1"}}))
            return batch._previous("b", spec)

        try:
            assert batch._previous("b", spec) == ("fly", "no finished run")
            assert stored() == ("reuse", "")
            assert stored(scenario_sha256="other") == ("fly", "scenario changed")
            assert stored(flight=None)[0] == "fly"                                   # run of an older runner
            changed = {**flight, "sha256": "F0", "files": {"client/supervisor.py": "old"}}
            assert stored(flight=changed) == ("fly", "flight version changed: client/supervisor.py")
            firmware = {**flight, "sha256": "F0", "firmware": {"binary_sha256": "old"}}
            assert stored(flight=firmware) == ("fly", "flight version changed: firmware")
            assert stored(law={"source_sha256": "L0"}) == ("fly", "law source changed")
            old_analysis = {"sha256": "A0", "files": {"analysis/metrics.py": "old"}}
            assert stored(analysis=old_analysis)[0] == "fly"                         # no .BIN: cannot recompute
            (rd / "logs" / "00000001.BIN").write_bytes(b"")
            assert stored(analysis=old_analysis) == ("reanalyze", "analysis changed: analysis/metrics.py")
            assert stored(flight=changed, analysis=old_analysis)[0] == "fly"         # flight change wins
        finally:
            batch.RUNS_ROOT, batch.mf.flight_info, batch.mf.analysis_info, batch.source_sha256 = saved


# ---------------------------------------------------------------- verdicts

def test_exit_code_needs_completed_pass_and_landed():
    v = lambda outcome, ok, landed: EXIT[verdict({"outcome": outcome, "pass": ok, "landed": landed})]  # noqa: E731
    assert v("ok", True, True) == 0
    assert v("error", False, False) == 1 and v("trip", False, True) == 2
    assert v("ok", False, True) == 3                    # script finished, task bounds not met
    assert v("ok", True, False) == 4                    # ... landing not confirmed


if __name__ == "__main__":
    tests = [(k, f) for k, f in sorted(globals().items()) if k.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
