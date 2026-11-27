"""Batch of scenario runs on parallel SITL instances.

    python sim/runner/batch.py scenarios/batches/t5.yaml [--workers N] [--base-instance 30] [--resume] [--only GLOB]

Each worker process owns one SITL instance number (ports 5760+10*I, ...) and flies its runs one
after another; the longest runs (by estimated wall time) start first. Outer loops run in
separate processes, not threads, so one loop's Python work never delays another's (GIL).
Results:
  ~/uavsim_runs/<batch>/<run_id>/    everything per run (runner/run.py)
  sim/results/<batch>/runs.csv       one row per run: group, repeat, overrides, outcome, pass, metrics
  sim/results/<batch>/summary.csv|md one row per sweep point (seed pooled): planned / completed / passed /
                                     landed counts, mean +- sd over completed runs (analysis/batch_table.py)
  sim/results/<batch>/early.csv      runs that ended early: end time and reason
  sim/results/<batch>/batch.json     batch file, base scenario hash, firmware / code / host info, timing,
                                     flight and analysis versions in the table, what --resume did
runs.csv is rewritten after every finished run. A run that ends in an exception (setup failure) is
retried once; trips are results and are not retried.

--resume never mixes experiments. An old run is used only if its scenario hash, flight version
(firmware, parameter files, airframe, client, flight procedure: runner/manifest.py) and law source are
the ones of this workspace. Then, if the analysis code is unchanged too, its result is taken as is; if
only the analysis changed, its stored logs are analysed again (the manifest records the new analysis
version). Anything else is flown again, and the reason is logged.

The exit code says whether the batch was executed (0: at least one run, none with outcome "error"), not
whether the runs passed: pass rates and landings are results, in summary.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from analysis.batch_table import early_runs, flag, summarize, to_markdown  # noqa: E402
from client.laws import source_sha256  # noqa: E402
from runner import manifest as mf  # noqa: E402
from runner.run import reanalyze, run_scenario  # noqa: E402
from runner.scenario import RunSpec, expand_batch, scenario_hash  # noqa: E402
from runner.seed import build as build_seed_shim  # noqa: E402
from sitl_runner import RUNS_ROOT  # noqa: E402

RESULTS = SIM / "results"
_slot: int | None = None


def _init(q, base_instance: int):
    global _slot
    _slot = base_instance + q.get()


def _previous(batch: str, spec: RunSpec) -> tuple[str, str]:
    """What an existing run directory is good for: ("reuse" | "reanalyze" | "fly", reason)."""
    rd = RUNS_ROOT / batch / spec.run_id
    try:
        man = json.loads((rd / "manifest.json").read_text())
        json.loads((rd / "metrics.json").read_text())["summary"]
    except (OSError, KeyError, ValueError):
        return "fly", "no finished run"
    if man.get("scenario_sha256") != scenario_hash(spec.scenario):
        return "fly", "scenario changed"
    if "flight" not in man:
        return "fly", "flight version not recorded (older runner)"
    now = mf.flight_info()
    if man["flight"]["sha256"] != now["sha256"]:
        return "fly", "flight version changed: " + ", ".join(mf.changed_files(man["flight"], now))
    law = man.get("law") or {}
    if law.get("source_sha256") != source_sha256(spec.scenario["controller"]["law"]):
        return "fly", "law source changed"
    if (man.get("analysis") or {}).get("sha256") == mf.analysis_info()["sha256"]:
        return "reuse", ""
    if not sorted((rd / "logs").glob("*.BIN")):
        return "fly", "analysis changed and no .BIN log to analyse again"
    return "reanalyze", "analysis changed: " + ", ".join(mf.changed_files(man.get("analysis") or {},
                                                                           mf.analysis_info()))


def _stored(batch: str, spec: RunSpec) -> dict:
    return json.loads((RUNS_ROOT / batch / spec.run_id / "metrics.json").read_text())["summary"]


def _work(batch: str, spec: RunSpec, action: str = "fly", retries: int = 1) -> dict:
    log = lambda s: print(f"[{spec.run_id}] {s}", flush=True)  # noqa: E731
    if action == "reanalyze":
        s = reanalyze(RUNS_ROOT / batch / spec.run_id, log)
        s["resumed"] = "reanalyzed"
        return s
    for attempt in range(retries + 1):
        s = run_scenario(spec.scenario, spec.run_id, batch, _slot, log)
        if s["outcome"] != "error":
            break
        log(f"attempt {attempt + 1} failed: {s['reason']}")
    s["attempts"] = attempt + 1
    s["instance"] = _slot
    return s


def _wall_estimate(spec: RunSpec, setup_fc_s: float = 60.0) -> float:
    """Rough wall time of a run: outer loop at speedup, boot/takeoff/landing (~60 s FC) at setup_speedup."""
    run = spec.scenario["run"]
    return spec.scenario["task"]["duration_s"] / run["speedup"] + setup_fc_s / (run["setup_speedup"] or run["speedup"])


def _row(spec: RunSpec, summary: dict) -> dict:
    return {"group": spec.group, "repeat": spec.repeat, **spec.overrides, **summary}


def _print(s: str):
    print(s, flush=True)


def run_batch(path: str | Path, workers: int | None = None, base_instance: int = 30, resume: bool = False,
              only: str | None = None, log=_print, overrides: dict | None = None,
              name: str | None = None) -> pd.DataFrame:
    """overrides: dotted key -> value for every run (e.g. the law under test); name: results/<name> instead
    of the batch file's name, so the same suite flown with different laws keeps separate tables."""
    meta, specs = expand_batch(path, overrides)
    if name:
        meta["name"] = name
    if only:
        specs = [s for s in specs if fnmatch.fnmatch(s.run_id, only)]
    workers = workers or meta["workers"]
    out_dir = RESULTS / meta["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    build_seed_shim()                                   # once, before workers start SITL
    # snapshot at batch start, inherited by the forked workers, so every manifest of the batch agrees
    mf.code_info(), mf.ardupilot_info(), mf.host_info(), mf.flight_info(), mf.analysis_info()
    rows, todo, resumed = {}, [], {"reused": 0, "reanalyzed": 0, "flown_again": {}}
    for s in specs:
        action, why = _previous(meta["name"], s) if resume else ("fly", "")
        if action == "reuse":
            rows[s.run_id] = _row(s, {**_stored(meta["name"], s), "resumed": "reused"})
            resumed["reused"] += 1
            continue
        if action == "reanalyze":
            resumed["reanalyzed"] += 1
        elif resume and why != "no finished run":
            kind = why.split(":")[0]
            resumed["flown_again"][kind] = resumed["flown_again"].get(kind, 0) + 1
        if why:
            log(f"{s.run_id}: {action} ({why})")
        todo.append((s, action))
    todo.sort(key=lambda sa: (-_wall_estimate(sa[0]) if sa[1] == "fly" else 0.0, sa[0].run_id))   # longest first
    n_fly = sum(a == "fly" for _, a in todo)
    log(f"batch {meta['name']}: {len(specs)} runs, {len(rows)} reused, {len(todo) - n_fly} to analyse again, "
        f"{n_fly} to fly on {workers} workers")

    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    wall0 = time.monotonic()

    def write():
        df = pd.DataFrame([rows[s.run_id] for s in specs if s.run_id in rows])
        df.to_csv(out_dir / "runs.csv", index=False)
        return df

    if todo:
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        for i in range(workers):
            q.put(i)
        with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init, initargs=(q, base_instance)) as ex:
            futs = {ex.submit(_work, meta["name"], s, a): s for s, a in todo}
            for n, f in enumerate(as_completed(futs), 1):
                s = futs[f]
                try:
                    rows[s.run_id] = _row(s, f.result())
                except Exception as e:                  # worker crashed outside run_scenario: a planned run
                    rows[s.run_id] = _row(s, {"run_id": s.run_id, "outcome": "error", "reason": repr(e),
                                              "completed": False, "pass": False, "pass_fails": "outcome error",
                                              "landed": False})      # without a result, never a pass
                r = rows[s.run_id]
                log(f"({n}/{len(todo)}) {s.run_id}: {r['outcome']} pass={r.get('pass')} landed={r.get('landed')} "
                    f"wall {r.get('wall_s', float('nan')):.0f} s")
                write()
    df = write()
    summ, early = summarize(df), early_runs(df)
    summ.to_csv(out_dir / "summary.csv", index=False)
    early.to_csv(out_dir / "early.csv", index=False)
    md = "\n\n".join([f"# {meta['name']}", meta["description"].strip(), to_markdown(summ, early=early)])
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    versions = {k: sorted(df[k].dropna().unique()) if k in df else [] for k in ("flight_sha", "analysis_sha")}
    counts = {"planned": len(specs), "completed": int((df.outcome == "ok").sum()) if len(df) else 0,
              "pass": int((flag(df.get("pass"), df.index) & (df.outcome == "ok")).sum()) if len(df) else 0,
              "landed": int(flag(df.get("landed"), df.index).sum()) if len(df) else 0}
    info = {**meta, "started_utc": started, "wall_s": time.monotonic() - wall0, "workers": workers,
            "base_instance": base_instance, "n_runs": len(specs), "counts": counts,
            "outcomes": df.outcome.value_counts().to_dict() if len(df) else {},
            "ardupilot": mf.ardupilot_info(), "code": mf.code_info(), "flight": mf.flight_info(),
            "analysis": mf.analysis_info(), "versions_in_table": versions, "resume": resumed if resume else None,
            "host": mf.host_info(), "runs_root": str(RUNS_ROOT / meta["name"])}
    (out_dir / "batch.json").write_text(json.dumps(info, indent=2, default=str))
    if len(versions["flight_sha"]) > 1:                  # cannot happen through --resume; say so if it does
        log(f"WARNING: runs of {len(versions['flight_sha'])} flight versions in one table: {versions['flight_sha']}")
    log(f"batch {meta['name']} finished in {info['wall_s']:.0f} s: {info['outcomes']}, {counts} -> "
        f"{out_dir / 'runs.csv'}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="run a scenario batch")
    ap.add_argument("batch")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--base-instance", type=int, default=30, help="first SITL instance (M2 tests use 0-2)")
    ap.add_argument("--resume", action="store_true",
                    help="keep finished runs of the same scenario, flight version and law; analyse them again "
                         "if only the analysis code changed")
    ap.add_argument("--only", help="fnmatch pattern on run_id")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a scenario key in every run (YAML value), e.g. controller.law=mylaws:Law; "
                         "repeatable")
    ap.add_argument("--name", help="results/<name> instead of the batch file's name")
    a = ap.parse_args()
    overrides = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        if not k or not _:
            ap.error(f"--set needs KEY=VALUE, got {kv!r}")
        overrides[k.strip()] = yaml.safe_load(v)
    df = run_batch(a.batch, a.workers, a.base_instance, a.resume, a.only, overrides=overrides, name=a.name)
    return 0 if len(df) and (df.outcome != "error").all() else 1


if __name__ == "__main__":
    sys.exit(main())
