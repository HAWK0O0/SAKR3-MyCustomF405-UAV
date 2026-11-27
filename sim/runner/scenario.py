"""Scenario files and batch expansion.

scenario = scenarios/defaults.yaml  <-deep merge-  scenario file  <-  batch overrides (dotted keys)

Keys are strict: a key that is not in defaults.yaml is an error (typos must not silently run the
default), except inside the free-form maps listed in FREE_FORM.

A batch file names a base scenario and one or more groups; each group has fixed overrides (`set`),
a `sweep` (Cartesian product over dotted keys; a value list or {range: [start, stop(, step)]}) and
`repeats`. A group may name its own `base` (a suite: one group per task), and `common: {set, sweep}`
applies to every group (a group's own keys win). Every combination becomes a RunSpec with a readable
run_id. Overrides passed to expand_batch (batch.py --set, e.g. the law under test) apply last.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SIM = Path(__file__).resolve().parents[1]
DEFAULTS = SIM / "scenarios/defaults.yaml"
FREE_FORM = {"environment.params", "controller.args", "pass"}
SHORT = {"run.seed": "s", "run.speedup": "x", "vehicle.mass_kg": "m", "vehicle.derate": "d",
         "environment.wind.speed_ms": "w", "environment.wind.dir_deg": "wd",
         "environment.wind.turbulence_ms": "wt", "controller.law": "law"}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def deep_merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        key = f"{path}.{k}" if path else k
        if key in FREE_FORM:                       # free-form map: merged without key checks
            out[k] = _merge_free(out.get(k) or {}, v or {})
        elif k not in out:
            raise KeyError(f"unknown scenario key {key!r}")
        elif isinstance(out[k], dict):
            if not isinstance(v, dict):
                raise TypeError(f"{key} must be a mapping")
            out[k] = deep_merge(out[k], v, key)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _merge_free(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge_free(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict)             else copy.deepcopy(v)
    return out


def _in_free(key: str) -> bool:
    return any(key == f or key.startswith(f + ".") for f in FREE_FORM)


def set_dotted(sc: dict, key: str, value) -> None:
    parts = key.split(".")
    d = sc
    for i, p in enumerate(parts[:-1]):
        if p not in d:
            if not _in_free(".".join(parts[:i + 1])):
                raise KeyError(f"unknown scenario key {key!r}")
            d[p] = {}
        d = d[p]
    if parts[-1] not in d and not _in_free(key):
        raise KeyError(f"unknown scenario key {key!r}")
    d[parts[-1]] = copy.deepcopy(value)


def validate(sc: dict) -> dict:
    v, task, run = sc["vehicle"], sc["task"], sc["run"]
    assert v["mass_kg"] > 0 and 0 < v["derate"] <= 1, "vehicle: mass > 0, 0 < derate <= 1"
    assert task["type"] in ("steps", "trajectory"), f"task.type {task['type']!r} not implemented"
    steps = task["steps"]
    assert steps and all(len(s) == 4 for s in steps), "task.steps: list of [t_s, dN, dE, dD]"
    ts = [s[0] for s in steps]
    assert ts[0] == 0 and ts == sorted(ts) and ts[-1] < task["duration_s"], \
        "task.steps: first at t=0, increasing, all before duration_s"
    if task["type"] == "trajectory":
        from client.trajectory import Trajectory
        tr = Trajectory.from_config(task["trajectory"])          # raises on a bad shape / value
        assert tr.end_s <= task["duration_s"], \
            f"task.trajectory ends at {tr.end_s:.1f} s, after duration_s {task['duration_s']}"
    for ev in task["events"]:
        kinds = [k for k in ("params", "push", "derate") if k in ev]
        assert set(ev) <= {"t_s", "params", "push", "derate"} and len(kinds) == 1 \
            and 0 <= ev["t_s"] < task["duration_s"], \
            "task.events: {t_s within duration, and one of params: {...}, push: {n, e, d, ms}, derate: k}"
        if "params" in ev:
            assert isinstance(ev["params"], dict), "task.events: params must be a mapping"
        elif "push" in ev:
            p = ev["push"]
            assert set(p) <= {"n", "e", "d", "ms"} and p.get("ms", 0) > 0, "task.events push: {n, e, d [m/s^2], ms > 0}"
        else:
            assert 0 < ev["derate"] <= 1, "task.events derate: 0 < k <= 1 (fraction of the thrust left)"
    gu = sc["environment"]["gusts"]
    assert gu["sigma_ms"] >= 0 and gu["tau_s"] > 0 and gu["rate_hz"] > 0, \
        "environment.gusts: sigma_ms >= 0, tau_s > 0, rate_hz > 0"
    for pm in task["pilot_modes"]:
        assert set(pm) == {"t_s", "mode"} and pm["t_s"] >= 0 and isinstance(pm["mode"], str),             "task.pilot_modes: [{t_s: >= 0, mode: NAME}, ...]"
    assert run["speedup"] > 0 and int(run["seed"]) >= 0, "run: speedup > 0, seed >= 0"
    assert run["setup_speedup"] is None or run["setup_speedup"] > 0, "run.setup_speedup: null or > 0"
    ht = sc["controller"]["hover_thrust"]
    assert ht == "fc" or (isinstance(ht, (int, float)) and 0 < ht < 1), "controller.hover_thrust: 'fc' or 0..1"
    assert (SIM / f"params/sensors_{sc['environment']['sensors']}.parm").exists(), \
        f"no params/sensors_{sc['environment']['sensors']}.parm"
    return sc


def load_scenario(src: str | Path | dict) -> dict:
    over = src if isinstance(src, dict) else _load_yaml(Path(src))
    return validate(deep_merge(_load_yaml(DEFAULTS), over))


def scenario_hash(sc: dict) -> str:
    return hashlib.sha256(json.dumps(sc, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class RunSpec:
    run_id: str
    group: str
    repeat: int
    overrides: dict = field(default_factory=dict)   # dotted key -> value (group set + sweep point)
    scenario: dict = field(default_factory=dict)


def _fmt(v) -> str:
    s = f"{v:g}" if isinstance(v, float) else str(v)
    return re.sub(r"[^A-Za-z0-9.+-]", "-", s)


def _run_id(label: str, point: dict, rep: int, repeats: int) -> str:
    parts = [f"{SHORT.get(k, k.split('.')[-1])}{_fmt(v)}" for k, v in point.items()]
    rid = f"{label}__{'_'.join(parts) or 'base'}"
    return f"{rid}__r{rep}" if repeats > 1 else rid


def expand_batch(path: str | Path, overrides: dict | None = None) -> tuple[dict, list[RunSpec]]:
    """Batch file -> runs. A group may name its own `base` scenario (relative to the batch file), so one
    batch can cover several tasks (a suite). `overrides` (dotted key -> value, e.g. the law under test
    from batch.py --set) apply to every run, after the group's set and sweep."""
    path = Path(path)
    b = _load_yaml(path)
    overrides = dict(overrides or {})
    base_path = (path.parent / b["base"]).resolve() if b.get("base") else None
    base = load_scenario(base_path) if base_path else None
    groups = b.get("groups") or [{"label": "main", "set": b.get("set", {}), "sweep": b.get("sweep", {}),
                                  "repeats": b.get("repeats", 1)}]
    common = b.get("common") or {}                     # set / sweep shared by all groups; a group's own wins
    groups = [{**g, "set": {**(common.get("set") or {}), **(g.get("set") or {})},
               "sweep": {**(common.get("sweep") or {}), **(g.get("sweep") or {})}} for g in groups]
    specs, seen, bases = [], set(), {}
    for g in groups:
        if g.get("base"):
            gb = (path.parent / g["base"]).resolve()
            bases[g["label"]] = str(gb)
            gbase = load_scenario(gb)
        elif base is not None:
            gbase = base
        else:
            raise ValueError(f"group {g['label']!r}: no base scenario (neither the batch nor the group names one)")
        label, fixed = g["label"], g.get("set") or {}
        sweep, repeats = dict(g.get("sweep") or {}), int(g.get("repeats", 1))
        for k, v in sweep.items():
            if isinstance(v, dict) and "range" in v:
                sweep[k] = list(range(*v["range"]))
        keys = list(sweep)
        for combo in itertools.product(*(sweep[k] for k in keys)):
            point = dict(zip(keys, combo))
            for rep in range(repeats):
                sc = copy.deepcopy(gbase)
                for k, v in {**fixed, **point, **overrides}.items():
                    set_dotted(sc, k, v)
                validate(sc)
                rid = _run_id(label, point, rep, repeats)
                if rid in seen:
                    raise ValueError(f"duplicate run id {rid}")
                seen.add(rid)
                specs.append(RunSpec(rid, label, rep, {**fixed, **point, **overrides}, sc))
    meta = {"name": b["name"], "description": b.get("description", ""), "file": str(path.resolve()),
            "base": str(base_path) if base_path else None, "group_bases": bases, "overrides": overrides,
            "workers": int(b.get("workers", 4)), "groups": groups,
            "base_sha256": scenario_hash(base) if base is not None else None}
    return meta, specs
