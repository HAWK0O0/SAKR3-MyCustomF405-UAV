"""Outer-loop laws a scenario can name (scenario key controller.law).

A law is any class with `step(state, setpoint, dt) -> Command` (see controller.py). It is built
with `cls.from_config(hover_thrust, **args)` if it has that classmethod, else `cls(hover_thrust, **args)`;
hover_thrust is read from the FC (MOT_THST_HOVER) at handover.

controller.law is a short name from REGISTRY or a "package.module:ClassName" path, so new laws
(later the RuleForge ones) plug in without changing the runner. The run manifest records the
class and the SHA-256 of its source file.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path

REGISTRY = {
    "pd": "client.controller:PDPositionController",   # T4/T6 test controller
}


def resolve(name: str) -> type:
    path = REGISTRY.get(name, name)
    if ":" not in path:
        raise ValueError(f"unknown law {name!r}; use one of {sorted(REGISTRY)} or 'module:Class'")
    mod, cls = path.split(":", 1)
    return getattr(importlib.import_module(mod), cls)


def source_sha256(name: str) -> str:
    """SHA-256 of the source file of a law (the manifest's law.source_sha256), without building it."""
    return hashlib.sha256(Path(inspect.getsourcefile(resolve(name))).read_bytes()).hexdigest()


def make_law(cfg: dict, hover_thrust: float):
    """cfg = scenario['controller'] -> (law instance, info for the manifest)."""
    cls = resolve(cfg["law"])
    args = dict(cfg.get("args") or {})
    law = cls.from_config(hover_thrust, **args) if hasattr(cls, "from_config") else cls(hover_thrust, **args)
    src = Path(inspect.getsourcefile(cls))
    info = {"law": cfg["law"], "class": f"{cls.__module__}:{cls.__qualname__}", "source": str(src),
            "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "args": args}
    return law, info
