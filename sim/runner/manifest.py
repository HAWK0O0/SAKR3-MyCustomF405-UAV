"""Run manifest: what exactly was run (firmware, parameters, model, code, law, seed).

Two versions decide what an old run is still good for (runner/batch.py --resume):
  flight   : everything that shapes the flight itself: firmware binary and its default parameter files,
             our parameter files, airframe spec and model code, the client (loop, supervisor, laws,
             MAVLink helpers) and the flight procedure (runner/run.py, SITL launch, seed shim).
             A run flown under another flight version is another experiment and is never reused.
  analysis : analysis/*.py (truth conversion, metrics, tables). If only this changed, the stored raw
             logs are analysed again and the manifest records the new analysis version.
code_info() stays the hash of all code, for traceability.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from sitl_runner import ARDUPILOT, COPTER_BIN, COPTER_DEFAULTS, COPTER_X_DEFAULTS

SIM = Path(__file__).resolve().parents[1]
CODE_DIRS = ("client", "runner", "analysis", "airframes")
CODE_FILES = ("sakr3.py", "sitl_runner.py", "tools/seed_preload.c")
FLIGHT_DIRS = ("client", "airframes", "params")          # every file in them (not only *.py)
FLIGHT_FILES = ("runner/run.py", "runner/pilot.py", "runner/seed.py", "sakr3.py", "sitl_runner.py",
                "tools/seed_preload.c")
ANALYSIS_DIRS = ("analysis",)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(ARDUPILOT), *args], capture_output=True, text=True).stdout.strip()


@lru_cache(maxsize=1)
def ardupilot_info() -> dict:
    tags = _git("tag", "--points-at", "HEAD").split()
    return {"commit": _git("rev-parse", "HEAD"), "vehicle_tags": [t for t in tags if t.startswith("Copter-")],
            "binary": str(COPTER_BIN), "binary_sha256": sha256_file(COPTER_BIN),
            "binary_mtime": os.path.getmtime(COPTER_BIN)}


@lru_cache(maxsize=1)
def code_info() -> dict:
    files = [p for d in CODE_DIRS for p in sorted((SIM / d).glob("*.py"))] + [SIM / f for f in CODE_FILES]
    per = {str(p.relative_to(SIM)).replace("\\", "/"): sha256_file(p) for p in files if p.exists()}
    combined = hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()
    return {"sha256": combined, "files": per}


def _tree(dirs, files=()) -> dict[str, str]:
    paths = [p for d in dirs for p in sorted((SIM / d).iterdir()) if p.is_file()] + [SIM / f for f in files]
    return {str(p.relative_to(SIM)).replace("\\", "/"): sha256_file(p) for p in paths if p.exists()}


def _combined(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def flight_info() -> dict:
    files = _tree(FLIGHT_DIRS, FLIGHT_FILES)
    firmware = {"binary_sha256": ardupilot_info()["binary_sha256"],
                "default_params": {p.name: sha256_file(p) for p in (COPTER_DEFAULTS, COPTER_X_DEFAULTS)}}
    return {"sha256": _combined({"files": files, "firmware": firmware}), "files": files, "firmware": firmware}


@lru_cache(maxsize=1)
def analysis_info() -> dict:
    files = _tree(ANALYSIS_DIRS)
    return {"sha256": _combined(files), "files": files}


def changed_files(old: dict, new: dict) -> list[str]:
    """Names whose hash differs between two info dicts (flight_info / analysis_info), incl. firmware."""
    fo, fn = old.get("files", {}), new.get("files", {})
    out = sorted(k for k in set(fo) | set(fn) if fo.get(k) != fn.get(k))
    if old.get("firmware") != new.get("firmware"):
        out.append("firmware")
    return out


@lru_cache(maxsize=1)
def host_info() -> dict:
    import pymavlink
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpus": os.cpu_count(),
            "pymavlink": getattr(pymavlink, "__version__", "?")}


def param_files(paths) -> list[dict]:
    return [{"path": str(p), "sha256": sha256_file(p)} for p in paths]
