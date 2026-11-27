"""Per-run random seed for ArduPilot SITL via an LD_PRELOAD shim (tools/seed_preload.c).

SITL draws all simulated noise from glibc random()/rand() and never seeds it, so the seed is
set from outside, without touching the ArduPilot source.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
SRC = SIM / "tools/seed_preload.c"
LIB = Path.home() / "uavsim_tools/libuavsim_seed.so"


def build(force: bool = False) -> Path:
    """Compile the shim if missing or older than its source."""
    if force or not LIB.exists() or LIB.stat().st_mtime < SRC.stat().st_mtime:
        LIB.parent.mkdir(parents=True, exist_ok=True)
        tmp = LIB.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", str(tmp), str(SRC)], check=True)
        tmp.replace(LIB)          # atomic, safe if several workers build at once
    return LIB


def sitl_env(seed: int) -> dict[str, str]:
    return {"LD_PRELOAD": str(build()), "UAVSIM_SEED": str(int(seed))}
