"""Read ArduPilot DataFlash (.BIN) logs into pandas DataFrames (via pymavlink DFReader).

Works for SITL and real-vehicle logs alike.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pymavlink import mavutil


def read_messages(bin_path: str | Path, types: list[str]) -> dict[str, pd.DataFrame]:
    """Return {msg_type: DataFrame}; every frame has a TimeUS column (FC boot time, us)."""
    rows: dict[str, list[dict]] = {t: [] for t in types}
    log = mavutil.mavlink_connection(str(bin_path), dialect="ardupilotmega")
    while True:
        msg = log.recv_match(type=types)
        if msg is None:
            break
        d = msg.to_dict()
        d.pop("mavpackettype", None)
        rows[msg.get_type()].append(d)
    return {t: pd.DataFrame(r) for t, r in rows.items()}


def read_params(bin_path: str | Path) -> dict[str, float]:
    log = mavutil.mavlink_connection(str(bin_path), dialect="ardupilotmega")
    params = {}
    while True:
        msg = log.recv_match(type="PARM")
        if msg is None:
            break
        params[msg.Name] = msg.Value
    return params


def window(df: pd.DataFrame, t0_us: int, t1_us: int) -> pd.DataFrame:
    return df[(df.TimeUS >= t0_us) & (df.TimeUS <= t1_us)]
