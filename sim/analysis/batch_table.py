"""Batch summary: one row per sweep point (all overrides except the seed).

Counts, each over all planned runs of the point (n):
  n_completed  outer loop flew the whole planned duration and the run was analysed (outcome "ok")
  n_pass       completed and within the scenario's pass bounds; pass_rate = n_pass / n
  n_landed     landing confirmed (seen disarmed)
A run without a verdict (worker crashed, no `pass` value) counts as planned and not passed.

Metric mean / sd are over the completed runs only. A run that stopped early integrates its error over
a shorter time (an IAE over 12 s is not comparable with one over 56 s), so it stays out of the
statistics and is listed with its end time and reason instead: n_early, early_t_end_s, early_reasons
per point, and one row per run from early_runs().
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

METRICS = ["iae_m_s", "hold_rms_h_m", "hold_rms_v_m", "seg1_overshoot_pct", "seg3_overshoot_pct",
           "seg1_final_err_m", "seg3_final_err_m", "max_tilt_deg", "thrust_p99", "clamped_frac", "lat_p95_ms"]
SEED_KEYS = {"run.seed", "repeat", "seed"}
COUNTS = ("n", "n_completed", "n_pass", "n_landed", "pass_rate")


def flag(col: pd.Series | None, index=None) -> pd.Series:
    """A verdict column as bool; missing values (NaN, no column) are False, never True."""
    if col is None:
        return pd.Series(False, index=index)
    return col.map(lambda v: v if isinstance(v, (bool, np.bool_)) else str(v).strip().lower() == "true").astype(bool)


def completed(df: pd.DataFrame) -> pd.Series:
    return (df["outcome"] == "ok") if "outcome" in df else pd.Series(False, index=df.index)


def _reason_kind(reason) -> str:
    """'altitude 1.92 m outside [2.0, 10.0]' -> 'altitude', 'fence: at ...' -> 'fence'."""
    return re.split(r"[:(\d]", str(reason))[0].strip() if isinstance(reason, str) and reason else "unknown"


def _by(df: pd.DataFrame) -> list[str]:
    # sweep keys except the seed; run speed as flown (also when not overridden)
    by = ["group"] + [c for c in df.columns if "." in c and c not in SEED_KEYS and not c.startswith("run.")]
    return by + [c for c in ("speedup", "setup_speedup") if c in df]


def summarize(df: pd.DataFrame, by: list[str] | None = None, metrics: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    by = by or _by(df)
    df = df.copy()
    for c in by:                                         # list / dict overrides (task.steps): group by their text
        df[c] = df[c].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    metrics = [m for m in (metrics or METRICS) if m in df]
    rows = []
    for key, g in df.groupby(by, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        done, passed = completed(g), flag(g.get("pass"), g.index)
        passed &= done                                   # a pass needs a completed run
        r["n"] = len(g)
        r["n_completed"] = int(done.sum())
        r["n_pass"] = int(passed.sum())
        r["n_landed"] = int(flag(g.get("landed"), g.index).sum())
        r["pass_rate"] = r["n_pass"] / r["n"]
        outcome = g["outcome"].fillna("error") if "outcome" in g else pd.Series("error", index=g.index)
        r["outcomes"] = ", ".join(f"{k} {v}" for k, v in outcome.value_counts().items())
        fails = g.loc[done & ~passed, "pass_fails"].dropna() if "pass_fails" in g else pd.Series(dtype=str)
        fails = fails[fails != ""]
        r["typical_fail"] = fails.str.split("; ").explode().str.split("=").str[0].mode().iat[0] if len(fails) else ""
        early = g[~done]
        r["n_early"] = len(early)
        t_end = pd.to_numeric(early["t_end_s"], errors="coerce") if "t_end_s" in early else pd.Series(dtype=float)
        r["early_t_end_s"] = ", ".join(f"{t:.2f}" if np.isfinite(t) else "-" for t in t_end)
        kinds = (early["reason"] if "reason" in early else pd.Series(dtype=str)).map(_reason_kind)
        r["early_reasons"] = ", ".join(f"{k} {v}" for k, v in kinds.value_counts().items())
        for m in metrics:
            v = pd.to_numeric(g.loc[done, m], errors="coerce")
            r[f"{m}_mean"] = float(v.mean()) if len(v) else float("nan")
            r[f"{m}_sd"] = float(v.std(ddof=1)) if v.notna().sum() > 1 else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def early_runs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run that did not complete: where it ended and why. Its IAE covers only t_end_s."""
    if df.empty:
        return df
    e = df[~completed(df)]
    cols = [c for c in ("run_id", "outcome", "t_end_s", "reason", "iae_m_s", "landed") if c in e]
    return e[cols].rename(columns={"iae_m_s": "iae_until_end_m_s"}).reset_index(drop=True)


def to_markdown(s: pd.DataFrame, metrics: list[str] | None = None, early: pd.DataFrame | None = None) -> str:
    if s.empty:
        return ""
    metrics = [m for m in (metrics or METRICS) if f"{m}_mean" in s]
    skip = ("outcomes", "typical_fail", "n_early", "early_t_end_s", "early_reasons")
    keys = [c for c in s.columns if not c.endswith(("_mean", "_sd")) and c not in skip]
    head = keys + ["outcomes"] + metrics + ["typical fail"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in s.iterrows():
        cells = [f"{r[k]:.0%}" if k == "pass_rate" else str(r[k]) for k in keys] + [r["outcomes"]]
        for m in metrics:
            mu, sd = r[f"{m}_mean"], r[f"{m}_sd"]
            cells.append("—" if not np.isfinite(mu) else (f"{mu:.3g} ± {sd:.2g}" if np.isfinite(sd) else f"{mu:.3g}"))
        cells.append(r["typical_fail"] or "")
        lines.append("| " + " | ".join(cells) + " |")
    out = "\n".join(lines) + "\n"
    out += ("\nn = planned runs; n_completed flew the whole task; n_pass also met the pass bounds "
            "(pass_rate = n_pass / n); n_landed = landing confirmed. Metrics: mean ± sd over completed runs only.\n")
    if early is not None and len(early):
        out += "\n## Runs that ended early (not in the metric statistics)\n\n"
        out += "| " + " | ".join(early.columns) + " |\n|" + "---|" * len(early.columns) + "\n"
        for _, r in early.iterrows():
            out += "| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |\n"
    return out
