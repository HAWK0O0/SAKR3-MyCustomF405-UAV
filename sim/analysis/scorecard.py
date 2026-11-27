"""Scorecard of one law on a suite (a batch whose groups are tasks, swept over theta and seeds).

Per task: planned runs, pass / completed / landed rates, the pass rate at the worst theta point, and the
task's headline metric (suite file: group `headline`; a pattern such as ev*_max_dev_m takes the max over
the matching metrics of a run), as mean over completed runs and at the worst theta point. Per task and
theta point: pass k/n and the headline mean (the grid in scorecard.png).

A law "works across the family" on a task only if every theta point passes; the worst cell says where
it does not. Pass bounds come from the task files and are proposals.

Usage (WSL, venv active):
    python sim/analysis/scorecard.py results/suite_core                      # one law
    python sim/analysis/scorecard.py results/suite_core results/suite_core__x  # side by side
"""
from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THETA = ["vehicle.mass_kg", "vehicle.derate"]


def _flag(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def headline_values(df: pd.DataFrame, pattern: str) -> pd.Series:
    cols = [c for c in df.columns if fnmatch.fnmatchcase(c, pattern)]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").max(axis=1, skipna=False)


def theta_label(r) -> str:
    return f"{r['vehicle.mass_kg']:g} kg / {r['vehicle.derate']:g}"


def score(result_dir: Path) -> dict:
    df = pd.read_csv(result_dir / "runs.csv")
    info = json.loads((result_dir / "batch.json").read_text())
    heads = {g["label"]: g.get("headline") for g in info["groups"]}
    df["passed"] = _flag(df["pass"]) & (df.outcome == "ok")
    df["completed_"] = df.outcome == "ok"
    df["landed_"] = _flag(df["landed"])
    df["theta"] = df.apply(theta_label, axis=1)
    tasks, cells = [], []
    for task in [g["label"] for g in info["groups"]]:
        d = df[df.group == task].copy()
        if d.empty:
            continue
        h = heads.get(task)
        d["headline"] = headline_values(d, h) if h else np.nan
        d.loc[~d.completed_, "headline"] = np.nan          # early-ended runs are not comparable
        per = d.groupby(["vehicle.mass_kg", "vehicle.derate"], sort=True)
        for (mass, der), c in per:
            cells.append({"task": task, "mass_kg": mass, "derate": der, "theta": theta_label(c.iloc[0]),
                          "n": len(c), "n_pass": int(c.passed.sum()), "pass_rate": float(c.passed.mean()),
                          "n_completed": int(c.completed_.sum()), "n_trip": int((c.outcome == "trip").sum()),
                          "headline": h,
                          "headline_mean": float(c.headline.mean()),
                          "fails": "; ".join(sorted(set(f.split("=")[0] for f in
                                                        c.loc[~c.passed, "pass_fails"].dropna().str.split("; ")
                                                        .explode())))})
        cd = pd.DataFrame([x for x in cells if x["task"] == task])
        worst = cd.sort_values(["pass_rate", "headline_mean"], ascending=[True, False]).iloc[0]
        tasks.append({"task": task, "n": len(d), "pass_rate": float(d.passed.mean()),
                      "completed_rate": float(d.completed_.mean()), "landed_rate": float(d.landed_.mean()),
                      "all_points_pass": bool((cd.pass_rate == 1.0).all()),
                      "worst_point": worst.theta, "worst_point_pass_rate": float(worst.pass_rate),
                      "headline": h, "headline_mean": float(d.headline.mean()),
                      "headline_worst_point": float(cd.headline_mean.max())})
    t = pd.DataFrame(tasks)
    overall = {"runs": int(len(df)), "pass_rate": float(df.passed.mean()),
               "tasks_all_points_pass": int(t.all_points_pass.sum()), "tasks": int(len(t)),
               "completed_rate": float(df.completed_.mean()), "landed_rate": float(df.landed_.mean()),
               "law": sorted(df["controller.law"].dropna().unique().tolist()) if "controller.law" in df else None,
               "overrides": info.get("overrides"), "flight_sha": sorted(df.flight_sha.dropna().unique().tolist())
               if "flight_sha" in df else None}
    return {"name": info["name"], "overall": overall, "tasks": t, "cells": pd.DataFrame(cells)}


def to_markdown(sc: dict) -> str:
    o, t = sc["overall"], sc["tasks"]
    lines = [f"# Scorecard: {sc['name']}", "",
             f"{o['runs']} runs; pass {o['pass_rate']:.0%}, completed {o['completed_rate']:.0%}, landed "
             f"{o['landed_rate']:.0%}; tasks passing at every theta point: {o['tasks_all_points_pass']} of {o['tasks']}."
             + (f" Overrides: `{o['overrides']}`." if o.get("overrides") else ""), "",
             "| task | runs | pass | worst theta point (pass) | headline | mean | worst point mean |",
             "|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r.task} | {r.n} | {r.pass_rate:.0%} | {r.worst_point} ({r.worst_point_pass_rate:.0%}) | "
                     f"`{r.headline}` | {r.headline_mean:.3g} | {r.headline_worst_point:.3g} |")
    c = sc["cells"]
    lines += ["", "Per theta point (pass k/n, trips, headline mean over completed runs; failing bounds):", "",
              "| task | theta | pass | trips | headline mean | failing |", "|---|---|---|---|---|---|"]
    for _, r in c.iterrows():
        head = f"{r.headline_mean:.3g}" if np.isfinite(r.headline_mean) else "— (none completed)"
        lines.append(f"| {r.task} | {r.theta} | {r.n_pass}/{r.n} | {r.n_trip} | {head} | {r.fails} |")
    return "\n".join(lines) + "\n"


def plot(sc: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
    c = sc["cells"]
    tasks = list(sc["tasks"].task)
    thetas = list(dict.fromkeys(c.sort_values(["derate", "mass_kg"], ascending=[False, True]).theta))
    mat = np.full((len(tasks), len(thetas)), np.nan)
    txt = [["" for _ in thetas] for _ in tasks]
    for _, r in c.iterrows():
        i, j = tasks.index(r.task), thetas.index(r.theta)
        mat[i, j] = r.pass_rate
        head = f"{r.headline_mean:.3g}" if np.isfinite(r.headline_mean) else "—"
        txt[i][j] = f"{r.n_pass}/{r.n}\n{head}" + (f"\ntrip {r.n_trip}" if r.n_trip else "")
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9ec5f4", "#2a78d6", "#123f80"])
    plt.rcParams.update({"font.size": 8.5, "text.color": INK, "axes.edgecolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "figure.facecolor": SURF})
    fig, ax = plt.subplots(figsize=(1.35 * len(thetas) + 3.2, 0.62 * len(tasks) + 1.8))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(tasks)):
        for j in range(len(thetas)):
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=7.5,
                    color="#fcfcfb" if np.isfinite(mat[i, j]) and mat[i, j] > 0.6 else INK)
    heads = dict(zip(sc["tasks"].task, sc["tasks"].headline))
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([f"{t}\n({heads[t]})" for t in tasks], fontsize=7.5)
    ax.set_xticks(range(len(thetas)))
    ax.set_xticklabels([th.replace(" / ", "\nderate ") for th in thetas], fontsize=7.5)
    ax.set_xlabel("theta point (take-off mass, available thrust)")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("pass rate", fontsize=7.5)
    o = sc["overall"]
    ax.set_title(f"{sc['name']}: pass {o['pass_rate']:.0%} of {o['runs']} runs; cell = passed/runs, the task's "
                 f"headline metric (mean over completed runs), trips", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def compare(scs: list[dict]) -> str:
    names = [s["name"] for s in scs]
    lines = ["| task | " + " | ".join(f"{n}: pass / worst point / headline" for n in names) + " |",
             "|---|" + "---|" * len(names)]
    tasks = list(dict.fromkeys(t for s in scs for t in s["tasks"].task))
    for task in tasks:
        cells = []
        for s in scs:
            r = s["tasks"][s["tasks"].task == task]
            cells.append("—" if r.empty else f"{r.pass_rate.iat[0]:.0%} / {r.worst_point_pass_rate.iat[0]:.0%} / "
                                                f"{r.headline_mean.iat[0]:.3g}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines.append("| overall | " + " | ".join(f"{s['overall']['pass_rate']:.0%}" for s in scs) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        print(__doc__)
        return 2
    scs = []
    for d in dirs:
        sc = score(d)
        (d / "scorecard.md").write_text(to_markdown(sc), encoding="utf-8")
        sc["tasks"].to_csv(d / "scorecard.csv", index=False)
        sc["cells"].to_csv(d / "scorecard_cells.csv", index=False)
        plot(sc, d / "scorecard.png")
        print(to_markdown(sc))
        scs.append(sc)
    if len(scs) > 1:
        md = compare(scs)
        (dirs[0] / "compare.md").write_text(md, encoding="utf-8")
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
