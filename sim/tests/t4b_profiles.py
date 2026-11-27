"""T4-B (M4): the T4 outer-loop flight under sensor profile B (optical flow + rangefinder), against A.

Batch scenarios/batches/m4_profiles.yaml, base scenarios/t4b_steps_flow.yaml (SAKR3 1.40 kg, still air,
T4 steps, PD test law without horizontal integrator), 60 seeds per group:
  flow          profile B, setup (boot/takeoff/landing) at 10x, loop at 1x
  flow_full_1x  profile B, everything at 1x
  gps           profile A, setup at 10x
T4-B passes if
  all_runs_ok         every run of the batch flies, lands and has all its parameters applied
  interface_B         the loop interface is unchanged under B: commands 50 Hz at the FC (49.5-50.5),
                      latency p95 <= 20 ms in every run
  ekf_height_B        the rangefinder gives the EKF its height: EKF-vs-truth height RMS mean < 2 cm,
                      every run < 3 cm
  setup_consistent_B  fast-forwarding the setup changes no metric under B either (T5 criterion: bias bound
                      |dmean| + 2 SE <= max(sigma_N, resolution), sigma_N from the flow group)
The A/B comparison of the tracking metrics is the result, not a check: the PD law holds the EKF's
position, and under B that estimate drifts.

Usage (WSL, venv active):  python sim/tests/t4b_profiles.py [--from-data] [--workers N] [--base-instance I]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM / "tests"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.batch_table import flag  # noqa: E402
from runner.batch import RESULTS, run_batch  # noqa: E402
from t5_repeatability import KEYS, LABELS, NL, _sd, censor_settle, compare, load_rows  # noqa: E402

BATCH = SIM / "scenarios/batches/m4_profiles.yaml"
OUT = RESULTS / "m4_profiles"
LAT_BUDGET_MS = 20.0
KEYS_B = {**KEYS, "est_err_h_rms_m": 0.002, "est_drift_h_m": 0.002}
LABELS_B = {**LABELS, "est_err_h_rms_m": "EKF horiz. err RMS [m]", "est_drift_h_m": "EKF horiz. drift in run [m]"}
TABLE = ["iae_m_s", "iae_h_m_s", "iae_v_m_s", "hold_rms_h_m", "hold_rms_v_m", "seg1_overshoot_pct",
         "seg3_overshoot_pct", "seg1_final_err_m", "seg3_final_err_m", "seg4_final_err_m", "max_tilt_deg",
         "est_err_h_rms_m", "est_err_h_max_m", "est_drift_h_m", "est_err_v_rms_m", "lat_p50_ms", "lat_p95_ms",
         "cmd_gap_max_ms"]


def analyze(df: pd.DataFrame) -> dict:
    df = censor_settle(df)
    flow, full, gps = (df[df.group == g] for g in ("flow", "flow_full_1x", "gps"))
    floor = {}
    for k, res in KEYS_B.items():
        s = _sd(flow[k])
        floor[k] = {"mean": float(np.nanmean(flow[k])), "sigma_N": s, "resolution": res,
                    "floor": max(s, res) if np.isfinite(s) else float("nan")}
    per = compare(flow[flow.outcome == "ok"], full[full.outcome == "ok"], floor, KEYS_B)
    table = {}
    for k in TABLE:
        a, b = gps[k].dropna(), flow[k].dropna()
        d = float(b.mean() - a.mean())
        se = float(np.sqrt(b.var(ddof=1) / len(b) + a.var(ddof=1) / len(a)))
        table[k] = {"A_mean": float(a.mean()), "A_sd": float(a.std(ddof=1)), "B_mean": float(b.mean()),
                    "B_sd": float(b.std(ddof=1)), "B_minus_A": d, "se": se, "n": [len(a), len(b)]}
    t4_pass = {g: float(flag(d.get("pass"), d.index).mean()) for g, d in df.groupby("group")}
    typical = {g: d.loc[~flag(d.get("pass"), d.index), "pass_fails"].dropna().str.split("; ").explode()
               .str.split("=").str[0].value_counts().head(4).to_dict() for g, d in df.groupby("group")}
    b_all = pd.concat([flow, full])
    checks = {
        "all_runs_ok": bool((df.outcome == "ok").all() and df.landed.astype(bool).all() and len(df) == 180),
        "interface_B": bool(b_all.cmd_rate_hz.between(49.5, 50.5).all() and (b_all.lat_p95_ms <= LAT_BUDGET_MS).all()),
        "ekf_height_B": bool(b_all.est_err_v_rms_m.mean() < 0.02 and (b_all.est_err_v_rms_m < 0.03).all()),
        "setup_consistent_B": all(m["consistent"] is not False for m in per.values()),
    }
    checks["T4B_pass"] = all(checks.values())
    return {"checks": checks, "inconsistent": [k for k, m in per.items() if m["consistent"] is False],
            "setup_consistency_B": per, "noise_floor_B": floor, "A_vs_B": table,
            "t4_step_shape_pass_rate": t4_pass, "t4_typical_fails": typical,
            "outcomes": df.groupby("group").outcome.value_counts().unstack(fill_value=0).to_dict(orient="index"),
            "wall_s_mean": df.groupby("group").wall_s.mean().to_dict(),
            "ekf_drift_B": {"median": float(flow.est_drift_h_m.median()),
                            "p90": float(flow.est_drift_h_m.quantile(0.9)), "max": float(flow.est_drift_h_m.max())}}


def plot(df: pd.DataFrame, out: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    SA, SB = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF})
    runs = {g: list(df[df.group == g].sort_values("seed").run_dir) for g in ("gps", "flow")}
    rows = {g: [load_rows(d) for d in dirs] for g, dirs in runs.items()}
    fig = plt.figure(figsize=(14.5, 8.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], hspace=0.42, wspace=0.42)
    sty = {"gps": (SA, "A: GPS + baro", 2), "flow": (SB, "B: flow + rangefinder", 1)}
    # the tighter group on top: A for the north step, B (rangefinder height) for the altitude step
    for ax, col, t0, t1, title, ylim, top in (
            (fig.add_subplot(gs[0, 0]), "t_n", 30, 46, "north step +2 m", (-1.2, 3.2), "gps"),
            (fig.add_subplot(gs[0, 1]), "t_d", 6, 22, "altitude step +1 m", (-0.4, 1.6), "flow")):
        for g in [x for x in ("flow", "gps") if x != top] + [top]:
            c, lab, _ = sty[g]
            z = 2 if g == top else 1
            for i, r in enumerate(rows[g]):
                w = r[(r.t >= t0) & (r.t <= t1)]
                y = -w.t_d + r.sp_z.iloc[0] if col == "t_d" else w.t_n - r.sp_x.iloc[0]
                ax.plot(w.t, y, color=c, lw=0.7, alpha=0.35, zorder=z, label=lab if i == 0 else None)
        ax.set_ylim(*ylim)
        ax.set_xlabel("outer-loop time [s]")
        ax.set_ylabel("true position rel. to handover [m]")
        ax.set_title(f"{title}, 60 seeds per profile", fontsize=9)
        ax.grid(color=GRID, lw=0.6)
        leg = ax.legend(frameon=False, fontsize=7.5, loc="upper right" if col == "t_d" else "upper left")
        for h in leg.legend_handles:
            h.set_alpha(1)
            h.set_linewidth(1.6)
    ax = fig.add_subplot(gs[1, 0])
    for g in ("flow", "gps"):
        c, lab, z = sty[g]
        grid = np.arange(0, 56, 0.1)
        errs = []
        for r in rows[g]:
            e = np.hypot(r.x - r.t_n, r.y - r.t_e)
            errs.append(np.interp(grid, r.t, e))
            ax.plot(r.t, e, color=c, lw=0.5, alpha=0.2, zorder=z)
        ax.plot(grid, np.median(errs, axis=0), color=c, lw=1.8, zorder=3, label=f"{lab} (median bold)")
    ax.set_xlabel("outer-loop time [s]")
    ax.set_ylabel("|EKF − true| horizontal [m]")
    ax.set_title("EKF horizontal position error", fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = fig.add_subplot(gs[1, 1])
    t = out["A_vs_B"]
    ks = ["hold_rms_h_m", "hold_rms_v_m", "seg3_final_err_m", "seg1_final_err_m", "est_err_v_rms_m"]
    names = ["hold RMS horizontal", "hold RMS vertical", "after north step:" + NL + "final err 3D",
             "after alt step:" + NL + "final err 3D", "EKF height err RMS"]
    yy = np.arange(len(ks))
    for off, g, key in ((-0.18, "gps", "A"), (0.18, "flow", "B")):
        c, lab, _ = sty[g]
        ax.barh(yy + off, [t[k][f"{key}_mean"] for k in ks], height=0.34, color=c, label=lab,
                xerr=[t[k][f"{key}_sd"] for k in ks], error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 2})
    ax.set_yticks(yy)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("true position error [m] (mean ± sd over 60 seeds)")
    ax.set_title("tracking on true position", fontsize=9)
    ax.grid(color=GRID, lw=0.6, axis="x")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax = fig.add_subplot(gs[:, 2])
    per = out["setup_consistency_B"]
    ks = list(KEYS_B)
    yy = np.arange(len(ks))
    ratio = [per[k]["ratio"] for k in ks]
    ax.barh(yy, ratio, height=0.6, color=[SB if r <= 1 else "#c0392b" for r in ratio])
    ax.axvline(1, color=INK2, lw=0.9, ls="--")
    for y, r in zip(yy, ratio):
        ax.text(r + 0.02, y, f"{r:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(yy)
    ax.set_yticklabels([LABELS_B[k] for k in ks], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("(|Δmean| + 2·SE) / noise floor under B")
    ax.set_title("profile B: setup at 10× vs everything at 1×" + NL + "(≤ 1: consistent)", fontsize=9)
    ax.grid(color=GRID, lw=0.6, axis="x")
    for a in fig.axes:
        a.spines[["top", "right"]].set_visible(False)
    ok = "passed" if out["checks"]["T4B_pass"] else "FAILED"
    fig.suptitle(f"T4-B: SAKR3 1.40 kg, T4 steps in still air, PD test law — sensor profile B vs A, "
                 f"60 seeds each (T4-B {ok})", fontsize=10.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main() -> int:
    a = sys.argv
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else None
    base = int(a[a.index("--base-instance") + 1]) if "--base-instance" in a else 30
    df = pd.read_csv(OUT / "runs.csv") if "--from-data" in a else run_batch(BATCH, workers, base, resume=True)
    out = analyze(df)
    (OUT / "t4b_checks.json").write_text(json.dumps(out, indent=2, default=str))
    plot(df, out, OUT / "t4b_profiles.png")
    print(json.dumps({k: out[k] for k in ("checks", "inconsistent", "t4_step_shape_pass_rate", "t4_typical_fails",
                                          "outcomes", "wall_s_mean", "ekf_drift_B")}, indent=2, default=str))
    for k, v in out["A_vs_B"].items():
        print(f"{k:20s} A {v['A_mean']:8.4f} ± {v['A_sd']:.4f}   B {v['B_mean']:8.4f} ± {v['B_sd']:.4f}   "
              f"B-A {v['B_minus_A']:+.4f} (2SE {2 * v['se']:.4f})")
    return 0 if out["checks"]["T4B_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
