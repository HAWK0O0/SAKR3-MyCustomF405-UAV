"""T5 (M3): noise floor and speedup consistency.

Batch scenarios/batches/t5.yaml, base scenarios/t5_steps_wind.yaml (T4 step sequence, 2 m/s wind
from 45 deg, 1 m/s turbulence, baro noise; PD test controller with horizontal integrator):
  same_seed : seed 1 five times, loop at 1x                -> spread with the seed fixed (sigma_R)
  seeds     : seeds 1-60 at loop speedup 1, 2, 5, 10, 20x  -> at 1x the noise floor sigma_N
  full_1x   : seeds 1-60, boot/takeoff/landing also at 1x  -> is fast-forwarding those phases harmless?
Everything except full_1x flies boot, takeoff and landing at 10x (run.setup_speedup) and only the
outer loop at the speedup under test.

Criterion (as fixed before the pilot run, results/m3_t5_pilot): a variant (speedup k, or the full-1x
setup) is consistent with the 1x reference for metric m if the upper ~95 % bound of its bias stays
within one run-to-run standard deviation at 1x:
    |mean_k - mean_1x| + 2 * sqrt(s_k^2/n_k + s_1x^2/n_1x)  <=  max(sigma_N(m), resolution(m))
The pilot paired seeds across speedups; a seed does not fix the in-flight noise (see the batch file),
so the comparison is unpaired (Welch standard error). The number of runs decides whether the test
can pass at all: with zero true bias the bound exceeds one sigma for 5 runs always, for 20 runs on
~24 % of metrics (results/m3_t5_n20), for 60 runs on ~0.05 %; 60 runs flag a bias of >= ~0.65 sigma.
Settling times are censored: a run that never enters the band counts as the segment length.
A loop speedup is usable if every metric is consistent, no run trips and latency p95 <= 20 ms.

T5 passes if every 1x run flies cleanly, the noise floor is defined for every metric, the seed sets
the noise realization (EKF height-reference error: sigma_N > 3 sigma_R), and fast-forwarding the
setup is consistent. The largest usable loop speedup is the second result.

Usage (WSL, venv active):
    python sim/tests/t5_repeatability.py [--from-data] [--workers N] [--base-instance I]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from runner.batch import RESULTS, run_batch  # noqa: E402

BATCH = SIM / "scenarios/batches/t5.yaml"
OUT = RESULTS / "m3_t5"
LAT_BUDGET_MS = 20.0
NL = "\n"
# task metrics compared across speedups, with the resolution below which a difference is irrelevant
KEYS = {
    "iae_m_s": 0.02, "iae_h_m_s": 0.02, "iae_v_m_s": 0.02,
    "hold_rms_h_m": 0.002, "hold_rms_v_m": 0.002,
    "seg1_overshoot_pct": 0.5, "seg3_overshoot_pct": 0.5,
    "seg1_rise_s": 0.05, "seg3_rise_s": 0.05,
    "seg1_settle_s": 0.1, "seg3_settle_s": 0.1,
    "seg1_final_err_m": 0.002, "seg3_final_err_m": 0.002,
    "max_tilt_deg": 0.1, "est_err_v_rms_m": 0.002,
}
LABELS = {
    "iae_m_s": "IAE 3D [m·s]", "iae_h_m_s": "IAE horizontal [m·s]", "iae_v_m_s": "IAE vertical [m·s]",
    "hold_rms_h_m": "hold RMS horiz. [m]", "hold_rms_v_m": "hold RMS vert. [m]",
    "seg1_overshoot_pct": "alt step overshoot [%]", "seg3_overshoot_pct": "north step overshoot [%]",
    "seg1_rise_s": "alt step rise [s]", "seg3_rise_s": "north step rise [s]",
    "seg1_settle_s": "alt step settle [s]", "seg3_settle_s": "north step settle [s]",
    "seg1_final_err_m": "alt step final err [m]", "seg3_final_err_m": "north step final err [m]",
    "max_tilt_deg": "max tilt [deg]", "est_err_v_rms_m": "EKF height err RMS [m]",
}


def _sd(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def load_rows(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / "rows.csv.gz")


def divergence(dirs: list[str]) -> dict:
    """Max / RMS distance of true trajectories from the first one, on the common time grid."""
    ref = load_rows(dirs[0])
    t = ref.t.values
    out = []
    for d in dirs[1:]:
        r = load_rows(d)
        p = np.column_stack([np.interp(t, r.t, r[c]) for c in ("t_n", "t_e", "t_d")])
        q = ref[["t_n", "t_e", "t_d"]].values
        dist = np.linalg.norm(p - q, axis=1)
        out.append((float(dist.max()), float(np.sqrt(np.mean(dist ** 2)))))
    return {"max_m": max(o[0] for o in out), "rms_m": float(np.mean([o[1] for o in out]))}


def compare(ref: pd.DataFrame, var: pd.DataFrame, floor: dict, keys=KEYS) -> dict:
    per = {}
    for k in keys:
        a, b = ref[k].dropna(), var[k].dropna()
        if len(a) < 4 or len(b) < 4:
            per[k] = {"n": [len(a), len(b)], "bias": float("nan"), "bound": float("nan"), "ratio": float("nan"),
                      "consistent": None}
            continue
        bias = float(b.mean() - a.mean())
        se = float(np.sqrt(b.var(ddof=1) / len(b) + a.var(ddof=1) / len(a)))
        bound = abs(bias) + 2 * se
        per[k] = {"n": [len(a), len(b)], "bias": bias, "se": se, "bound": bound,
                  "ratio": bound / floor[k]["floor"], "consistent": bool(bound <= floor[k]["floor"])}
    return per


def censor_settle(df: pd.DataFrame) -> pd.DataFrame:
    """seg{i}_settle_s NaN (never settled) -> segment length, so every run has a value."""
    from runner.scenario import load_scenario
    steps = load_scenario(SIM / "scenarios/t5_steps_wind.yaml")["task"]["steps"]
    df = df.copy()
    for i in (1, 3):
        df[f"seg{i}_settle_s"] = df[f"seg{i}_settle_s"].fillna(steps[i + 1][0] - steps[i][0])
    return df


def analyze(df: pd.DataFrame) -> dict:
    df = censor_settle(df)
    same = df[df.group == "same_seed"]
    seeds = df[df.group == "seeds"]
    full = df[df.group == "full_1x"]
    ref = seeds[seeds.speedup == 1]
    floor = {}
    for k, res in KEYS.items():
        sig_r, sig_n = _sd(same[k]), _sd(ref[k])
        floor[k] = {"mean_1x": float(np.nanmean(ref[k])), "sigma_R": sig_r, "sigma_N": sig_n,
                    "resolution": res, "floor": max(sig_n, res) if np.isfinite(sig_n) else float("nan"),
                    "n_finite_1x": int(ref[k].notna().sum())}
    variants = {"full_1x": full, **{f"x{x:g}": seeds[seeds.speedup == x] for x in sorted(seeds.speedup.unique())
                                    if x != 1}}
    cons = {}
    for name, v in variants.items():
        ok_runs = bool((v.outcome == "ok").all()) and len(v) == len(ref)
        per = compare(ref[ref.outcome == "ok"], v[v.outcome == "ok"], floor)
        lat95 = float(v.lat_p95_ms.max())
        c = {"metrics": per, "n": len(v), "outcomes": v.outcome.value_counts().to_dict(), "all_runs_ok": ok_runs,
             "lat_p50_mean_ms": float(v.lat_p50_ms.mean()), "lat_p95_max_ms": lat95,
             "lat_ok": bool(lat95 <= LAT_BUDGET_MS),
             "metrics_consistent": all(m["consistent"] is not False for m in per.values()),
             "inconsistent": [k for k, m in per.items() if m["consistent"] is False],
             "speedup_achieved_min": float(v.speedup_achieved.min()),
             "wall_s_mean": float(v.wall_s.mean())}
        c["usable"] = c["all_runs_ok"] and c["lat_ok"] and c["metrics_consistent"]
        cons[name] = c
    usable = 1.0
    for x in sorted(float(n[1:]) for n in cons if n.startswith("x")):
        if not cons[f"x{x:g}"]["usable"]:
            break
        usable = x
    one_x = pd.concat([same, ref, full])
    e = floor["est_err_v_rms_m"]
    checks = {
        "all_1x_runs_ok": bool((one_x.outcome == "ok").all() and len(one_x) == 125),
        "noise_floor_defined": all(np.isfinite(v["floor"]) for v in floor.values()),
        "seed_sets_noise_realization": bool(e["sigma_N"] > 3 * max(e["sigma_R"], 1e-6)),
        "setup_fast_forward_consistent": cons["full_1x"]["all_runs_ok"] and cons["full_1x"]["metrics_consistent"],
    }
    checks["T5_pass"] = all(checks.values())
    lat = seeds.groupby("speedup").agg(lat_p50=("lat_p50_ms", "mean"), lat_p95=("lat_p95_ms", "mean"),
                                       lat_p95_max=("lat_p95_ms", "max"), client_ms_p50=("client_ms_p50", "mean"),
                                       cmd_gap_max=("cmd_gap_max_ms", "max"), wall_s=("wall_s", "mean"),
                                       trips=("outcome", lambda o: int((o != "ok").sum())))
    return {"checks": checks,
            "speedup": {"max_usable_loop_speedup": usable, "loop_speedup_2x_usable": cons["x2"]["usable"]},
            "noise_floor": floor, "consistency": cons,
            "divergence": {"same_seed": divergence(list(same.sort_values("repeat").run_dir)),
                           "seeds_1x": divergence(list(ref.sort_values("seed").run_dir[:5]))},
            "timing_by_speedup": lat.reset_index().to_dict(orient="records"),
            "wall_s_mean": {"seeds_1x_fast_setup": float(ref.wall_s.mean()), "full_1x": float(full.wall_s.mean())}}


def plot(df: pd.DataFrame, out: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    S1, S2 = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURF,
                         "figure.facecolor": SURF})
    seeds = df[df.group == "seeds"]
    xs = sorted(seeds.speedup.unique())
    hi = max(xs)
    fig = plt.figure(figsize=(14.5, 8.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.3], hspace=0.42, wspace=0.55)
    panels = [(fig.add_subplot(gs[0, 0]), "t_d", 6, 22, "altitude step +1 m", (-0.4, 1.6)),
              (fig.add_subplot(gs[0, 1]), "t_n", 30, 46, "north step +2 m", (-0.3, 2.6))]
    for ax, col, t0, t1, title, ylim in panels:
        n_out = 0
        for x, c, lab, z in ((hi, S2, f"{hi:g}×", 1), (1.0, S1, "1×", 2)):     # 1x drawn on top
            for i, rd in enumerate(seeds[seeds.speedup == x].sort_values("seed").run_dir):
                r = load_rows(rd)
                w = r[(r.t >= t0) & (r.t <= t1)]
                y, y0 = (-w[col], -r.sp_z.iloc[0]) if col == "t_d" else (w[col], r.sp_x.iloc[0])
                y = y - y0
                n_out += int(x == hi and (y.min() < ylim[0] or y.max() > ylim[1]))
                ax.plot(w.t, y, color=c, lw=0.7, alpha=0.4, zorder=z, label=lab if i == 0 else None)
        ax.set_ylim(*ylim)
        ax.set_xlabel("outer-loop time [s]")
        ax.set_ylabel("true position rel. to handover [m]")
        ax.set_title(f"{title}, 60 seeds ({n_out} of the {hi:g}× runs leave the plot)", fontsize=8.5)
        ax.grid(color=GRID, lw=0.6)
        leg = ax.legend(frameon=False, fontsize=8, title="loop speedup", title_fontsize=8, loc="lower right")
        for h in leg.legend_handles:
            h.set_alpha(1)
            h.set_linewidth(1.6)
    ax = fig.add_subplot(gs[1, 0])
    for k, c, lab in (("lat_p50_ms", S1, "p50"), ("lat_p95_ms", S2, "p95")):
        ax.plot(seeds.speedup, seeds[k], "o", ms=3.5, color=c, mec=SURF, mew=0.6, alpha=0.6)
        g = seeds.groupby("speedup")[k].median()
        ax.plot(g.index, g.values, color=c, lw=1.4, label=f"latency {lab} (dots: runs, line: median)")
    ax.axhline(LAT_BUDGET_MS, color=INK2, lw=0.9, ls="--")
    ax.text(xs[0], LAT_BUDGET_MS * 1.12, "one control period (20 ms)", color=INK2, fontsize=7.5, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}×" for x in xs])
    ax.minorticks_off()
    ax.set_yticks([5, 10, 20, 50, 100, 500, 2000])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "500", "2000"])
    ax.set_xlabel("loop speedup")
    ax.set_ylabel("state sample → FC applies command [ms]")
    ax.set_title("loop latency in simulated time", fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = fig.add_subplot(gs[1, 1])
    fl = out["noise_floor"]
    ks = list(KEYS)
    yy = np.arange(len(ks))
    ax.barh(yy, [fl[k]["sigma_R"] / fl[k]["sigma_N"] for k in ks], height=0.6, color=S1)
    ax.axvline(1, color=INK2, lw=0.9, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels([LABELS[k] for k in ks], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("σ (seed 1, 5 runs) / σ (seeds 1–60)")
    ax.set_title("spread left with the seed fixed, 1×", fontsize=9)
    ax.grid(color=GRID, lw=0.6, axis="x")
    ax = fig.add_subplot(gs[:, 2])
    cons = out["consistency"]
    cols = ["full_1x"] + [f"x{x:g}" for x in xs if x != 1]
    names = {"full_1x": "1×, setup" + NL + "also 1×", **{f"x{x:g}": f"{x:g}×" for x in xs}}
    mat = np.array([[cons[c]["metrics"][k]["ratio"] for c in cols] for k in ks], float)
    cmap = LinearSegmentedColormap.from_list("div", ["#256abf", "#9ec5f4", "#f0efec", "#f2a58f", "#c0392b"])
    im = ax.imshow(np.clip(mat, 0.05, 20), cmap=cmap, norm=LogNorm(0.05, 20), aspect="auto")
    for i in range(len(ks)):
        for j in range(len(cols)):
            v = mat[i, j]
            ax.text(j, i, "n/a" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color=INK, fontweight="bold" if np.isfinite(v) and v > 1 else "normal")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([names[c] + NL + ("usable" if cons[c]["usable"] else "not usable") for c in cols],
                       fontsize=7.5)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([LABELS[k] for k in ks], fontsize=7.5)
    ax.set_title("bias bound / noise floor vs. 1× reference" + NL
                 + "(≤ 1: consistent, bold > 1; usable also needs latency p95 ≤ 20 ms, no trips)", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("(|Δmean| + 2·SE) / floor", fontsize=7.5)
    for a in fig.axes[:-1]:
        a.spines[["top", "right"]].set_visible(False)
    sp = out["speedup"]["max_usable_loop_speedup"]
    fig.suptitle(f"T5: SAKR3 1.40 kg, T4 steps in 2 m/s wind + turbulence + baro noise, 60 seeds — "
                 f"largest consistent loop speedup {sp:g}×", fontsize=10.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main() -> int:
    a = sys.argv
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else None
    base = int(a[a.index("--base-instance") + 1]) if "--base-instance" in a else 30
    df = pd.read_csv(OUT / "runs.csv") if "--from-data" in a else run_batch(BATCH, workers, base, resume=True)
    out = analyze(df)
    (OUT / "t5_checks.json").write_text(json.dumps(out, indent=2, default=str))
    plot(df, out, OUT / "t5_repeatability.png")
    print(json.dumps({k: out[k] for k in ("checks", "speedup", "divergence", "wall_s_mean")}, indent=2))
    for name, c in out["consistency"].items():
        print(f"{name:8s} usable={c['usable']} runs_ok={c['all_runs_ok']} lat p50 {c['lat_p50_mean_ms']:.1f} "
              f"p95max {c['lat_p95_max_ms']:.1f} ms  inconsistent={c['inconsistent']}")
    return 0 if out["checks"]["T5_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
