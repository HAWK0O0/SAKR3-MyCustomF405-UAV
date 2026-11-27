"""Plot altitude vs time for the T1 run, coloured by flight phase."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "uavsim_runs/T1_takeoff_hover_land"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else run_dir / "t1_altitude.png"

df = pd.read_csv(run_dir / "t1_local_position.csv")
df["t"] = (df.time_boot_ms - df.time_boot_ms.iloc[0]) / 1000.0
df["alt"] = -df.z

colors = {"climb": "#2a7ab0", "hover": "#2e9e5b", "land": "#c0602a"}
names = {"climb": "climb (takeoff cmd)", "hover": "hover 10 s", "land": "LAND mode"}
fig, ax = plt.subplots(figsize=(8, 3.6))
for ph, g in df.groupby("phase", sort=False):
    ax.plot(g.t, g.alt, ".", ms=3, color=colors.get(ph, "k"), label=names.get(ph, ph))
ax.axhline(5.0, ls="--", lw=0.8, color="grey")
ax.text(df.t.iloc[-1], 5.08, "target 5 m", ha="right", va="bottom", fontsize=8, color="grey")
ax.set_xlabel("time since takeoff command [s]")
ax.set_ylabel("altitude [m]")
ax.set_title("T1: default SITL quad, scripted takeoff / hover / land")
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(out)
