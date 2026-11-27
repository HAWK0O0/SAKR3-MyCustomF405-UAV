"""Launch a headless ArduCopter SITL instance in its own run directory.

Runs inside WSL2 (Ubuntu-22.04), with the uavsim venv active:
    source ~/venvs/uavsim/bin/activate

The command line is the one Tools/autotest/sim_vehicle.py builds (same binary, --model,
--defaults chain copter.parm[,copter-X.parm], -I, --speedup, --home), launched directly
instead of through sim_vehicle's run_in_terminal_window.sh, which opens an xterm under WSLg
and makes process cleanup unreliable for batch runs.

model examples: "quad" (ArduPilot default + frame), "x:/abs/path/sakr3_m1.40.json".
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

ARDUPILOT = Path(os.environ.get("ARDUPILOT_ROOT", Path.home() / "ardupilot"))
COPTER_BIN = ARDUPILOT / "build/sitl/bin/arducopter"
COPTER_DEFAULTS = ARDUPILOT / "Tools/autotest/default_params/copter.parm"
COPTER_X_DEFAULTS = ARDUPILOT / "Tools/autotest/default_params/copter-X.parm"
RUNS_ROOT = Path(os.environ.get("UAVSIM_RUNS", Path.home() / "uavsim_runs"))

# ArduPilot's standard SITL test location (CMAC, Canberra): lat, lon, alt[m], heading[deg]
DEFAULT_HOME = "-35.363261,149.165230,584,353"


@dataclass
class SitlInstance:
    run_name: str
    model: str = "quad"
    instance: int = 0
    speedup: float = 1.0
    home: str = DEFAULT_HOME
    extra_defaults: list[str] = field(default_factory=list)
    wipe: bool = True
    env: dict[str, str] = field(default_factory=dict)   # extra environment (e.g. LD_PRELOAD seed shim)

    proc: subprocess.Popen | None = None
    run_dir: Path | None = None

    @property
    def tcp_port(self) -> int:
        return 5760 + 10 * self.instance

    @property
    def connection(self) -> str:
        return f"tcp:127.0.0.1:{self.tcp_port}"

    def prepare_dir(self) -> Path:
        """Create (and if wipe, clear) the run directory; call before writing model files into it."""
        if self.run_dir is None:
            self.run_dir = RUNS_ROOT / self.run_name
            if self.run_dir.exists() and self.wipe:
                shutil.rmtree(self.run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def defaults_chain(self) -> list[str]:
        """Parameter files in load order (later files override earlier ones)."""
        chain = [str(COPTER_DEFAULTS)]
        if self.model.lower().startswith("x"):
            chain.append(str(COPTER_X_DEFAULTS))
        return chain + [str(p) for p in self.extra_defaults]

    def start(self) -> "SitlInstance":
        if not COPTER_BIN.exists():
            raise FileNotFoundError(f"{COPTER_BIN} not built; run ./waf copter")
        self.prepare_dir()

        defaults = ",".join(self.defaults_chain())
        cmd = [
            str(COPTER_BIN),
            "--model", self.model,
            "--speedup", str(self.speedup),
            "--home", self.home,
            "--defaults", defaults,
            "-I", str(self.instance),
        ]
        if self.wipe:
            cmd.append("--wipe")
        env_txt = "".join(f"{k}={v} " for k, v in self.env.items())
        (self.run_dir / "cmdline.txt").write_text(env_txt + " ".join(cmd) + "\n")
        self._log = open(self.run_dir / "sitl_stdout.log", "w")
        self.proc = subprocess.Popen(cmd, cwd=self.run_dir, stdout=self._log,
                                     stderr=subprocess.STDOUT, env={**os.environ, **self.env})
        time.sleep(1.0)
        if self.proc.poll() is not None:
            raise RuntimeError(f"SITL exited early, see {self.run_dir}/sitl_stdout.log")
        return self

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "_log", None):
            self._log.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
