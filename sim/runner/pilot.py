"""A pilot / GCS on a second MAVLink link, for takeover tests (scenario key task.pilot_modes).

Connects to SITL SERIAL1 (tcp 5762 + 10 * instance) with its own system id and sets FC modes at given
outer-loop times. It is polled from the outer loop's step hook and, once the loop has stopped, from
monitor(), so a pilot who takes over can also be the one who lands. Nothing here goes through the
client's own link: what the client sends is counted separately (client.mav.TxLog).
"""
from __future__ import annotations

from client import mav

PILOT_SYSID = 250


class Pilot:
    def __init__(self, sitl, changes: list[dict], log=print):
        self.pending = sorted(({"t_s": float(c["t_s"]), "mode": str(c["mode"])} for c in changes),
                              key=lambda c: c["t_s"])
        self.applied: list[dict] = []
        self.log = log
        self.link = mav.mavutil.mavlink_connection(f"tcp:127.0.0.1:{sitl.tcp_port + 2}",
                                                   source_system=PILOT_SYSID)
        if self.link.wait_heartbeat(timeout=10) is None:
            raise TimeoutError("no heartbeat on SERIAL1 (pilot link)")

    def poll(self, t: float):
        while self.pending and t >= self.pending[0]["t_s"]:
            c = self.pending.pop(0)
            self.link.set_mode(self.link.mode_mapping()[c["mode"]])
            self.applied.append({**c, "t_applied": t})
            self.log(f"pilot link: set mode {c['mode']} at t={t:.2f}")

    def close(self):
        self.link.close()
