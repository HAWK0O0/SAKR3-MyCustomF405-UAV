"""SAKR3 SITL instances: frame model for a (mass, derate) variant + firmware parameter chain."""
from __future__ import annotations

from pathlib import Path

from airframes.make_airframe import Variant, write_variant
from client import mav
from client.outer_loop import request_streams
from sitl_runner import SitlInstance

SIM = Path(__file__).resolve().parent
BASE_PARM = SIM / "params/sakr3_base.parm"
OUTER_LOOP_PARM = SIM / "params/outer_loop.parm"


def sakr3_sitl(run_name: str, variant: Variant, instance: int = 0, extra_parms=(), speedup: float = 1.0,
               spec: dict | None = None) -> SitlInstance:
    sitl = SitlInstance(run_name=run_name, instance=instance, speedup=speedup)
    run_dir = sitl.prepare_dir()
    jpath, ppath = write_variant(variant, run_dir / "model", spec)
    # SITL's AP_Filesystem strips a leading '/', so the model path must be relative to run_dir
    sitl.model = f"x:{jpath.relative_to(run_dir)}"
    sitl.extra_defaults = [str(BASE_PARM)] + ([str(ppath)] if ppath else []) + [str(p) for p in extra_parms]
    return sitl


def parm_values(paths) -> dict[str, float]:
    """name -> value over .parm files in load order (later files win). Like AP_Param, a line is
    `NAME VALUE ...`; '#' lines and anything after the value are ignored."""
    out = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            tok = line.replace(",", " ").replace("=", " ").split()
            if len(tok) >= 2 and not tok[0].startswith("#"):
                try:
                    out[tok[0]] = float(tok[1])
                except ValueError:
                    pass
    return out


def param_check(fc_first: dict[str, float], paths) -> list[str]:
    """Problems with parameters our files set, against the FC's PARM log (first value per name).

    SITL ignores a name it does not know (e.g. one renamed between releases) without an error, so
    every run checks that each parameter of its own files arrived with its value.
    """
    out = []
    for k, v in parm_values(paths).items():
        if k not in fc_first:
            out.append(f"{k} unknown to the firmware")
        elif abs(fc_first[k] - v) > 1e-4 * max(1.0, abs(v)):
            out.append(f"{k}={fc_first[k]:g}, file says {v:g}")
    return out


def needs_ekf_origin(sitl: SitlInstance) -> bool:
    """No GPS in the parameter chain (GPS1_TYPE 0): the client has to set the EKF origin."""
    return parm_values(sitl.extra_defaults).get("GPS1_TYPE", 1) == 0


def fly_to_hover(m, alt: float, log=print, settle_s: float = 3.0, ekf_origin=None):
    """Streams, EKF wait, GUIDED, arm, FC takeoff to alt, settle (FC time). Returns (LOCAL_POSITION_NED, yaw).

    ekf_origin (lat, lon, alt_m): set this EKF origin first and wait for relative position (no GPS).
    """
    request_streams(m)
    if ekf_origin is not None:
        mav.set_ekf_origin(m, *ekf_origin, log=lambda s: None)
        mav.wait_ekf_position(m, log=lambda s: None, relative=True)
    else:
        mav.wait_ekf_position(m, log=lambda s: None)
    mav.set_mode(m, "GUIDED")
    mav.arm_with_retry(m)
    mav.takeoff(m, alt, log=log)
    lpn = mav.wait_hover(m, alt, settle_s)
    att = m.recv_match(type="ATTITUDE", blocking=True, timeout=2)
    return lpn, att.yaw
