"""Generate SITL frame models for the SAKR3 parameter family.

SITL (libraries/SITL/SIM_Frame.cpp, Copter-4.7.3) derives the motor model from the
JSON fields, not from a "max thrust" field:

    per-motor thrust(c) = (mass*g / num_motors / hoverThrOut) * vs^2 * f(c) * rho/rho_ref
    f(c) = (1-expo)*c + expo*c^2,     vs = battery voltage / maxVoltage
    effective prop area  ~ (mass*g)^3 / (2*rho*P^2),   P = refCurrent*refVoltage
    drag area areaCd     = (mass*g*tan(refAngle) - momentum_drag) / (0.5*rho*refSpd^2)

So changing only "mass" would scale thrust with it (thrust/weight unchanged) and also change
drag and prop area. To vary *only* the take-off mass we rescale, per mass point m:

    hoverThrOut = m*g / T_max                    -> max thrust stays T_max
    refCurrent  = I0 * (m/m0)^1.5                -> prop area and max outflow velocity unchanged
    refSpd      = solved so areaCd(m) = areaCd(m0) -> body drag unchanged

Thrust derating is *not* done in the model: like on the real vehicle it is done by lowering the
ArduCopter parameter MOT_SPIN_MAX, so the firmware commands less of the same physical motor range.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml
from scipy.optimize import brentq

G = 9.80665
SPEC_PATH = Path(__file__).with_name("sakr3_spec.yaml")


def load_spec(path: Path = SPEC_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def isa_density(alt_m: float) -> float:
    """ISA troposphere density; same model family as AP_Baro::get_air_density_for_alt_amsl."""
    T0, L, p0, R, M_air = 288.15, 0.0065, 101325.0, 8.31446, 0.0289644
    T = T0 - L * alt_m
    p = p0 * (T / T0) ** (G * M_air / (R * L))
    return p * M_air / (R * T)


def _drag_terms(mass, refSpd, refAngle, mdrag, disc_area, rho):
    a = math.radians(refAngle)
    drag_force = mass * G * math.tan(a)
    airspeed_bf = refSpd * math.cos(a)
    ref_thrust = mass * G / math.cos(a)
    momentum_drag = math.cos(a) * mdrag * airspeed_bf * math.sqrt(ref_thrust * rho * disc_area)
    return drag_force, momentum_drag


def _area_cd_raw(mass, refSpd, spec) -> float:
    rho = isa_density(spec["refAlt"])
    df, md = _drag_terms(mass, refSpd, spec["refAngle"], spec["mdrag_coef"], spec["disc_area_m2"], rho)
    return (df - md) / (0.5 * rho * refSpd ** 2)


def area_cd(mass, refSpd, spec) -> float:
    cd = _area_cd_raw(mass, refSpd, spec)
    if cd <= 0:
        raise ValueError("momentum drag exceeds total drag at reference point; adjust refSpd/refAngle")
    return cd


def _solve_refSpd(mass, target_cd, spec) -> float:
    # momentum drag is linear in refSpd, so areaCd > 0 only below v_crit
    rho = isa_density(spec["refAlt"])
    df, md1 = _drag_terms(mass, 1.0, spec["refAngle"], spec["mdrag_coef"], spec["disc_area_m2"], rho)
    v_crit = df / md1
    return brentq(lambda v: _area_cd_raw(mass, v, spec) - target_cd, 0.5, 0.999 * v_crit)


def frame_model(mass: float, spec: dict | None = None) -> dict:
    """SITL JSON frame model for take-off mass `mass` [kg], physics otherwise fixed."""
    spec = spec or load_spec()
    m0 = spec["base_mass_kg"]
    T_max = spec["max_total_thrust_n"]
    cd0 = area_cd(m0, spec["refSpd"], spec)
    refSpd = _solve_refSpd(mass, cd0, spec) if mass != m0 else spec["refSpd"]
    return {
        "mass": mass,
        "diagonal_size": spec["arm_length_m"],
        "refSpd": refSpd,
        "refAngle": spec["refAngle"],
        "refVoltage": spec["refVoltage"],
        "refCurrent": spec["refCurrent"] * (mass / m0) ** 1.5,
        "refAlt": spec["refAlt"],
        "refTempC": spec["refTempC"],
        "refBatRes": spec["refBatRes"],
        "maxVoltage": spec["maxVoltage"],
        "battCapacityAh": spec["battCapacityAh"],
        "propExpo": spec["propExpo"],
        "refRotRate": spec["refRotRate"],
        "hoverThrOut": mass * G / T_max,
        "pwmMin": spec["pwmMin"],
        "pwmMax": spec["pwmMax"],
        "spin_min": spec["spin_min"],
        "spin_max": spec["spin_max"],
        "slew_max": spec["slew_max"],
        "disc_area": spec["disc_area_m2"],
        "mdrag_coef": spec["mdrag_coef"],
        "moment_inertia": list(spec["moment_inertia"]),
        "num_motors": 4,
    }


def thrust_curve(c: float, expo: float) -> float:
    return (1 - expo) * c + expo * c * c


def thrust_curve_inv(f: float, expo: float) -> float:
    if expo == 0:
        return f
    return (-(1 - expo) + math.sqrt((1 - expo) ** 2 + 4 * expo * f)) / (2 * expo)


def spin_max_for_derate(k: float, spec: dict | None = None) -> float:
    """MOT_SPIN_MAX that caps total thrust at k*T_max (k=1 -> no derate)."""
    spec = spec or load_spec()
    c = thrust_curve_inv(k, spec["propExpo"])
    return spec["spin_min"] + c * (spec["spin_max"] - spec["spin_min"])


RHO_SEA_LEVEL = 1.225


def predicted_hover_throttle(mass: float, voltage: float, mot_spin_max: float | None = None,
                             spec: dict | None = None, alt_amsl: float | None = None) -> float:
    """Predicted CTUN.ThO in steady hover.

    Assumes MOT_BAT_VOLT_MAX=0 (no voltage compensation). ArduCopter always applies air-density
    compensation (AP_Motors_Thrust_Linearization::get_compensation_gain): the actuator demand is
    ThO / density_ratio, density_ratio = rho(alt)/1.225. CTUN.ThO is therefore a sea-level-equivalent
    thrust demand, i.e. ThO = density_ratio * f(actuator).
    """
    spec = spec or load_spec()
    e = spec["propExpo"]
    smin, smax = spec["spin_min"], spec["spin_max"]
    smax_p = smax if mot_spin_max is None else mot_spin_max
    alt = spec["refAlt"] if alt_amsl is None else alt_amsl
    vs = voltage / spec["maxVoltage"]
    rho_model = isa_density(alt) / isa_density(spec["refAlt"])    # SITL thrust scales with rho/rho_ref
    f_need = mass * G / (spec["max_total_thrust_n"] * vs ** 2 * rho_model)
    c_model = thrust_curve_inv(f_need, e)                         # model command (physical range)
    c_param = c_model * (smax - smin) / (smax_p - smin)           # firmware actuator in its own range
    density_ratio = isa_density(alt) / RHO_SEA_LEVEL
    return density_ratio * thrust_curve(c_param, e)


@dataclass
class Variant:
    name: str
    mass: float
    derate: float = 1.0        # fraction of T_max available

    def param_overrides(self, spec: dict | None = None) -> dict:
        if self.derate >= 1.0:
            return {}
        return {"MOT_SPIN_MAX": round(spin_max_for_derate(self.derate, spec), 4)}


def write_variant(v: Variant, out_dir: Path, spec: dict | None = None) -> tuple[Path, Path | None]:
    """Write <name>.json (frame model) and, if needed, <name>.parm (firmware overrides)."""
    spec = spec or load_spec()
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"{v.name}.json"
    jpath.write_text(json.dumps(frame_model(v.mass, spec), indent=2))
    ppath = None
    ov = v.param_overrides(spec)
    if ov:
        ppath = out_dir / f"{v.name}.parm"
        ppath.write_text("".join(f"{k} {val}\n" for k, val in ov.items()))
    return jpath, ppath


if __name__ == "__main__":
    spec = load_spec()
    for m in (1.2, 1.3, 1.4, 1.5, 1.6):
        fm = frame_model(m, spec)
        print(f"m={m:.2f}  hoverThrOut={fm['hoverThrOut']:.4f}  refCurrent={fm['refCurrent']:.2f}  "
              f"refSpd={fm['refSpd']:.3f}  areaCd={area_cd(m, fm['refSpd'], spec):.5f}  "
              f"ThO_pred@12.4V={predicted_hover_throttle(m, 12.4, spec=spec):.4f}")
    for k in (0.90, 0.85):
        print(f"derate {k:.2f}: MOT_SPIN_MAX={spin_max_for_derate(k, spec):.4f}")
