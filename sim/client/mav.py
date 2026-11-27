"""Thin pymavlink helpers shared by tests and (later) the outer-loop client.

Nothing here is SITL-specific: the same calls work against the real vehicle
over telemetry/USB, only the connection string changes.
"""
from __future__ import annotations

import math
import time

from pymavlink import mavutil

M = mavutil.mavlink


def connect(conn_str: str, timeout: float = 30):
    m = mavutil.mavlink_connection(conn_str, source_system=255)
    if m.wait_heartbeat(timeout=timeout) is None:
        raise TimeoutError(f"no heartbeat on {conn_str}")
    return m


def set_message_rate(m, msg_id: int, hz: float):
    m.mav.command_long_send(m.target_system, m.target_component,
                            M.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)


def request_streams(m, hz: float = 20):
    for msg_id in (M.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
                   M.MAVLINK_MSG_ID_ATTITUDE,
                   M.MAVLINK_MSG_ID_GLOBAL_POSITION_INT):
        set_message_rate(m, msg_id, hz)


def set_mode(m, name: str, timeout: float = 10):
    m.set_mode(m.mode_mapping()[name])
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb and mavutil.mode_string_v10(hb) == name:
            return
    raise TimeoutError(f"mode {name} not confirmed")


def wait_ekf_position(m, timeout: float = 120, log=print, relative: bool = False):
    """Wait until the EKF has horizontal and vertical position.

    relative=False: absolute horizontal position (GPS). relative=True: position relative to the start
    (optical flow); ArduCopter accepts that for GUIDED once the EKF reports it (PRED_POS_HORIZ_REL when
    disarmed).
    """
    set_message_rate(m, M.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 5)
    horiz = M.EKF_POS_HORIZ_REL | M.EKF_PRED_POS_HORIZ_REL if relative else M.EKF_POS_HORIZ_ABS
    need = M.EKF_ATTITUDE | M.EKF_VELOCITY_HORIZ | horiz | M.EKF_POS_VERT_ABS
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type=["EKF_STATUS_REPORT", "STATUSTEXT"], blocking=True, timeout=1)
        if msg is None:
            continue
        if msg.get_type() == "STATUSTEXT":
            log(f"FC: {msg.text}")
        elif (msg.flags & need) == need and not (msg.flags & M.EKF_CONST_POS_MODE):
            return time.time() - t0
    raise TimeoutError(f"EKF never reached {'relative' if relative else 'absolute'} position")


def set_ekf_origin(m, lat: float, lon: float, alt: float, timeout: float = 60, log=print) -> float:
    """SET_GPS_GLOBAL_ORIGIN until the FC reports that origin in GPS_GLOBAL_ORIGIN. Returns seconds waited.

    Needed without GPS: GUIDED takeoff targets an altitude above the origin, and ORGN (used for the truth
    frame) is only logged once an origin exists. EKF3 rejects the origin until the filter has initialised,
    so the message is repeated every second. On the real vehicle the companion computer does the same.
    """
    want = (int(round(lat * 1e7)), int(round(lon * 1e7)), int(round(alt * 1000)))
    t0 = time.time()
    last_send = -1.0
    while time.time() - t0 < timeout:
        if time.time() - last_send >= 1.0:
            m.mav.set_gps_global_origin_send(m.target_system, *want)
            last_send = time.time()
        msg = m.recv_match(type=["GPS_GLOBAL_ORIGIN", "STATUSTEXT"], blocking=True, timeout=0.2)
        if msg is None:
            continue
        if msg.get_type() == "STATUSTEXT":
            log(f"FC: {msg.text}")
        elif (msg.latitude, msg.longitude) == want[:2] and abs(msg.altitude - want[2]) <= 10:
            return time.time() - t0
    raise TimeoutError("EKF origin not accepted")


def command_ack(m, command: int, params, timeout: float = 3, retries: int = 5, log=print):
    params = list(params) + [0] * (7 - len(params))
    for _ in range(retries):
        m.mav.command_long_send(m.target_system, m.target_component, command, 0, *params)
        end = time.time() + timeout
        while time.time() < end:
            ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
            if ack and ack.command == command:
                if ack.result == M.MAV_RESULT_ACCEPTED:
                    return
                log(f"command {command} rejected, result={ack.result}")
                break
        time.sleep(1)
    raise RuntimeError(f"command {command} not accepted")


def arm_with_retry(m, timeout: float = 90):
    """Prearm checks can still be settling; retry until armed. Returns seconds waited."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        m.mav.command_long_send(m.target_system, m.target_component,
                                M.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        end = time.time() + 3
        while time.time() < end:
            m.recv_match(blocking=True, timeout=0.5)
            if m.motors_armed():
                return time.time() - t0
    raise TimeoutError("could not arm (prearm checks not passing)")


def takeoff(m, alt: float, log=print):
    command_ack(m, M.MAV_CMD_NAV_TAKEOFF, (0, 0, 0, 0, 0, 0, alt), log=log)


def wait_hover(m, alt: float, settle_s: float = 3.0, timeout: float = 60):
    """After takeoff(): wait until within 0.2 m of alt and still for settle_s. Returns last LOCAL_POSITION_NED.

    settle_s is FC time (time_boot_ms), so SITL speedup does not change the flight; timeout is wall time.
    """
    t0, t_ok = time.time(), None
    while time.time() - t0 < timeout:
        msg = m.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=1)
        if msg is None:
            continue
        if not m.motors_armed():
            raise RuntimeError("disarmed while climbing")
        steady = abs(-msg.z - alt) < 0.2 and abs(msg.vz) < 0.1
        if not steady:
            t_ok = None
        elif t_ok is None:
            t_ok = msg.time_boot_ms
        elif (msg.time_boot_ms - t_ok) / 1e3 > settle_s:
            return msg
    raise TimeoutError(f"never settled at {alt} m")


def land_and_wait(m, timeout: float = 60) -> bool:
    set_mode(m, "LAND")
    t0 = time.time()
    while time.time() - t0 < timeout:
        m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if not m.motors_armed():
            return True
    return False


def get_param(m, name: str, timeout: float = 5, retries: int = 3) -> float:
    for _ in range(retries):
        m.mav.param_request_read_send(m.target_system, m.target_component, name.encode(), -1)
        end = time.time() + timeout / retries
        while time.time() < end:
            p = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.2)
            if p and p.param_id == name:
                return p.param_value
    raise TimeoutError(f"param {name} not received")


def set_param(m, name: str, value: float, timeout: float = 5, retries: int = 3) -> float:
    """PARAM_SET and wait for the FC's PARAM_VALUE echo; returns the value the FC reports."""
    for _ in range(retries):
        set_param_nowait(m, name, value)
        end = time.time() + timeout / retries
        while time.time() < end:
            p = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.2)
            if p and p.param_id == name and abs(p.param_value - value) <= 1e-6 * max(1.0, abs(value)):
                return p.param_value
    raise TimeoutError(f"param {name}={value} not confirmed")


def set_param_nowait(m, name: str, value: float):
    """PARAM_SET without waiting for the PARAM_VALUE echo (for timed SIM_ events inside the loop)."""
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(), float(value),
                         M.MAV_PARAM_TYPE_REAL32)


class TxLog:
    """Everything this client sends on a link, by kind and wall time (pymavlink's send callback).

    Independent of the code that sends: it sees m.set_mode(), command_long_send() and the attitude
    targets alike, so "the client sent no mode command after the takeover" can be checked rather than
    assumed. Kinds: mode, attitude_target, position_target, command, param_set, other (heartbeats,
    requests).
    """
    CONTROL = ("mode", "attitude_target", "position_target", "command", "param_set")

    def __init__(self, m):
        self.sent: list[tuple[float, str]] = []
        m.mav.set_send_callback(self._on_send)

    @staticmethod
    def kind(msg) -> str:
        typ = msg.get_type()
        if typ == "SET_MODE":
            return "mode"
        if typ in ("COMMAND_LONG", "COMMAND_INT"):
            return "mode" if msg.command == M.MAV_CMD_DO_SET_MODE else "command"
        if typ == "SET_ATTITUDE_TARGET":
            return "attitude_target"
        if typ.startswith("SET_POSITION_TARGET"):
            return "position_target"
        return "param_set" if typ == "PARAM_SET" else "other"

    def _on_send(self, msg, *args, **kwargs):
        self.sent.append((time.monotonic(), self.kind(msg)))

    def counts(self, since: float | None = None) -> dict[str, int]:
        """Messages per kind, all or those sent after wall time `since` (time.monotonic())."""
        out = dict.fromkeys(self.CONTROL + ("other",), 0)
        for t, k in self.sent:
            if since is None or t > since:
                out[k] += 1
        return out


def drain(m) -> int:
    """Discard everything already buffered (stale state must not reach the controller)."""
    n = 0
    while m.recv_match(blocking=False) is not None:
        n += 1
    return n


def send_heartbeat(m):
    m.mav.heartbeat_send(M.MAV_TYPE_ONBOARD_CONTROLLER, M.MAV_AUTOPILOT_INVALID, 0, 0, 0)


# SET_ATTITUDE_TARGET type_mask: use attitude quaternion + thrust, ignore body rates
ATT_TARGET_MASK = (M.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
                   | M.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
                   | M.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """ZYX Euler [rad] -> [w, x, y, z] (MAVLink / ArduPilot convention, NED)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def quat_to_euler(q) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def send_attitude_target(m, roll: float, pitch: float, yaw: float, thrust: float, time_boot_ms: int = 0):
    """Attitude [rad, NED] + normalized collective thrust (needs GUID_OPTIONS bit 3, see params/outer_loop.parm)."""
    m.mav.set_attitude_target_send(time_boot_ms & 0xFFFFFFFF, m.target_system, m.target_component,
                                   ATT_TARGET_MASK, euler_to_quat(roll, pitch, yaw), 0, 0, 0, thrust)
