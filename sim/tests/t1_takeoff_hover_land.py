"""T1: default quad SITL — automatic arm, takeoff, hover, land with no human input.

Pass criteria: vehicle arms, reaches target altitude, holds it during hover,
lands and disarms; a dataflash .BIN log is produced.

Usage (inside WSL, uavsim venv active):
    python tests/t1_takeoff_hover_land.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymavlink import mavutil  # noqa: E402

from sitl_runner import SitlInstance  # noqa: E402

TARGET_ALT = 5.0     # m
HOVER_S = 10.0       # s
TIMEOUT_S = 120.0


def wait_heartbeat(m, timeout=30):
    hb = m.wait_heartbeat(timeout=timeout)
    if hb is None:
        raise TimeoutError("no heartbeat")
    return hb


def set_mode(m, name):
    m.set_mode(m.mode_mapping()[name])
    t0 = time.time()
    while time.time() - t0 < 10:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb and mavutil.mode_string_v10(hb) == name:
            return
    raise TimeoutError(f"mode {name} not confirmed")


def request_streams(m, hz=20):
    for msg_id in (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
                   mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
                   mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT):
        m.mav.command_long_send(m.target_system, m.target_component,
                                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)


def wait_ekf_position(m, timeout=90):
    """Wait until EKF reports absolute horizontal position and is not in const-pos mode."""
    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                            mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT,
                            int(1e6 / 5), 0, 0, 0, 0, 0)
    need = (mavutil.mavlink.EKF_ATTITUDE | mavutil.mavlink.EKF_VELOCITY_HORIZ
            | mavutil.mavlink.EKF_POS_HORIZ_ABS | mavutil.mavlink.EKF_POS_VERT_ABS)
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type=["EKF_STATUS_REPORT", "STATUSTEXT"], blocking=True, timeout=1)
        if msg is None:
            continue
        if msg.get_type() == "STATUSTEXT":
            print("FC:", msg.text)
        elif (msg.flags & need) == need and not (msg.flags & mavutil.mavlink.EKF_CONST_POS_MODE):
            return time.time() - t0
    raise TimeoutError("EKF never reached absolute position")


def command_ack(m, command, params, timeout=3, retries=5):
    for _ in range(retries):
        m.mav.command_long_send(m.target_system, m.target_component, command, 0, *params)
        end = time.time() + timeout
        while time.time() < end:
            ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
            if ack and ack.command == command:
                if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    return
                print(f"command {command} rejected, result={ack.result}")
                break
        time.sleep(1)
    raise RuntimeError(f"command {command} not accepted")


def arm_with_retry(m, timeout=90):
    """Prearm checks (EKF/GPS) take a while after boot; retry until armed."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        m.mav.command_long_send(m.target_system, m.target_component,
                                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                                1, 0, 0, 0, 0, 0, 0)
        end = time.time() + 3
        while time.time() < end:
            m.recv_match(blocking=True, timeout=0.5)
            if m.motors_armed():
                return time.time() - t0
    raise TimeoutError("could not arm (prearm checks not passing)")


def main() -> int:
    result = {"test": "T1", "pass": False}
    rows = []
    with SitlInstance(run_name="T1_takeoff_hover_land") as sitl:
        m = mavutil.mavlink_connection(sitl.connection, source_system=255)
        wait_heartbeat(m)
        request_streams(m)
        result["ekf_wait_s"] = round(wait_ekf_position(m), 1)
        set_mode(m, "GUIDED")
        result["arm_wait_s"] = round(arm_with_retry(m), 1)

        command_ack(m, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    (0, 0, 0, 0, 0, 0, TARGET_ALT))
        phase, t_phase, t0 = "climb", time.time(), time.time()
        alt = 0.0
        while time.time() - t0 < TIMEOUT_S:
            msg = m.recv_match(type=["LOCAL_POSITION_NED", "HEARTBEAT", "STATUSTEXT"],
                               blocking=True, timeout=1)
            if msg is None:
                continue
            if msg.get_type() == "STATUSTEXT":
                print("FC:", msg.text)
                continue
            if msg.get_type() == "LOCAL_POSITION_NED":
                alt = -msg.z
                rows.append([msg.time_boot_ms, phase, msg.x, msg.y, msg.z,
                             msg.vx, msg.vy, msg.vz])
            if phase in ("climb", "hover") and not m.motors_armed():
                print(f"unexpected disarm during {phase}")
                break
            if phase == "climb" and alt > 0.95 * TARGET_ALT:
                phase, t_phase = "hover", time.time()
            elif phase == "hover" and time.time() - t_phase > HOVER_S:
                set_mode(m, "LAND")
                phase, t_phase = "land", time.time()
            elif phase == "land" and not m.motors_armed():
                phase = "done"
                break
        result["final_phase"] = phase

        hover = [r for r in rows if r[1] == "hover"]
        if hover:
            alts = [-r[4] for r in hover]
            result["hover_alt_mean_m"] = round(sum(alts) / len(alts), 3)
            result["hover_alt_min_m"] = round(min(alts), 3)
            result["hover_alt_max_m"] = round(max(alts), 3)
            result["hover_samples"] = len(hover)
            result["hover_rate_hz"] = round(
                len(hover) / ((hover[-1][0] - hover[0][0]) / 1000.0), 1)
        m.close()
        time.sleep(2)  # let SITL flush the dataflash log

        run_dir = sitl.run_dir
        with open(run_dir / "t1_local_position.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_boot_ms", "phase", "x", "y", "z", "vx", "vy", "vz"])
            w.writerows(rows)
        bins = sorted((run_dir / "logs").glob("*.BIN")) if (run_dir / "logs").exists() else []
        result["bin_logs"] = [b.name for b in bins]
        result["pass"] = (phase == "done" and bool(hover) and bool(bins)
                          and abs(result["hover_alt_mean_m"] - TARGET_ALT) < 0.5)
        result["run_dir"] = str(run_dir)
        (run_dir / "t1_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
