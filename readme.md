# SAKR3 / MyCustomF405 — Custom STM32F405 UAV Platform (ArduPilot)

Custom two-board **STM32F405RGT6** quadrotor platform running **ArduPilot (ArduCopter)** (`MyCustomF405` target), prepared for Python/MAVLink controller integration and SITL validation.

## 1. System Architecture Tree

```text
[Laptop / Companion Computer (Fang's Python Controller)]
       │
       ▼  MAVLink (pymavlink / MAVSDK / ROS 2, GUIDED mode)
[915 MHz Telemetry Radio]  or  [USB OTG1 / SBC UART]
       │
       ▼
[Top PCB: SAKR3 Flight Controller (STM32F405RGT6 @ 168 MHz, ArduCopter)]
 ├── Sensors ──► GY-91 10-DOF (MPU-9250 IMU + BMP280 Baro via SPI)
 │           ├── MTF-01P (Optical Flow + 12m LiDAR)
 │           └── NEO-7M (GPS + GLONASS via UART)
 ├── Video   ──► RunCam WiFiLink 2 (OpenIPC HD Video)
 └── Power   ◄── 8-pin GH1.25 (9V, 5V, 3.3V, INA219 I2C SDA/SCL)
       │
       ▼  400 Hz PWM (TIM3/TIM4/TIM1/TIM12)
[Bottom PCB: SAKR2.1 Power Regulator & Distribution Board]
 ├── Input   ◄── 3S 4200mAh LiPo (XT60, LM5050MKX + IRF1404PBF ORing, 5A Fuse, INA219)
 └── Output  ──► 4× Hobbywing Skywalker 40A V2 ESCs ──► 4× DJI 2212 920KV + 1045 Props
```

## 2. Repository Structure

```text
SAKR3-MyCustomF405-UAV/
├── README.md
└── hwdef/
    └── MyCustomF405/
        ├── hwdef.dat          # ArduPilot ChibiOS board definition
        ├── hwdef-bl.dat       # Bootloader definition
        └── defaults.parm      # Default board parameters
```

## 3. Two-Board Stacked Avionics (47 × 47 mm)

- **Bottom Board (`SAKR2.1` Power Board):** Dual-battery ideal-diode ORing (`25V-IN1`/`25V-IN2` via **XT60**), 2× **LM5050MKX** + 2× **IRF1404PBF**, **2920L500/30GR** 5A fuse, **INA219BIDR** current/voltage sensor (10 mΩ shunt), and regulated `9V` / `5V` / `3.3V` rails.
- **Top Board (`SAKR3` Flight Controller):** **STM32F405RGT6** running **ArduCopter** (`MyCustomF405` target with `TIM3`/`TIM4`/`TIM1`/`TIM12` PWM mapping to avoid ChibiOS `TIM2` systick conflict), with dedicated headers for `GPS`, `Telemetry`, `WiFi Link`, `SBC`, `SBUS1`, `PWM&SERVO1`, `I2C`, and `SPI`.

## 4. Sensors & Propulsion Specs (For Simulation & Tuning)

| Component | Specification |
|---|---|
| **All-Up Weight (AUW)** | **1.20 – 1.60 kg** (with battery & payload) |
| **Max Total Thrust** | **~2.40 kg** (~23.5 N, ~600g per motor on 3S + 1045 props) |
| **Estimated Inertia** | `Ix = 0.02 kg·m²`, `Iy = 0.02 kg·m²`, `Iz = 0.04 kg·m²` |
| **Motors & Props** | 4× **DJI 2212 / 920KV** + **1045** propellers |
| **ESCs** | 4× **Hobbywing Skywalker 40A V2** (40A/60A, 400 Hz PWM) |
| **Battery & Cabling** | **YUAN H 3S 4200mAh 35C** LiPo (11.1V, 290.5g) with **XT60** |
| **10-DOF IMU + Baro** | **GY-91** (**MPU-9250**: Gyro/Accel/Mag + **BMP280** Barometer) |
| **GPS** | u-blox **NEO-7M** (GPS + GLONASS, UART) |
| **Optical Flow / LiDAR** | **MTF-01P** (Optical Flow + 12m Rangefinder) |
| **Telemetry & Video** | **915 MHz** MAVLink Telemetry + **RunCam WiFiLink 2** (OpenIPC) |

## 5. MAVLink Interface (Python / SITL)

Fang's Python controller function can connect via `pymavlink`, `MAVSDK`, or `ROS 2 / MAVROS` in `GUIDED` / `GUIDED_NOGPS` mode:
- **State Feedback:** `LOCAL_POSITION_NED`, `ATTITUDE`, `ATTITUDE_QUATERNION`, `GLOBAL_POSITION_INT`
- **Control Commands:** `SET_POSITION_TARGET_LOCAL_NED` or `SET_ATTITUDE_TARGET`
