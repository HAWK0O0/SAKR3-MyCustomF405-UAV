# UAV SITL simulation platform

ArduCopter SITL running in WSL2, plus Python/pymavlink outer-loop tooling.

## Environment (WSL2, `Ubuntu-22.04`)

| Item | Location / version |
|---|---|
| ArduPilot source | `~/ardupilot`, tag `Copter-4.7.3` (shallow clone) |
| SITL binary | `~/ardupilot/build/sitl/bin/arducopter` |
| Python venv | `~/venvs/uavsim` (Python 3.10, created with `uv`) |
| Run directories | `~/uavsim_runs/<run_name>/` (SITL writes `eeprom.bin`, `logs/*.BIN` here) |
| This folder in WSL | `sim/` of this repository, e.g. `/mnt/d/.../SAKR3-MyCustomF405-UAV/sim` |
| Outputs | `results/` (not tracked in git; the commands below regenerate it) |

System packages were installed with `Tools/environment_install/install-prereqs-ubuntu.sh -y`.
The venv holds the ArduPilot build/SITL Python deps (pymavlink, MAVProxy, empy==3.3.4, pexpect, ...)
plus numpy/scipy/pandas/matplotlib/pyyaml.

Rebuild after changing ArduPilot source:

```bash
source ~/venvs/uavsim/bin/activate
cd ~/ardupilot && ./waf copter
```

## Layout

| Path | Content |
|---|---|
| `sitl_runner.py` | starts a headless `arducopter` in its own run dir; same command line as `sim_vehicle.py` (`--model`, defaults chain `copter.parm[,copter-X.parm]`, `-I`, `--speedup`, `--home`), launched directly to avoid sim_vehicle's xterm under WSLg. MAVLink on `tcp:127.0.0.1:5760+10*instance` |
| `airframes/sakr3_spec.yaml` | SAKR3 base airframe spec; every value tagged readme / estimate / ap_default |
| `airframes/make_airframe.py` | SITL frame model for any (mass, thrust derate) point; hover-throttle predictor |
| `params/sakr3_base.parm` | firmware-side params (X frame, motor range, expo, no voltage compensation) |
| `params/outer_loop.parm` | firmware settings for the external outer loop (`GUID_OPTIONS=8`, `GUID_TIMEOUT=3`, `ATC_ANGLE_BOOST=0`); same file for the real vehicle |
| `sakr3.py` | SITL instance for a SAKR3 variant (model + parameter chain); `fly_to_hover()` (sets the EKF origin first when there is no GPS); `param_check()`: did every value of our .parm files reach the FC |
| `client/mav.py` | pymavlink helpers (connect, streams, mode, EKF wait, arm, takeoff, params, `send_attitude_target`); `TxLog` counts everything the client sends, by kind |
| `client/controller.py` | outer-loop interface (`State`, `Setpoint`, `Command`, `law.step(state, setpoint, dt)`) and the PD test controller |
| `client/supervisor.py` | controller-fault check (non-finite command), command clamps (total tilt, thrust) + trip conditions; BRAKE stopping-point model for the fence |
| `client/outer_loop.py` | event-driven loop over MAVLink, one control step per FC position sample; records everything |
| `client/laws.py` | registry of outer-loop laws a scenario can name (`pd`, or any `module:Class`) |
| `client/trajectory.py` | smooth reference trajectories (circle, figure 8) with velocity / acceleration for feedforward |
| `runner/disturbance.py` | seeded low-frequency gust process (written to `SIM_WIND_*`) and NED pushes (`SIM_SHOVE_*`) |
| `scenarios/tasks/*.yaml` | the task library: `hover_gust`, `steps`, `circle`, `figure8`, `push`, `motor_fault`, `derate_drop` |
| `analysis/scorecard.py` | scorecard of a law on a suite (pass rate per task and theta point, headline metric), and laws side by side |
| `analysis/replay.py`, `analysis/replay_viewer.html` | replay viewer: exports finished runs (`results/<batch>/replay/`) for a self-contained HTML page with a 3D view, timeline and plots |
| `params/sensors_gps.parm` | sensor profile A (GPS + baro): SITL noise settings, estimates until M5 |
| `params/sensors_flow.parm` | sensor profile B (optical flow + rangefinder, no GPS; MTF-01P): sensors, EKF3 sources, estimates until M5 |
| `scenarios/defaults.yaml` | every scenario key with its default and meaning |
| `scenarios/*.yaml` | test scenarios (`t4_steps`, `t4b_steps_flow`, `t5_steps_wind`, `t6e_takeover`); `scenarios/batches/*.yaml` batch definitions (`t5`, `family_demo`, `m4_profiles`, `tasks_smoke`, `suite_core`) |
| `runner/scenario.py` | scenario loading (strict keys, deep merge) and batch expansion (groups, sweeps, repeats) |
| `runner/run.py` | one run: scenario → SITL + outer loop → truth, metrics, manifest; `reanalyze()` from stored logs; also a CLI for a single scenario |
| `runner/pilot.py` | a pilot / GCS on a second MAVLink link that sets FC modes at given times (`task.pilot_modes`, takeover tests) |
| `runner/batch.py` | parallel batch on separate SITL instances / processes → `results/<batch>/runs.csv` |
| `runner/seed.py`, `tools/seed_preload.c` | per-run noise seed for SITL (LD_PRELOAD shim, no ArduPilot change) |
| `runner/manifest.py` | firmware commit/tag + binary hash, parameter-file and code hashes, law source hash; flight version and analysis version |
| `analysis/binlog.py` | `.BIN` → pandas via pymavlink DFReader |
| `analysis/truth.py` | simulator ground truth (`SIM2`/`SIM`) in the outer loop's EKF-origin frame |
| `analysis/metrics.py` | per-run metrics on true position (IAE, per-step rise/overshoot/settle/final error, hold RMS, tilt, thrust headroom, latency); pass rules |
| `analysis/batch_table.py` | batch summary per sweep point (planned / completed / passed / landed, mean ± sd over completed runs, early-ended runs listed) as CSV and Markdown |
| `analysis/outer_loop.py` | latency on the FC clock (state sample → GUIA), command intervals, step metrics |
| `tests/` | acceptance tests (`m2_common.py`: FC takeoff → outer loop → land, data saved for re-analysis); `test_unit.py`: checks without SITL (supervisor, loop on a fake link, batch table, resume rules, exit codes) |
| `results/` | result tables and figures |

## Outer-loop interface (M2)

The FC streams `LOCAL_POSITION_NED` and `ATTITUDE` at 50 Hz. Each new position sample triggers
`supervisor.check → law.step → supervisor.command_fault / clamp → SET_ATTITUDE_TARGET` (attitude
quaternion + normalized collective thrust, body rates ignored). The loop is paced by the FC's time, not by a
client timer, so it runs unchanged at any SITL speedup and on the real vehicle. There it runs on a
companion computer on UART4 = `SERIAL4`, and only the connection string changes.

Firmware semantics confirmed in the 4.7.3 source and in T4/T6:
- `GUID_OPTIONS` bit 3 makes the thrust field a collective thrust in the units of `CTUN.ThO`. The
  default reads it as a climb rate, with 0.5 meaning hold altitude. `ATC_ANGLE_BOOST=0` stops the
  FC adding its own 1/cos(tilt) on top, so the law owns the collective thrust.
- The FC logs every accepted command in `GUIA` with its arrival time, so latency is measured on
  the FC clock with no clock sync.
- Without new commands, the FC holds the last attitude and thrust for `GUID_TIMEOUT` (3 s). It then
  levels the attitude and switches to its own climb-rate-zero altitude hold, while horizontal
  position drifts freely. Commands that arrive again take over at once.
- `SET_ATTITUDE_TARGET` is silently ignored outside GUIDED. A mode change sends a HEARTBEAT
  immediately (`mode.cpp`), so the loop notices within one state period.
- BRAKE (the supervisor's fallback) does not stop hard. It brings velocity to zero with time
  constant 7.5/`PSC_NE_JERK` = 1.5 s, starting from zero acceleration: about 1.6 m to stop from
  1 m/s and about 5.3 m from 3 m/s. The fence therefore trips on the predicted stopping point.
- pymavlink keeps `target_component` at 0, so the FC's heartbeat is identified by sysid + autopilot type.

Client safety rules (review of 2026-11-26):
- **A non-finite command is a controller fault.** If the law raises or returns NaN/inf in any field,
  nothing is sent for that step and the loop trips into BRAKE in the same step (trip reason
  `controller fault: ...`). It is not replaced by a "safe" command.
- **`safety.max_tilt_deg` limits the total tilt**, acos(cos roll · cos pitch), the same measure as the
  trip on measured tilt. Over the limit, the thrust axis is brought back onto the cone in the same
  horizontal direction. (Before, each axis was limited on its own: 25° + 25° gave 34.8°.)
- **After a takeover the client sends nothing.** When the FC reports a mode the client did not command
  (pilot, GCS on another link, FC failsafe), the loop stops and the run procedure only watches until
  the vehicle is disarmed: no LAND, no other mode, no attitude target, no parameter. The same holds
  if someone changes the mode during the BRAKE hold after a trip. Whoever took over lands.

Client timing (changed in M3): the loop's timers run on FC time (1 Hz heartbeat, settle and monitor
durations). A gap between state samples longer than `state_timeout_s` of FC time trips the loop;
`state_timeout_s` of wall time with no data at all means a dead link. Python's garbage collector is off
while the loop runs: a full collection took 25–30 ms, i.e. one late command at 1×, and a trip at 10×.

## Scenarios and batches (M3)

A **scenario** is `scenarios/defaults.yaml` overridden by a scenario file (deep merge). Keys are strict,
so a typo fails instead of silently flying the default. It holds the family point (`vehicle.mass_kg`,
`vehicle.derate`), wind and sensor profile, task (handover altitude, step setpoints, timed parameter
events), controller (`law` + `args`), supervisor limits, metric settings, pass bounds, and `run`
(`seed`, `speedup`, `setup_speedup`). `task.pilot_modes` lets a pilot on a second link change the FC
mode at given times (takeover tests).

`controller.hover_thrust` sets what the law is told about the hover thrust. With `fc`, the default,
it gets `MOT_THST_HOVER` read at handover. In SITL that starts at 0.58 from `sakr3_base.parm` and
barely moves in the 3 s hover before handover (0.580–0.592 across 1.2–1.6 kg), so it carries a
little information about the variant, and more with a longer hover. With a number, every variant
gets the same prior. Which of the two RuleForge laws should see is a research decision, still open.

`runner/run.py` flies one scenario into `~/uavsim_runs/<batch>/<run_id>/`:
- `scenario.yaml` and the generated `scenario.parm`;
- `rows.csv.gz`, one row per control step with the simulator truth attached;
- `metrics.json`;
- `manifest.json`: ArduPilot commit and `Copter-4.7.3` tag, binary SHA-256, every parameter file in
  load order with its hash, frame model, seed-shim hash, the law's class and source hash, the hashes
  of all client/runner/analysis code, the FC parameters actually in effect, timing and outcome;
- the `.BIN` log.

A **batch** (`scenarios/batches/*.yaml`) names a base scenario and groups. Each group has `set`
overrides, a `sweep` (Cartesian product; lists or `{range: [a, b]}`) and `repeats`.

`runner/batch.py` flies the runs on N parallel SITL instances, one process per instance, longest run
first. It writes `results/<batch>/runs.csv` (one row per run), `summary.csv`/`summary.md` (one row per
sweep point with the seeds pooled), `early.csv` and `batch.json`. Code and firmware hashes are taken
once at batch start.

**Three verdicts per run**, kept apart:

| Field | Meaning |
|---|---|
| `completed` (`outcome` = `ok`) | the script ran to its end and the outer loop flew the whole planned duration. `trip`: the loop stopped early; `error`: setup, SITL or analysis failed |
| `pass` | completed, and every bound of the scenario's `pass` section holds (the task criteria) |
| `landed` | the vehicle was seen disarmed at the end |

`runner/run.py` exits with 0 only if all three hold (1 error, 2 trip, 3 task bounds not met, 4 landing
not confirmed). `runner/batch.py` exits with 0 if the batch was executed without an `error` run; pass
rates are results, not an exit code.

**The batch table** counts per sweep point: `n` planned runs, `n_completed`, `n_pass`, `n_landed`;
`pass_rate = n_pass / n`. A run without a verdict (crashed worker) is a planned run that did not pass.
Metric mean ± sd are over completed runs only: a run that stopped at 12 s of 56 s has an IAE over 12 s,
which is not comparable. Those runs are listed with end time and reason (`early.csv`, second table in
`summary.md`).

**`--resume` never mixes experiments.** Each manifest records a *flight version* (firmware binary and
its default parameters, our parameter files, airframe spec, client, flight procedure) and an *analysis
version* (`analysis/*.py`). An old run is kept only if scenario hash, flight version and law source are
those of the workspace. If only the analysis version differs, its stored logs are analysed again
(`reanalyze()`, needs the `.BIN`) and the manifest records the new analysis version next to the old
one. Everything else is flown again, with the reason logged. Runs flown before 2026-11-26 carry no
flight version and are always flown again.

**Metrics are on the simulator's true state**, not the EKF. `analysis/truth.py` takes `SIM2` (position
relative to SITL home) and shifts it into the EKF-origin frame the setpoints live in (`ORGN`). With
baro noise off, EKF − truth is under 1 cm on every axis, so the conversion is exact to the resolution
of the logged origin.

Per run, `analysis/metrics.py` computes:
- IAE of the true position error (3D, horizontal, vertical);
- per step: 10–90 % rise, overshoot, settling time into `band_m` (or "not settled"), final error and
  spread over the last 2 s, cross-track error;
- hold RMS;
- EKF-vs-truth error;
- max true tilt and clamp fraction;
- latency and command gaps at the FC;
- achieved speedup.

Pass/fail compares any metric (fnmatch patterns allowed) against an upper bound.

**Speed policy (from T5).** Boot, EKF convergence, takeoff and landing are flown by the FC's own
controllers and run at `setup_speedup: 10`; `SIM_SPEEDUP` is switched at runtime. The outer loop runs
at 1× (2× is consistent too). One 56 s scenario then takes 64 s of wall time. 20 instances run in
parallel; T5's 365 runs took 14 min.

What M3 found out about SITL (Copter-4.7.3):
- **Its noise is never seeded.** IMU, baro and turbulence noise all draw from glibc `random()`, and
  nothing seeds it. `tools/seed_preload.c` (LD_PRELOAD) sets it from `run.seed`. With the vehicle idle,
  the same seed gives bit-identical sensor samples and another seed gives different ones; seed 1 is
  the same as no shim.
- **A seed does not fix a flight.** The single random stream is also consumed by the turbulence, from
  takeoff on, and takeoff happens at a slightly different simulated moment in every run (MAVLink
  timing). Five runs with seed 1 still spread 3.4 cm RMS, against 4.1 cm across seeds. What the seed
  does govern is the vertical: the EKF height-reference error σ is 1.2 mm with the seed fixed vs 33 mm
  across seeds. Compare variants with unpaired statistics.
- **Baro noise shows up as a fixed EKF height error.** With `SIM_BARO_RND 0.1` the EKF height is off
  by 5.4 ± 3.3 cm (up to 12 cm) for the whole flight, already on the ground. True-position metrics
  include it. The PD law without a horizontal integrator holds 6–9 cm off even in still air: the EKF
  reports ~2 cm/s of velocity while the vehicle is still. That is why the PD test controller's
  horizontal integrator is on by default in scenarios (`t4_steps` turns it off to replicate T4).
- **Loop latency grows with the speedup.** In simulated time, p50/p95 latency is 5.5/8.3 ms at 1×,
  8.0/13.2 at 2×, 15.6/27 at 5×, 22/77 at 10× and 43/294 at 20×. The client's own processing is
  0.15 ms of wall time. SITL paces itself on wall time and pauses when the client falls behind
  reading, so every wall-clock delay on the link costs S× as much simulated time. At 20×, 6 of 60
  runs crash or nearly hit the fence.
- **The family demo** (`scenarios/batches/family_demo.yaml`, `results/m3_family_demo/summary.md`):
  mass 1.2/1.4/1.6 kg × derate 1.0/0.85 × 3 seeds, 18 runs in 66 s. All points pass 67–100 %, except
  1.6 kg + 15 % derate at 0 %: altitude 0.31–0.35 m low. The PD test law starts from a hover prior
  of 0.59 against a true ~0.81; its altitude integrator (limit 2 m/s²) and thrust cap (0.9, thrust
  p99 0.89) can't close the gap. This is the kind of θ-dependence the platform is meant to expose.
- **Limitations of the models.** SITL has no horizontal GPS noise, and its turbulence is
  high-frequency (a new random direction every physics step), so it barely moves the vehicle. Under 20
  parallel instances, the worst command gap even at 1× was 52 ms, against 25 ms alone.

## Sensor profile B: optical flow + rangefinder (M4)

`environment.sensors: flow` selects `params/sensors_flow.parm`, the indoor case: no GPS, the EKF takes
horizontal velocity from optical flow and height from the downward rangefinder (`EK3_SRC1_VELXY 5`,
`EK3_SRC1_POSZ 2`, `POSXY`/`VELZ` none, yaw from the compass). The real sensor is the MicoAir MTF-01P
(flow + ToF lidar in one module). On the real vehicle it talks MAVLink (`FLOW_TYPE 5`, `RNGFND1_TYPE 10`,
module set to `mav_apm`); which UART it is on is still open. SITL uses its own simulated sensors
(`FLOW_TYPE 10`, `RNGFND1_TYPE 100`), driven by the true state: flow at 100 Hz with 2 samples (20 ms)
of delay and ±0.05 rad/s uniform noise, range ±2 cm uniform, 0.01–8 m. The EKF fuses at most one sample
per 50 ms of each either way. The noise, delay, range limits and mount height are estimates (datasheet
figures from retailer listings: 100 Hz, 42° FOV, 2 cm / 2 %, 10–12 m), to be replaced from real logs in M5.

Without GPS nothing gives the EKF an origin, and then GUIDED takeoff is refused (it targets an altitude
above the origin) and no `ORGN` is logged. The client therefore sends `SET_GPS_GLOBAL_ORIGIN` (SITL home)
once the EKF has initialised, as a companion computer would on the real vehicle (`mav.set_ekf_origin`,
done by `fly_to_hover` whenever the parameter chain has `GPS1_TYPE 0`). Truth is then referred
vertically to the landed position, not to the origin altitude: with rangefinder height the EKF's zero
is the floor where it initialised, and SITL's ground at CMAC is 0.10 m above the home altitude.

What building profile B found out:
- **A renamed parameter is ignored without an error.** 4.7 renamed `RNGFND1_GNDCLEAR` to `RNGFND1_GNDCLR`.
  The old name was dropped silently, the default 0.10 m applied, and the EKF height was 5 cm off in
  every run (EKF3 keeps the floor at least ground clearance below the vehicle). Every run now checks
  that each parameter of our own .parm files reached the FC with its value (`sakr3.param_check`, from
  the log's `PARM`); a mismatch makes the run an error. Profile A's files pass it too (60 runs of T4-B).
- **The rangefinder must read in range when landed.** EKF3 takes only samples with status Good. SITL's
  vehicle rests with its body origin on the ground, so a sensor there reads 0, below any minimum range.
  Without range data on the ground there is no flow ground offset, no relative position, and arming in
  GUIDED fails ("Need Position Estimate"). The module is therefore placed 5 cm above the body origin
  (`SIM_SONAR_POS_Z`, `SIM_FLOW_POS_Z`, `FLOW_POS_Z` −0.05), giving a 5 cm landed reading
  (`RNGFND1_GNDCLR 0.05`, `RNGFND1_MIN 0.01`).
- **Height is better than with GPS + baro, position much worse.** In T4-B (below) the EKF height error is
  0.8 cm RMS against 5.4 cm with baro. The horizontal EKF error grows instead: median 0.34 m over the
  56 s run (90 % ≤ 0.59 m, max 0.76 m), against 1 cm with GPS. Even with all sensor noise off it drifts
  0.14 m. The PD law holds the EKF's position, so the true position drifts along: hold RMS 0.34 m
  instead of 0.076 m.
- **Fast setup is still harmless under B**: everything at 1× vs setup at 10× agrees on all 17 metrics
  (bias bound ≤ 0.94 σ).
- The EKF velocity is ~2–3 cm/s off the true velocity in hover under both profiles (seen with all noise
  off too); it is not specific to flow.

## Task library, disturbances and the evaluation suite

**Task types.** `task.type: steps` (piecewise-constant setpoints) or `trajectory`: a smooth closed curve
(`client/trajectory.py`: circle or figure 8, optional altitude excursion) that starts and ends at rest at
the handover point, with smoothstep speed ramps. The `Setpoint` a law receives now carries the reference's
velocity and acceleration as well (zero for steps). A law may use them as feedforward or ignore them. The
PD test law uses them unless `controller.args.feedforward: false`; for steps its output is bit-identical
to before.

**Disturbances**, all at outer-loop times (FC clock) and recorded in the manifest:
- `environment.gusts`: a seeded low-frequency gust process, a first-order Gauss-Markov process per
  horizontal component, on top of the mean wind. The client writes the total to `SIM_WIND_SPD/DIR` at
  `rate_hz`. SITL has no gusts of its own (its turbulence is white-ish, `SIM_WIND_TC` is unused in
  4.7.3). The profile is drawn up front from `run.seed`, so it does not depend on timing, and it is
  stored per run (`gusts.csv.gz`).
- `task.events`: `params` (any parameter, e.g. a partial motor failure `SIM_ENGINE_MUL`/`SIM_ENGINE_FAIL`);
  `push` (an NED acceleration for some ms, through `SIM_SHOVE_*` in the body frame); `derate: k` (from
  then on k × the variant's available thrust, via the same `MOT_SPIN_MAX` mapping as `vehicle.derate`).
  The last one is a θ change in flight that the law is not told about.

**Metrics added.** Trajectories: `traj_rms_m` (3D, horizontal, vertical) and `traj_max_m` over the motion.
Events: `ev{i}_max_dev_m` is the largest change of the true position error after the event, relative to
its mean over the second before, so a standing offset (flow drift, a hover bias) is not counted.
`ev{i}_recover_s` is the time until that change stays within `band_m` (NaN if it never does before the
next event).

**The task library** (`scenarios/tasks/`): `hover_gust` (40 s, 2 m/s mean + 1 m/s gusts), `steps` (T4
sequence), `circle` (r 1.5 m, 8 s per lap), `figure8` (2 m half-width, 12 s, ±0.5 m altitude), `push`
(3 pushes of 1.5 m/s), `motor_fault` (motor 1 at 80 % PWM), `derate_drop` (thrust to 85 % in flight).
Each has pass bounds, written before its first flight. They are proposals, not tuned to the PD law.

**Suites and scorecards.** A batch group may name its own base scenario and `common:` sets / sweeps all
groups, so one batch file covers several tasks. `scenarios/batches/suite_core.yaml` runs every task at
every θ point (mass 1.2/1.4/1.6 kg × derate 1.0/0.85) with 3 seeds: 126 runs in 4.7 min on 20 workers.
`runner/batch.py --set KEY=VALUE --name NAME` flies the same suite with another law or setting into its own
table (`--set controller.law=mymodule:MyLaw`). `analysis/scorecard.py` gives per task the pass rate, the
worst θ point and a headline metric, and draws the tasks × θ grid (`scorecard.png`); with several result
folders it puts the laws side by side (`compare.md`). This is the interface RuleForge will use to
evaluate a candidate law; nothing of RuleForge is integrated.

PD test law on the core suite (`results/suite_core/`, 2026-11-26): pass 38 % of 126 runs, all landed.
- **Trajectories**: 3/3 at every θ point except 1.6 kg + 15 % derate (0/3), the hover-prior case of the
  family demo: traj RMS 0.44 m there, 0.15–0.25 m elsewhere. Steps the same, plus one 1.4 kg run over
  the overshoot bound.
- **`hover_gust`**: hold RMS 0.30–0.42 m against the 0.3 m bound.
- **`push`**: max deviation 0.64–0.82 m, but it never returns within 0.1 m in 8 s. The horizontal
  integrator, wound up during the push, leaves a ~0.2 m undershoot that decays slowly.
- **`motor_fault`**: 1.0–1.5 m deviation. The 1.6 kg variants and 1.4 kg derated lose ~3 m of altitude
  and trip the 2 m floor.
- **`derate_drop`**: 0.35–0.45 m sag; 1.6 kg + 15 % derate cannot hold altitude after the drop (trip).

The same suite without feedforward (`results/suite_core__pd_noff/`) differs only where it should:
circle / figure 8 pass 83 % → 0 %, traj RMS 0.23 → 1.4–1.5 m; every other task is unchanged within its
3-seed spread. 3 seeds per point screen a law; a statistical comparison needs ~60 (T5).

## Replay viewer

The platform runs headless, so flights are looked at afterwards. `python analysis/replay.py results/<batch>`
exports every run of a batch (8 s for 126 runs, ~250 KB each) into `results/<batch>/replay/`.
`--run <run_dir>` adds single runs to `results/replay/`. Open `index.html` there by double-click: it is
one self-contained page, no internet, no server. It shows exactly the data the metrics were computed from.

- **3D view.** Shows the true path, the reference (dashed) and the EKF estimate, and the vehicle at its
  true attitude (X frame, front rotors filled, drawn ×1/×2/×4). Also a drop line to the floor, the wind at
  the vehicle (gusts included) and an arrow during a push. Views: 3D / top / from south / from east /
  follow; drag rotates, shift-drag pans, the wheel zooms.
- **Timeline.** Play at 0.25–4×, slider, ←/→ step, space to play/pause. "Whole flight" adds takeoff and
  landing (truth from the `.BIN`, FC modes marked).
- **Plots** on one time axis, one quantity each: east, north, up (true, reference, EKF), error to the
  reference horizontal and vertical (with the settling band), EKF − true, tilt (true, commanded), thrust
  command. Every plot has a hover readout and click-to-seek. Steps, events, mode changes and the trip are
  vertical lines.
- **Info panel.** The three verdicts, θ, sensors, wind, law and its arguments, the bounds not met, the
  events (click to jump), and the key metrics against their pass bounds.
- **Run picker and comparison.** Filter by task and θ; "compare with" overlays a second run of the batch
  (its true path and a ghost vehicle in aqua, its true series in the plots). The URL keeps the selection
  (`index.html#run=<id>&cmp=<id>`).

Colours keep one meaning everywhere: blue true state, grey dashed reference / command, orange EKF, aqua
comparison. The palette is validated for colour-vision deficiency, and the legend always labels it.
Frame: east / north / up relative to the handover point; time relative to the start of the outer loop.
Light mode only.

## How the parameter family is represented

SITL has no "max thrust" field: per-motor thrust is `mass*g/(n*hoverThrOut) * vs^2 * f(c)`, so changing
`mass` alone keeps thrust/weight constant. `make_airframe.frame_model(m)` therefore sets, per mass point,
`hoverThrOut = m*g/T_max` (max thrust fixed at 23.5 N), `refCurrent ∝ m^1.5` (prop area and outflow
velocity unchanged) and solves `refSpd` so body drag area is unchanged. Thrust derating is done the same
way as on the real vehicle: lowering the firmware parameter `MOT_SPIN_MAX`.

Two implementation facts found while building this:
- SITL's filesystem strips a leading `/`, so the model JSON path must be relative to the run directory.
- ArduCopter applies air-density compensation (always on, also on the real F405 build): `CTUN.ThO` is a
  sea-level-equivalent thrust demand, `ThO = (rho/1.225) * f(actuator)`. At the 584 m test site this is ×0.945.

## Acceptance tests

| Test | Status |
|---|---|
| T1 default quad: auto arm / takeoff 5 m / hover 10 s / land | passed 2026-11-25 (hover mean 4.98 m) |
| T2 SAKR3 model 1.40 kg: steady hover, hover throttle = prediction ±0.03 | passed 2026-11-25 (ThO 0.5853 vs 0.5847, alt std 3.5 mm) |
| T3 mass 1.2–1.6 kg + derate 10/15 % (incl. 1.55 kg + 15 %): ThO monotonic, slope, derate, drag constant | passed 2026-11-25 (all 8 points within 0.001 of prediction) |
| T4 PD outer loop via SET_ATTITUDE_TARGET, ~50 Hz, latency | passed 2026-11-25. Commands at the FC 50.00 Hz (max interval 25 ms); latency from state sample to FC applying the command p50 5.4 ms, p95 8.2 ms, max 8.4 ms; hover std 1.4 mm alt / 5 mm horizontal; 1 m altitude steps overshoot about 20 % and 2 m north steps about 2 %; final error < 4 cm |
| T5 noise floor + speedup consistency (`tests/t5_repeatability.py`, `results/m3_t5/`) | passed 2026-11-25, 365 runs: T4 steps in 2 m/s wind + 1 m/s turbulence + baro noise, 60 seeds per variant, unpaired comparison against the 1× runs. All 125 1× runs flew cleanly; noise floor for 15 metrics (e.g. IAE 15.6 ± 0.66 m·s, hold RMS horizontal 0.127 ± 0.005 m, vertical 0.059 ± 0.030 m, altitude-step overshoot 19.5 ± 6.1 %). Fast-forwarded boot/takeoff/landing changes no metric (bias bound ≤ 0.68 σ). Loop speedup: 2× consistent on every metric (≤ 0.61 σ, latency p95 13 ms); 5× consistent on metrics but latency p95 27 ms > one control period, so not used; 10× and 20× biased (north-step rise, max tilt up to 4 σ; 6/60 runs at 20× fail). Criterion: the bias bound \|Δmean\| + 2·SE must stay ≤ max(σ at 1×, resolution). Earlier designs, kept: `m3_t5_pilot` (5 seeds, paired) and `m3_t5_n20` (20 seeds): too few runs for that bound, they failed even where no bias exists |
| T6 command dropout + safety reactions | passed 2026-11-25 (re-run 2026-11-26 with the M3 client changes: all checks pass; T4 too, latency p50/p95/max 5.4/8.2/8.4 ms, hover std 1.5 mm / 5.3 mm). Dropout 1 s: FC holds the last command, even a 9° tilt (2 m step overshoot 14 % vs 2 % without dropout). Dropout 5 s: FC holds 3.02 s, then levels and holds altitude (max dev 8 cm). Loop resumes every time. Fence trip: BRAKE within 20 ms, stopped 4 cm past the 6 m fence from 3 m/s. GCS mode change on a second link: loop stops at once and sends no mode of its own. Re-run 2026-11-26 after the review fixes, events A–D (the takeover case moved to T6-E): all checks pass |
| T6-E takeover through the formal entry (`tests/t6e_takeover.py`, `scenarios/t6e_takeover.yaml`) | passed 2026-11-26. A pilot on the second link sets BRAKE at 8 s and LAND at 14 s. The loop detects it in the same state period and stops; after that the client's link carries 0 mode commands, 0 attitude targets, 0 other commands (counter cross-checked: 401 attitude targets counted = 401 recorded by the loop, and it saw the GUIDED command of the takeoff). FC log: modes after handover are exactly BRAKE (8.003 s) and LAND (14.003 s), no attitude target accepted after the change. Negative control with the old behaviour (client sends LAND after every loop): LAND appears in the FC log at 8.010 s and the test fails. The earlier version of this check let the GCS switch to LAND while the procedure sent LAND itself, so it could not tell the two apart |
| Unit checks without SITL (`tests/test_unit.py`) | 12 pass: total-tilt clamp with both axes at the limit, NaN/inf/exception from the law not sent and tripping in the same step, takeover sends no mode, a run without a verdict counts 1/2 not 100 %, early-ended runs out of the IAE mean, resume rules, exit codes |
| Task library + core suite (`tests/test_unit_tasks.py`, `tasks_smoke`, `suite_core`) | 2026-11-26. Unit checks without SITL, 12 pass: trajectory derivatives equal finite differences, start and end at rest; the PD law's output for steps is bit-identical to before feedforward; gust std and correlation time come out as set, and the same seed gives the same profile; pushes rotate into the body frame. Every task flies once (`tasks_smoke`, 7/7 completed and landed). The core suite ran with the PD law and with the PD law without feedforward, 2 × 126 runs, all landed; results above. T4, T6 and T6-E re-run with the changed client: all pass |
| T1-B T1 under sensor profile B (`tests/t1b_flow_hover.py`, `results/m4_t1b.json`): SAKR3 1.40 kg, origin set by the client, GUIDED takeoff 5 m, hover 10 s, land | passed 2026-11-26. All parameters applied; profile B active (no GPS, EKF "fusing optical flow", rangefinder Good in flight). Hover 4.94 m (EKF) / 4.94 m (true); EKF height error 1.9 cm RMS. Reported: true horizontal drift 0.10 m during the 10 s hover while the EKF saw 0.01 m |
| T4-B T4 under profile B vs A (`tests/t4b_profiles.py`, `results/m4_profiles/`), 60 seeds each: B with fast setup, B all at 1×, A | passed 2026-11-26, 180 runs in 12 min, all flew and landed. Interface unchanged under B: 50 Hz at the FC, latency p50/p95 5.4/8.2 ms (A 5.5/8.3). EKF height error 0.8 cm RMS (A 5.4 cm). Fast setup consistent under B (≤ 0.94 σ). Result: hold RMS horizontal 0.34 ± 0.14 m (A 0.076 ± 0.001), vertical 0.011 ± 0.002 m (A 0.054 ± 0.031); EKF horizontal drift in the run 0.35 ± 0.17 m (A 0.012); T4 step-shape bounds hold in 28 % of B runs vs 67 % of A runs (B: the return step never reaches 90 % on true position) |

Run:

```bash
source ~/venvs/uavsim/bin/activate
cd <repo>/sim
python tests/t1_takeoff_hover_land.py
python tests/t2_t3_hover_family.py      # 8 SITL instances in parallel, ~2 min
python tests/t2_t3_hover_family.py --from-csv   # re-evaluate saved run
python tests/t4_outer_loop.py --instance 0   # ~2.5 min
python tests/t6_dropout.py --instance 1      # ~2.5 min
python tests/t6e_takeover.py --instance 1    # takeover, ~1.5 min
python tests/test_unit.py                    # no SITL, seconds
python tests/t6_dropout.py --from-data       # re-analyse saved flight data

python runner/run.py scenarios/t4_steps.yaml --instance 0 [--seed 3] [--speedup 2]   # one scenario
python runner/batch.py scenarios/batches/family_demo.yaml [--workers 18] [--resume]   # a batch
python tests/t5_repeatability.py              # T5: 365 runs, ~15 min on 20 workers (--from-data: re-analyse)
python tests/t1b_flow_hover.py --instance 0   # T1-B, ~2 min
python tests/t4b_profiles.py                  # T4-B: 180 runs, ~12 min on 20 workers (--from-data: re-analyse)
python tests/test_unit_tasks.py               # no SITL: trajectories, feedforward, gusts, pushes, event metrics
python runner/batch.py scenarios/batches/tasks_smoke.yaml               # every task once, ~1 min
python runner/batch.py scenarios/batches/suite_core.yaml                # core suite, 126 runs, ~5 min
python runner/batch.py scenarios/batches/suite_core.yaml --set controller.law=mymodule:MyLaw --name suite_core__mylaw
python analysis/scorecard.py results/suite_core [results/suite_core__mylaw]   # scorecard(s), side by side
python analysis/replay.py results/suite_core   # replay viewer -> results/suite_core/replay/index.html
python analysis/replay.py --run ~/uavsim_runs/adhoc/<run_id>   # single runs -> results/replay/index.html
```

Use instance numbers that don't overlap when several things run at once (`--base-instance` for batches).
Batches default to 30 and up, the M2 tests to 0–2.

## Still estimates (replace when collaborator data arrives)

Arm length 0.225 m, refSpd/refAngle (drag), hover current 15 A, battery resistance, motor slew
(time constant), and that ArduCopter motor params (MOT_THST_EXPO 0.65, MOT_SPIN_MIN/MAX 0.15/0.95,
PWM 1000–2000) match the real vehicle. The 23.5 N max thrust is the readme figure, assumed at full pack voltage.

For the real vehicle (M5), still open:
- companion link baud and sysid/compid (SITL uses sysid 255 = GCS);
- latency over UART instead of local TCP;
- whether the real firmware has the same BRAKE/PSC parameters (`PSC_NE_JERK`);
- `FS_GCS_ENABLE` (0 in SITL, so the dropped heartbeats in T6 trigger no FC failsafe).
- profile B: which UART the MTF-01P uses (its protocol is MAVLink, `FLOW_TYPE 5` / `RNGFND1_TYPE 10`), its
  real flow noise and delay, landed range reading and mount position, and whether the real setup takes
  height from the rangefinder (`EK3_SRC1_POSZ 2`, assumed here) or from the baro. A real indoor log
  (hover over a textured floor, with the full parameter export) would settle all of them.

Platform, open:
- Sensor noise is estimated in both profiles (A: `SIM_BARO_RND 0.1`, IMU at SITL's floor; B: flow
  ±0.05 rad/s, range ±2 cm, flow delay 20 ms); replace with values from real logs (M5). Under B the flow
  noise sets the EKF drift, and with it most of the horizontal tracking error.
- SITL lacks horizontal GPS noise. Low-frequency gusts are now added by the client (`environment.gusts`).
- Which tasks, pass bounds and θ dimensions RuleForge is judged on is a research decision; the task
  library, its bounds and the θ grid of `suite_core` are proposals. θ is mass and thrust derate so far;
  the frame model also takes inertia, motor slew and drag, and the in-flight θ events cover thrust loss
  and partial motor failure.
- What the law may know about θ (`controller.hover_thrust`), see above.
- Profile B in SITL has flat ground and a uniformly textured floor: no terrain steps, no flow quality
  loss over a plain floor or in the dark. Tasks at lower, more typical indoor heights (1–2 m) would need
  the supervisor's 2 m altitude floor lowered.
