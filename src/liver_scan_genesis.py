"""
Autonomous liver scan state machine in Genesis — data collection for imitation learning.

Ports the i4h-workflows robotic_ultrasound liver_scan_sm.py to Genesis + raysim,
removing all Isaac Sim / RTI DDS dependencies.

State machine (mirrors i4h UltrasoundStateMachine):
    SETUP    — move end-effector above phantom (scan_start_world)
    APPROACH — descend to contact point (scan_contact_world)
    CONTACT  — hold N steps (simulates force settle; no force sensor needed)
    SCANNING — sweep to scan_end_world, render US, record every step
    DONE     — save HDF5 episode, reset, repeat

HDF5 output format is compatible with the i4h training pipeline:
    data_{ep}.hdf5
      data/demo_0/
        observations/
          robot_obs   [N, 7]   EE pos (m) + quat wxyz
          joint_pos   [N, n]   joint angles
          torso_obs   [N, 7]   phantom pos + quat (constant, fixed body)
          us_image    [N, H, W] B-mode uint8 (scanning steps only use last image)
        abs_action    [N, 7]   IK target pos + quat
        action        [N, 7]   same as abs_action (absolute; rel not needed for pi0)
        state         [N, 1]   ScanState index

Usage:
    conda run -n genesis python src/liver_scan_genesis.py
    conda run -n genesis python src/liver_scan_genesis.py --cpu
"""

import argparse
import datetime
import json
import os
import sys
import tomllib
from enum import Enum

import cv2
import h5py
import numpy as np
import genesis as gs
import genesis.utils.geom as gu
from i4h_asset_helper.assets import (
    _S3_BUCKETS, _get_asset_env, _get_s3_client,
    get_i4h_asset_hash, get_i4h_asset_version, get_i4h_local_asset_path,
)

_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RS_CFG_PATH = os.path.join(_REPO_ROOT, "config", "raysim_isaac.toml")
_SM_CFG_PATH = os.path.join(_REPO_ROOT, "config", "liver_scan.toml")

with open(_RS_CFG_PATH, "rb") as _f:
    _RS_CFG = tomllib.load(_f)
with open(_SM_CFG_PATH, "rb") as _f:
    _SM_CFG = tomllib.load(_f)

_RAYSIM_ROOT = os.path.join(_REPO_ROOT, "i4h-sensor-simulation", "ultrasound-raytracing")
if _RAYSIM_ROOT not in sys.path:
    sys.path.insert(0, _RAYSIM_ROOT)

import raysim.cuda as rs


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class ScanState(Enum):
    SETUP    = 0  # move EE above phantom
    APPROACH = 1  # descend to contact point
    CONTACT  = 2  # hold (force settle simulation)
    SCANNING = 3  # sweep across liver, record data
    DONE     = 4  # episode finished


# ---------------------------------------------------------------------------
# HDF5 recorder
# ---------------------------------------------------------------------------

class HDF5Recorder:
    """Buffers one episode in memory, flushes to HDF5 on save_episode()."""

    def __init__(self, output_dir: str, task_name: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir  = output_dir
        self.task_name   = task_name
        self.episode_idx = 0
        self._reset()

    def _reset(self):
        self._robot_obs  = []   # [7]  EE pos + quat
        self._joint_pos  = []   # [n]  joint angles
        self._torso_obs  = []   # [7]  phantom pos + quat (constant)
        self._abs_action = []   # [7]  IK target pos + quat
        self._state      = []   # [1]  ScanState.value
        self._us_image   = []   # [H, W] uint8 (None → placeholder zeros)

    def record(self, robot_obs, joint_pos, torso_obs, abs_action, state_idx,
               us_image=None):
        self._robot_obs .append(np.asarray(robot_obs,  dtype=np.float32))
        self._joint_pos .append(np.asarray(joint_pos,  dtype=np.float32))
        self._torso_obs .append(np.asarray(torso_obs,  dtype=np.float32))
        self._abs_action.append(np.asarray(abs_action, dtype=np.float32))
        self._state     .append([state_idx])
        if us_image is not None:
            self._us_image.append(us_image.astype(np.uint8))
        else:
            self._us_image.append(np.zeros(1, dtype=np.uint8))  # placeholder

    def save_episode(self) -> str:
        N     = len(self._robot_obs)
        fname = os.path.join(self.output_dir, f"data_{self.episode_idx}.hdf5")
        with h5py.File(fname, "w") as f:
            f.attrs["num_samples"] = N
            f.attrs["sim"]         = True
            f.attrs["total"]       = N
            f.attrs["env_args"]    = json.dumps({"task": self.task_name})

            grp = f.create_group("data/demo_0")
            grp.attrs["num_samples"] = N

            obs = grp.create_group("observations")
            obs.create_dataset("robot_obs",  data=np.stack(self._robot_obs))
            obs.create_dataset("joint_pos",  data=np.stack(self._joint_pos))
            obs.create_dataset("torso_obs",  data=np.stack(self._torso_obs))
            obs.create_dataset("us_image",   data=np.stack(self._us_image))

            grp.create_dataset("abs_action", data=np.stack(self._abs_action))
            grp.create_dataset("action",     data=np.stack(self._abs_action))
            grp.create_dataset("state",      data=np.array(self._state, dtype=np.int32))

        print(f"  → saved {fname}  ({N} steps)")
        self.episode_idx += 1
        self._reset()
        return fname


# ---------------------------------------------------------------------------
# Coordinate helpers (identical to echoworld_test.py)
# ---------------------------------------------------------------------------

def _euler_deg_to_R(ex, ey, ez):
    rx, ry, rz = np.radians(ex), np.radians(ey), np.radians(ez)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0,           1, 0          ],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,           1]])
    return Rz @ Ry @ Rx


def _quat_wxyz_to_euler_xyz(quat_wxyz):
    R  = gu.quat_to_R(quat_wxyz)
    ry = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    rx = np.arctan2(R[2, 1] / np.cos(ry), R[2, 2] / np.cos(ry))
    rz = np.arctan2(R[1, 0] / np.cos(ry), R[0, 0] / np.cos(ry))
    return np.array([rx, ry, rz], dtype=np.float32)


PROBE_TCP_OFFSET_M = np.array(_RS_CFG["sensor"]["tcp_offset_m"])
_ORGAN_EULER_DEG   = tuple(_RS_CFG["world"]["organ_euler_deg"])
_ORIENT_EULER_DEG  = tuple(_RS_CFG["world"]["orient_euler_deg"])

_ORGAN_MATERIALS = {
    "Liver.obj":        "liver",
    "Kidney.obj":       "muscle",
    # "Gallbladder.obj":  "water",
    # "Pancreas.obj":     "muscle",
    # "Colon.obj":        "muscle",
    # "Small_bowel.obj":  "muscle",
    # "Stomach.obj":      "muscle",
    "Heart.obj":        "muscle",
    "Bone.obj":         "bone",
    # "Back_muscles.obj": "muscle",
    # "Spleen.obj":       "muscle",
    # "Vessels.obj":      "blood",
    # "Tumor1.obj":       "liver",
    # "Tumor2.obj":       "liver",
    # "Lungs.obj":        "muscle",
    # "Skin.obj":         "muscle",
}

# Colours for Genesis viewer (RGB, medical convention)
_ORGAN_COLORS = {
    "Liver.obj":        (0.50, 0.08, 0.08),
    "Kidney.obj":       (0.75, 0.18, 0.12),
    # "Gallbladder.obj":  (0.40, 0.62, 0.10),
    # "Pancreas.obj":     (0.90, 0.60, 0.55),
    # "Colon.obj":        (0.80, 0.50, 0.40),
    # "Small_bowel.obj":  (0.88, 0.68, 0.58),
    # "Stomach.obj":      (0.82, 0.55, 0.50),
    "Heart.obj":        (0.85, 0.08, 0.08),
    "Bone.obj":         (0.92, 0.88, 0.78),
    # "Back_muscles.obj": (0.60, 0.12, 0.12),
    # "Spleen.obj":       (0.45, 0.08, 0.18),
    # "Vessels.obj":      (0.55, 0.03, 0.08),
    # "Tumor1.obj":       (0.85, 0.82, 0.10),
    # "Tumor2.obj":       (0.85, 0.82, 0.10),
    # "Lungs.obj":        (0.90, 0.70, 0.65),
    # "Skin.obj":         (0.88, 0.72, 0.58),
}

def _build_raysim_world(organ_dir):
    mats  = rs.Materials()
    world = rs.World("water")
    for obj, mat in _ORGAN_MATERIALS.items():
        p = os.path.join(organ_dir, obj)
        if os.path.exists(p):
            world.add(rs.Mesh(p, mats.get_index(mat)))
    return world, mats


def _raysim_render(probe_face_m, R_probe, rs_origin_m,
                   rs_probe_tpl, rs_sim_params, rs_simulator):
    R_organ   = _euler_deg_to_R(*_ORGAN_EULER_DEG).T
    R_orient  = _euler_deg_to_R(*_ORIENT_EULER_DEG)
    d         = (probe_face_m - rs_origin_m) * 1000.0
    pos_mm    = (R_organ @ d).astype(np.float32)
    R_rs      = R_organ @ R_probe @ R_orient
    euler_xyz = _quat_wxyz_to_euler_xyz(gu.R_to_quat(R_rs))
    rs_probe_tpl.set_pose(rs.Pose(position=pos_mm, rotation=euler_xyz))
    bmode   = rs_simulator.simulate(rs_probe_tpl, rs_sim_params)
    DR      = 60.0
    display = np.clip((bmode + DR) / DR, 0.0, 1.0)
    return (display * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu",       action="store_true", default=False)
    parser.add_argument("--vis",       action="store_true", default=False,
                        help="Open Genesis viewer (slower)")
    parser.add_argument("--no-record", action="store_true", default=False,
                        help="Run state machine without saving HDF5 data")
    args = parser.parse_args()

    # --- Config ----------------------------------------------------------
    ep_cfg  = _SM_CFG["episode"]
    rob_cfg = _SM_CFG["robot"]
    pos_cfg = _SM_CFG["positions"]
    tr_cfg  = _SM_CFG["transitions"]
    sc_cfg  = _SM_CFG["scanning"]
    out_cfg = _SM_CFG["output"]

    NUM_EPISODES   = ep_cfg["num_episodes"]
    MAX_STEPS      = ep_cfg["max_steps"]
    RESET_STEPS    = ep_cfg["reset_steps"]
    POS_TOL        = tr_cfg["pos_tolerance_m"]
    CONTACT_STEPS  = tr_cfg["contact_steps"]
    SCAN_STEPS     = sc_cfg["scan_steps"]
    HOLD_STEPS     = sc_cfg["hold_steps"]

    DOWN_QUAT      = gu.xyz_to_quat(np.radians(rob_cfg["down_euler_deg"]))
    SCAN_START_W   = np.array(pos_cfg["scan_start_world"])
    SCAN_CONTACT_W = np.array(pos_cfg["scan_contact_world"])
    SCAN_END_W     = np.array(pos_cfg["scan_end_world"])
    SCAN_INCREMENT = (SCAN_END_W - SCAN_CONTACT_W) / SCAN_STEPS

    DATA_DIR  = os.path.join(_REPO_ROOT, out_cfg["data_dir"],
                              datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    TASK_NAME = out_cfg["task_name"]

    # --- Assets ----------------------------------------------------------
    local_dir       = get_i4h_local_asset_path()
    organ_dir       = os.path.join(local_dir, "Props", "ABDPhantom", "Organs")
    robot_usd_path  = os.path.join(_REPO_ROOT, "assets", "data", "Robots",
                                   "Franka", "fr3_US.usd")
    PHANTOM_POS_M   = np.array(_RS_CFG["world"]["phantom_pos"])
    RS_ORIGIN_M     = PHANTOM_POS_M

    # phantom torso_obs (constant — fixed rigid body)
    TORSO_OBS = np.concatenate([PHANTOM_POS_M, np.array([1., 0., 0., 0.])])

    # --- Genesis scene ---------------------------------------------------
    import taichi as ti
    ti.init(arch=ti.cpu if args.cpu else ti.cuda, device_memory_GB=2)
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    # --- raysim ----------------------------------------------------------
    rs_world, rs_mats = _build_raysim_world(organ_dir)
    rs_sim   = rs.RaytracingUltrasoundSimulator(rs_world, rs_mats)

    _s = _RS_CFG["sim"]
    rs_params = rs.SimParams()
    rs_params.t_far              = _s["t_far"]
    rs_params.buffer_size        = _s["buffer_size"]
    rs_params.max_depth          = _s["max_depth"]
    rs_params.min_intensity      = _s["min_intensity"]
    rs_params.use_scattering     = _s["use_scattering"]
    rs_params.conv_psf           = _s["conv_psf"]
    rs_params.median_clip_filter = _s["median_clip_filter"]
    rs_params.b_mode_size        = (_s["b_mode_width"], _s["b_mode_height"])
    rs_params.contact_epsilon    = _s["contact_epsilon"]

    _p = _RS_CFG["probe"]
    if _p["type"] == "curvilinear":
        rs_probe = rs.CurvilinearProbe(
            num_elements_x=_p["num_elements_x"],
            sector_angle=_p["sector_angle"],
            radius=_p["radius"],
            frequency=_p["frequency"],
            speed_of_sound=_p["speed_of_sound"],
            pulse_duration=_p["pulse_duration"],
        )
    else:
        raise ValueError(f"Unsupported probe type: {_p['type']!r}")

    # --- Genesis scene build ---------------------------------------------
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0, 0, -9.8),
            box_box_detection=True,
            constraint_timeconst=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 1.0, 1.0),
            camera_lookat=(0.4, 0.0, 0.2),
            camera_fov=50,
            max_FPS=60,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane())

    robot = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.USD(
            file=robot_usd_path,
            requires_jac_and_IK=True,
            recompute_inertia=True,
        ),
    )

    for obj_name, color in _ORGAN_COLORS.items():
        obj_path = os.path.join(organ_dir, obj_name)
        if not os.path.exists(obj_path):
            continue
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=obj_path,
                scale=0.001,
                pos=PHANTOM_POS_M,
                fixed=True,
                euler=_ORGAN_EULER_DEG,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=color),
        )

    scene.build()

    n_dofs     = robot.n_dofs
    motors_dof = np.arange(n_dofs)
    ee_link    = robot.get_link("/fr3/fr3_link8")
    probe_link = robot.get_link("/fr3/linear")

    robot.set_dofs_kp([4500] * n_dofs)
    robot.set_dofs_kv([450]  * n_dofs)
    robot.set_dofs_force_range([-500] * n_dofs, [500] * n_dofs)

    # --- Recorder --------------------------------------------------------
    recorder = None if args.no_record else HDF5Recorder(DATA_DIR, TASK_NAME)

    # --- Episode loop ----------------------------------------------------
    if recorder:
        print(f"\n[scan] Output: {DATA_DIR}")
    else:
        print("\n[scan] --no-record: running without saving data")
    print(f"[scan] Collecting {NUM_EPISODES} episodes\n")

    for ep in range(NUM_EPISODES):
        print(f"Episode {ep + 1}/{NUM_EPISODES}")

        # Reset robot to scan_start
        q0 = robot.inverse_kinematics(link=ee_link,
                                       pos=SCAN_START_W,
                                       quat=DOWN_QUAT)
        robot.set_qpos(q0, motors_dof)
        for _ in range(RESET_STEPS):
            robot.control_dofs_position(q0, motors_dof)
            scene.step()

        # Episode state
        state         = ScanState.SETUP
        step_in_state = 0
        scan_step     = 0
        scan_pos      = SCAN_CONTACT_W.copy()
        target_pos    = SCAN_START_W.copy()
        target_quat   = DOWN_QUAT.copy()
        total_steps   = 0
        last_display  = None

        while state != ScanState.DONE and total_steps < MAX_STEPS:
            # IK step
            q, _ = robot.inverse_kinematics(
                link=ee_link, pos=target_pos, quat=target_quat,
                return_error=True)
            robot.control_dofs_position(q, motors_dof)
            scene.step()
            total_steps += 1

            # Current EE state
            ee_pos    = ee_link.get_pos().cpu().numpy()
            ee_quat   = ee_link.get_quat().cpu().numpy()
            joint_pos = robot.get_qpos().cpu().numpy()

            probe_body_m = probe_link.get_pos().cpu().numpy()
            probe_quat   = probe_link.get_quat().cpu().numpy()
            R_probe      = gu.quat_to_R(probe_quat)
            probe_face_m = probe_body_m + R_probe @ PROBE_TCP_OFFSET_M

            # Observations
            robot_obs  = np.concatenate([ee_pos, ee_quat])
            abs_action = np.concatenate([target_pos, target_quat])

            # Render US (every step during SCANNING, else skip)
            us_image = None
            if state == ScanState.SCANNING:
                try:
                    us_image = _raysim_render(
                        probe_face_m, R_probe, RS_ORIGIN_M,
                        rs_probe, rs_params, rs_sim)
                    last_display = us_image
                    cv2.imshow("Ultrasound (raysim)", us_image)
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"  [US] {e}")

            # Record every step
            if recorder:
                recorder.record(robot_obs, joint_pos, TORSO_OBS, abs_action,
                                state.value, us_image)

            # ---- State transitions -------------------------------------
            if state == ScanState.SETUP:
                target_pos  = SCAN_START_W
                target_quat = DOWN_QUAT
                dist = np.linalg.norm(ee_pos - SCAN_START_W)
                if dist < POS_TOL:
                    state = ScanState.APPROACH
                    step_in_state = 0
                    print(f"  SETUP → APPROACH  (step {total_steps})")

            elif state == ScanState.APPROACH:
                target_pos  = SCAN_CONTACT_W
                target_quat = DOWN_QUAT
                dist = np.linalg.norm(ee_pos - SCAN_CONTACT_W)
                if dist < POS_TOL:
                    state = ScanState.CONTACT
                    step_in_state = 0
                    print(f"  APPROACH → CONTACT  (step {total_steps})")

            elif state == ScanState.CONTACT:
                target_pos  = SCAN_CONTACT_W
                target_quat = DOWN_QUAT
                step_in_state += 1
                if step_in_state >= CONTACT_STEPS:
                    state     = ScanState.SCANNING
                    scan_pos  = SCAN_CONTACT_W.copy()
                    scan_step = 0
                    step_in_state = 0
                    print(f"  CONTACT → SCANNING  (step {total_steps})")

            elif state == ScanState.SCANNING:
                if scan_step < SCAN_STEPS:
                    scan_pos   = SCAN_CONTACT_W + SCAN_INCREMENT * scan_step
                    target_pos = scan_pos.copy()
                    target_quat = DOWN_QUAT
                    scan_step += 1
                elif scan_step < SCAN_STEPS + HOLD_STEPS:
                    scan_step += 1
                else:
                    state = ScanState.DONE
                    print(f"  SCANNING → DONE  (step {total_steps})")

        # Save episode
        if recorder:
            if total_steps >= MAX_STEPS:
                print(f"  timeout at {total_steps} steps — saving partial episode")
            recorder.save_episode()

    cv2.destroyAllWindows()
    if recorder:
        print(f"\n[scan] Done. {NUM_EPISODES} episodes saved to {DATA_DIR}")
    else:
        print(f"\n[scan] Done. (no data saved)")


if __name__ == "__main__":
    main()
