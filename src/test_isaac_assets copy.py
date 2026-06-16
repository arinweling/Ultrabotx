"""
Keyboard Controls:
↑  - Move end-effector forward
↓  - Move end-effector backward
←  - Move end-effector left
→  - Move end-effector right
n  - Move end-effector up
m  - Move end-effector down
j  - Rotate end-effector counterclockwise
k  - Rotate end-effector clockwise
[  - Decrease scan depth  (-10 mm)
]  - Increase scan depth  (+10 mm)
,  - Decrease probe frequency (-0.5 MHz)
.  - Increase probe frequency (+0.5 MHz)
\\ - Reset robot pose
esc - Quit

Plus default viewer controls (press 'i' in viewer to see them)

Differences from test.py:
  - Robot loaded from i4h panda_assembly.usda instead of local MJCF
  - Phantom/probe assets remain the same (already from i4h asset helper)
"""

import argparse
import os
import sys
import tomllib

import cv2
import numpy as np
import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind
from i4h_asset_helper.assets import (
    _get_s3_client, _S3_BUCKETS, _get_asset_env,
    get_i4h_local_asset_path, get_i4h_asset_hash, get_i4h_asset_version,
)

_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "raysim_isaac.toml")

with open(_CONFIG_PATH, "rb") as _f:
    _CFG = tomllib.load(_f)

# raysim is built inside the ultrasound-raytracing repo; add it to the path
_RAYSIM_ROOT = os.path.join(_REPO_ROOT, "i4h-sensor-simulation", "ultrasound-raytracing")
if _RAYSIM_ROOT not in sys.path:
    sys.path.insert(0, _RAYSIM_ROOT)

import raysim.cuda as rs


# --- Asset download -----------------------------------------------------------

def _download_asset(sub_path: str, local_dir: str) -> None:
    """Download all files under sub_path from S3 if not already present."""
    bucket   = _S3_BUCKETS[_get_asset_env()]
    ver      = get_i4h_asset_version()
    h        = get_i4h_asset_hash(version=ver)
    prefix   = f"Assets/Isaac/Healthcare/{ver}/{h}/"
    s3       = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + sub_path):
        for obj in page.get("Contents", []):
            rel  = obj["Key"][len(prefix):]
            dest = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest):
                print(f"  Downloading {rel}")
                s3.download_file(bucket, obj["Key"], dest)


# --- Coordinate helpers ------------------------------------------------------

def _quat_wxyz_to_euler_xyz(quat_wxyz):
    R = gu.quat_to_R(quat_wxyz)
    ry = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    rx = np.arctan2(R[2, 1] / np.cos(ry), R[2, 2] / np.cos(ry))
    rz = np.arctan2(R[1, 0] / np.cos(ry), R[0, 0] / np.cos(ry))
    return np.array([rx, ry, rz], dtype=np.float32)


# --- raysim world setup ------------------------------------------------------

PROBE_TCP_OFFSET_M = np.array(_CFG["sensor"]["tcp_offset_m"])

def _euler_deg_to_R(ex, ey, ez):
    rx, ry, rz = np.radians(ex), np.radians(ey), np.radians(ez)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

US_SCAN_DEPTH_MM = _CFG["sim"]["t_far"]

_ORGAN_MATERIALS = {
    "Liver.obj":        "liver",
    "Kidney.obj":       "muscle",
    "Gallbladder.obj":  "water",
    "Pancreas.obj":     "muscle",
    "Colon.obj":        "muscle",
    "Small_bowel.obj":  "muscle",
    "Stomach.obj":      "muscle",
    "Heart.obj":        "muscle",
    "Bone.obj":         "bone",
    "Back_muscles.obj": "muscle",
    "Spleen.obj":       "muscle",
    "Vessels.obj":      "blood",
    "Tumor1.obj":       "liver",
    "Tumor2.obj":       "liver",
    "Lungs.obj":        "muscle",
    # "Skin.obj":         "muscle",
}

# Colours for Genesis viewer (RGB, medical convention)
_ORGAN_COLORS = {
    "Liver.obj":        (0.50, 0.08, 0.08),
    "Kidney.obj":       (0.75, 0.18, 0.12),
    "Gallbladder.obj":  (0.40, 0.62, 0.10),
    "Pancreas.obj":     (0.90, 0.60, 0.55),
    "Colon.obj":        (0.80, 0.50, 0.40),
    "Small_bowel.obj":  (0.88, 0.68, 0.58),
    "Stomach.obj":      (0.82, 0.55, 0.50),
    "Heart.obj":        (0.85, 0.08, 0.08),
    "Bone.obj":         (0.92, 0.88, 0.78),
    "Back_muscles.obj": (0.60, 0.12, 0.12),
    "Spleen.obj":       (0.45, 0.08, 0.18),
    "Vessels.obj":      (0.55, 0.03, 0.08),
    "Tumor1.obj":       (0.85, 0.82, 0.10),
    "Tumor2.obj":       (0.85, 0.82, 0.10),
    "Lungs.obj":        (0.90, 0.70, 0.65),
    # "Skin.obj":         (0.88, 0.72, 0.58),
}


_ORGAN_EULER_DEG  = tuple(_CFG["world"]["organ_euler_deg"])
_ORIENT_EULER_DEG = tuple(_CFG["world"]["orient_euler_deg"])


def build_raysim_world(organ_dir):
    materials = rs.Materials()
    world = rs.World("water")
    loaded = []
    for obj_name, mat_name in _ORGAN_MATERIALS.items():
        obj_path = os.path.join(organ_dir, obj_name)
        if not os.path.exists(obj_path):
            continue
        mat_idx = materials.get_index(mat_name)
        world.add(rs.Mesh(obj_path, mat_idx))
        loaded.append(obj_name)
    print(f"raysim meshes: {', '.join(loaded)}")
    return world, materials


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    local_dir = get_i4h_local_asset_path()

    robot_usd_path = os.path.join(_REPO_ROOT, "assets", "data", "Robots", "Franka", "fr3_US.usd")

    PHANTOM_POS_M = np.array(_CFG["world"]["phantom_pos"])
    RS_ORIGIN_M   = PHANTOM_POS_M

    # --- Genesis scene --------------------------------------------------------
    import taichi as ti
    ti.init(arch=ti.cpu if args.cpu else ti.cuda, device_memory_GB=2)

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    # --- raysim setup ---------------------------------------------------------
    organ_dir = os.path.join(local_dir, "Props", "ABDPhantom", "Organs")
    rs_world, rs_materials = build_raysim_world(organ_dir)
    rs_simulator = rs.RaytracingUltrasoundSimulator(rs_world, rs_materials)

    _s = _CFG["sim"]
    rs_sim_params = rs.SimParams()
    rs_sim_params.t_far              = _s["t_far"]
    rs_sim_params.buffer_size        = _s["buffer_size"]
    rs_sim_params.max_depth          = _s["max_depth"]
    rs_sim_params.min_intensity      = _s["min_intensity"]
    rs_sim_params.use_scattering     = _s["use_scattering"]
    rs_sim_params.conv_psf           = _s["conv_psf"]
    rs_sim_params.median_clip_filter = _s["median_clip_filter"]
    rs_sim_params.b_mode_size        = (_s["b_mode_width"], _s["b_mode_height"])
    rs_sim_params.contact_epsilon    = _s["contact_epsilon"]

    _p = _CFG["probe"]
    _probe_type = _p["type"]
    if _probe_type == "curvilinear":
        rs_probe_template = rs.CurvilinearProbe(
            num_elements_x=_p["num_elements_x"],
            sector_angle=_p["sector_angle"],
            radius=_p["radius"],
            frequency=_p["frequency"],
            speed_of_sound=_p["speed_of_sound"],
            pulse_duration=_p["pulse_duration"],
        )
    elif _probe_type == "linear":
        rs_probe_template = rs.LinearArrayProbe(
            num_elements_x=_p["num_elements_x"],
            width=_p["width"],
            frequency=_p["frequency"],
            speed_of_sound=_p["speed_of_sound"],
            pulse_duration=_p["pulse_duration"],
        )
    elif _probe_type == "phased":
        rs_probe_template = rs.PhasedArrayProbe(
            num_elements_x=_p["num_elements_x"],
            width=_p["width"],
            sector_angle=_p["sector_angle"],
            frequency=_p["frequency"],
            speed_of_sound=_p["speed_of_sound"],
            pulse_duration=_p["pulse_duration"],
        )
    else:
        raise ValueError(f"Unknown probe type: {_probe_type!r}")
    print(f"Probe: {_probe_type}, {_p['num_elements_x']} elements, {_p['frequency']} MHz")
    np.set_printoptions(precision=7, suppress=True)

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
            enable_gui=True,
            camera_pos=(1.5, 1.0, 1.0),
            camera_lookat=(0.4, 0.0, 0.2),
            camera_fov=50,
            max_FPS=60,
        ),
        show_viewer=True,
    )

    plane = scene.add_entity(gs.morphs.Plane())

    # --- Robot from local fr3_US.usd -----------------------------------------
    robot = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.USD(
            file=robot_usd_path,
            requires_jac_and_IK=True,
            recompute_inertia=True,
        ),
    )
    print(f"Robot USD loaded: {robot_usd_path}")

    # Phantom outer skin
    phantom_path = os.path.join(local_dir, "Props", "ABDPhantom", "phantom.usda")
    # scene.add_stage(
    #     gs.morphs.USD(file=phantom_path, pos=PHANTOM_POS_M, euler=(0, 0, 180), fixed=True)
    # )
    print("Loaded phantom skin (USD)")

    # Individual organs
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
                collision=True,
            ),
            surface=gs.surfaces.Default(color=color),
        )
    print(f"Loaded {sum(1 for n in _ORGAN_COLORS if os.path.exists(os.path.join(organ_dir, n)))} organs")

    target = scene.add_entity(
        gs.morphs.Mesh(file="meshes/axis.obj", scale=0.15, collision=False),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    # --- RealSense D435 depth camera (world-fixed, overhead view) -------------
    # D435 specs: 640x480 depth, 87°H x 58°V FOV, 0.1–3m range
    # pos_offset is in world frame (plane is at origin with no rotation)
    # Camera is placed 1.2m above, 0.9m behind the phantom, angled 40° down
    _cam_x   = PHANTOM_POS_M[0] + 0.3
    _cam_y   = PHANTOM_POS_M[1]
    _cam_z   = 0.5    
    _cam_pos = (_cam_x, _cam_y, _cam_z)
    _cam_euler = (0.0, 40.0, 180.0)

    realsense = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.DepthCameraPattern(
                res=(640, 480),
                fov_horizontal=87.0,
                fov_vertical=58.0,
            ),
            entity_idx=plane.idx,
            link_idx_local=0,
            pos_offset=_cam_pos,
            euler_offset=_cam_euler,
            max_range=3.0,
            min_range=0.1,
            return_world_frame=True,
        )
    )

    # Visual mesh so the camera is visible in the Genesis viewer
    _CAMERA_MESH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "genesis-world", "genesis", "assets", "meshes", "camera", "camera.glb",
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=_CAMERA_MESH,
            scale=0.02,
            pos=_cam_pos,
            euler=(-_cam_euler[1], _cam_euler[0], _cam_euler[2] - 90.0),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(0.1, 0.1, 0.1, 1.0)),
    )
    print(f"RealSense D435 sensor added at ({_cam_x:.2f}, {_cam_y:.2f}, {_cam_z:.2f})")

    scene.build()

    # Print available links so we can find the right names
    print("Robot links:", [l.name for l in robot.links])

    # --- Robot control --------------------------------------------------------
    # fr3_US.usd links: base, fr3_link0..8, linear (US probe fixed to fr3_link8)
    n_dofs     = robot.n_dofs
    motors_dof = np.arange(n_dofs)
    ee_link    = robot.get_link("/fr3/fr3_link8")
    probe_link = robot.get_link("/fr3/linear")

    # USD joint gains are too weak (stiffness=100, damping=1) — override with Franka defaults
    robot.set_dofs_kp([4500] * n_dofs)
    robot.set_dofs_kv([450]  * n_dofs)
    robot.set_dofs_force_range([-500] * n_dofs, [500] * n_dofs)

    robot_init_pos  = np.array([0.5, 0.0, 0.50])
    robot_init_quat = gu.xyz_to_quat(np.array([0, np.pi, 0]))

    target_pos  = robot_init_pos.copy()
    target_quat = robot_init_quat.copy()

    dpos = 0.002
    drot = 0.01

    def reset_robot():
        target_pos[:]  = robot_init_pos.copy()
        target_quat[:] = robot_init_quat.copy()
        target.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.set_qpos(q, motors_dof)

    reset_robot()

    scan_depth_mm  = [US_SCAN_DEPTH_MM]
    ddepth         = 10.0
    probe_freq_mhz = [_CFG["probe"]["frequency"]]
    dfreq          = 0.5

    def change_freq(delta):
        probe_freq_mhz[0] = max(1.0, min(20.0, probe_freq_mhz[0] + delta))
        rs_probe_template.set_frequency(probe_freq_mhz[0])
        print(f"\rUS freq: {probe_freq_mhz[0]:.1f} MHz    ", end="", flush=True)

    def move(delta):
        target_pos[:] += np.array(delta, dtype=gs.np_float)

    def rotate(delta):
        target_quat[:] = gu.transform_quat_by_quat(
            target_quat, gu.xyz_to_quat(np.array([0, 0, delta]))
        )

    def change_depth(delta):
        scan_depth_mm[0] = max(10.0, scan_depth_mm[0] + delta)
        rs_sim_params.t_far = scan_depth_mm[0]
        print(f"\rUS depth: {scan_depth_mm[0]:.0f} mm    ", end="", flush=True)

    is_running = True

    def stop():
        global is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("move_forward",  Key.UP,            KeyAction.HOLD,    callback=move,         args=((-dpos, 0, 0),)),
        Keybind("move_back",     Key.DOWN,           KeyAction.HOLD,    callback=move,         args=((dpos, 0, 0),)),
        Keybind("move_left",     Key.LEFT,           KeyAction.HOLD,    callback=move,         args=((0, -dpos, 0),)),
        Keybind("move_right",    Key.RIGHT,          KeyAction.HOLD,    callback=move,         args=((0, dpos, 0),)),
        Keybind("move_up",       Key.N,              KeyAction.HOLD,    callback=move,         args=((0, 0, dpos),)),
        Keybind("move_down",     Key.M,              KeyAction.HOLD,    callback=move,         args=((0, 0, -dpos),)),
        Keybind("rotate_ccw",    Key.J,              KeyAction.HOLD,    callback=rotate,       args=(drot,)),
        Keybind("rotate_cw",     Key.K,              KeyAction.HOLD,    callback=rotate,       args=(-drot,)),
        Keybind("depth_inc",     Key.BRACKETRIGHT,  KeyAction.RELEASE, callback=change_depth, args=(ddepth,)),
        Keybind("depth_dec",     Key.BRACKETLEFT,   KeyAction.RELEASE, callback=change_depth, args=(-ddepth,)),
        Keybind("freq_inc",      Key.PERIOD,         KeyAction.RELEASE, callback=change_freq,  args=(dfreq,)),
        Keybind("freq_dec",      Key.COMMA,          KeyAction.RELEASE, callback=change_freq,  args=(-dfreq,)),
        Keybind("reset",         Key.BACKSLASH,      KeyAction.RELEASE, callback=reset_robot),
        Keybind("quit",          Key.ESCAPE,         KeyAction.RELEASE, callback=stop),
    )

    # --- Simulation loop ------------------------------------------------------
    _us_step  = 0
    US_EVERY  = 10
    DEPTH_EVERY = 10  # read RealSense depth every N sim steps

    try:
        while is_running:
            target.set_qpos(np.concatenate([target_pos, target_quat]))
            q, _ = robot.inverse_kinematics(
                link=ee_link, pos=target_pos, quat=target_quat, return_error=True
            )
            robot.control_dofs_position(q, motors_dof)
            scene.step()

            probe_body_m  = probe_link.get_pos().cpu().numpy()
            probe_face_q  = probe_link.get_quat().cpu().numpy()
            R_probe       = gu.quat_to_R(probe_face_q)
            probe_face_m  = probe_body_m + R_probe @ PROBE_TCP_OFFSET_M

            _us_step += 1

            if _us_step % DEPTH_EVERY == 0:
                depth_np = realsense.read_image().cpu().numpy()
                # Exclude no-hit pixels (clamped to max_range=3.0) and non-finite values
                valid = np.isfinite(depth_np) & (depth_np < 2.99)
                if valid.any():
                    # Percentile clip so a few far/near outliers don't crush the gradient
                    p2, p98 = np.percentile(depth_np[valid], [2, 98])
                    norm = np.clip((depth_np - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
                    norm[~valid] = 0.0
                    # TURBO has better perceptual uniformity than JET; display at half res for speed
                    depth_vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    depth_vis = cv2.resize(depth_vis, (320, 240), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("RealSense D435 Depth", depth_vis)
                    cv2.waitKey(1)

            if _us_step % US_EVERY != 0:
                continue

            R_organ  = _euler_deg_to_R(*_ORGAN_EULER_DEG).T
            R_orient = _euler_deg_to_R(*_ORIENT_EULER_DEG)
            d        = (probe_face_m - RS_ORIGIN_M) * 1000.0
            pos_mm   = (R_organ @ d).astype(np.float32)
            R_rs     = R_organ @ R_probe @ R_orient
            euler_xyz = _quat_wxyz_to_euler_xyz(gu.R_to_quat(R_rs))

            rs_probe_template.set_pose(rs.Pose(
                position=pos_mm.astype(np.float32),
                rotation=euler_xyz,
            ))

            try:
                bmode = rs_simulator.simulate(rs_probe_template, rs_sim_params)
                DR = 60.0
                display = np.clip((bmode + DR) / DR, 0.0, 1.0)
                display = (display * 255).astype(np.uint8)
                cv2.imshow("Ultrasound (raysim)", display)
                cv2.waitKey(1)
            except Exception as e:
                print(f"[US] {e}")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
