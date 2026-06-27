"""
Keyboard Controls:
↑ / ↓  - Sweep probe along its face
← / →  - Slide probe across its face
m / n  - Press / release probe normal to the face
j / k  - Rotate probe footprint
q / e  - Rock probe side-to-side
t / g  - Tilt probe heel-toe
[  - Decrease scan depth  (-10 mm)
]  - Increase scan depth  (+10 mm)
,  - Decrease probe frequency (-0.5 MHz)
.  - Increase probe frequency (+0.5 MHz)
\\ - Reset robot pose
esc - Quit

Plus default viewer controls (press 'i' in viewer to see them)
"""

import argparse
import os
import re
import sys
import tomllib

import cv2
import numpy as np
import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind
from i4h_asset_helper.assets import get_i4h_local_asset_path

_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "raysim_custom.toml")

with open(_CONFIG_PATH, "rb") as _f:
    _CFG = tomllib.load(_f)

# raysim is built inside the ultrasound-raytracing repo; add it to the path
_RAYSIM_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "i4h-sensor-simulation", "ultrasound-raytracing",
)
if _RAYSIM_ROOT not in sys.path:
    sys.path.insert(0, _RAYSIM_ROOT)

import raysim.cuda as rs


# --- Coordinate helpers ------------------------------------------------------

def _quat_wxyz_to_euler_xyz(quat_wxyz):
    R = gu.quat_to_R(quat_wxyz)
    ry = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    rx = np.arctan2(R[2, 1] / np.cos(ry), R[2, 2] / np.cos(ry))
    rz = np.arctan2(R[1, 0] / np.cos(ry), R[0, 0] / np.cos(ry))
    return np.array([rx, ry, rz], dtype=np.float32)


# --- raysim world setup ------------------------------------------------------

PROBE_TCP_OFFSET_M = np.array(_CFG["sensor"]["tcp_offset_m"])


def _probe_control_axes():
    press_axis = PROBE_TCP_OFFSET_M.astype(np.float64)
    press_norm = np.linalg.norm(press_axis)
    if press_norm < 1e-9:
        press_axis = np.array([0.0, 0.0, 1.0])
    else:
        press_axis /= press_norm

    slide_axis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(slide_axis, press_axis))) > 0.95:
        slide_axis = np.array([0.0, 1.0, 0.0])
    slide_axis = slide_axis - press_axis * np.dot(slide_axis, press_axis)
    slide_axis /= np.linalg.norm(slide_axis)

    sweep_axis = np.cross(press_axis, slide_axis)
    sweep_axis /= np.linalg.norm(sweep_axis)
    return slide_axis, sweep_axis, press_axis


PROBE_SLIDE_AXIS, PROBE_SWEEP_AXIS, PROBE_PRESS_AXIS = _probe_control_axes()

def _euler_deg_to_R(ex, ey, ez):
    rx, ry, rz = np.radians(ex), np.radians(ey), np.radians(ez)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

US_SCAN_DEPTH_MM = _CFG["sim"]["t_far"]

_ORGAN_MATERIALS = {
    # "Liver.obj":        "liver",
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
    # "Liver.obj":        (0.50, 0.08, 0.08),
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


_ORGAN_EULER_DEG = tuple(_CFG["world"]["organ_euler_deg"])


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
    parser.add_argument("-v", "--vis", action="store_true", default=True,
                        help="Open interactive viewer")
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    local_dir = get_i4h_local_asset_path()

    # Coordinate systems
    # -----------------
    # Genesis : metres, Z-up, origin at world floor centre.
    # raysim  : millimetres, same Z-up axes, origin at the phantom USD local origin.
    #
    # The phantom USD (metersPerUnit=1, geometry scale=0.003) is centred at its
    # local (0,0,0).  The raysim OBJ organs share the same origin.  Their combined
    # bounding box in raysim mm is approximately:
    #   X [-133, +160]  Y [-91, +107]  Z [-121, +162]
    # Setting PHANTOM_POS_M.z = 0.121 places the bottom of the organs (Z = -121 mm)
    # exactly on the Genesis ground plane (z = 0).
    PHANTOM_POS_M = np.array(_CFG["world"]["phantom_pos"])  # where phantom USD is placed in Genesis
    RS_ORIGIN_M   = PHANTOM_POS_M                  # raysim (0,0,0) in Genesis metres

    # (kept for reference — probe face now comes from the MJCF probe_link body)
    # PROBE_FACE_OFFSET_M    = 0.08
    # PROBE_LATERAL_OFFSET_M = -0.025

    # --- Genesis scene --------------------------------------------------------
    # Limit Taichi's GPU memory pool before Genesis calls ti.init() internally.
    # RTX 3060 has 6 GB; 2 GB for Genesis leaves ~4 GB for raysim's OptiX BVH.
    import taichi as ti
    ti.init(arch=ti.cpu if args.cpu else ti.cuda, device_memory_GB=2)

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    # --- raysim setup (after gs.init so both share the same CUDA context) -----
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
        raise ValueError(f"Unknown probe type in config: {_probe_type!r}")
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
        morph=gs.morphs.MJCF(file="/home/arin/Ultrabotx/xml/franka_emika_panda/panda_with_probe.xml"),
    )

    # Phantom outer skin (USD)
    phantom_path = os.path.join(local_dir, "Props", "ABDPhantom", "phantom.usda")

    scene.add_stage(
        gs.morphs.USD(file=phantom_path, pos=PHANTOM_POS_M, euler=(0, 0, 180), fixed=True)
    )
    print("Loaded phantom skin (USD)")

    # Individual organs as coloured meshes (OBJ in mm LPS coords → scale 0.001, euler 90° X)
    organ_dir = os.path.join(local_dir, "Props", "ABDPhantom", "Organs")
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
    print(f"Loaded {sum(1 for n in _ORGAN_COLORS if os.path.exists(os.path.join(organ_dir, n)))} organs")

    probe_usd_path = os.path.join(local_dir, "Props", "ClariusUltrasoundProbe", "fixture_nomtl.usda")
    probe = scene.add_entity(
        material=gs.materials.Kinematic(),
        morph=gs.morphs.USD(file=probe_usd_path, scale=100, collision=True),
        surface=gs.surfaces.Default(color=(0.753, 0.753, 1.0)),
    )
    print("Clarius probe USD loaded")

    target = scene.add_entity(
        gs.morphs.Mesh(file="meshes/axis.obj", scale=0.15, collision=False),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    scene.build()

    # --- Robot control --------------------------------------------------------
    n_dofs     = robot.n_dofs
    motors_dof = np.arange(n_dofs)
    ee_link    = robot.get_link("attachment")
    probe_link = robot.get_link("probe")

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

    scan_depth_mm = [US_SCAN_DEPTH_MM]  # mutable so closures can modify it
    ddepth = 10.0  # mm per keypress

    probe_freq_mhz = [_CFG["probe"]["frequency"]]
    dfreq = 0.5  # MHz per keypress

    def change_freq(delta):
        probe_freq_mhz[0] = max(1.0, min(20.0, probe_freq_mhz[0] + delta))
        rs_probe_template.set_frequency(probe_freq_mhz[0])
        print(f"\rUS freq: {probe_freq_mhz[0]:.1f} MHz    ", end="", flush=True)

    def move_probe(local_delta):
        target_pos[:] += gu.quat_to_R(target_quat) @ np.array(local_delta, dtype=gs.np_float)

    def rotate_probe(local_axis, delta):
        target_quat[:] = gu.transform_quat_by_quat(
            target_quat, gu.xyz_to_quat(np.array(local_axis, dtype=gs.np_float) * delta)
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
        Keybind("probe_sweep_forward", Key.UP,           KeyAction.HOLD,    callback=move_probe,   args=(PROBE_SWEEP_AXIS * dpos,)),
        Keybind("probe_sweep_back",    Key.DOWN,         KeyAction.HOLD,    callback=move_probe,   args=(-PROBE_SWEEP_AXIS * dpos,)),
        Keybind("probe_slide_left",    Key.LEFT,         KeyAction.HOLD,    callback=move_probe,   args=(-PROBE_SLIDE_AXIS * dpos,)),
        Keybind("probe_slide_right",   Key.RIGHT,        KeyAction.HOLD,    callback=move_probe,   args=(PROBE_SLIDE_AXIS * dpos,)),
        Keybind("probe_release",       Key.N,            KeyAction.HOLD,    callback=move_probe,   args=(-PROBE_PRESS_AXIS * dpos,)),
        Keybind("probe_press",         Key.M,            KeyAction.HOLD,    callback=move_probe,   args=(PROBE_PRESS_AXIS * dpos,)),
        Keybind("probe_rotate_ccw",    Key.J,            KeyAction.HOLD,    callback=rotate_probe, args=(PROBE_PRESS_AXIS, drot)),
        Keybind("probe_rotate_cw",     Key.K,            KeyAction.HOLD,    callback=rotate_probe, args=(PROBE_PRESS_AXIS, -drot)),
        Keybind("probe_rock_left",     Key.Q,            KeyAction.HOLD,    callback=rotate_probe, args=(PROBE_SWEEP_AXIS, drot)),
        Keybind("probe_rock_right",    Key.E,            KeyAction.HOLD,    callback=rotate_probe, args=(PROBE_SWEEP_AXIS, -drot)),
        Keybind("probe_tilt_up",       Key.T,            KeyAction.HOLD,    callback=rotate_probe, args=(PROBE_SLIDE_AXIS, drot)),
        Keybind("probe_tilt_down",     Key.G,            KeyAction.HOLD,    callback=rotate_probe, args=(PROBE_SLIDE_AXIS, -drot)),
        Keybind("depth_inc",     Key.BRACKETRIGHT, KeyAction.RELEASE, callback=change_depth, args=(ddepth,)),
        Keybind("depth_dec",     Key.BRACKETLEFT,  KeyAction.RELEASE, callback=change_depth, args=(-ddepth,)),
        Keybind("freq_inc", Key.PERIOD, KeyAction.RELEASE, callback=change_freq, args=(dfreq,)),
        Keybind("freq_dec", Key.COMMA,  KeyAction.RELEASE, callback=change_freq, args=(-dfreq,)),
        Keybind("reset",  Key.BACKSLASH, KeyAction.RELEASE, callback=reset_robot),
        Keybind("quit",   Key.ESCAPE,   KeyAction.RELEASE, callback=stop),
    )

    # --- Simulation loop ------------------------------------------------------
    _us_step = 0
    US_EVERY  = 10  # run raysim every N physics steps

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

            probe.set_pos(probe_body_m)
            probe.set_quat(probe_face_q)

            _us_step += 1
            if _us_step % US_EVERY != 0:
                continue

            # OBJs are in raw LPS mm; Genesis loads them with _ORGAN_EULER_DEG applied.
            # Transform probe from Genesis frame into the OBJ frame (inverse rotation).
            R_gen2rs = _euler_deg_to_R(*_ORGAN_EULER_DEG).T
            d      = (probe_face_m - RS_ORIGIN_M) * 1000.0
            pos_mm = (R_gen2rs @ d).astype(np.float32)

            euler_xyz = _quat_wxyz_to_euler_xyz(gu.R_to_quat(R_gen2rs @ R_probe))

            rs_probe_template.set_pose(rs.Pose(
                position=pos_mm.astype(np.float32),
                rotation=euler_xyz,
            ))

            try:
                bmode = rs_simulator.simulate(rs_probe_template, rs_sim_params)
                # bmode is in dB (negative values); normalise to [0,255] for display
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
