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
  - Phantom replaced with a calibration cuboid (no organ meshes)
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

_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "raysim.toml")

with open(_CONFIG_PATH, "rb") as _f:
    _CFG = tomllib.load(_f)

# raysim is built inside the ultrasound-raytracing repo; add it to the path
_RAYSIM_ROOT = os.path.join(_REPO_ROOT, "i4h-sensor-simulation", "ultrasound-raytracing")
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

# linear.usd origin (z=0 in STL) is the transducer face — no offset needed
PROBE_TCP_OFFSET_M = np.array([0.0, 0.032, 0.0])

_CUBE_OBJ_PATH  = "/tmp/raysim_cube.obj"
_CUBE_HALF_X_MM = 50.0   # half-extents of calibration cuboid in raysim mm
_CUBE_HALF_Y_MM = 50.0
_CUBE_HALF_Z_MM = 1.0    # 2 mm total height

US_SCAN_DEPTH_MM = _CFG["sim"]["t_far"]

# Remaps Genesis world axes → raysim world axes (applied to position vector and orientation)
_FRAME_EULER_DEG  = (0, 0, 0)
# Extra rotation on the probe scan direction in raysim space (does not affect position)
_ORIENT_EULER_DEG = (-90, 0, 90)


def _euler_deg_to_R(ex, ey, ez):
    rx, ry, rz = np.radians(ex), np.radians(ey), np.radians(ez)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx


def _write_cube_obj(path, hx, hy, hz):
    """Write a cuboid OBJ with per-face normals (required by raysim)."""
    lines = [
        "# calibration cuboid\n",
        f"v -{hx} -{hy} -{hz}\n", f"v  {hx} -{hy} -{hz}\n",
        f"v  {hx}  {hy} -{hz}\n", f"v -{hx}  {hy} -{hz}\n",
        f"v -{hx} -{hy}  {hz}\n", f"v  {hx} -{hy}  {hz}\n",
        f"v  {hx}  {hy}  {hz}\n", f"v -{hx}  {hy}  {hz}\n",
        "vn  0  0 -1\n", "vn  0  0  1\n",
        "vn  0 -1  0\n", "vn  0  1  0\n",
        "vn -1  0  0\n", "vn  1  0  0\n",
        "f 1//1 3//1 2//1\n", "f 1//1 4//1 3//1\n",
        "f 5//2 6//2 7//2\n", "f 5//2 7//2 8//2\n",
        "f 1//3 2//3 6//3\n", "f 1//3 6//3 5//3\n",
        "f 4//4 7//4 3//4\n", "f 4//4 8//4 7//4\n",
        "f 1//5 5//5 8//5\n", "f 1//5 8//5 4//5\n",
        "f 2//6 3//6 7//6\n", "f 2//6 7//6 6//6\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)


def build_raysim_cube():
    _write_cube_obj(_CUBE_OBJ_PATH, _CUBE_HALF_X_MM, _CUBE_HALF_Y_MM, _CUBE_HALF_Z_MM)
    materials = rs.Materials()
    world = rs.World("water")
    mat_idx = materials.get_index("muscle")
    world.add(rs.Mesh(_CUBE_OBJ_PATH, mat_idx))
    print(f"raysim cuboid: {_CUBE_HALF_X_MM*2:.0f}x{_CUBE_HALF_Y_MM*2:.0f}x{_CUBE_HALF_Z_MM*2:.0f}mm at raysim origin")
    return world, materials


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    robot_usd_path = os.path.join(_REPO_ROOT, "assets", "data", "Robots", "Franka", "fr3_US.usd")

    CUBE_HALF_M = np.array([_CUBE_HALF_X_MM, _CUBE_HALF_Y_MM, _CUBE_HALF_Z_MM]) / 1000.0
    CUBE_POS_M  = np.array([0.5, 0.0, 0.30])   # cuboid centre in Genesis metres
    RS_ORIGIN_M = CUBE_POS_M                    # raysim (0,0,0) maps to this point

    # --- Genesis scene --------------------------------------------------------
    import taichi as ti
    ti.init(arch=ti.cpu if args.cpu else ti.cuda, device_memory_GB=2)

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    # --- raysim setup ---------------------------------------------------------
    rs_world, rs_materials = build_raysim_cube()
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
            camera_pos=(1.5, 1.0, 1.0),
            camera_lookat=(0.4, 0.0, 0.2),
            camera_fov=50,
            max_FPS=60,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

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

    # Calibration cuboid — fixed, raysim cube is centred at RS_ORIGIN_M
    scene.add_entity(
        gs.morphs.Box(
            size=tuple(CUBE_HALF_M * 2),
            pos=CUBE_POS_M,
            fixed=True,
        )
    )
    print(f"Calibration cuboid: {CUBE_HALF_M*2*1000} mm at {CUBE_POS_M}")

    target = scene.add_entity(
        gs.morphs.Mesh(file="meshes/axis.obj", scale=0.15, collision=False),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    scene.build()

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
        Keybind("move_forward",  Key.UP,           KeyAction.HOLD,    callback=move,         args=((-dpos, 0, 0),)),
        Keybind("move_back",     Key.DOWN,          KeyAction.HOLD,    callback=move,         args=((dpos, 0, 0),)),
        Keybind("move_left",     Key.LEFT,          KeyAction.HOLD,    callback=move,         args=((0, -dpos, 0),)),
        Keybind("move_right",    Key.RIGHT,         KeyAction.HOLD,    callback=move,         args=((0, dpos, 0),)),
        Keybind("move_up",       Key.N,             KeyAction.HOLD,    callback=move,         args=((0, 0, dpos),)),
        Keybind("move_down",     Key.M,             KeyAction.HOLD,    callback=move,         args=((0, 0, -dpos),)),
        Keybind("rotate_ccw",    Key.J,             KeyAction.HOLD,    callback=rotate,       args=(drot,)),
        Keybind("rotate_cw",     Key.K,             KeyAction.HOLD,    callback=rotate,       args=(-drot,)),
        Keybind("depth_inc",     Key.BRACKETRIGHT,  KeyAction.RELEASE, callback=change_depth, args=(ddepth,)),
        Keybind("depth_dec",     Key.BRACKETLEFT,   KeyAction.RELEASE, callback=change_depth, args=(-ddepth,)),
        Keybind("freq_inc",      Key.PERIOD,        KeyAction.RELEASE, callback=change_freq,  args=(dfreq,)),
        Keybind("freq_dec",      Key.COMMA,         KeyAction.RELEASE, callback=change_freq,  args=(-dfreq,)),
        Keybind("reset",         Key.BACKSLASH,     KeyAction.RELEASE, callback=reset_robot),
        Keybind("quit",          Key.ESCAPE,        KeyAction.RELEASE, callback=stop),
    )

    # --- Simulation loop ------------------------------------------------------
    _us_step = 0
    US_EVERY  = 10

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
            if _us_step % US_EVERY != 0:
                continue

            R_frame   = _euler_deg_to_R(*_FRAME_EULER_DEG)
            R_orient  = _euler_deg_to_R(*_ORIENT_EULER_DEG)
            d         = (probe_face_m - RS_ORIGIN_M) * 1000.0
            pos_mm    = (R_frame @ d).astype(np.float32)
            R_rs      = R_frame @ R_probe @ R_orient
            euler_xyz = _quat_wxyz_to_euler_xyz(gu.R_to_quat(R_rs))

            rs_probe_template.set_pose(rs.Pose(
                position=pos_mm,
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
