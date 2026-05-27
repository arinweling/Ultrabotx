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
space (hold) - Close gripper
\\ - Reset robot pose
esc - Quit

Plus default viewer controls (press 'i' in viewer to see them)
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np
import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind
from i4h_asset_helper.assets import get_i4h_local_asset_path

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
    """Convert wxyz quaternion to XYZ Euler angles (radians)."""
    R = gu.quat_to_R(quat_wxyz)
    ry = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    rx = np.arctan2(R[2, 1] / np.cos(ry), R[2, 2] / np.cos(ry))
    rz = np.arctan2(R[1, 0] / np.cos(ry), R[0, 0] / np.cos(ry))
    return np.array([rx, ry, rz], dtype=np.float32)


# --- raysim world setup ------------------------------------------------------

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
}


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
    phantom_path = os.path.join(local_dir, "Props", "ABDPhantom", "phantom.usda")
    if not os.path.exists(phantom_path):
        raise FileNotFoundError(f"phantom.usda not found at {phantom_path}. Run download_phantom.py first.")

    PHANTOM_ORIGIN_M = np.array([0.5, 0.0, 0.0])

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

    rs_sim_params = rs.SimParams()
    rs_sim_params.conv_psf    = True
    rs_sim_params.buffer_size = 4096
    rs_sim_params.t_far       = 180.0
    rs_sim_params.b_mode_size = (400, 400)
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
        morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    # Phantom fixed — raysim meshes are in OBJ mm coords relative to this origin
    phantom_entities = scene.add_stage(
        gs.morphs.USD(file=phantom_path, pos=(0.5, 0.0, 0.0), fixed=True)
    )
    print(f"Loaded phantom: {len(phantom_entities)} entity/entities")

    # Ultrasound probe — box proxy rigidly attached to end-effector.
    # Dimensions approximate a Clarius handheld probe (6 cm face, 3 cm wide, 2 cm thick).
    probe = scene.add_entity(
        material=gs.materials.Kinematic(),
        morph=gs.morphs.Box(size=(0.03, 0.06, 0.02)),
        surface=gs.surfaces.Default(color=(0.2, 0.6, 1.0, 1.0)),
    )
    print("Probe box proxy created")

    target = scene.add_entity(
        gs.morphs.Mesh(file="meshes/axis.obj", scale=0.15, collision=False),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    scene.build()

    # --- Robot control --------------------------------------------------------
    n_dofs      = robot.n_dofs
    motors_dof  = np.arange(n_dofs - 2)
    fingers_dof = np.arange(n_dofs - 2, n_dofs)
    ee_link     = robot.get_link("hand")

    robot_init_pos  = np.array([0.5, 0.0, 0.40])
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
        robot.set_qpos(q[:-2], motors_dof)

    reset_robot()

    def move(delta):
        target_pos[:] += np.array(delta, dtype=gs.np_float)

    def rotate(delta):
        target_quat[:] = gu.transform_quat_by_quat(
            target_quat, gu.xyz_to_quat(np.array([0, 0, delta]))
        )

    def toggle_gripper(close=True):
        force = -1.0 if close else 1.0
        robot.control_dofs_force(np.array([force, force]), fingers_dof)

    is_running = True

    def stop():
        global is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("move_forward",  Key.UP,        KeyAction.HOLD,    callback=move,           args=((-dpos, 0, 0),)),
        Keybind("move_back",     Key.DOWN,       KeyAction.HOLD,    callback=move,           args=((dpos, 0, 0),)),
        Keybind("move_left",     Key.LEFT,       KeyAction.HOLD,    callback=move,           args=((0, -dpos, 0),)),
        Keybind("move_right",    Key.RIGHT,      KeyAction.HOLD,    callback=move,           args=((0, dpos, 0),)),
        Keybind("move_up",       Key.N,          KeyAction.HOLD,    callback=move,           args=((0, 0, dpos),)),
        Keybind("move_down",     Key.M,          KeyAction.HOLD,    callback=move,           args=((0, 0, -dpos),)),
        Keybind("rotate_ccw",    Key.J,          KeyAction.HOLD,    callback=rotate,         args=(drot,)),
        Keybind("rotate_cw",     Key.K,          KeyAction.HOLD,    callback=rotate,         args=(-drot,)),
        Keybind("reset",         Key.BACKSLASH,  KeyAction.RELEASE, callback=reset_robot),
        Keybind("close_gripper", Key.SPACE,      KeyAction.PRESS,   callback=toggle_gripper, args=(True,)),
        Keybind("open_gripper",  Key.SPACE,      KeyAction.RELEASE, callback=toggle_gripper, args=(False,)),
        Keybind("quit",          Key.ESCAPE,     KeyAction.RELEASE, callback=stop),
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
            robot.control_dofs_position(q[:-2], motors_dof)
            scene.step()

            hand_pos  = ee_link.get_pos().cpu().numpy()
            hand_quat = ee_link.get_quat().cpu().numpy()
            approach  = gu.quat_to_R(hand_quat)[:, 2]

            probe_face_m = hand_pos + approach * 0.10
            probe.set_pos(probe_face_m)
            probe.set_quat(hand_quat)

            _us_step += 1
            if _us_step % US_EVERY != 0:
                continue

            # Probe pose in raysim frame: mm, relative to phantom OBJ origin
            pos_mm    = (probe_face_m - PHANTOM_ORIGIN_M) * 1000.0
            euler_xyz = _quat_wxyz_to_euler_xyz(hand_quat)

            rs_probe = rs.CurvilinearProbe(
                rs.Pose(
                    position=pos_mm.astype(np.float32),
                    rotation=euler_xyz,
                )
            )

            try:
                bmode = rs_simulator.simulate(rs_probe, rs_sim_params)
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
