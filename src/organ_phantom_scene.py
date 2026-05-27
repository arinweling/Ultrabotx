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

import numpy as np
import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind
from i4h_asset_helper.assets import get_i4h_local_asset_path

# OBJ files are in mm; scale 0.001 converts to metres (verified from vertex coords)
ORGAN_SCALE = 0.001


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True,
                        help="Open interactive viewer")
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    local_dir = get_i4h_local_asset_path()
    phantom_path = os.path.join(local_dir, "Props", "ABDPhantom", "phantom.usda")
    organs_base = os.path.join(local_dir, "Props", "ABDPhantom", "Organs")

    if not os.path.exists(phantom_path):
        raise FileNotFoundError(
            f"phantom.usda not found at {phantom_path}. "
            "Run download_phantom.py first."
        )

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
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

    # Franka Panda at origin
    robot = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    # Assembled phantom as a static backdrop to the right of the robot
    phantom_entities = scene.add_stage(
        gs.morphs.USD(file=phantom_path, pos=(0.5, -0.6, 0.0), fixed=False)
    )
    print(f"Loaded phantom: {len(phantom_entities)} entity/entities")

    # Individual organs as dynamic rigid bodies, scattered in front of the robot
    # Positions in metres; scale=0.003 matches phantom.usda's xformOp:scale
    organ_layout = [
        ("Liver.obj",       (0.30, -0.30, 0.5)),
        ("Heart.obj",       (0.30, -0.10, 0.5)),
        ("Kidney.obj",      (0.30,  0.10, 0.5)),
        ("Spleen.obj",      (0.30,  0.30, 0.5)),
        ("Stomach.obj",     (0.45, -0.30, 0.5)),
        ("Gallbladder.obj", (0.45, -0.10, 0.5)),
        ("Pancreas.obj",    (0.45,  0.10, 0.5)),
        ("Tumor1.obj",      (0.45,  0.30, 0.5)),
        ("Colon.obj",       (0.60, -0.20, 0.5)),
        ("Small_bowel.obj", (0.60,  0.00, 0.5)),
        ("Vessels.obj",     (0.60,  0.20, 0.5)),
        ("Bone.obj",        (0.60,  0.40, 0.5)),
        ("Tumor2.obj",      (0.30,  0.50, 0.5)),
        ("Back_muscles.obj",(0.45,  0.50, 0.5)),
    ]

    loaded = []
    for filename, pos in organ_layout:
        obj_path = os.path.join(organs_base, filename)
        if not os.path.exists(obj_path):
            print(f"Skipping {filename} (not downloaded)")
            continue
        scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Mesh(file=obj_path, scale=ORGAN_SCALE, pos=pos),
        )
        loaded.append(filename)

    print(f"Loaded {len(loaded)} organs: {', '.join(o.replace('.obj', '') for o in loaded)}")

    # Visual marker for end-effector target
    target = scene.add_entity(
        gs.morphs.Mesh(file="meshes/axis.obj", scale=0.15, collision=False),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    scene.build()

    # Robot control state
    n_dofs = robot.n_dofs
    motors_dof = np.arange(n_dofs - 2)
    fingers_dof = np.arange(n_dofs - 2, n_dofs)
    ee_link = robot.get_link("hand")

    robot_init_pos = np.array([0.5, 0.0, 0.55])
    robot_init_quat = gu.xyz_to_quat(np.array([0, np.pi, 0]))

    target_pos = robot_init_pos.copy()
    target_quat = robot_init_quat.copy()

    dpos = 0.002
    drot = 0.01

    def reset_robot():
        target_pos[:] = robot_init_pos.copy()
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

    try:
        while is_running:
            target.set_qpos(np.concatenate([target_pos, target_quat]))
            q, _ = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)
            robot.control_dofs_position(q[:-2], motors_dof)
            scene.step()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
