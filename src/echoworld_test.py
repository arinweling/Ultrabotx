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
a  - Toggle EchoWorld auto-guidance ON/OFF
\\ - Reset robot pose
esc - Quit

Plus default viewer controls (press 'i' in viewer to see them)

Differences from test.py:
  - Robot loaded from i4h panda_assembly.usda instead of local MJCF
  - Phantom/probe assets remain the same (already from i4h asset helper)
  - EchoWorld model can optionally auto-guide the probe (--echoworld-ckpt)
"""

import argparse
import os
import sys
import tomllib
from collections import deque

import cv2
import numpy as np
import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind
from i4h_asset_helper.assets import (
    _get_s3_client, _S3_BUCKETS, _get_asset_env,
    get_i4h_local_asset_path, get_i4h_asset_hash, get_i4h_asset_version,
)

_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH  = os.path.join(_REPO_ROOT, "config", "raysim_isaac.toml")
_EW_CFG_PATH  = os.path.join(_REPO_ROOT, "config", "echoworld.toml")

with open(_CONFIG_PATH, "rb") as _f:
    _CFG = tomllib.load(_f)

with open(_EW_CFG_PATH, "rb") as _f:
    _EW_CFG = tomllib.load(_f)

# raysim is built inside the ultrasound-raytracing repo; add it to the path
_RAYSIM_ROOT = os.path.join(_REPO_ROOT, "i4h-sensor-simulation", "ultrasound-raytracing")
if _RAYSIM_ROOT not in sys.path:
    sys.path.insert(0, _RAYSIM_ROOT)

# EchoWorld finetune code
_ECHOWORLD_ROOT = os.path.join(_REPO_ROOT, "EchoWorld", "finetune")
if _ECHOWORLD_ROOT not in sys.path:
    sys.path.insert(0, _ECHOWORLD_ROOT)

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


def _euler_deg_to_R(ex, ey, ez):
    rx, ry, rz = np.radians(ex), np.radians(ey), np.radians(ez)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx


# --- raysim world setup ------------------------------------------------------

PROBE_TCP_OFFSET_M = np.array(_CFG["sensor"]["tcp_offset_m"])

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


# --- EchoWorld guidance -------------------------------------------------------

class EchoWorldGuide:
    """Wraps the EchoWorld ViT model for online probe guidance inference.

    Maintains a sliding window of the last `num_frames` B-mode images and
    probe poses (raysim hexa format).  Each call to `infer()` returns the
    predicted 6-DoF delta (mm + deg) for the chosen target standard plane.
    `hexa_to_genesis()` converts that delta to a Genesis-world position and
    rotation correction that can be applied to target_pos / target_quat.
    """

    IMG_MEAN    = [0.193, 0.193, 0.193]
    IMG_STD     = [0.224, 0.224, 0.224]
    LABEL_SCALE = 200.0
    IMG_SIZE    = 224

    def __init__(self, ckpt_path, num_frames=5, target_plane=0,
                 action_scale=0.02, R_organ_inv=None):
        """
        Args:
            ckpt_path:    Path to EchoWorld checkpoint (keys: "encoder", "predictor").
            num_frames:   Temporal context window size.
            target_plane: Which of the 10 standard cardiac planes to target (0-9).
            action_scale: Scales predicted delta before applying to robot.
            R_organ_inv:  3×3 matrix that maps raysim frame → Genesis world.
                          Equals _euler_deg_to_R(*_ORGAN_EULER_DEG) in this sim.
        """
        import torch
        from torchvision import transforms as TVT

        self.torch        = torch
        self.num_frames   = num_frames
        self.target_plane = target_plane
        self.action_scale = action_scale
        self.R_organ_inv  = R_organ_inv if R_organ_inv is not None else np.eye(3)
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fan_mask  = self._build_fan_mask()
        self.transform = TVT.Compose([
            TVT.ToTensor(),
            TVT.Normalize(self.IMG_MEAN, self.IMG_STD),
        ])

        # ring buffer: each entry is (img_tensor [3,224,224], hexa np[6])
        self.buffer: deque = deque(maxlen=num_frames)

        self.model = self._load_model(ckpt_path, num_frames)
        print(f"[EchoWorld] model ready — target plane {target_plane}, "
              f"action_scale {action_scale}, device {self.device}")

    # ------------------------------------------------------------------

    def _build_fan_mask(self):
        size = self.IMG_SIZE
        mask = np.zeros((size, size), dtype=np.uint8)
        r = size / 256.0
        cv2.ellipse(
            mask,
            center=(int(128 * r), int(9 * r)),
            axes=(int(237 * r), int(237 * r)),
            angle=0, startAngle=45, endAngle=135,
            color=255, thickness=-1,
        )
        return mask

    def _load_model(self, ckpt_path, num_frames):
        from models.jhj_models import ViT_Cardiac_Seq_Model
        model = ViT_Cardiac_Seq_Model(
            model_name="vit_small",
            timestep=num_frames,
            modelpath=ckpt_path,
            pred_depth=4,
            pred_emb_dim=192,
            pred_num_heads=4,
            pred_mlp_ratio=2,
        )
        model.eval()
        return model.to(self.device)

    def _preprocess(self, bmode_uint8: np.ndarray) -> "torch.Tensor":
        """uint8 grayscale B-mode → normalised float tensor [3, 224, 224]."""
        from PIL import Image
        img = cv2.resize(bmode_uint8, (self.IMG_SIZE, self.IMG_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        img = cv2.bitwise_and(img, self.fan_mask)
        img_rgb = np.stack([img, img, img], axis=-1)
        return self.transform(Image.fromarray(img_rgb))

    # ------------------------------------------------------------------

    def push(self, bmode_uint8: np.ndarray, pos_mm: np.ndarray, euler_xyz_rad: np.ndarray):
        """Add a US frame to the sliding window.

        Args:
            bmode_uint8:   Grayscale B-mode image [H, W] uint8.
            pos_mm:        Probe face position in raysim frame (mm) [3].
            euler_xyz_rad: Probe orientation in raysim frame (radians) [3].
        """
        img_t = self._preprocess(bmode_uint8)
        hexa  = np.array([
            pos_mm[0], pos_mm[1], pos_mm[2],
            np.degrees(euler_xyz_rad[0]),
            np.degrees(euler_xyz_rad[1]),
            np.degrees(euler_xyz_rad[2]),
        ], dtype=np.float64)
        self.buffer.append((img_t, hexa))

    def infer(self):
        """Run EchoWorld inference.

        Returns:
            np.ndarray [6] — predicted hexa delta (mm + deg in raysim frame)
            for the target plane, or None if the buffer is not yet full.
        """
        from autous.robot.transformation import Transformation

        if len(self.buffer) < self.num_frames:
            return None

        frames = list(self.buffer)

        # [1, T, 3, 224, 224]
        imgs = self.torch.stack([f[0] for f in frames], dim=0).unsqueeze(0)

        # Compute T-1 inter-frame actions via hexa_diff_inv, then scale
        hexas   = [f[1] for f in frames]
        actions = [
            Transformation.hexa_diff_inv(hexas[i], hexas[i + 1])
            for i in range(len(hexas) - 1)
        ]
        acts = (
            self.torch.tensor(np.stack(actions), dtype=self.torch.float32)
            .unsqueeze(0)            # [1, T-1, 6]
            / self.LABEL_SCALE
        )

        imgs = imgs.to(self.device)
        acts = acts.to(self.device)

        with self.torch.no_grad():
            pred = self.model(imgs, acts)  # [1, 60]

        pred = pred[0].cpu().numpy()       # [60]
        p    = self.target_plane
        return pred[p * 6 : p * 6 + 6]    # [6] = (tx_mm, ty_mm, tz_mm, rx_deg, ry_deg, rz_deg)

    def hexa_to_genesis(self, delta_hexa: np.ndarray):
        """Convert a raysim-frame hexa delta to Genesis-world dpos and dR.

        The translation is scaled by action_scale and mapped back through the
        organ frame transform.  The rotation delta is similarly remapped and
        scaled via linear interpolation then re-orthogonalised.

        Returns:
            dpos_m:    Position delta in Genesis world metres [3].
            dR_world:  3×3 rotation delta matrix (apply as new_R = dR @ old_R).
        """
        # Translation: raysim mm → Genesis world metres, scaled
        dpos_rs = delta_hexa[:3] / 1000.0            # raysim metres
        dpos_m  = self.R_organ_inv @ dpos_rs * self.action_scale

        # Rotation delta in raysim frame
        rx, ry, rz = np.radians(delta_hexa[3:])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx),  np.cos(rx)]])
        Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                       [0,           1, 0          ],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz),  np.cos(rz), 0],
                       [0,           0,           1]])
        dR_rs = Rz @ Ry @ Rx   # XYZ intrinsic

        # Map to Genesis world frame: R_organ_inv @ dR_rs @ R_organ
        R_organ   = self.R_organ_inv.T
        dR_world  = self.R_organ_inv @ dR_rs @ R_organ

        # Scale the rotation by interpolation toward identity, then re-orthogonalise
        dR_scaled = (1.0 - self.action_scale) * np.eye(3) + self.action_scale * dR_world
        U, _, Vt  = np.linalg.svd(dR_scaled)
        dR_final  = U @ Vt

        return dpos_m, dR_final


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("--cpu",       action="store_true", default=False)
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
                collision=False,
            ),
            surface=gs.surfaces.Default(color=color),
        )
    print(f"Loaded {sum(1 for n in _ORGAN_COLORS if os.path.exists(os.path.join(organ_dir, n)))} organs")

    target = scene.add_entity(
        gs.morphs.Mesh(file="meshes/axis.obj", scale=0.15, collision=False),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    scene.build()

    print("Robot links:", [l.name for l in robot.links])

    # --- Robot control --------------------------------------------------------
    n_dofs     = robot.n_dofs
    motors_dof = np.arange(n_dofs)
    ee_link    = robot.get_link("/fr3/fr3_link8")
    probe_link = robot.get_link("/fr3/linear")

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

    # --- EchoWorld guidance ---------------------------------------------------
    _ew_ckpt   = _EW_CFG["model"]["ckpt_path"]
    _ew_frames = _EW_CFG["model"]["num_frames"]
    _ew_plane  = _EW_CFG["model"]["target_plane"]
    _ew_scale  = _EW_CFG["guidance"]["action_scale"]

    guide      = None
    auto_guide = [False]

    if _ew_ckpt:
        # R_organ_inv maps raysim frame → Genesis world
        R_organ_inv = _euler_deg_to_R(*_ORGAN_EULER_DEG)  # = (_euler_deg_to_R(...).T).T
        guide = EchoWorldGuide(
            ckpt_path    = _ew_ckpt,
            num_frames   = _ew_frames,
            target_plane = _ew_plane,
            action_scale = _ew_scale,
            R_organ_inv  = R_organ_inv,
        )

    def toggle_auto():
        if guide is None:
            print("\r[EchoWorld] no model loaded — pass --echoworld-ckpt to enable", end="")
            return
        auto_guide[0] = not auto_guide[0]
        print(f"\r[EchoWorld] auto-guidance {'ON ' if auto_guide[0] else 'OFF'}   ", end="", flush=True)

    # --- Key bindings ---------------------------------------------------------
    scan_depth_mm  = [_CFG["sim"]["t_far"]]
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
        Keybind("toggle_auto",   Key.A,              KeyAction.RELEASE, callback=toggle_auto),
        Keybind("reset",         Key.BACKSLASH,      KeyAction.RELEASE, callback=reset_robot),
        Keybind("quit",          Key.ESCAPE,         KeyAction.RELEASE, callback=stop),
    )

    # --- Simulation loop ------------------------------------------------------
    _us_step = 0
    US_EVERY  = _EW_CFG["guidance"]["us_every"]

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
                bmode   = rs_simulator.simulate(rs_probe_template, rs_sim_params)
                DR      = 60.0
                display = np.clip((bmode + DR) / DR, 0.0, 1.0)
                display = (display * 255).astype(np.uint8)
                cv2.imshow("Ultrasound (raysim)", display)
                cv2.waitKey(1)

                # --- EchoWorld inference ------------------------------------
                if guide is not None:
                    guide.push(display, pos_mm, euler_xyz)
                    delta = guide.infer()
                    if delta is not None:
                        print(
                            f"\r[EchoWorld] plane {_ew_plane}: "
                            f"tx={delta[0]:+.1f}mm  ty={delta[1]:+.1f}mm  tz={delta[2]:+.1f}mm  "
                            f"| auto={'ON' if auto_guide[0] else 'off'}   ",
                            end="", flush=True,
                        )
                        if auto_guide[0]:
                            dpos_m, dR = guide.hexa_to_genesis(delta)
                            target_pos[:] += dpos_m
                            # Apply rotation delta: new_R = dR @ current_R → new_quat
                            R_cur = gu.quat_to_R(target_quat)
                            target_quat[:] = gu.R_to_quat(dR @ R_cur)

            except Exception as e:
                print(f"[US] {e}")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
