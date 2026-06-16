"""
Keyboard Controls:
  ↑ / ↓       Move probe forward / backward (X axis)
  ← / →       Move probe left / right (Y axis)
  n / m       Move probe up / down (Z axis)
  j / k       Rotate probe CCW / CW around Z
  q / e       Tilt probe forward / backward (Y rotation)
  [ / ]       Decrease / increase imaging depth offset (+5 mm)
  \\ / Esc    Reset robot pose / Quit

Genesis + SonoGym integration (no raysim).
--sim net  (default) : USSimulatorNetwork — learned pix2pix CT→US (requires models/)
--sim conv           : USSimulatorConv   — physics-based, no model weights needed
"""

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as ScipyR

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS_DIR   = os.path.join(_REPO_ROOT, "assets", "data")
_SONOGYM_CFGS = os.path.join(
    _REPO_ROOT, "SonoGym", "source", "spinal_surgery",
    "spinal_surgery", "lab", "sensors", "cfgs",
)

# Direct import of both simulators — bypasses SonoGym's package __init__.py
# which pulls in isaaclab/gymnasium.
# simulate_US_network also imports `from spinal_surgery import PROJECT_DIR`, so
# we inject a minimal mock module before the import.
import types as _types
_sg_mock = _types.ModuleType("spinal_surgery")
_sg_mock.PACKAGE_DIR = os.path.join(
    _REPO_ROOT, "SonoGym", "source", "spinal_surgery", "spinal_surgery"
)
_sg_mock.PROJECT_DIR = _REPO_ROOT   # models/ lives at repo root
sys.modules.setdefault("spinal_surgery", _sg_mock)

_US_SIM_DIR = os.path.join(
    _REPO_ROOT, "SonoGym", "source", "spinal_surgery",
    "spinal_surgery", "lab", "sensors", "ultrasound",
)
sys.path.insert(0, _US_SIM_DIR)
from simulate_US_conv    import USSimulatorConv
from simulate_US_network import USSimulatorNetwork

# ── Scene layout (mirrors SonoGym robotic_US_guidance.yaml) ──────────────────
_PATIENT_ID       = "s0010"
_LABEL_RES        = 0.0015          # m per voxel
_PATIENT_POS      = np.array([0.2, -0.45, 1.0])   # world metres
_PATIENT_EULER_YXZ = [-90.0, -90.0, 0.0]          # intrinsic YXZ degrees
_ROBOT_BASE_POS   = np.array([0.0, -0.75, 0.5])
_BED_POS          = np.array([0.0, 0.0, 0.3])
_BED_EULER_XYZ    = [90.0, 0.0, 90.0]

# ── Standalone transform math (replaces isaaclab.utils.math) ─────────────────

def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion(s) → 3×3 rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([1., -1., -1., -1.])


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _subtract_frame_transforms(p1, q1, p2, q2):
    """Pose of frame2 expressed in frame1. Inputs: (N,3)/(N,4) float32 tensors."""
    R1  = _quat_to_mat(q1)
    dp  = (p2 - p1).unsqueeze(-1)
    p_rel = torch.bmm(R1.transpose(-1, -2), dp).squeeze(-1)
    q_rel = _quat_mul(_quat_conj(q1), q2)
    return p_rel, q_rel


def _transform_points(points, pos, quat):
    """points: (P,3), pos: (N,3), quat: (N,4) → (N, P, 3)"""
    R = _quat_to_mat(quat)
    return torch.einsum("nij,pj->npi", R, points) + pos.unsqueeze(1)


# ── Volume slicer ─────────────────────────────────────────────────────────────

class VolumeSlicer:
    """
    Slices a 2D B-mode image from a 3D patient volume.
    mode='net'  — USSimulatorNetwork: CT slice → learned pix2pix → US
    mode='conv' — USSimulatorConv:    label slice → physics model → US
    Standalone — no isaaclab required.
    """
    IMG_SIZE   = (150, 200)   # (W, H) pixels
    IMG_RES    = 0.0005       # m / pixel
    LABEL_RES  = 0.0015       # m / voxel
    HEIGHT_IMG = 0.13         # m — EE-to-skin gap; imaging starts at body surface

    def __init__(self, label_map_np: np.ndarray, us_cfg: dict,
                 label_conv: dict, device: str,
                 ct_map_np: np.ndarray = None, us_net_cfg: dict = None):
        self.device = device
        self.mode   = "net" if (ct_map_np is not None and us_net_cfg is not None) else "conv"

        if self.mode == "net":
            self.vol = torch.tensor(ct_map_np, dtype=torch.float32, device=device)
            self.us_sim = USSimulatorNetwork(us_net_cfg, device=device)
        else:
            # Build label conversion LUT (NIfTI seg labels → acoustic labels)
            lut = torch.zeros(512, dtype=torch.uint8, device=device)
            for src, dst in label_conv.items():
                if 0 <= src < 512:
                    lut[src] = int(dst)
            raw = torch.clamp(
                torch.tensor(label_map_np.astype(np.int32), dtype=torch.long, device=device),
                0, 511,
            )
            self.vol = lut[raw].to(torch.uint8)
            self.us_sim = USSimulatorConv(us_cfg, device=device)

        self.vol_shape = torch.tensor(
            list(label_map_np.shape), dtype=torch.long, device=device
        )

        # Pixel grid in probe-local frame: x=lateral, y=elevation, z=depth
        W, H = self.IMG_SIZE
        x_g, z_g, y_g = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=device) - W // 2,
            torch.arange(H, dtype=torch.float32, device=device),
            torch.zeros(1, dtype=torch.float32, device=device),
            indexing="ij",
        )
        self.img_coords = (
            torch.stack([x_g, y_g, z_g], dim=-1).reshape(-1, 3) * self.IMG_RES
        )  # (W*H, 3) metres

    def slice_us(
        self,
        world_human_pos:  torch.Tensor,   # (1, 3)
        world_human_quat: torch.Tensor,   # (1, 4) wxyz
        world_ee_pos:     torch.Tensor,   # (1, 3)
        world_ee_quat:    torch.Tensor,   # (1, 4) wxyz
    ) -> np.ndarray:
        """Returns (H, W) uint8 B-mode image."""
        h_pos, h_quat = _subtract_frame_transforms(
            world_human_pos, world_human_quat,
            world_ee_pos,    world_ee_quat,
        )
        # Probe Z axis in patient frame — flip to point into body (toward +Y)
        R_ee   = _quat_to_mat(h_quat)
        normal = R_ee[:, :, 2]
        body_y = h_pos.new_tensor([[0., 1., 0.]])
        sign   = torch.sign((normal * body_y).sum(-1, keepdim=True))
        sign[sign == 0] = 1.0
        normal = normal * sign

        img_center = h_pos + self.HEIGHT_IMG * normal

        pts = _transform_points(self.img_coords, img_center, h_quat)   # (1, P, 3)
        vox = (pts / self.LABEL_RES).long()
        vox = torch.clamp(
            vox,
            torch.zeros(3, dtype=torch.long, device=self.device),
            self.vol_shape - 1,
        )
        v = vox[0]   # (P, 3)

        W, H = self.IMG_SIZE
        # Reshape: pixel grid is (W, H) order → transpose to (H, W) for simulators
        raw_slice = self.vol[v[:, 0], v[:, 1], v[:, 2]].reshape(W, H).t()  # (H, W)

        with torch.no_grad():
            if self.mode == "net":
                # Network expects (N, 1, W, H) — note SonoGym's W/H naming
                ct_in = raw_slice.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
                us_t  = self.us_sim.simulate_US_image(ct_in)   # (1, 1, H, W)
                us    = us_t[0, 0].cpu().numpy()
            else:
                label_in = raw_slice.unsqueeze(0).float()      # (1, H, W)
                us_t     = self.us_sim.simulate_US_image(label_in, if_noise=True)
                us       = us_t[0].cpu().numpy()

        lo, hi = us.min(), us.max()
        return ((us - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scipy_quat_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """scipy as_quat() is [x,y,z,w]; Genesis wants [w,x,y,z]."""
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _np_to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genesis + SonoGym US scan")
    parser.add_argument("--patient",  default=_PATIENT_ID,
                        help="Patient ID from assets/data/HumanModels (default: s0010)")
    parser.add_argument("--sim",      default="net", choices=["net", "conv"],
                        help="US simulator: net=learned pix2pix (default), conv=physics")
    parser.add_argument("--no-vis",   action="store_true",
                        help="Disable Genesis viewer (headless)")
    parser.add_argument("--cpu",      action="store_true",
                        help="Use CPU backend")
    parser.add_argument("--us-every", type=int, default=5,
                        help="Render US every N sim steps")
    args = parser.parse_args()

    device_str = "cpu" if args.cpu else "cuda"

    # ── Load SonoGym configs ──────────────────────────────────────────────────
    yaml = YAML()
    us_cfg_raw     = yaml.load(open(os.path.join(_SONOGYM_CFGS, "us_cfg.yaml")))
    label_conv_raw = yaml.load(open(os.path.join(_SONOGYM_CFGS, "label_conversion.yaml")))
    us_net_cfg_raw = yaml.load(open(os.path.join(_SONOGYM_CFGS, "us_generative_cfg.yaml")))

    # ruamel CommentedMap fails on float-key lookup — convert to plain Python dicts
    us_cfg = dict(us_cfg_raw)
    us_cfg["label_to_ac_params_dict"] = {
        int(k): {str(sk): float(sv) for sk, sv in v.items()}
        for k, v in us_cfg_raw["label_to_ac_params_dict"].items()
    }
    us_cfg["system_params"] = {str(k): v for k, v in us_cfg_raw["system_params"].items()}

    label_conv = {int(k): int(v) for k, v in label_conv_raw.items()
                  if not isinstance(v, dict)}

    # Network config: point model paths to repo-local models/ directory
    _models_dir = os.path.join(_REPO_ROOT, "models")
    us_net_cfg = dict(us_net_cfg_raw)
    us_net_cfg["model_path"] = [
        os.path.join(_models_dir, "pix2pix_rand_down_up.pth"),
        os.path.join(_models_dir, "pix2pix_rand_down_up_2.pth"),
        os.path.join(_models_dir, "pix2pix_rand_down_up_3.pth"),
        os.path.join(_models_dir, "pix2pix_rand_down_up_4.pth"),
        os.path.join(_models_dir, "pix2pix_rand_down_up_5.pth"),
    ]
    us_net_cfg["train_data_sample_path"] = os.path.join(_models_dir, "training_samples")
    us_net_cfg["model"] = dict(us_net_cfg_raw["model"])

    # ── Load patient data ─────────────────────────────────────────────────────
    stl_dir  = os.path.join(_ASSETS_DIR, "HumanModels", "selected_dataset_stl", args.patient)
    raw_dir  = os.path.join(_ASSETS_DIR, "HumanModels", "selected_dataset", args.patient)
    usd_dir  = os.path.join(_ASSETS_DIR, "HumanModels", "selected_dataset_body_from_urdf",
                             args.patient, "combined_wrapwrap")

    label_map_path = os.path.join(stl_dir, "combined_label_map.nii.gz")
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Label map not found: {label_map_path}")

    print(f"[data] Loading patient {args.patient} ({args.sim} mode) …")
    label_np = nib.load(label_map_path).get_fdata().astype(np.int32)
    print(f"[data] Label map: {label_np.shape},  labels {int(label_np.min())}–{int(label_np.max())}")

    ct_np = None
    if args.sim == "net":
        ct_path = os.path.join(raw_dir, "ct.nii.gz")
        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"CT not found: {ct_path}")
        ct_np = nib.load(ct_path).get_fdata().astype(np.float32)
        print(f"[data] CT: {ct_np.shape},  HU {ct_np.min():.0f}–{ct_np.max():.0f}")

    # ── Volume slicer + US simulator ──────────────────────────────────────────
    sim_label = "USSimulatorNetwork (pix2pix)" if args.sim == "net" else "USSimulatorConv"
    print(f"[US] Building {sim_label} …")
    slicer = VolumeSlicer(
        label_map_np = label_np,
        us_cfg       = us_cfg,
        label_conv   = label_conv,
        device       = device_str,
        ct_map_np    = ct_np                      if args.sim == "net" else None,
        us_net_cfg   = us_net_cfg                 if args.sim == "net" else None,
    )
    print(f"[US] Ready — mode: {slicer.mode}")

    # ── Patient world pose ────────────────────────────────────────────────────
    patient_pos  = _PATIENT_POS.copy()
    patient_quat = _scipy_quat_to_wxyz(
        ScipyR.from_euler("yxz", _PATIENT_EULER_YXZ, degrees=True).as_quat()
    )  # wxyz for Genesis

    # ── Genesis scene ─────────────────────────────────────────────────────────
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0, 0, -9.8),
            box_box_detection=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.2, -1.5, 1.8),
            camera_lookat=(0.2, -0.45, 1.0),
            camera_fov=50,
            max_FPS=60,
        ),
        show_viewer=not args.no_vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # Hospital bed (visual only)
    bed_stl = os.path.join(_ASSETS_DIR, "MedicalBed", "stl", "hospital_bed.stl")
    if os.path.exists(bed_stl):
        _bed_rot = ScipyR.from_euler("xyz", _BED_EULER_XYZ, degrees=True)
        _bed_quat = _scipy_quat_to_wxyz(_bed_rot.as_quat())
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=bed_stl,
                scale=0.001,
                pos=_BED_POS,
                quat=_bed_quat,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=(0.85, 0.85, 0.85, 1.0)),
        )

    # Patient body — fixed static body, collision enabled so the probe can't sink in
    body_stl = os.path.join(stl_dir, "combined_wrapwrap.stl")
    if os.path.exists(body_stl):
        scene.add_entity(
            morph=gs.morphs.Mesh(
                file=body_stl,
                scale=_LABEL_RES,
                pos=patient_pos,
                quat=patient_quat,
                fixed=True,
                collision=True,
            ),
            surface=gs.surfaces.Default(color=(0.9, 0.75, 0.65, 0.8)),
        )
        print(f"[scene] Patient body loaded: {body_stl}")
    else:
        print(f"[warn] Body STL not found: {body_stl}")

    # Robot (fr3 with US probe attachment)
    robot_usd = os.path.join(_ASSETS_DIR, "Robots", "Franka", "fr3_US.usd")
    robot = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.USD(
            file=robot_usd,
            pos=_ROBOT_BASE_POS,
            requires_jac_and_IK=True,
            recompute_inertia=True,
        ),
    )
    print(f"[scene] Robot loaded: {robot_usd}")

    # EE target marker
    target = scene.add_entity(
        gs.morphs.Mesh(file="meshes/axis.obj", scale=0.12, collision=False),
        surface=gs.surfaces.Default(color=(1.0, 0.4, 0.4, 1.0)),
    )

    scene.build()

    # ── Robot control setup ───────────────────────────────────────────────────
    n_dofs     = robot.n_dofs
    motors_dof = np.arange(n_dofs)
    ee_link    = robot.get_link("/fr3/fr3_link8")

    robot.set_dofs_kp([4500] * n_dofs)
    robot.set_dofs_kv([450]  * n_dofs)
    robot.set_dofs_force_range([-500] * n_dofs, [500] * n_dofs)

    # Compute correct probe start position from surface map.
    # Patient +Y maps to world -Z; probe images downward, orientation Ry(π) ✓.
    # EE is placed HEIGHT=0.13 m above (in patient -Y = world +Z) the body surface.
    _surf_pt = os.path.join(stl_dir, "body_lowest_y_array.pt")
    if os.path.exists(_surf_pt):
        _surf      = torch.load(_surf_pt, map_location="cpu").numpy().astype(float)
        _xc, _zc   = 140, 135   # centre of SonoGym scan range
        _surf_y    = _surf[_xc, _zc]
        _HEIGHT    = 0.13        # m above surface
        _ee_local  = np.array([_xc * _LABEL_RES,
                                _surf_y * _LABEL_RES - _HEIGHT,
                                _zc * _LABEL_RES])
        _R         = ScipyR.from_euler("yxz", _PATIENT_EULER_YXZ, degrees=True).as_matrix()
        robot_init_pos = patient_pos + _R @ _ee_local
        print(f"[probe] surface_y={_surf_y:.0f} vox  EE={robot_init_pos.round(4)}")
    else:
        robot_init_pos = np.array([0.0, -0.24, 1.07])
        print("[probe] surface map not found, using fallback position")
    robot_init_quat = gu.xyz_to_quat(np.array([0.0, np.pi, 0.0]))

    target_pos  = robot_init_pos.copy()
    target_quat = robot_init_quat.copy()

    def reset_robot():
        target_pos[:]  = robot_init_pos
        target_quat[:] = robot_init_quat
        target.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.set_qpos(q, motors_dof)

    reset_robot()

    # ── Keybindings ───────────────────────────────────────────────────────────
    dpos  = 0.003
    drot  = 0.015
    dh    = 0.005   # depth offset step (m)

    def move(delta):
        target_pos[:] += np.array(delta, dtype=gs.np_float)

    def rotate_z(delta):
        target_quat[:] = gu.transform_quat_by_quat(
            target_quat, gu.xyz_to_quat(np.array([0, 0, delta]))
        )

    def tilt_y(delta):
        target_quat[:] = gu.transform_quat_by_quat(
            target_quat, gu.xyz_to_quat(np.array([0, delta, 0]))
        )

    def change_depth(delta):
        slicer.HEIGHT_IMG = max(0.0, slicer.HEIGHT_IMG + delta)
        print(f"\r[US] depth offset: {slicer.HEIGHT_IMG*1000:.0f} mm    ",
              end="", flush=True)

    is_running = [True]

    def stop():
        is_running[0] = False

    scene.viewer.register_keybinds(
        Keybind("fwd",    Key.UP,           KeyAction.HOLD,    callback=move,         args=((-dpos, 0, 0),)),
        Keybind("back",   Key.DOWN,         KeyAction.HOLD,    callback=move,         args=((dpos, 0, 0),)),
        Keybind("left",   Key.LEFT,         KeyAction.HOLD,    callback=move,         args=((0, -dpos, 0),)),
        Keybind("right",  Key.RIGHT,        KeyAction.HOLD,    callback=move,         args=((0, dpos, 0),)),
        Keybind("up",     Key.N,            KeyAction.HOLD,    callback=move,         args=((0, 0, dpos),)),
        Keybind("down",   Key.M,            KeyAction.HOLD,    callback=move,         args=((0, 0, -dpos),)),
        Keybind("rot_ccw",Key.J,            KeyAction.HOLD,    callback=rotate_z,     args=(drot,)),
        Keybind("rot_cw", Key.K,            KeyAction.HOLD,    callback=rotate_z,     args=(-drot,)),
        Keybind("tilt_f", Key.Q,            KeyAction.HOLD,    callback=tilt_y,       args=(drot,)),
        Keybind("tilt_b", Key.E,            KeyAction.HOLD,    callback=tilt_y,       args=(-drot,)),
        Keybind("d_inc",  Key.BRACKETRIGHT, KeyAction.RELEASE, callback=change_depth, args=(dh,)),
        Keybind("d_dec",  Key.BRACKETLEFT,  KeyAction.RELEASE, callback=change_depth, args=(-dh,)),
        Keybind("reset",  Key.BACKSLASH,    KeyAction.RELEASE, callback=reset_robot),
        Keybind("quit",   Key.ESCAPE,       KeyAction.RELEASE, callback=stop),
    )

    # Patient pose as torch tensors (static — computed once)
    t_human_pos  = _np_to_tensor(patient_pos,  device_str)
    t_human_quat = _np_to_tensor(patient_quat, device_str)

    # ── Simulation loop ───────────────────────────────────────────────────────
    step = 0
    try:
        while is_running[0]:
            target.set_qpos(np.concatenate([target_pos, target_quat]))
            q, _ = robot.inverse_kinematics(
                link=ee_link, pos=target_pos, quat=target_quat, return_error=True
            )
            robot.control_dofs_position(q, motors_dof)
            scene.step()
            step += 1

            if step % args.us_every != 0:
                continue

            ee_pos_np  = ee_link.get_pos().cpu().numpy()
            ee_quat_np = ee_link.get_quat().cpu().numpy()   # wxyz

            t_ee_pos  = _np_to_tensor(ee_pos_np,  device_str)
            t_ee_quat = _np_to_tensor(ee_quat_np, device_str)

            try:
                us_img = slicer.slice_us(t_human_pos, t_human_quat,
                                         t_ee_pos,    t_ee_quat)
                # Exact same display as SonoGym: figsize=(2,3), gray cmap, plt.pause
                plt.figure(1, figsize=(2, 3))
                plt.clf()
                plt.imshow(us_img, cmap="gray", vmin=0, vmax=255)
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.pause(0.0001)
            except Exception as exc:
                print(f"\r[US] error: {exc}    ", end="", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
