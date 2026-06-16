import argparse
import os
import sys
import tomllib
import cv2
import numpy as np
import open3d as o3d
import genesis as gs
import genesis.utils.geom as gu
from i4h_asset_helper.assets import get_i4h_local_asset_path

_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "raysim_isaac.toml")

with open(_CONFIG_PATH, "rb") as _f:
    _CFG = tomllib.load(_f)

_ORGAN_COLORS = {
    # "Liver.obj":        (0.50, 0.08, 0.08),
    # "Kidney.obj":       (0.75, 0.18, 0.12),
    # "Gallbladder.obj":  (0.40, 0.62, 0.10),
    # "Pancreas.obj":     (0.90, 0.60, 0.55),
    # "Colon.obj":        (0.80, 0.50, 0.40),
    # "Small_bowel.obj":  (0.88, 0.68, 0.58),
    # "Stomach.obj":      (0.82, 0.55, 0.50),
    # "Heart.obj":        (0.85, 0.08, 0.08),
    # "Bone.obj":         (0.92, 0.88, 0.78),
    # "Back_muscles.obj": (0.60, 0.12, 0.12),
    # "Spleen.obj":       (0.45, 0.08, 0.18),
    # "Vessels.obj":      (0.55, 0.03, 0.08),
    # "Tumor1.obj":       (0.85, 0.82, 0.10),
    # "Tumor2.obj":       (0.85, 0.82, 0.10),
    # "Lungs.obj":        (0.90, 0.70, 0.65),
    "Skin.obj":         (0.88, 0.72, 0.58),
}

_ORGAN_EULER_DEG  = tuple(_CFG["world"]["organ_euler_deg"])
PHANTOM_POS_M = np.array(_CFG["world"]["phantom_pos"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Show camera view while sweeping")
    parser.add_argument("--cam-offset", nargs=3, type=float, default=[-0.09, 0.0, 0.06], help="XYZ offset of camera from phantom center (meters)")
    args = parser.parse_args()

    cam_offset = np.array(args.cam_offset, dtype=np.float32)

    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            enable_gui=False,
            camera_pos=(1.5, 1.0, 1.0),
            camera_lookat=(0.4, 0.0, 0.2),
        )
    )

    plane = scene.add_entity(gs.morphs.Plane())

    local_dir = get_i4h_local_asset_path()
    organ_dir = os.path.join(local_dir, "Props", "ABDPhantom", "Organs")

    # 1. Load organs
    print("Loading organs...")
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

    # Dummy entity to attach camera to
    # Dummy entity for the camera – made larger and colored for visibility
    cam_node = scene.add_entity(
        morph=gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0, 0, 0), collision=False),
        surface=gs.surfaces.Default(color=(1, 0, 0, 1)),
        material=gs.materials.Rigid(gravity_compensation=1.0),
    )

    res_w, res_h = 640, 480
    fov_h, fov_v = 87.0, 58.0
    camera = scene.add_sensor(
        gs.sensors.DepthCamera(
            pattern=gs.sensors.DepthCameraPattern(
                res=(res_w, res_h),
                fov_horizontal=fov_h,
                fov_vertical=fov_v,
            ),
            entity_idx=cam_node.idx,
            max_range=2.0,
            min_range=0.05,
            return_world_frame=True,
        )
    )

    scene.build()

    print("Generating point cloud...")
    
    fx = (res_w / 2.0) / np.tan(np.radians(fov_h) / 2.0)
    fy = (res_h / 2.0) / np.tan(np.radians(fov_v) / 2.0)
    cx, cy = res_w / 2.0, res_h / 2.0
    
    points_all = []

    # Place camera at a static offset from phantom center (default mimics wrist placement)
    pos = PHANTOM_POS_M + cam_offset
    
    # Compute orientation to look at phantom center
    forward = PHANTOM_POS_M - pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0, 0, 1])
    if np.abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([1, 0, 0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    R = np.column_stack((right, -up, forward))
    quat = gu.R_to_quat(R)
    cam_node.set_qpos(np.concatenate([pos, quat]))
    scene.step()
    # Debug print of camera pose (position and quaternion)
    if args.debug:
        print(f"Camera position: {pos}, quaternion: {quat}")
    scene.step()  # settle
    
    depth = camera.read_image().cpu().numpy()
    
    if args.debug:
        # Same CLAHE depth visualization logic
        valid_vis = np.isfinite(depth) & (depth < 1.99)
        if valid_vis.any():
            p2, p98 = np.percentile(depth[valid_vis], [2, 98])
            norm = np.clip((depth - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
            depth_8u = (norm * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            depth_enhanced = clahe.apply(depth_8u)
            depth_enhanced[~valid_vis] = 0
            depth_vis = cv2.applyColorMap(depth_enhanced, cv2.COLORMAP_TURBO)
            cv2.imshow("Debug Camera View", depth_vis)
            cv2.waitKey(50)  # Pause to let user see
    
    # Unproject
    u, v = np.meshgrid(np.arange(res_w), np.arange(res_h))
    valid = (depth > 0.05) & (depth < 1.9)
    z_cam = depth[valid]
    x_cam = (u[valid] - cx) * z_cam / fx
    y_cam = (v[valid] - cy) * z_cam / fy
    pts_cam = np.stack((x_cam, y_cam, z_cam), axis=-1)
    pts_world = pts_cam @ R.T + pos
    points_all.append(pts_world)

    if args.debug:
        cv2.destroyAllWindows()

    pts_concat = np.concatenate(points_all, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_concat)
    
    print(f"Captured {len(pcd.points)} points (static placement).")
    
    # 2. Crop to phantom boundaries
    print("Cropping...")
    bbox_min = PHANTOM_POS_M - np.array([0.2, 0.2, 0.05]) # Cropping slightly above ground
    bbox_max = PHANTOM_POS_M + np.array([0.2, 0.2, 0.2])
    bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
    pcd = pcd.crop(bbox)
    
    # Downsample for speed and uniformity
    pcd = pcd.voxel_down_sample(voxel_size=0.002)

    # 3. Compute normals
    print("Computing normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # Coloring points using normal directions to fulfill "colored point cloud"
    pcd.colors = pcd.normals
    
    # 4. Poisson surface reconstruction
    print("Poisson reconstruction...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # 5. Density map and 6. Prune
    print("Pruning low density...")
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Save
    out_pcd = "phantom_cloud.ply"
    out_mesh = "phantom_mesh.obj"
    o3d.io.write_point_cloud(out_pcd, pcd)
    o3d.io.write_triangle_mesh(out_mesh, mesh)
    
    print(f"Saved point cloud to {out_pcd}")
    print(f"Saved mesh to {out_mesh}")
    
    # Optional Visualization
    print("Visualizing result...")
    o3d.visualization.draw_geometries([mesh], window_name="Reconstructed Phantom")

if __name__ == "__main__":
    main()
