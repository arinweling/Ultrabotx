#!/usr/bin/env python3
"""Extract Clarius HD3C3 probe mesh from USDA and export as simplified OBJ for MuJoCo collision."""

import sys
import numpy as np
import trimesh
from pxr import Usd, UsdGeom

USDA_PATH = "/home/arin/.cache/i4h-assets/132c82d/Props/ClariusUltrasoundProbe/fixture.usda"
OUT_PATH = "/home/arin/Ultrabotx/xml/franka_emika_panda/probe_assets/probe_col.obj"
TARGET_FACES = 2000  # simplified face count for collision mesh


def extract_all_meshes(stage):
    all_verts = []
    all_faces = []
    vert_offset = 0

    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        fc = mesh.GetFaceVertexCountsAttr().Get()
        fi = mesh.GetFaceVertexIndicesAttr().Get()
        if pts is None or fc is None or fi is None:
            continue

        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for p in pts:
            wp = xf.Transform(p)
            all_verts.append((wp[0], wp[1], wp[2]))

        idx = 0
        for count in fc:
            face_verts = [fi[idx + i] + vert_offset for i in range(count)]
            if count == 3:
                all_faces.append(face_verts)
            elif count == 4:
                all_faces.append([face_verts[0], face_verts[1], face_verts[2]])
                all_faces.append([face_verts[0], face_verts[2], face_verts[3]])
            else:
                for i in range(1, count - 1):
                    all_faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])
            idx += count
        vert_offset += len(pts)

    return np.array(all_verts, dtype=np.float32), np.array(all_faces, dtype=np.int32)


def main():
    print(f"Loading {USDA_PATH} ...")
    stage = Usd.Stage.Open(USDA_PATH)

    verts, faces = extract_all_meshes(stage)
    print(f"  Raw mesh: {len(verts)} vertices, {len(faces)} faces")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.merge_vertices()
    print(f"  After merge: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # Decimate to target face count
    if len(mesh.faces) > TARGET_FACES:
        mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
        print(f"  After decimation: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # Center mesh at centroid
    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    bb = mesh.bounding_box
    extents = bb.extents
    print(f"  Centered mesh extents (m): {extents}  =>  {extents*1000} mm")

    mesh.export(OUT_PATH)
    print(f"Exported to {OUT_PATH}")
    print()
    print("Use in MuJoCo XML:")
    print(f'  <mesh name="probe_col" file="../probe_assets/probe_col.obj"/>')
    print(f'  Geom body offset from attachment: center = {center}')


if __name__ == "__main__":
    main()
