"""
Convert a NV-Segment-CTMR segmentation NIfTI to per-label OBJ meshes for raysim.

Uses the VTK pipeline (vtkDiscreteFlyingEdges3D + Windowed Sinc smoothing +
vtkQuadricDecimation + vtkPolyDataNormals) — same quality as the i4h-workflows
CT_to_USD converter, but outputs individual OBJs with raysim acoustic materials.

Usage (genesis conda env):
    conda run -n genesis python scripts/seg_to_obj.py \
        --seg output/eval_all/image/image_trans.nii.gz \
        --out_dir output/obj_all \
        [--min_voxels 50] [--smooth_factor 0.5] [--reduction 0.9]
"""

import argparse
import json
import os
import pathlib

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


_REPO_ROOT = pathlib.Path(__file__).parent.parent
_META_PATH = _REPO_ROOT.parent / "NV-Segment-CTMR" / "NV-Segment-CTMR" / "configs" / "metadata.json"

# raysim built-in materials: water, liver, muscle, bone, blood, fat, air
RAYSIM_MATERIAL_MAP = {
    "liver":       "liver",
    "spleen":      "liver",
    "kidney":      "liver",
    "pancreas":    "muscle",
    "gallbladder": "water",
    "stomach":     "muscle",
    "bowel":       "muscle",
    "colon":       "muscle",
    "rectum":      "muscle",
    "duodenum":    "muscle",
    "esophagus":   "muscle",
    "bladder":     "water",
    "uterus":      "muscle",
    "prostate":    "muscle",
    "adrenal":     "muscle",
    "heart":       "muscle",
    "ventricle":   "muscle",
    "atrium":      "muscle",
    "myocardium":  "muscle",
    "aorta":       "blood",
    "artery":      "blood",
    "vein":        "blood",
    "vena":        "blood",
    "portal":      "blood",
    "vessel":      "blood",
    "pulmonary":   "blood",
    "lung":        "air",
    "rib":         "bone",
    "vertebrae":   "bone",
    "vertebra":    "bone",
    "spine":       "bone",
    "sternum":     "bone",
    "sacrum":      "bone",
    "femur":       "bone",
    "humerus":     "bone",
    "hip":         "bone",
    "iliac":       "bone",
    "cartilage":   "bone",
    "disc":        "muscle",
    "cord":        "muscle",
    "muscles":     "muscle",
    "muscle":      "muscle",
    "gluteus":     "muscle",
    "iliopsoas":   "muscle",
    "autochthon":  "muscle",
    "fat":         "fat",
    "skin":        "muscle",
    "tissue":      "muscle",
    "breast":      "fat",
    "thyroid":     "muscle",
    "tumor":       "liver",
    "lesion":      "bone",
    "cyst":        "water",
    "bone":        "bone",
}


def label_to_material(name: str) -> str:
    name_lower = name.lower()
    for keyword, material in RAYSIM_MATERIAL_MAP.items():
        if keyword in name_lower:
            return material
    return "muscle"


def sanitize_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def label_to_mesh_vtk(reader, label_idx: int, smooth_factor: float, reduction: float) -> vtk.vtkPolyData | None:
    """
    VTK pipeline: FlyingEdges -> Windowed Sinc smooth -> Quadric decimate
                  -> Normals -> SForm transform -> RAS-to-LPS.
    Returns vtkPolyData or None if the label produces no geometry.
    """
    # Surface extraction
    flying_edges = vtk.vtkDiscreteFlyingEdges3D()
    flying_edges.SetInputConnection(reader.GetOutputPort())
    flying_edges.ComputeGradientsOff()
    flying_edges.ComputeNormalsOff()
    flying_edges.SetValue(0, label_idx)
    flying_edges.Update()

    if flying_edges.GetOutput().GetNumberOfPoints() == 0:
        return None

    # Windowed Sinc smoothing — volume-preserving, feature-preserving
    n_iter   = int(20 + smooth_factor * 40)
    passband = pow(10.0, -4.0 * smooth_factor)
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(flying_edges.GetOutputPort())
    smoother.SetNumberOfIterations(n_iter)
    smoother.SetPassBand(passband)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    # Quadric decimation with volume preservation
    decimation = vtk.vtkQuadricDecimation()
    decimation.SetInputConnection(smoother.GetOutputPort())
    decimation.SetTargetReduction(reduction)
    decimation.VolumePreservationOn()
    decimation.Update()

    # Smooth, consistent normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(decimation.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.Update()

    # Apply NIfTI SForm matrix (voxel -> world mm)
    sform = reader.GetSFormMatrix()
    if sform is None or sform.IsIdentity():
        sform = vtk.vtkMatrix4x4()
    sform_transform = vtk.vtkTransform()
    sform_transform.SetMatrix(sform)
    world_transformer = vtk.vtkTransformPolyDataFilter()
    world_transformer.SetTransform(sform_transform)
    world_transformer.SetInputConnection(normals.GetOutputPort())
    world_transformer.Update()

    # RAS -> LPS (flip X and Y — matches existing ABDPhantom OBJ convention)
    ras2lps_mat = vtk.vtkMatrix4x4()
    ras2lps_mat.SetElement(0, 0, -1)
    ras2lps_mat.SetElement(1, 1, -1)
    ras2lps = vtk.vtkTransform()
    ras2lps.SetMatrix(ras2lps_mat)
    lps_transformer = vtk.vtkTransformPolyDataFilter()
    lps_transformer.SetTransform(ras2lps)
    lps_transformer.SetInputConnection(world_transformer.GetOutputPort())
    lps_transformer.Update()

    return lps_transformer.GetOutput()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg", default=str(_REPO_ROOT / "output" / "eval_all" / "image" / "image_trans.nii.gz"),
                        help="Segmentation NIfTI from NV-Segment-CTMR")
    parser.add_argument("--out_dir", default=str(_REPO_ROOT / "output" / "obj_all"),
                        help="Output directory for OBJ files")
    parser.add_argument("--min_voxels", type=int, default=50,
                        help="Skip labels with fewer voxels (noise filter)")
    parser.add_argument("--smooth_factor", type=float, default=0.5,
                        help="Windowed Sinc smoothing strength 0-1 (0=off, 1=max)")
    parser.add_argument("--reduction", type=float, default=0.9,
                        help="Quadric decimation target reduction 0-1 (0.9 = 90%% fewer faces)")
    parser.add_argument("--metadata", default=str(_META_PATH))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.metadata) as f:
        meta = json.load(f)
    label_names = meta["network_data_format"]["outputs"]["pred"]["channel_def"]

    vtk.vtkOutputWindow().SetGlobalWarningDisplay(0)

    print(f"Loading {args.seg}")
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(args.seg)
    reader.Update()

    # Get unique labels via numpy
    scalars = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars()).astype(np.int32)
    unique_labels = sorted(set(scalars.tolist()) - {0})
    print(f"Found {len(unique_labels)} non-background labels\n")

    material_index = {}

    for label_idx in unique_labels:
        name = label_names.get(str(label_idx), f"class_{label_idx}")
        n_vox = int((scalars == label_idx).sum())

        if n_vox < args.min_voxels:
            print(f"  skip  {label_idx:3d} {name} ({n_vox} voxels < {args.min_voxels})")
            continue

        material = label_to_material(name)
        fname    = sanitize_filename(name) + ".obj"
        out_path = os.path.join(args.out_dir, fname)

        try:
            polydata = label_to_mesh_vtk(reader, label_idx, args.smooth_factor, args.reduction)
            if polydata is None or polydata.GetNumberOfPoints() == 0:
                print(f"  empty {label_idx:3d} {name} — skipping")
                continue

            writer = vtk.vtkOBJWriter()
            writer.SetFileName(out_path)
            writer.SetInputData(polydata)
            writer.Write()

            material_index[fname] = material
            n_faces = polydata.GetNumberOfCells()
            print(f"  saved {label_idx:3d} {name:40s} -> {fname}  [{material}]  ({n_vox} vox, {n_faces} faces)")
        except Exception as e:
            print(f"  ERROR {label_idx:3d} {name}: {e}")

    index_path = os.path.join(args.out_dir, "material_index.json")
    with open(index_path, "w") as f:
        json.dump(material_index, f, indent=2)
    print(f"\nWrote material index -> {index_path}")
    print(f"OBJ files ready in:    {args.out_dir}")


if __name__ == "__main__":
    main()
