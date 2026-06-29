"""
Convert a NV-Segment-CTMR segmentation NIfTI to per-label OBJ meshes for raysim.

Uses the VTK pipeline (vtkDiscreteFlyingEdges3D + CleanPolyData + LargestRegion +
Windowed Sinc smoothing + vtkQuadricDecimation + CleanPolyData + FillHoles +
vtkPolyDataNormals) — outputs individual OBJs with raysim acoustic materials.

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
    VTK pipeline: FlyingEdges -> Clean -> LargestComponent -> Windowed Sinc
                  -> Decimate -> Clean -> FillHoles -> Normals -> RAS-to-LPS.
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

    # Merge coincident points and remove degenerate triangles from voxel surface.
    # FlyingEdges leaves duplicate vertices at shared voxel faces — without this
    # step, every subsequent filter sees a non-manifold soup and breaks.
    clean1 = vtk.vtkCleanPolyData()
    clean1.SetInputConnection(flying_edges.GetOutputPort())
    clean1.PointMergingOn()
    clean1.Update()

    if clean1.GetOutput().GetNumberOfPoints() == 0:
        return None

    # Keep only the largest connected component — eliminates floating noise
    # islands that segmentation produces for small, isolated voxel clusters.
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputConnection(clean1.GetOutputPort())
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()

    # Windowed Sinc smoothing removes staircase voxel artifacts while
    # preserving volume. passband ~ 0.1 at default (not 0.01 — that over-smooths).
    # NonManifoldSmoothing is off: it worsens butterfly artifacts at voxel corners.
    n_iter   = int(20 + smooth_factor * 40)
    passband = max(0.001, pow(10.0, -2.0 * smooth_factor))
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(connectivity.GetOutputPort())
    smoother.SetNumberOfIterations(n_iter)
    smoother.SetPassBand(passband)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOff()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    # Quadric decimation with volume preservation
    decimation = vtk.vtkQuadricDecimation()
    decimation.SetInputConnection(smoother.GetOutputPort())
    decimation.SetTargetReduction(reduction)
    decimation.VolumePreservationOn()
    decimation.Update()

    # Clean after decimation — quadric decimation can produce zero-area triangles
    # at high reduction rates that cause rendering artifacts and broken topology.
    clean2 = vtk.vtkCleanPolyData()
    clean2.SetInputConnection(decimation.GetOutputPort())
    clean2.PointMergingOn()
    clean2.Update()

    # Close small holes torn by decimation (e.g. in thin structures like vessels)
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputConnection(clean2.GetOutputPort())
    fill_holes.SetHoleSize(100.0)
    fill_holes.Update()

    # Consistent normals, auto-oriented outward. Without AutoOrientNormals, inward-
    # pointing patches appear invisible in one-sided rendering ("half broken").
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(fill_holes.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.ComputePointNormalsOn()
    normals.Update()

    # vtkNIFTIImageReader already applies DataSpacing and DataOrigin to the image
    # data, so FlyingEdges output is in physical mm. Applying SForm on top would
    # square the voxel spacing. Only flip RAS→LPS (matches ABDPhantom OBJ convention).
    ras2lps_mat = vtk.vtkMatrix4x4()
    ras2lps_mat.SetElement(0, 0, -1)
    ras2lps_mat.SetElement(1, 1, -1)
    ras2lps = vtk.vtkTransform()
    ras2lps.SetMatrix(ras2lps_mat)
    lps_transformer = vtk.vtkTransformPolyDataFilter()
    lps_transformer.SetTransform(ras2lps)
    lps_transformer.SetInputConnection(normals.GetOutputPort())
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
    parser.add_argument("--reduction", type=float, default=0.7,
                        help="Quadric decimation target reduction 0-1 (0.7 = 70%% fewer faces)")
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

    # Get unique labels via numpy.
    # GetScalars() can return None when the NIfTI reader stores data under a named
    # array rather than the active-scalar slot — fall back to GetArray(0).
    point_data = reader.GetOutput().GetPointData()
    scalars_vtk = point_data.GetScalars()
    if scalars_vtk is None:
        scalars_vtk = point_data.GetArray(0)
    if scalars_vtk is None:
        raise RuntimeError(f"No scalar data found in {args.seg} — check the file is a valid NIfTI segmentation.")
    scalars = vtk_to_numpy(scalars_vtk).astype(np.int32)
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
