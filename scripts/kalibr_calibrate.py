"""
Kalibr single-camera calibration pipeline for a laptop webcam.

Steps:
  capture    — capture calibration images from webcam
  bag        — convert images to ROS bag via Docker
  calibrate  — run kalibr_calibrate_cameras via Docker
  convert    — convert Kalibr YAML output to StereoParams.xml (MarkerPose format)
  all        — run all steps in sequence

Usage:
    python kalibr_calibrate.py capture
    python kalibr_calibrate.py bag
    python kalibr_calibrate.py calibrate
    python kalibr_calibrate.py convert
    python kalibr_calibrate.py all
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np
import yaml


# ── Config ────────────────────────────────────────────────────────────────────
CAPTURE_DIR   = 'calib_images'
BAG_FILE      = 'calib.bag'
TARGET_FILE   = 'aprilgrid.yaml'
RESULT_PREFIX = 'kalibr_result'
OUTPUT_XML    = 'StereoParams.xml'

CAMERA_TOPIC  = '/cam0/image_raw'
CAMERA_INDEX  = 1
FRAME_W, FRAME_H = 1280, 720

DOCKER_IMAGE  = 'ghcr.io/ethz-asl/kalibr:latest'

APRILGRID_YAML = """\
target_type: 'aprilgrid'
tagCols: 6
tagRows: 4
tagSize: 0.030
tagSpacing: 0.3
"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"[ERROR] Command failed (exit {result.returncode})")
        sys.exit(result.returncode)


def ensure_target_file():
    if not os.path.exists(TARGET_FILE):
        with open(TARGET_FILE, 'w') as f:
            f.write(APRILGRID_YAML)
        print(f"Created {TARGET_FILE}")


# ── Step 1: capture ───────────────────────────────────────────────────────────
def step_capture():
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # Count existing images so we don't overwrite
    existing = [f for f in os.listdir(CAPTURE_DIR) if f.endswith('.png')]
    idx = len(existing)

    print("=" * 60)
    print("CALIBRATION CAPTURE")
    print("=" * 60)
    print("  SPACE  — capture frame")
    print("  D      — delete last frame")
    print("  ESC    — finish and exit")
    print()
    print("Tips:")
    print("  • Hold the board steady when pressing SPACE")
    print("  • Cover all corners and edges of the frame")
    print("  • Tilt the board ±30° in both axes")
    print("  • Vary distance from ~40cm to ~120cm")
    print("  • Aim for 60-80 images")
    print("=" * 60)

    last_saved = None
    cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Capture', 960, 540)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        display = frame.copy()
        cv2.putText(display, f"Captured: {idx}  |  SPACE=capture  D=delete  ESC=done",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.imshow('Capture', display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC
            break

        elif key == ord(' '):
            path = os.path.join(CAPTURE_DIR, f'frame_{idx:04d}.png')
            cv2.imwrite(path, frame)
            last_saved = path
            idx += 1
            print(f"  Saved {path}  (total: {idx})")

        elif key == ord('d') and last_saved and os.path.exists(last_saved):
            os.remove(last_saved)
            idx -= 1
            print(f"  Deleted {last_saved}  (total: {idx})")
            last_saved = None

    cap.release()
    cv2.destroyAllWindows()

    saved = len([f for f in os.listdir(CAPTURE_DIR) if f.endswith('.png')])
    print(f"\nDone. {saved} images in '{CAPTURE_DIR}/'")
    if saved < 40:
        print("[WARN] Fewer than 40 images — calibration quality may suffer.")


# ── Step 2: bag ───────────────────────────────────────────────────────────────
def step_bag():
    images = sorted(f for f in os.listdir(CAPTURE_DIR) if f.endswith('.png'))
    if not images:
        print(f"[ERROR] No images found in '{CAPTURE_DIR}/'")
        sys.exit(1)

    print(f"Converting {len(images)} images → {BAG_FILE} ...")

    cwd = os.path.abspath('.')
    run([
        'docker', 'run', '--rm',
        '-v', f'{cwd}/{CAPTURE_DIR}:/images',
        '-v', f'{cwd}:/output',
        DOCKER_IMAGE,
        'kalibr_bagcreater',
        '--folder',     '/images',
        '--output-bag', f'/output/{BAG_FILE}',
        '--freq',       '10',
        '--topic',      CAMERA_TOPIC,
    ])
    print(f"Bag saved: {BAG_FILE}")


# ── Step 3: calibrate ─────────────────────────────────────────────────────────
def step_calibrate():
    ensure_target_file()

    if not os.path.exists(BAG_FILE):
        print(f"[ERROR] Bag file not found: {BAG_FILE}. Run 'bag' step first.")
        sys.exit(1)

    cwd = os.path.abspath('.')
    run([
        'docker', 'run', '--rm',
        '-v', f'{cwd}:/data',
        DOCKER_IMAGE,
        'kalibr_calibrate_cameras',
        '--bag',         f'/data/{BAG_FILE}',
        '--target',      f'/data/{TARGET_FILE}',
        '--models',      'pinhole-radtan',
        '--topics',      CAMERA_TOPIC,
        '--output-path', '/data',
        '--dont-show-report',
    ])

    results = [f for f in os.listdir('.') if f.startswith('camchain') and f.endswith('.yaml')]
    if results:
        print(f"\nCalibration complete. Result file(s): {results}")
    else:
        print("\n[WARN] No camchain YAML found — check Docker output above for errors.")


# ── Step 4: convert ───────────────────────────────────────────────────────────
def step_convert():
    # Find the camchain yaml
    candidates = sorted(f for f in os.listdir('.') if 'camchain' in f and f.endswith('.yaml'))
    if not candidates:
        print("[ERROR] No camchain YAML found. Run 'calibrate' step first.")
        sys.exit(1)

    yaml_file = candidates[-1]
    print(f"Reading: {yaml_file}")

    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    if 'cam0' not in data:
        print(f"[ERROR] 'cam0' not found in {yaml_file}")
        sys.exit(1)

    cam = data['cam0']
    fx, fy, cx, cy = cam['intrinsics']
    dist = np.array(cam['distortion_coeffs'], dtype=np.float64).reshape(1, -1)

    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float64)

    print(f"\nIntrinsics:")
    print(f"  fx={fx:.4f}  fy={fy:.4f}  cx={cx:.4f}  cy={cy:.4f}")
    print(f"  distortion: {dist}")

    # Reprojection error (if reported)
    if 'reprojection_error' in cam:
        err = cam['reprojection_error']
        flag = "OK" if err < 0.5 else "HIGH — consider recapturing"
        print(f"  reprojection error: {err:.4f} px  [{flag}]")

    # Write XML with K1/dist1 keys (single camera — K2/dist2/R/t left empty)
    fs = cv2.FileStorage(OUTPUT_XML, cv2.FILE_STORAGE_WRITE)
    fs.write('K1',    K)
    fs.write('dist1', dist)

    # Placeholders so MarkerPose doesn't crash if it tries to read stereo params
    fs.write('K2',    K)
    fs.write('dist2', dist)
    fs.write('R',     np.eye(3,    dtype=np.float64))
    fs.write('t',     np.zeros((3,1), dtype=np.float64))
    fs.release()

    print(f"\nSaved: {OUTPUT_XML}")
    print("Note: K2/dist2/R/t are placeholders — replace with right-camera")
    print("      calibration when you set up stereo.")


# ── Main ──────────────────────────────────────────────────────────────────────
STEPS = {
    'capture':   step_capture,
    'bag':       step_bag,
    'calibrate': step_calibrate,
    'convert':   step_convert,
}

def main():
    parser = argparse.ArgumentParser(
        description='Kalibr single-camera calibration pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join(f'  {k}' for k in [*STEPS, 'all'])
    )
    parser.add_argument('step', choices=[*STEPS, 'all'],
                        help='Pipeline step to run')
    args = parser.parse_args()

    if args.step == 'all':
        for name, fn in STEPS.items():
            print(f"\n{'='*60}")
            print(f"  STEP: {name.upper()}")
            print(f"{'='*60}")
            fn()
    else:
        STEPS[args.step]()


if __name__ == '__main__':
    main()
