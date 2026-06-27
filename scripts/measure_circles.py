"""
Measures inner and outer diameters of the 3-circle MarkerPose target.

Marker layout: 3 concentric-ring circles at vertices of a right-angled triangle.
  - Each circle = white inner disk + black outer ring on white background
  - Known real-world distances: short leg = 25mm, long leg = 40mm
    (hypotenuse = sqrt(25^2 + 40^2) = 47.17mm)

Usage:
    python measure_circles.py <image_path>
"""

import sys
import cv2
import numpy as np


KNOWN_DISTANCES_MM = sorted([25.0, 40.0, np.sqrt(25**2 + 40**2)])
MIN_CIRCULARITY = 0.50
MIN_AREA_PX = 80


def circularity(cnt):
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim == 0:
        return 0.0
    return 4 * np.pi * area / (perim ** 2)


def circle_from_contour(cnt):
    """Returns (center_xy, radius) using minimum enclosing circle."""
    (x, y), r = cv2.minEnclosingCircle(cnt)
    return np.array([x, y]), r


def detect_rings(gray, debug_prefix=None):
    """
    Returns list of (outer_center, outer_radius_px, inner_radius_px).

    Strategy:
      - Invert threshold so black rings become white foreground.
      - Background (white) and inner disks (white) become black → holes.
      - RETR_TREE gives parent-child hierarchy:
          Level 0: outer boundary of each ring
          Level 1: inner boundary of each ring (the white-disk hole)
    """
    # Normalize lighting before thresholding
    gray = cv2.equalizeHist(gray)

    # Adaptive threshold handles uneven lighting better than global Otsu.
    # Block size must be odd and larger than the ring width in pixels.
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,   # increase if rings are large in the image
        C=10
    )

    # Clean up noise: small specks and broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_1_threshold.png", thresh)
        print(f"  [debug] threshold saved: {debug_prefix}_1_threshold.png")

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return []

    hier = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]

    # Collect circular contours with their hierarchy info
    circles = []
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < MIN_AREA_PX:
            continue
        if circularity(cnt) < MIN_CIRCULARITY:
            continue
        center, radius = circle_from_contour(cnt)
        parent_idx = hier[i][3]
        child_idx  = hier[i][2]
        circles.append({
            'idx': i,
            'center': center,
            'radius': radius,
            'parent': parent_idx,
            'child': child_idx,
        })

    # Outer rings: have a child contour (the inner hole)
    outer = [c for c in circles if c['child'] != -1]

    # Inner holes: have a parent that is an outer ring
    outer_indices = {c['idx'] for c in outer}
    inner = [c for c in circles if c['parent'] in outer_indices]

    # Build (outer_center, outer_r, inner_r) triples
    rings = []
    for o in outer:
        matching_inner = [c for c in inner if c['parent'] == o['idx']]
        if not matching_inner:
            print(f"  Warning: outer contour {o['idx']} has no matching inner hole")
            continue
        # Take largest inner contour if multiple
        best_inner = max(matching_inner, key=lambda c: c['radius'])
        rings.append((o['center'], o['radius'], best_inner['radius']))

    if debug_prefix:
        dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i, cnt in enumerate(contours):
            cv2.drawContours(dbg, contours, i, (200, 200, 200), 1)
        for o in outer:
            cx, cy = int(o['center'][0]), int(o['center'][1])
            cv2.circle(dbg, (cx, cy), int(o['radius']), (0, 0, 255), 2)
            cv2.putText(dbg, f"out r={o['radius']:.0f}px", (cx+5, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        for ic in inner:
            cx, cy = int(ic['center'][0]), int(ic['center'][1])
            cv2.circle(dbg, (cx, cy), int(ic['radius']), (0, 255, 0), 2)
            cv2.putText(dbg, f"in r={ic['radius']:.0f}px", (cx+5, cy+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,0), 1)
        cv2.imwrite(f"{debug_prefix}_2_detected.png", dbg)
        print(f"  [debug] detections saved: {debug_prefix}_2_detected.png")
        print(f"  [debug] all circular contours found: {len([c for c in circles])}")
        print(f"  [debug] outer rings: {len(outer)}, inner holes: {len(inner)}")

    return rings


def assign_scale(rings):
    """
    Compute mm/pixel scale by matching sorted pixel distances
    to sorted known real distances.
    """
    centers = [r[0] for r in rings]
    n = len(centers)

    pixel_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            pixel_dists.append(d)
    pixel_dists.sort()

    if len(pixel_dists) != len(KNOWN_DISTANCES_MM):
        print(f"  Warning: expected {len(KNOWN_DISTANCES_MM)} distances, "
              f"got {len(pixel_dists)}. Scale may be inaccurate.")
        n_pairs = min(len(pixel_dists), len(KNOWN_DISTANCES_MM))
        pixel_dists = pixel_dists[:n_pairs]
        known = KNOWN_DISTANCES_MM[:n_pairs]
    else:
        known = KNOWN_DISTANCES_MM

    scales = [known[k] / pixel_dists[k] for k in range(len(pixel_dists))]
    scale = float(np.mean(scales))

    print("\nDistance calibration:")
    for k in range(len(pixel_dists)):
        print(f"  {pixel_dists[k]:.1f} px  →  {known[k]:.2f} mm  "
              f"(scale = {known[k]/pixel_dists[k]:.4f} mm/px)")
    print(f"  Mean scale: {scale:.4f} mm/px  (std: {np.std(scales):.4f})")

    return scale


def draw_results(img, rings, scale):
    vis = img.copy()
    for i, (center, r_outer, r_inner) in enumerate(rings):
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(vis, (cx, cy), int(r_outer), (0, 0, 255), 2)
        cv2.circle(vis, (cx, cy), int(r_inner), (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 3, (255, 0, 0), -1)
        label = f"C{i+1}: D_out={2*r_outer*scale:.1f}mm D_in={2*r_inner*scale:.1f}mm"
        cv2.putText(vis, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Image: {image_path}  ({img.shape[1]}x{img.shape[0]} px)")

    debug_prefix = None
    if '--debug' in sys.argv:
        debug_prefix = image_path.rsplit('.', 1)[0]
        print("[debug mode on]")

    rings = detect_rings(gray, debug_prefix=debug_prefix)
    print(f"\nDetected {len(rings)} ring(s)")

    if len(rings) < 2:
        print("Need at least 2 rings to establish scale. Check image or MIN_AREA_PX threshold.")
        sys.exit(1)

    if len(rings) != 3:
        print(f"Warning: expected 3 rings, found {len(rings)}. Proceeding anyway.")

    scale = assign_scale(rings)

    print("\nCircle measurements:")
    print(f"  {'Circle':<10} {'Outer diam (mm)':<20} {'Inner diam (mm)':<20} {'Ring width (mm)'}")
    print(f"  {'-'*65}")
    for i, (center, r_outer, r_inner) in enumerate(rings):
        d_out = 2 * r_outer * scale
        d_in  = 2 * r_inner * scale
        ring_w = (r_outer - r_inner) * scale
        print(f"  Circle {i+1:<4}  {d_out:<20.2f} {d_in:<20.2f} {ring_w:.2f}")

    # Save annotated image
    out_path = image_path.rsplit('.', 1)[0] + '_measured.png'
    vis = draw_results(img, rings, scale)
    cv2.imwrite(out_path, vis)
    print(f"\nAnnotated image saved: {out_path}")


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        print("Usage: python measure_circles.py <image_path> [--debug]")
        sys.exit(1)
    main(args[0])
