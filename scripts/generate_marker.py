"""
Generates a print-ready SVG of the MarkerPose 3-circle marker.

Marker spec (measured):
  - Outer diameter: 20mm (black ring)
  - Inner diameter: 10mm (white disk)
  - Triangle: right-angle at c0, legs 40mm (→ cX) and 25mm (↑ cY)

Usage:
    python generate_marker.py [output.svg]
"""

import sys

# ── Marker geometry (mm) ──────────────────────────────────────────────────────
LEG_X   = 40.0   # c0 → cX (long leg, horizontal)
LEG_Y   = 25.0   # c0 → cY (short leg, vertical)
R_OUTER = 10.0   # outer black ring radius
R_INNER =  4.0   # inner white disk radius
MARGIN  = 15.0   # whitespace from SVG edge to circle center

# ── Circle centres (SVG coords: y increases downward) ────────────────────────
#   c0 at bottom-left, cX to the right, cY directly above c0
cx0, cy0 = MARGIN + R_OUTER,          MARGIN + LEG_Y + R_OUTER
cxX, cyX = cx0 + LEG_X,               cy0
cxY, cyY = cx0,                        cy0 - LEG_Y

W = cxX + R_OUTER + MARGIN
H = cy0 + R_OUTER + MARGIN


# ── SVG helpers ───────────────────────────────────────────────────────────────
def circ(cx, cy, r, fill, stroke='none', sw=0):
    return (f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:.3f}"/>')


def line(x1, y1, x2, y2, stroke='#aaaaaa', sw=0.25, dash=''):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>')


def txt(x, y, content, size=3.0, anchor='middle', color='#555555', rotate=None):
    rot = f' transform="rotate({rotate},{x:.3f},{y:.3f})"' if rotate else ''
    return (f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size}" '
            f'text-anchor="{anchor}" font-family="sans-serif" '
            f'fill="{color}"{rot}>{content}</text>')


def marker_circle(cx, cy):
    """Black ring with white inner disk."""
    return '\n  '.join([
        circ(cx, cy, R_OUTER, fill='black'),
        circ(cx, cy, R_INNER, fill='white'),
    ])


def dim_line_h(x1, x2, y, label):
    """Horizontal dimension annotation."""
    tick = 1.5
    return '\n  '.join([
        line(x1, y - tick, x1, y + tick),
        line(x2, y - tick, x2, y + tick),
        line(x1, y, x2, y),
        txt((x1 + x2) / 2, y - 2, label, size=3.0, color='#888888'),
    ])


def dim_line_v(x, y1, y2, label):
    """Vertical dimension annotation."""
    tick = 1.5
    return '\n  '.join([
        line(x - tick, y1, x + tick, y1),
        line(x - tick, y2, x + tick, y2),
        line(x, y1, x, y2),
        txt(x - 2, (y1 + y2) / 2, label, size=3.0, color='#888888',
            anchor='middle', rotate=-90),
    ])


# ── Build SVG ─────────────────────────────────────────────────────────────────
def generate(output_path):
    dim_y_x = cx0 - R_OUTER - 7          # x position of vertical dim line
    dim_x_y = cy0 + R_OUTER + 9          # y position of horizontal dim line

    svg = f"""\
<svg xmlns="http://www.w3.org/2000/svg"
     width="{W:.3f}mm" height="{H:.3f}mm"
     viewBox="0 0 {W:.3f} {H:.3f}">

  <!-- Background -->
  <rect width="{W:.3f}" height="{H:.3f}" fill="white"/>

  <!-- Cut border -->
  <rect x="0.5" y="0.5" width="{W-1:.3f}" height="{H-1:.3f}"
        fill="none" stroke="#cccccc" stroke-width="0.3" stroke-dasharray="2,2"/>


  <!-- ── Circles ── -->
  {marker_circle(cx0, cy0)}
  {marker_circle(cxX, cyX)}
  {marker_circle(cxY, cyY)}

</svg>
"""

    with open(output_path, 'w') as f:
        f.write(svg)

    print(f"Saved : {output_path}")
    print(f"Canvas: {W:.1f} x {H:.1f} mm")
    print(f"  c0  (origin)  : ({cx0:.1f}, {cy0:.1f}) mm from top-left")
    print(f"  cX  (+x axis) : ({cxX:.1f}, {cyX:.1f}) mm from top-left")
    print(f"  cY  (+y axis) : ({cxY:.1f}, {cyY:.1f}) mm from top-left")
    print()
    print("Print at 100% scale (no 'fit to page'). Verify with ruler before use.")


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'markerpose_target.svg'
    generate(out)
