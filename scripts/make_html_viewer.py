#!/usr/bin/env python3
"""Generate a standalone browser viewer for pin-axis demo outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_ply_xyzrgb(path: Path) -> dict:
    lines = path.read_text(encoding="ascii").splitlines()
    vertex_count = None
    data_start = None
    for idx, line in enumerate(lines):
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
        if line == "end_header":
            data_start = idx + 1
            break
    if vertex_count is None or data_start is None:
        raise ValueError(f"Not a supported XYZRGB ASCII PLY: {path}")

    points = []
    colors = []
    for line in lines[data_start : data_start + vertex_count]:
        x, y, z, r, g, b = line.split()[:6]
        points.append([round(float(x), 6), round(float(y), 6), round(float(z), 6)])
        colors.append([int(r), int(g), int(b)])
    return {"points": points, "colors": colors}


def compact_json(value) -> str:
    return json.dumps(value, separators=(",", ":"))


def build_html(scene: dict, axes: dict, centerlines: dict, result: dict) -> str:
    scene_json = compact_json(scene)
    axes_json = compact_json(axes)
    center_json = compact_json(centerlines)
    result_summary = {
        "detections": len(result["detection"]["detections"]),
        "evaluation": result["evaluation"],
    }
    summary_json = compact_json(result_summary)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pin Axis 3D Viewer</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #111316;
      color: #f3f4f6;
      font-family: Arial, sans-serif;
    }}
    #view {{
      width: 100vw;
      height: 100vh;
      display: block;
      cursor: grab;
    }}
    #view:active {{ cursor: grabbing; }}
    #panel {{
      position: fixed;
      top: 14px;
      left: 14px;
      width: min(380px, calc(100vw - 28px));
      background: rgba(17, 19, 22, 0.86);
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 8px;
      padding: 12px 14px;
      box-sizing: border-box;
      backdrop-filter: blur(6px);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 16px;
      font-weight: 700;
    }}
    .metric {{
      font-size: 13px;
      line-height: 1.45;
      color: #d7dce2;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: #eef1f5;
      user-select: none;
    }}
    input[type="checkbox"] {{
      width: 15px;
      height: 15px;
    }}
    button {{
      border: 1px solid rgba(255,255,255,0.22);
      background: #20262d;
      color: #f3f4f6;
      border-radius: 6px;
      padding: 5px 9px;
      font-size: 13px;
      cursor: pointer;
    }}
    #hint {{
      margin-top: 8px;
      font-size: 12px;
      line-height: 1.35;
      color: #aeb7c2;
    }}
  </style>
</head>
<body>
<canvas id="view"></canvas>
<div id="panel">
  <h1>3D Pin-Axis Alignment Prototype</h1>
  <div class="metric" id="metrics"></div>
  <div class="controls">
    <label><input id="toggleScene" type="checkbox" checked> cloud</label>
    <label><input id="toggleAxes" type="checkbox" checked> detected axes</label>
    <label><input id="toggleCenter" type="checkbox" checked> gripper lines</label>
    <button id="reset">Reset View</button>
  </div>
  <div id="hint">Drag to orbit. Wheel to zoom. Red = detected pin axes. Blue = virtual gripper centerlines. Green dots in RViz export = pregrasp targets.</div>
</div>
<script>
const scene = {scene_json};
const axes = {axes_json};
const centerlines = {center_json};
const summary = {summary_json};

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d', {{ alpha: false }});
const toggles = {{
  scene: document.getElementById('toggleScene'),
  axes: document.getElementById('toggleAxes'),
  center: document.getElementById('toggleCenter')
}};

document.getElementById('metrics').textContent =
  `${{summary.evaluation.matched_count}}/${{summary.evaluation.truth_count}} pins matched, ` +
  `${{summary.evaluation.false_positive_count}} false positives, ` +
  `${{summary.evaluation.mean_angular_error_deg}} deg mean angle error, ` +
  `${{summary.evaluation.mean_axis_lateral_error_m}} m mean lateral error`;

let yaw = -0.78;
let pitch = -0.72;
let zoom = 610;
let panX = 0;
let panY = 0;
let dragging = false;
let lastX = 0;
let lastY = 0;

const allPoints = scene.points.concat(axes.points, centerlines.points);
const bounds = allPoints.reduce((acc, p) => {{
  for (let i = 0; i < 3; i++) {{
    acc.min[i] = Math.min(acc.min[i], p[i]);
    acc.max[i] = Math.max(acc.max[i], p[i]);
  }}
  return acc;
}}, {{ min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] }});
const center = [0, 1, 2].map(i => (bounds.min[i] + bounds.max[i]) / 2);

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  canvas.style.width = window.innerWidth + 'px';
  canvas.style.height = window.innerHeight + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  render();
}}

function rotatePoint(p) {{
  let x = p[0] - center[0];
  let y = p[1] - center[1];
  let z = p[2] - center[2];

  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);

  const x1 = cy * x - sy * y;
  const y1 = sy * x + cy * y;
  const z1 = z;

  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1;
  return [x1, y2, z2];
}}

function project(p) {{
  const r = rotatePoint(p);
  const scale = zoom / (1.2 + r[2]);
  return [
    window.innerWidth / 2 + panX + r[0] * scale,
    window.innerHeight / 2 + panY - r[1] * scale,
    r[2],
    scale
  ];
}}

function rgb(c, alpha = 1) {{
  return `rgba(${{c[0]}},${{c[1]}},${{c[2]}},${{alpha}})`;
}}

function drawPoints(cloud, radius, alpha) {{
  const projected = cloud.points.map((p, i) => {{
    const q = project(p);
    return {{ x: q[0], y: q[1], z: q[2], c: cloud.colors[i] }};
  }});
  projected.sort((a, b) => a.z - b.z);
  for (const p of projected) {{
    ctx.fillStyle = rgb(p.c, alpha);
    ctx.fillRect(p.x - radius, p.y - radius, radius * 2, radius * 2);
  }}
}}

function drawLineCloud(cloud, alpha) {{
  ctx.lineWidth = 2;
  for (let i = 0; i + 1 < cloud.points.length; i += 2) {{
    const a = project(cloud.points[i]);
    const b = project(cloud.points[i + 1]);
    ctx.strokeStyle = rgb(cloud.colors[i], alpha);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }}
}}

function drawAxesGizmo() {{
  const origin = project(center);
  const length = 0.05;
  const axesLocal = [
    [[center[0] + length, center[1], center[2]], '#ef4444', 'X'],
    [[center[0], center[1] + length, center[2]], '#22c55e', 'Y'],
    [[center[0], center[1], center[2] + length], '#60a5fa', 'Z']
  ];
  ctx.lineWidth = 2;
  ctx.font = '12px Arial';
  for (const [end, color, label] of axesLocal) {{
    const p = project(end);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(origin[0], origin[1]);
    ctx.lineTo(p[0], p[1]);
    ctx.stroke();
    ctx.fillText(label, p[0] + 4, p[1] + 4);
  }}
}}

function render() {{
  ctx.fillStyle = '#111316';
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);
  drawAxesGizmo();
  if (toggles.scene.checked) drawPoints(scene, 1.15, 0.82);
  if (toggles.axes.checked) drawLineCloud(axes, 0.95);
  if (toggles.center.checked) drawLineCloud(centerlines, 0.95);
}}

canvas.addEventListener('mousedown', e => {{
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
}});
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;
  yaw += dx * 0.008;
  pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.008));
  render();
}});
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  zoom *= Math.exp(-e.deltaY * 0.001);
  zoom = Math.max(120, Math.min(2500, zoom));
  render();
}}, {{ passive: false }});
Object.values(toggles).forEach(input => input.addEventListener('change', render));
document.getElementById('reset').addEventListener('click', () => {{
  yaw = -0.78;
  pitch = -0.72;
  zoom = 610;
  panX = 0;
  panY = 0;
  render();
}});
window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    demo_dir = args.demo_dir
    scene = read_ply_xyzrgb(demo_dir / "scene_cloud.ply")
    axes = read_ply_xyzrgb(demo_dir / "detected_axes.ply")
    centerlines = read_ply_xyzrgb(demo_dir / "gripper_centerlines.ply")
    result = json.loads((demo_dir / "result.json").read_text(encoding="utf-8"))

    output = args.output or demo_dir / "viewer.html"
    output.write_text(build_html(scene, axes, centerlines, result), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
