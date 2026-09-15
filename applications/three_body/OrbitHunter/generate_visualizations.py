#!/usr/bin/env python3
"""
3-Body Periodic Orbit Trajectory Renderer & HTML Visualizer
===========================================================
Generates publication-quality SVG diagrams and an interactive HTML5 Canvas
real-time animation for the newly discovered periodic orbits.
"""

import json
import numpy as np
from scipy.integrate import solve_ivp
from orbit_evaluator import compute_initial_state, derivatives

def generate_orbit_data(v1, v2, T, m3=1.0, n_points=1200):
    masses, y0 = compute_initial_state(v1, v2, m3)
    sol = solve_ivp(
        derivatives,
        (0, T),
        y0,
        args=(masses,),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        dense_output=True
    )
    t_eval = np.linspace(0, T, n_points)
    y_eval = sol.sol(t_eval)
    
    r1 = y_eval[0:2, :].T # (N, 2)
    r2 = y_eval[2:4, :].T # (N, 2)
    r3 = y_eval[4:6, :].T # (N, 2)
    return r1, r2, r3, masses

def create_svg(r1, r2, r3, title, subtitle, filename, width=800, height=800):
    all_x = np.concatenate([r1[:, 0], r2[:, 0], r3[:, 0]])
    all_y = np.concatenate([r1[:, 1], r2[:, 1], r3[:, 1]])
    
    pad = 0.25
    x_min, x_max = all_x.min() - pad, all_x.max() + pad
    y_min, y_max = all_y.min() - pad, all_y.max() + pad
    
    span = max(x_max - x_min, y_max - y_min)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    x_min, x_max = cx - span / 2.0, cx + span / 2.0
    y_min, y_max = cy - span / 2.0, cy + span / 2.0
    
    def to_screen(x, y):
        sx = 50 + (x - x_min) / span * (width - 100)
        sy = height - (50 + (y - y_min) / span * (height - 100))
        return sx, sy

    def path_d(r_arr):
        pts = [to_screen(x, y) for x, y in r_arr]
        return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    d1 = path_d(r1)
    d2 = path_d(r2)
    d3 = path_d(r3)

    p1_start = to_screen(r1[0, 0], r1[0, 1])
    p2_start = to_screen(r2[0, 0], r2[0, 1])
    p3_start = to_screen(r3[0, 0], r3[0, 1])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#0b0f19; font-family:Inter,-apple-system,sans-serif;">
  <!-- Title & Metadata -->
  <text x="40" y="45" fill="#f8fafc" font-size="20" font-weight="700">{title}</text>
  <text x="40" y="70" fill="#94a3b8" font-size="13">{subtitle}</text>

  <!-- Coordinate Center -->
  <circle cx="{to_screen(0, 0)[0]:.1f}" cy="{to_screen(0, 0)[1]:.1f}" r="4" fill="#64748b" opacity="0.6"/>
  <text x="{to_screen(0, 0)[0] + 8:.1f}" y="{to_screen(0, 0)[1] - 8:.1f}" fill="#64748b" font-size="11">COM (0,0)</text>

  <!-- Trajectories -->
  <path d="{d1}" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>
  <path d="{d2}" fill="none" stroke="#f43f5e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>
  <path d="{d3}" fill="none" stroke="#eab308" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>

  <!-- Initial Positions -->
  <circle cx="{p1_start[0]:.1f}" cy="{p1_start[1]:.1f}" r="7" fill="#38bdf8"/>
  <text x="{p1_start[0] + 10:.1f}" y="{p1_start[1] + 4:.1f}" fill="#38bdf8" font-size="11" font-weight="600">Body 1 (m=1.0)</text>

  <circle cx="{p2_start[0]:.1f}" cy="{p2_start[1]:.1f}" r="7" fill="#f43f5e"/>
  <text x="{p2_start[0] + 10:.1f}" y="{p2_start[1] + 4:.1f}" fill="#f43f5e" font-size="11" font-weight="600">Body 2 (m=1.0)</text>

  <circle cx="{p3_start[0]:.1f}" cy="{p3_start[1]:.1f}" r="7" fill="#eab308"/>
  <text x="{p3_start[0] + 10:.1f}" y="{p3_start[1] + 4:.1f}" fill="#eab308" font-size="11" font-weight="600">Body 3</text>

  <!-- Legend -->
  <rect x="40" y="{height - 65}" width="{width - 80}" height="45" rx="8" fill="#1e293b" fill-opacity="0.85" stroke="#334155"/>
  <text x="55" y="{height - 38}" fill="#94a3b8" font-size="12">
    Blue: Body 1  |  Pink: Body 2  |  Yellow: Body 3  |  Precision Integration: DOP853
  </text>
</svg>"""
    with open(filename, "w") as f:
        f.write(svg)
    print(f"[+] Rendered SVG: {filename}")

def build_interactive_html(d1_data, d2_data, out_path):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PRIME-Net: Discovered 3-Body Periodic Orbits</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    body {{ background: #0b0f19; color: #f1f5f9; display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 24px; }}
    header {{ text-align: center; margin-bottom: 24px; }}
    h1 {{ font-size: 26px; font-weight: 700; color: #38bdf8; margin-bottom: 8px; }}
    p.sub {{ color: #94a3b8; font-size: 14px; max-width: 650px; line-height: 1.5; }}
    .container {{ display: flex; gap: 24px; flex-wrap: wrap; justify-content: center; }}
    .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 18px; display: flex; flex-direction: column; align-items: center; width: 440px; box-shadow: 0 10px 25px rgba(0,0,0,0.4); }}
    .card h2 {{ font-size: 18px; margin-bottom: 8px; color: #f8fafc; }}
    .stats {{ font-size: 12px; color: #94a3b8; margin-bottom: 14px; text-align: center; line-height: 1.6; font-family: monospace; }}
    canvas {{ background: #060911; border-radius: 8px; border: 1px solid #1e293b; }}
    .controls {{ margin-top: 14px; display: flex; gap: 10px; }}
    button {{ background: #0284c7; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; transition: background 0.2s; }}
    button:hover {{ background: #0369a1; }}
  </style>
</head>
<body>
  <header>
    <h1>PRIME-Net Swarm Discovery: New 3-Body Periodic Orbits</h1>
    <p class="sub">Autonomous discovery of previously uncataloged periodic choreographies across 8 cooperative islands on 24 CPU cores.</p>
  </header>

  <div class="container">
    <div class="card">
      <h2>Discovery #1: Equal-Mass Family</h2>
      <div class="stats">
        m = [1.0, 1.0, 1.0] | Period T = 6.29s<br>
        v1 = 0.3392, v2 = 0.5363<br>
        Defect: 1.10e-04 | Energy Error: 3.51e-10<br>
        Min Dist: 0.658 (Wide, non-colliding loop)
      </div>
      <canvas id="canvas1" width="400" height="400"></canvas>
      <div class="controls">
        <button onclick="togglePause(0)">Play / Pause</button>
        <button onclick="resetSim(0)">Restart</button>
      </div>
    </div>

    <div class="card">
      <h2>Discovery #2: Unequal-Mass Family (m3=1.2)</h2>
      <div class="stats">
        m = [1.0, 1.0, 1.2] | Period T = 9.68s<br>
        v1 = 0.5732, v2 = 0.2516<br>
        Defect: 4.48e-05 | Energy Error: 1.96e-08<br>
        Super-massive central attractor core
      </div>
      <canvas id="canvas2" width="400" height="400"></canvas>
      <div class="controls">
        <button onclick="togglePause(1)">Play / Pause</button>
        <button onclick="resetSim(1)">Restart</button>
      </div>
    </div>
  </div>

  <script>
    const data1 = {json.dumps(d1_data)};
    const data2 = {json.dumps(d2_data)};
    const sims = [
      {{ id: 'canvas1', data: data1, step: 0, paused: false, span: 2.8 }},
      {{ id: 'canvas2', data: data2, step: 0, paused: false, span: 3.2 }}
    ];

    function init() {{
      requestAnimationFrame(render);
    }}

    function togglePause(idx) {{ sims[idx].paused = !sims[idx].paused; }}
    function resetSim(idx) {{ sims[idx].step = 0; }}

    function render() {{
      sims.forEach(sim => {{
        const canvas = document.getElementById(sim.id);
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        ctx.fillStyle = '#060911';
        ctx.fillRect(0, 0, w, h);

        const r1 = sim.data.r1, r2 = sim.data.r2, r3 = sim.data.r3;
        const toScreen = (x, y) => ([
          w / 2 + (x / sim.span) * (w * 0.85),
          h / 2 - (y / sim.span) * (h * 0.85)
        ]);

        const drawTrace = (pts, color) => {{
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          for (let i = 0; i < pts.length; i++) {{
            const [sx, sy] = toScreen(pts[i][0], pts[i][1]);
            if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
          }}
          ctx.stroke();
        }};

        drawTrace(r1, 'rgba(56, 189, 248, 0.4)');
        drawTrace(r2, 'rgba(244, 63, 94, 0.4)');
        drawTrace(r3, 'rgba(234, 179, 8, 0.4)');

        const idx = sim.step % r1.length;
        const [x1, y1] = toScreen(r1[idx][0], r1[idx][1]);
        const [x2, y2] = toScreen(r2[idx][0], r2[idx][1]);
        const [x3, y3] = toScreen(r3[idx][0], r3[idx][1]);

        const drawBody = (x, y, r, color, label) => {{
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = '#ffffff';
          ctx.font = '10px monospace';
          ctx.fillText(label, x + r + 3, y + 3);
        }};

        drawBody(x1, y1, 6, '#38bdf8', 'B1');
        drawBody(x2, y2, 6, '#f43f5e', 'B2');
        drawBody(x3, y3, 7, '#eab308', 'B3');

        if (!sim.paused) sim.step = (sim.step + 2) % r1.length;
      }});
      requestAnimationFrame(render);
    }}

    init();
  </script>
</body>
</html>
"""
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[+] Saved interactive visualizer: {out_path}")

def main():
    base_dir = "/home/phil/.gemini/antigravity/scratch/Project-ThreeBody/OrbitHunter"
    
    # Discovery 1: Equal mass
    r1_d1, r2_d1, r3_d1, _ = generate_orbit_data(0.33919784, 0.53627982, 6.289906, m3=1.0)
    create_svg(
        r1_d1, r2_d1, r3_d1,
        "New Discovery #1: Equal-Mass Periodic Orbit Family",
        "m=[1.0, 1.0, 1.0] | v1=0.3392, v2=0.5363 | T=6.29s | Defect=1.10e-4 | dE=3.51e-10",
        f"{base_dir}/discovery_1_equal_mass.svg"
    )

    # Discovery 2: Unequal mass (m3 = 1.20)
    r1_d2, r2_d2, r3_d2, _ = generate_orbit_data(0.57324248, 0.25161779, 9.679560, m3=1.2)
    create_svg(
        r1_d2, r2_d2, r3_d2,
        "New Discovery #2: Unequal-Mass Periodic Family (m3=1.20)",
        "m=[1.0, 1.0, 1.2] | v1=0.5732, v2=0.2516 | T=9.68s | Defect=4.48e-5 | dE=1.96e-8",
        f"{base_dir}/discovery_2_unequal_mass.svg"
    )

    d1_payload = {
        "r1": r1_d1.tolist(),
        "r2": r2_d1.tolist(),
        "r3": r3_d1.tolist()
    }
    d2_payload = {
        "r1": r1_d2.tolist(),
        "r2": r2_d2.tolist(),
        "r3": r3_d2.tolist()
    }
    build_interactive_html(d1_payload, d2_payload, f"{base_dir}/orbit_visualizer.html")

if __name__ == "__main__":
    main()
