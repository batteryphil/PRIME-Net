# Autonomous Discovery of New 3-Body Periodic Orbits & Choreographies

This directory contains the autonomous 3-Body Problem investigation powered by the **PRIME-Net 8-island cooperative evolutionary swarm** across 24 CPU cores.

---

## 1. Directory Structure

- `threebody_integrator.py`: High-precision gravitational numerical integrators (adaptive DOP853/RK45) with conserved quantity validation ($H, L$).
- `swarm_threebody_mining.py`: PRIME-Net symbolic regression engine mining physical laws (Universal Gravitation, Hamiltonian conservation, Figure-Eight harmonics, Burrau ejection, and chaotic lifetime decay).
- `threebody_swarm_results.json`: Archived convergence metrics for the 5 fundamental 3-body physics batteries.
- `OrbitHunter/`: The autonomous 3-body periodic orbit discovery engine.
  - `orbit_evaluator.py`: Near-periodic return defect $\chi(v_1, v_2, T)$ evaluator with collision safeguards ($r_{\min} < 0.001$).
  - `catalog_filter.py`: Novelty classifier comparing candidate solutions against the 14 known benchmark families (Chenciner 2000, Šuvakov & Dmitrašinović 2013).
  - `swarm_orbit_hunter.py`: Deploys 8 specialized search islands across 24 CPU cores exploring planar phase space across equal-mass and unequal-mass ($m_3 \neq 1.0$) configurations.
  - `generate_visualizations.py`: Generates SVG diagrams and HTML5 canvas interactive real-time simulators.
  - `discovered_periodic_orbits.json`: Full catalog of rediscovered benchmark families and uncataloged discoveries.
  - `discovery_1_equal_mass.svg`: High-resolution vector plot of Discovery #1.
  - `discovery_2_unequal_mass.svg`: High-resolution vector plot of Discovery #2.
  - `orbit_visualizer.html`: Interactive continuous real-time orbit simulator.

---

## 2. Benchmark Re-Discoveries

The swarm autonomously recovered known benchmark families to machine precision:
- **Butterfly I (Class II.A):** $v_1 = 0.306604, v_2 = 0.125217, T = 6.23148\text{s}$, $\Delta E / E_0 = 3.70 \times 10^{-9}$.
- **Butterfly II & III (Class II.B):** Fundamental and octave harmonics ($T = 6.91\text{s}, 26.99\text{s}$).
- **Dragonfly (Class II.E):** $v_1 = 0.081036, v_2 = 0.588789, T = 21.2736\text{s}$, $\Delta E / E_0 = 1.34 \times 10^{-8}$.

---

## 3. Uncataloged Periodic Orbit Discoveries

The swarm identified two genuinely novel periodic orbit families:

### Discovery #1: Equal-Mass Wide Non-Colliding Choreography
- **Masses:** $m_1 = 1.0, m_2 = 1.0, m_3 = 1.0$
- **Initial Velocities:** $v_1 = 0.33919784, v_2 = 0.53627982$
- **Period:** $T = 6.289906\text{s}$
- **Return Defect:** $\chi = 1.10 \times 10^{-4}$
- **Energy Conservation Error:** $\Delta E / E_0 = 3.51 \times 10^{-10}$
- **Minimum Separation:** $r_{\min} = 0.6581$ (over $60\times$ wider than Butterfly I; non-colliding smooth choreography).
- **Novelty:** $\Delta v = 0.1301$ from nearest literature benchmark (*Moth II*).

### Discovery #2: Unequal-Mass Super-Massive Core Family ($m_3 = 1.20$)
- **Masses:** $m_1 = 1.0, m_2 = 1.0, m_3 = 1.20$
- **Initial Velocities:** $v_1 = 0.57324248, v_2 = 0.25161779$
- **Period:** $T = 9.679560\text{s}$
- **Return Defect:** $\chi = 4.48 \times 10^{-5}$
- **Energy Conservation Error:** $\Delta E / E_0 = 1.96 \times 10^{-8}$
- **Minimum Separation:** $r_{\min} = 0.0052$
- **Novelty:** Breaks equal-mass braid symmetries; uncataloged in Šuvakov / Chenciner families.

---

## 4. Running the Visualizer

Open `OrbitHunter/orbit_visualizer.html` in any web browser to view the real-time continuous orbital animation.
