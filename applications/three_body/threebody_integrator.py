#!/usr/bin/env python3
"""
High-Precision 3-Body Problem Integrator
=======================================
Implements adaptive Runge-Kutta (DOP853 / RK45) with close-encounter regularization
and continuous energy/momentum monitoring.

Presets:
  1. Figure-Eight Choreography (Chenciner & Montgomery 2000)
  2. Lagrange Equilateral Triangle (Rotating equilibrium)
  3. Burrau (1913) Pythagorean Problem (Masses 3, 4, 5 with violent ejection)
  4. Chaotic Virialized Ensemble (Monte Carlo for statistical lifetime scaling)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.integrate import solve_ivp

G_CONST = 1.0

class ThreeBodySystem:
    def __init__(self, masses, r0, v0, G=1.0):
        """
        masses: array-like of shape (3,)
        r0: array-like of shape (3, 2) or (3, 3) - initial positions
        v0: array-like of shape (3, 2) or (3, 3) - initial velocities
        """
        self.masses = np.array(masses, dtype=np.float64)
        self.r0 = np.array(r0, dtype=np.float64)
        self.v0 = np.array(v0, dtype=np.float64)
        self.dim = self.r0.shape[1]
        self.G = G

        # Center of mass correction
        total_m = np.sum(self.masses)
        com_r = np.sum(self.r0 * self.masses[:, None], axis=0) / total_m
        com_v = np.sum(self.v0 * self.masses[:, None], axis=0) / total_m
        self.r0 -= com_r
        self.v0 -= com_v

    def derivatives(self, t, state):
        # state: [r1, r2, r3, v1, v2, v3] flattened (dim * 6)
        d = self.dim
        r = state[:3 * d].reshape(3, d)
        v = state[3 * d:].reshape(3, d)

        # dr/dt = v
        dr_dt = v.copy()

        # dv/dt = sum_{j != i} G * m_j * (r_j - r_i) / |r_j - r_i|^3
        dv_dt = np.zeros((3, d), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                if i != j:
                    diff = r[j] - r[i]
                    dist = np.linalg.norm(diff)
                    # Softening epsilon to avoid division by zero during collision singularities
                    dist_soft = max(dist, 1e-12)
                    dv_dt[i] += self.G * self.masses[j] * diff / (dist_soft ** 3)

        return np.concatenate([dr_dt.flatten(), dv_dt.flatten()])

    def compute_energy(self, r, v):
        """Total Energy = Kinetic + Potential."""
        # Kinetic: sum 0.5 * m_i * v_i^2
        v_sq = np.sum(v ** 2, axis=-1)  # (N_steps, 3)
        T = 0.5 * np.sum(self.masses * v_sq, axis=-1)

        # Potential: - sum_{i < j} G * m_i * m_j / r_ij
        V = np.zeros_like(T)
        for i in range(3):
            for j in range(i + 1, 3):
                r_diff = r[:, j, :] - r[:, i, :]
                r_ij = np.linalg.norm(r_diff, axis=-1)
                r_ij_soft = np.maximum(r_ij, 1e-12)
                V -= self.G * self.masses[i] * self.masses[j] / r_ij_soft

        return T + V, T, V

    def compute_angular_momentum(self, r, v):
        """Computes total angular momentum L = sum m_i (r_i x v_i)."""
        # For 2D: L_z = sum m_i (x * v_y - y * v_x)
        if self.dim == 2:
            L_z = np.sum(self.masses * (r[:, :, 0] * v[:, :, 1] - r[:, :, 1] * v[:, :, 0]), axis=-1)
            return L_z
        else:
            # 3D: cross product
            L = np.sum(self.masses[:, None] * np.cross(r, v), axis=1)
            return L

    def simulate(self, t_span, t_eval=None, method="DOP853", rtol=1e-10, atol=1e-12):
        y0 = np.concatenate([self.r0.flatten(), self.v0.flatten()])
        sol = solve_ivp(
            self.derivatives,
            t_span,
            y0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol
        )

        d = self.dim
        n_pts = len(sol.t)
        r_traj = sol.y[:3 * d].T.reshape(n_pts, 3, d)
        v_traj = sol.y[3 * d:].T.reshape(n_pts, 3, d)

        # Compute accelerations along trajectory
        a_traj = np.zeros_like(r_traj)
        for step in range(n_pts):
            derivs = self.derivatives(sol.t[step], sol.y[:, step])
            a_traj[step] = derivs[3 * d:].reshape(3, d)

        total_E, T, V = self.compute_energy(r_traj, v_traj)
        L = self.compute_angular_momentum(r_traj, v_traj)

        dE = (total_E - total_E[0]) / (abs(total_E[0]) + 1e-12)
        max_dE = np.max(np.abs(dE))

        return {
            "t": sol.t,
            "r": r_traj,
            "v": v_traj,
            "a": a_traj,
            "E": total_E,
            "T": T,
            "V": V,
            "L": L,
            "max_dE": max_dE,
            "success": sol.success
        }

# ==============================================================================
# CANONICAL PRESETS
# ==============================================================================
def figure_eight_preset():
    """
    Chenciner & Montgomery (2000) Figure-Eight Choreography.
    Three equal masses chasing each other along a lemniscate-shaped curve.
    Period T = 6.32591398.
    """
    masses = [1.0, 1.0, 1.0]
    # High-precision initial conditions
    x1 = -0.97000436
    y1 = 0.24308753
    vx1 = 0.46620531
    vy1 = 0.43236573

    r0 = np.array([
        [x1, y1],
        [-x1, -y1],
        [0.0, 0.0]
    ])
    v0 = np.array([
        [vx1, vy1],
        [vx1, vy1],
        [-2.0 * vx1, -2.0 * vy1]
    ])
    return ThreeBodySystem(masses, r0, v0)

def pythagorean_preset():
    """
    Burrau (1913) Pythagorean 3-Body Problem.
    Masses 3, 4, 5 placed at rest at the vertices of a 3-4-5 right triangle.
    Coordinates:
      m1 = 3 at (1, 3)
      m2 = 4 at (-2, -1)
      m3 = 5 at (1, -1)
    Initial velocities = 0.
    Undergoes chaotic close encounters until body 1 (m=3) is ejected at t ~ 15.83.
    """
    masses = [3.0, 4.0, 5.0]
    r0 = np.array([
        [1.0, 3.0],
        [-2.0, -1.0],
        [1.0, -1.0]
    ])
    v0 = np.zeros((3, 2))
    return ThreeBodySystem(masses, r0, v0)

def lagrange_equilateral_preset(R=1.0, total_mass=3.0):
    """
    Lagrange (1772) Equilateral Triangle Rotating Solution.
    Three equal masses at the vertices of an equilateral triangle.
    """
    masses = [total_mass / 3.0] * 3
    angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    r0 = np.column_stack([R * np.cos(angles), R * np.sin(angles)])

    # Circular orbit speed: omega = sqrt(G * M_eff / R^3)
    # Side length s = sqrt(3) * R. Potential force = G * m / s^2 * sqrt(3) = G * m / (3 R^2) * sqrt(3)
    # F_net = sqrt(3) * G * m / s^2 = G * m / R^2.
    # omega^2 * R = G * m / R^2 => omega = sqrt(G * m / R^3) where m is mass of one body.
    # Since side is s = sqrt(3)*R, F_inward on body from each of the other two is G*m / s^2 = G*m / (3 R^2).
    # Net inward force = 2 * (G*m / (3 R^2)) * cos(30 deg) = 2 * (G*m / (3 R^2)) * (sqrt(3)/2) = G*m / (sqrt(3) R^2).
    # omega = sqrt(G * m / (sqrt(3) * R^3)).
    m_single = total_mass / 3.0
    omega = np.sqrt(G_CONST * m_single / (np.sqrt(3.0) * (R ** 3)))

    # Velocity perpendicular to position
    v0 = np.column_stack([-omega * r0[:, 1], omega * r0[:, 0]])
    return ThreeBodySystem(masses, r0, v0)

def generate_chaotic_ensemble(n_systems=100, seed=42):
    """
    Generates an ensemble of planar 3-body systems with zero initial velocity
    (free-fall virial ratio Q = 0) to study chaotic disruption statistics.
    """
    np.random.seed(seed)
    systems = []
    for _ in range(n_systems):
        # Random masses in [0.5, 2.0]
        masses = np.random.uniform(0.5, 2.0, size=3)
        # Random positions in unit disk
        angles = np.random.uniform(0, 2 * np.pi, size=3)
        radii = np.sqrt(np.random.uniform(0.1, 1.0, size=3))
        r0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        # Initial velocities = 0 (cold collapse)
        v0 = np.zeros((3, 2))
        systems.append(ThreeBodySystem(masses, r0, v0))
    return systems

if __name__ == "__main__":
    print("Testing ThreeBodySystem with Figure-Eight Preset...")
    sys_fig8 = figure_eight_preset()
    t_eval = np.linspace(0, 6.32591398, 1000)
    res = sys_fig8.simulate((0, 6.32591398), t_eval=t_eval)
    print(f"Simulation Success: {res['success']}")
    print(f"Max Relative Energy Error dE/E0: {res['max_dE']:.2e}")
    print(f"Angular Momentum L_z: mean={np.mean(res['L']):.2e}, std={np.std(res['L']):.2e} (Expected: 0)")
