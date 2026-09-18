"""Gravity + inertial compensation of a wrist F/T sensor.

Model (sensor-site frame S, R = rotation S -> world, g = world gravity vector):

    F_meas = F_contact_on_load + m * Rᵀ(a_site + ω̇×c + ω×(ω×c) - g)     … (1)
    T_meas = T_contact          + c × (m Rᵀ(a - g)) + I ω̇ + ω×Iω          … (2)

where the sensor reports the wrench the *parent* applies to the *load* (M1 sign convention), m is
the mass distal to the sensor, c its COM offset in S, a the site's linear acceleration (world).
Rearranged, the contact wrench on the load is

    F_contact = F_meas − m Rᵀ(a − g)                      (linear-accel + gravity terms)
    T_contact = T_meas − c × (m Rᵀ(a − g))                (torque of the same force at the COM)

The rotational inertia terms are dropped (Track A has no rotation except the tilt experiment,
which is slow). Sign check at rest: a = 0, Rᵀ(−g) = +g_mag·ẑ, so F_meas = m·g·ẑ = +0.981 N ✓.

Identification (`identify_mass_com`): with the load at rest in several orientations,
F_meas = m·Rᵀ(−g) is linear in m, and T_meas = c × F_meas is linear in c, so both are ordinary
least-squares problems — the same procedure used on a real robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G_WORLD = np.array([0.0, 0.0, -9.81])


@dataclass
class LoadParams:
    mass: float  # kg
    com: np.ndarray  # (3,) COM offset from the sensor site, sensor frame, m
    residual_force_rms: float = float("nan")
    residual_torque_rms: float = float("nan")


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def identify_mass_com(
    ft_static: np.ndarray, R_static: np.ndarray, g: np.ndarray = G_WORLD
) -> LoadParams:
    """Least-squares mass + COM from N static readings.

    ft_static: (N,6) sensor-frame wrenches at rest; R_static: (N,3,3) site->world rotations.
    Needs at least 2 distinct orientations for the COM to be observable (one for the mass).
    """
    ft_static = np.asarray(ft_static, dtype=np.float64)
    R_static = np.asarray(R_static, dtype=np.float64)
    n = len(ft_static)
    # force: F = m * (Rᵀ(-g))  -> stack to (3N,) = m * (3N,)
    gs = np.stack([R.T @ (-g) for R in R_static])  # (N,3)
    A = gs.reshape(-1, 1)
    b = ft_static[:, :3].reshape(-1)
    m = float(np.linalg.lstsq(A, b, rcond=None)[0][0])
    f_res = b - A[:, 0] * m
    # torque: T = c × F = -[F]_x c  -> stack (3N,3) @ c
    F = ft_static[:, :3]
    A_t = np.concatenate([-_skew(f) for f in F], axis=0)  # (3N,3)
    b_t = ft_static[:, 3:].reshape(-1)
    c, *_ = np.linalg.lstsq(A_t, b_t, rcond=None)
    t_res = b_t - A_t @ c
    return LoadParams(
        mass=m,
        com=np.asarray(c),
        residual_force_rms=float(np.sqrt(np.mean(f_res**2)) * np.sqrt(3)) if n else float("nan"),
        residual_torque_rms=float(np.sqrt(np.mean(t_res**2)) * np.sqrt(3)) if n else float("nan"),
    )


def gravity_wrench(load: LoadParams, R: np.ndarray, g: np.ndarray = G_WORLD) -> np.ndarray:
    """Predicted static sensor reading (sensor frame) for the load at orientation R."""
    f = load.mass * (np.asarray(R).T @ (-g))
    return np.concatenate([f, np.cross(load.com, f)])


def inertial_wrench(load: LoadParams, R: np.ndarray, a_world: np.ndarray) -> np.ndarray:
    """Extra sensor reading due to linear acceleration ``a_world`` of the site."""
    f = load.mass * (np.asarray(R).T @ np.asarray(a_world))
    return np.concatenate([f, np.cross(load.com, f)])


def compensate(
    ft_raw: np.ndarray,
    R: np.ndarray,
    load: LoadParams,
    a_world: np.ndarray | None = None,
    g: np.ndarray = G_WORLD,
) -> np.ndarray:
    """Contact wrench on the load (sensor frame) = raw − gravity − inertial. Single sample."""
    out = np.asarray(ft_raw, dtype=np.float64) - gravity_wrench(load, R, g)
    if a_world is not None:
        out = out - inertial_wrench(load, R, a_world)
    return out


def compensate_series(
    ft_raw: np.ndarray,
    R: np.ndarray,
    load: LoadParams,
    a_world: np.ndarray | None = None,
    g: np.ndarray = G_WORLD,
) -> np.ndarray:
    """Batched version. ft_raw (T,6), R (T,3,3), a_world (T,3) or None."""
    T = len(ft_raw)
    out = np.empty((T, 6))
    for i in range(T):
        out[i] = compensate(ft_raw[i], R[i], load, None if a_world is None else a_world[i], g)
    return out


def estimate_accel(t: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Causal finite-difference acceleration from a velocity series (T,3) -> (T,3)."""
    t = np.asarray(t, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    a = np.zeros_like(vel)
    if len(t) > 1:
        dt = np.diff(t)[:, None]
        a[1:] = np.diff(vel, axis=0) / dt
    return a
