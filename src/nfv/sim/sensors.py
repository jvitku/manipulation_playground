"""Wrist wrench pipeline: raw MuJoCo sensor -> tool-weight compensation -> 200 Hz low-pass.

The compensation math is imported from ``fvb.ft.compensate`` (the Stage-0 module this course
builds on): the raw sensor reads the wrench the parent applies to the load, so at rest it shows
the tool weight ``m·Rᵀ(−g)``; subtracting :func:`fvb.ft.compensate.gravity_wrench` leaves the
contact wrench. Because this is a simulator we take the load parameters from the model itself
instead of least-squares identification: the mass is ``body_subtreemass`` of the tool body
(constant) and the COM is ``data.subtree_com`` (exact every step, so finger motion cannot bias
the compensation). Linear-acceleration terms are not compensated — the tool moves at <= 0.5 m/s
with ~0.2 kg distal mass, so inertial spikes stay well under the 1 N tolerances of M0 and the
low-pass soaks the transients.

Sign convention (the course's, documented once here and unit-tested in tests/nfv):
    The compensated wrench is the EXTERNAL CONTACT wrench expressed in the wrist-site frame
    (site axes = world axes while yaw = 0), with the sign such that
      * contact pushing UP on the tool reads NEGATIVE F_z (pressing on a table: F_z < 0);
      * a mass resting ON the tool (pushing down) reads POSITIVE F_z = +m·9.81.

Filtering: a first-order causal low-pass (the online form of ``fvb.ft.filters.lowpass_iir1``,
same alpha formula) running at the 200 Hz controller rate, cutoff ``fc`` = 20 Hz by default.
The controller and the policy both consume the filtered stream; nothing downstream sees raws.

Shapes: a wrench is a (6,) float64 ``[fx, fy, fz, tx, ty, tz]`` in N / N·m.
"""

from __future__ import annotations

import mujoco
import numpy as np

from fvb.ft.compensate import LoadParams, compensate

CTRL_RATE_HZ = 200.0  # the stream's nominal rate (see controllers.py for the exact clock)
DEFAULT_CUTOFF_HZ = 20.0


class OnlineLowpass:
    """One-pole causal low-pass, per-channel; alpha as in fvb.ft.filters.lowpass_iir1."""

    def __init__(self, fc: float = DEFAULT_CUTOFF_HZ, fs: float = CTRL_RATE_HZ):
        dt = 1.0 / fs
        rc = 1.0 / (2 * np.pi * fc)
        self.alpha = dt / (rc + dt)
        self._y: np.ndarray | None = None

    def reset(self) -> None:
        self._y = None  # next sample re-seeds the state

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._y is None:
            self._y = x.copy()
        else:
            self._y = self._y + self.alpha * (x - self._y)
        return self._y.copy()


class WrenchSensor:
    """Compensated + filtered wrist wrench of the nfv gantry (one instance per model)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site: str = "wrist",
        body: str = "tool",
        force_sensor: str = "wrist_force",
        torque_sensor: str = "wrist_torque",
        fc: float = DEFAULT_CUTOFF_HZ,
    ):
        self.model, self.data = model, data
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        f_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, force_sensor)
        t_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, torque_sensor)
        if min(self._site_id, self._body_id, f_id, t_id) < 0:
            raise KeyError(f"missing site/body/sensor among {site, body, force_sensor}")
        self._f_adr = int(model.sensor_adr[f_id])
        self._t_adr = int(model.sensor_adr[t_id])
        # mass distal to the wrist: the tool body's subtree (palm + fingers + task tool geometry)
        self.load_mass = float(model.body_subtreemass[self._body_id])
        self._filter = OnlineLowpass(fc=fc)

    def reset(self) -> None:
        self._filter.reset()

    def raw(self) -> np.ndarray:
        """Unmodified sensordata wrench (6,), sensor-site frame."""
        f = self.data.sensordata[self._f_adr : self._f_adr + 3]
        t = self.data.sensordata[self._t_adr : self._t_adr + 3]
        return np.concatenate([f, t]).astype(np.float64)

    def _load_now(self) -> tuple[LoadParams, np.ndarray]:
        """Current load (exact COM from the model) and the site->world rotation R."""
        r = np.asarray(self.data.site_xmat[self._site_id], dtype=np.float64).reshape(3, 3)
        com_world = np.asarray(self.data.subtree_com[self._body_id], dtype=np.float64)
        site_world = np.asarray(self.data.site_xpos[self._site_id], dtype=np.float64)
        com_site = r.T @ (com_world - site_world)  # (3,) COM offset in the sensor frame
        return LoadParams(mass=self.load_mass, com=com_site), r

    def compensated(self) -> np.ndarray:
        """Contact wrench (6,) = raw − tool gravity wrench, sensor-site frame, unfiltered."""
        load, r = self._load_now()
        return compensate(self.raw(), r, load)

    def sample(self) -> np.ndarray:
        """One 200 Hz pipeline sample: compensated then low-passed. Call once per ctrl tick."""
        return self._filter(self.compensated())
