"""Causal filters and spike metrics for F/T time series."""

from __future__ import annotations

import numpy as np
from scipy import signal


def lowpass_iir1(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """First-order causal low-pass (exponential smoothing), cutoff ``fc`` Hz at rate ``fs``."""
    x = np.asarray(x, dtype=np.float64)
    dt = 1.0 / fs
    rc = 1.0 / (2 * np.pi * fc)
    alpha = dt / (rc + dt)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def butter_lowpass(x: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    """Causal Butterworth low-pass (``lfilter``, not ``filtfilt`` — no look-ahead)."""
    x = np.asarray(x, dtype=np.float64)
    b, a = signal.butter(order, fc / (fs / 2), btype="low")
    zi = signal.lfilter_zi(b, a)
    y, _ = signal.lfilter(b, a, x, axis=0, zi=np.outer(zi, x[0]) if x.ndim > 1 else zi * x[0])
    return y


def contact_onset_index(fmag: np.ndarray, threshold: float, baseline_n: int = 10) -> int | None:
    """First index where |F| - baseline exceeds ``threshold``. None if never."""
    fmag = np.asarray(fmag)
    base = float(np.median(fmag[:baseline_n])) if len(fmag) >= baseline_n else 0.0
    idx = np.nonzero(fmag - base > threshold)[0]
    return int(idx[0]) if len(idx) else None


def spike_metrics(
    fmag: np.ndarray,
    t: np.ndarray,
    onset: int | None,
    settle_frac: float = 0.25,
    window_s: float = 0.1,
) -> dict:
    """Onset-transient metrics.

    ``steady`` is the median of the last ``settle_frac`` of the series. The *spike* is the
    excess of |F| over ``steady`` within ``window_s`` after ``onset`` (so a slow quasi-static
    wind-up is not counted as a spike); ``spike_width_s`` is the time that excess stays above
    half its peak inside the window.
    """
    fmag = np.asarray(fmag, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = len(fmag)
    steady = float(np.median(fmag[int(n * (1 - settle_frac)) :])) if n else float("nan")
    peak = float(np.max(fmag)) if n else float("nan")
    if onset is None or onset >= n:
        return {
            "peak": peak,
            "spike_height": 0.0,
            "spike_width_s": 0.0,
            "steady": steady,
            "t_peak": float("nan"),
        }
    win = (t >= t[onset]) & (t <= t[onset] + window_s)
    excess = np.where(win, fmag - steady, -np.inf)
    ipk = int(np.argmax(excess))
    height = max(float(excess[ipk]), 0.0)
    if height <= 0:
        return {
            "peak": peak,
            "spike_height": 0.0,
            "spike_width_s": 0.0,
            "steady": steady,
            "t_peak": float(t[ipk]),
        }
    above = win & (fmag - steady >= 0.5 * height)
    width = float(np.sum(above) * np.median(np.diff(t))) if n > 1 else 0.0
    return {
        "peak": peak,
        "spike_height": height,
        "spike_width_s": width,
        "steady": steady,
        "t_peak": float(t[ipk]),
    }
