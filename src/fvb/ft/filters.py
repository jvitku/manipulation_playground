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
    fmag: np.ndarray, t: np.ndarray, onset: int | None, settle_frac: float = 0.25
) -> dict:
    """Peak height above the eventual steady value, and width at half that height.

    ``steady`` is the median of the last ``settle_frac`` of the series.
    """
    fmag = np.asarray(fmag, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = len(fmag)
    steady = float(np.median(fmag[int(n * (1 - settle_frac)) :])) if n else float("nan")
    if onset is None or onset >= n:
        return {
            "peak": float(np.max(fmag)) if n else float("nan"),
            "spike_height": 0.0,
            "spike_width_s": 0.0,
            "steady": steady,
            "t_peak": float("nan"),
        }
    seg = fmag[onset:]
    ipk = int(np.argmax(seg)) + onset
    peak = float(fmag[ipk])
    height = peak - steady
    half = steady + 0.5 * height
    above = fmag[onset:] >= half
    # width of the contiguous region around the peak that stays above half height
    lo = ipk
    while lo > onset and above[lo - onset - 1]:
        lo -= 1
    hi = ipk
    while hi + 1 < n and above[hi + 1 - onset]:
        hi += 1
    width = float(t[hi] - t[lo]) if height > 0 else 0.0
    return {
        "peak": peak,
        "spike_height": float(height),
        "spike_width_s": width,
        "steady": steady,
        "t_peak": float(t[ipk]),
    }
