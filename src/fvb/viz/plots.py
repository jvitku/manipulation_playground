"""Plot helpers. Every axis is labelled with a unit and a frame."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FT_LABELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_ft_timeseries(
    t: np.ndarray,
    ft: np.ndarray,
    path: str | Path,
    title: str = "",
    frame: str = "sensor-site",
    onset_t: float | None = None,
    t_hf: np.ndarray | None = None,
    ft_hf: np.ndarray | None = None,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i, lab in enumerate(FT_LABELS[:3]):
        if t_hf is not None and ft_hf is not None:
            axes[0].plot(t_hf, ft_hf[:, i], lw=0.5, alpha=0.5, color=f"C{i}")
        axes[0].plot(
            t, ft[:, i], label=lab, color=f"C{i}", marker="." if len(t) < 400 else None, ms=3, lw=1
        )
    axes[0].set_ylabel(f"force [N] ({frame} frame)")
    axes[0].legend(loc="best")
    for i, lab in enumerate(FT_LABELS[3:]):
        axes[1].plot(t, ft[:, 3 + i], label=lab, lw=1)
    axes[1].set_ylabel(f"torque [N·m] ({frame} frame)")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(loc="best")
    if onset_t is not None:
        for ax in axes:
            ax.axvline(onset_t, color="k", ls="--", lw=0.8, label="contact onset")
    axes[0].set_title(title)
    for ax in axes:
        ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_fmag(t, ft, path, title="", onset_t=None, t_hf=None, ft_hf=None) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if t_hf is not None and ft_hf is not None:
        ax.plot(
            t_hf, np.linalg.norm(ft_hf[:, :3], axis=1), lw=0.5, alpha=0.6, label="|F| physics rate"
        )
    ax.plot(t, np.linalg.norm(ft[:, :3], axis=1), lw=1.2, label="|F| control rate")
    if onset_t is not None:
        ax.axvline(onset_t, color="k", ls="--", lw=0.8, label="contact onset")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|F| [N]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_force_vs_depth(depth, ft, path, title="") -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(depth * 1e3, ft[:, 2], ".-", ms=3, lw=0.8)
    ax.set_xlabel("insertion depth [mm] (peg tip below hole rim; negative = above)")
    ax.set_ylabel("Fz [N] (sensor-site frame)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_lateral_vs_offset(offsets_mm, fx, fy, path, title="") -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(offsets_mm, fx, "o-", label="steady Fx")
    ax.plot(offsets_mm, fy, "s-", label="steady Fy")
    ax.set_xlabel("lateral x offset [mm]")
    ax.set_ylabel("force [N] (sensor-site frame)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_heatmap(
    mat, xlabels, ylabels, path, title="", xlabel="", ylabel="", cbar="", fmt="{:.1f}"
) -> Path:
    fig, ax = plt.subplots(figsize=(1.2 * len(xlabels) + 3, 0.6 * len(ylabels) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xlabels)), [str(x) for x in xlabels])
    ax.set_yticks(range(len(ylabels)), [str(y) for y in ylabels])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label=cbar)
    return _save(fig, path)


def write_mp4(frames: list[np.ndarray], path: str | Path, fps: int = 30) -> Path | None:
    """Write frames (H,W,3 uint8) to MP4. Returns None if ffmpeg/imageio fails."""
    try:
        import imageio.v2 as imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        w = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=None)
        for f in frames:
            w.append_data(f)
        w.close()
        return path
    except Exception as e:  # noqa: BLE001
        print(f"[viz] mp4 write failed: {type(e).__name__}: {e}")
        return None


def plot_peak_force_distribution(peaks_by_label: dict[str, tuple[list, list]], path, title=""):
    """Histogram of per-episode peak |F|: control-rate (ft_comp) vs physics-rate (ft_raw_hf)."""
    fig, axes = plt.subplots(
        1, len(peaks_by_label), figsize=(4.5 * len(peaks_by_label), 3.6), squeeze=False
    )
    for ax, (label, (pk_c, pk_hf)) in zip(axes[0], peaks_by_label.items(), strict=True):
        pk_c, pk_hf = np.asarray(pk_c), np.asarray(pk_hf)
        hi = max(pk_hf.max(), pk_c.max()) if len(pk_c) else 1.0
        bins = np.linspace(0, hi * 1.05, 20)
        ax.hist(
            pk_hf, bins=bins, alpha=0.6, label=f"physics rate (median {np.median(pk_hf):.1f} N)"
        )
        ax.hist(pk_c, bins=bins, alpha=0.6, label=f"control rate (median {np.median(pk_c):.1f} N)")
        ax.set_xlabel("episode peak |F| [N] (compensated / raw hf, sensor frame)")
        ax.set_ylabel("episodes")
        ax.set_title(f"{label} (n={len(pk_c)})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    return _save(fig, path)
