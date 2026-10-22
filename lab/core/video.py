"""Video helper for the lab: stream RGB frames into an H.264 MP4 (imageio + imageio-ffmpeg)."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def frames_to_mp4(frames_iter: Iterable[np.ndarray], path: str | Path, fps: int = 20) -> None:
    """Write (H, W, 3) uint8 frames to `path` as a browser-playable MP4 (yuv420p, libx264).

    Frames are consumed lazily, so a generator that simulates while it renders never holds the
    whole video in memory. The file appears atomically (written to a temp name, then renamed);
    sizes need not be multiples of 16. Raises ValueError if the iterator yields no frame.
    """
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part.mp4")
    n = 0
    w = imageio.get_writer(
        tmp, fps=fps, codec="libx264", quality=7, macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )  # fmt: skip
    try:
        for f in frames_iter:
            w.append_data(np.ascontiguousarray(f))
            n += 1
    finally:
        w.close()
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError("frames_to_mp4: no frames")
    os.replace(tmp, path)
