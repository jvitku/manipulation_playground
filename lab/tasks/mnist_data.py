"""MNIST loader: gzipped IDX files baked into the image, parsed with numpy only."""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np

_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 [N,28,28], labels uint8 [N]) for split "train" or "test"."""
    if split not in _FILES:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    root = Path(os.environ.get("LAB_MNIST_DIR", "/opt/mnist"))
    paths = [root / name for name in _FILES[split]]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"MNIST files not found ({', '.join(missing)}); "
            "set LAB_MNIST_DIR to a directory holding the IDX .gz files"
        )
    return _read_idx(paths[0]), _read_idx(paths[1])
