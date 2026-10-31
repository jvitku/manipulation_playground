import importlib

import numpy as np
import pytest


def test_load_mnist_shapes():
    from lab.tasks.mnist_data import load_mnist

    for split, n in (("train", 60000), ("test", 10000)):
        x, y = load_mnist(split)
        assert x.shape == (n, 28, 28) and x.dtype == np.uint8
        assert y.shape == (n,) and y.dtype == np.uint8
        assert int(y.min()) == 0 and int(y.max()) == 9


def test_missing_dir_raises(monkeypatch, tmp_path):
    from lab.tasks.mnist_data import load_mnist

    monkeypatch.setenv("LAB_MNIST_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="LAB_MNIST_DIR"):
        load_mnist("train")


def test_import_without_files(monkeypatch):
    monkeypatch.setenv("LAB_MNIST_DIR", "/nonexistent/mnist")
    import lab.tasks.mnist_data as m

    importlib.reload(m)
