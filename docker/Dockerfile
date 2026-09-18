FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/mpl

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libosmesa6 libegl1 libgles2 libglfw3 \
      ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
      "mujoco==3.3.0" "robosuite==1.5.2" "numpy<2" \
      scipy matplotlib imageio imageio-ffmpeg h5py tqdm pytest ruff
# robosuite asks for a private macros file; create it once at build time
RUN python -c "import robosuite, os, runpy; p=os.path.join(os.path.dirname(robosuite.__file__),'scripts','setup_macros.py'); runpy.run_path(p, run_name='__main__')" || true
# caches must be writable by the (non-root) uid that `make` runs the container as
RUN mkdir -p /tmp/numba /tmp/mpl && chmod -R 1777 /tmp/numba /tmp/mpl

COPY . .
RUN pip install --no-cache-dir -e .
CMD ["bash"]
