"""Policy server / client over TCP (stdlib ``multiprocessing.connection``), PLAN §13 V0.

The simulator (sim image: Python 3.11, numpy < 2) and the policy (VLA image: Python 3.12,
numpy >= 2) run in different containers. Arrays are sent as plain ``(dtype, shape, bytes)``
records, never as pickled ``ndarray``s: a numpy-2 pickle references ``numpy._core``, which numpy
1.26 cannot import. Messages are dicts: ``{"type": "reset"}`` or ``{"type": "act", "obs": ...}``;
replies are ``{"ok": True, ...}`` or ``{"ok": False, "error": str}``.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from multiprocessing.connection import Client, Listener
from typing import Any

import numpy as np

_ND = "__ndarray__"


def pack(obj: Any) -> Any:
    """Recursively replace ndarrays by version-independent records."""
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        return {_ND: True, "dtype": a.dtype.str, "shape": list(a.shape), "data": a.tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(pack(v) for v in obj)
    return obj


def unpack(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get(_ND):
            a = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
            return a.reshape(obj["shape"]).copy()
        return {k: unpack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(unpack(v) for v in obj)
    return obj


class PolicyServer:
    """Serves ``act(obs) -> action chunk`` and ``reset()`` to one client at a time."""

    def __init__(
        self,
        act: Callable[[dict], np.ndarray],
        reset: Callable[[], None] | None = None,
        host: str = "0.0.0.0",
        port: int = 6010,
        authkey: bytes = b"fvb",
    ):
        self.act, self.reset = act, reset or (lambda: None)
        self.address, self.authkey = (host, port), authkey

    def serve(self, max_connections: int | None = None) -> None:
        with Listener(self.address, authkey=self.authkey) as listener:
            n = 0
            while max_connections is None or n < max_connections:
                with listener.accept() as conn:
                    n += 1
                    self._handle(conn)

    def _handle(self, conn) -> None:
        while True:
            try:
                msg = unpack(conn.recv())
            except (EOFError, ConnectionResetError):
                return
            try:
                if msg["type"] == "reset":
                    self.reset()
                    reply = {"ok": True}
                elif msg["type"] == "act":
                    t0 = time.perf_counter()
                    action = np.asarray(self.act(msg["obs"]), dtype=np.float32)
                    reply = {"ok": True, "action": action, "infer_s": time.perf_counter() - t0}
                elif msg["type"] == "close":
                    conn.send(pack({"ok": True}))
                    return
                else:
                    reply = {"ok": False, "error": f"unknown message type {msg['type']!r}"}
            except Exception as e:  # noqa: BLE001  (report to the client, keep serving)
                reply = {"ok": False, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
            conn.send(pack(reply))


class PolicyClient:
    """Sim-side handle. Raises TimeoutError if the server does not answer within ``timeout``
    seconds, so an infrastructure stall is never mistaken for a policy failure."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6010,
        authkey: bytes = b"fvb",
        timeout: float = 1.0,
        connect_retries: int = 50,
    ):
        last = None
        for _ in range(connect_retries):
            try:
                self.conn = Client((host, port), authkey=authkey)
                break
            except (ConnectionRefusedError, OSError) as e:
                last = e
                time.sleep(0.2)
        else:
            raise ConnectionError(f"cannot reach policy server {host}:{port}: {last}")
        self.timeout = timeout
        self.last_infer_s = float("nan")

    def _call(self, msg: dict) -> dict:
        self.conn.send(pack(msg))
        if not self.conn.poll(self.timeout):
            raise TimeoutError(f"policy server did not answer within {self.timeout} s")
        reply = unpack(self.conn.recv())
        if not reply.get("ok"):
            raise RuntimeError(f"policy server error: {reply.get('error')}")
        return reply

    def reset(self) -> None:
        self._call({"type": "reset"})

    def act(self, obs: dict) -> np.ndarray:
        reply = self._call({"type": "act", "obs": obs})
        self.last_infer_s = reply.get("infer_s", float("nan"))
        return reply["action"]

    def close(self) -> None:
        try:
            self._call({"type": "close"})
        finally:
            self.conn.close()
