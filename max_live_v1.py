"""Small lifecycle wrapper for Hyponoia's Max/MSP OSC bridge."""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from pythonosc.udp_client import SimpleUDPClient


LISTEN_IP = "127.0.0.1"
LISTEN_PORT = 7401
MAX_IP = "127.0.0.1"
MAX_PORT = 7402


def udp_port_is_available(host: str = LISTEN_IP, port: int = LISTEN_PORT) -> bool:
    """Return True when Hyponoia can safely start its UDP receiver."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        probe.close()


class MaxLiveController:
    """Start and stop only the receiver process owned by this app session."""

    def __init__(
        self,
        project_dir: str | Path,
        *,
        popen: Callable[..., Any] = subprocess.Popen,
        port_probe: Callable[..., bool] = udp_port_is_available,
        client_factory: Callable[..., Any] = SimpleUDPClient,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self._popen = popen
        self._port_probe = port_probe
        self._client_factory = client_factory
        self.process: Any = None
        self._log_handle: Any = None

    @property
    def owns_running_receiver(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def snapshot(self) -> dict[str, Any]:
        available = self._port_probe(LISTEN_IP, LISTEN_PORT)
        active = self.owns_running_receiver or not available
        return {
            "receiver_active": active,
            "owned_by_app": self.owns_running_receiver,
            "listen": f"{LISTEN_IP}:{LISTEN_PORT}",
            "max_output": f"{MAX_IP}:{MAX_PORT}",
            "message": (
                "Hyponoia receiver is active and waiting for Max/MSP."
                if active
                else "Live connection is stopped."
            ),
        }

    def start(self) -> dict[str, Any]:
        if self.owns_running_receiver:
            return self.snapshot()
        if not self._port_probe(LISTEN_IP, LISTEN_PORT):
            result = self.snapshot()
            result["message"] = "Port 7401 is already active. Another Hyponoia receiver may be running."
            return result
        log_dir = self.project_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = (log_dir / "max_live.log").open("a", encoding="utf-8")
        try:
            self.process = self._popen(
                [sys.executable, str(self.project_dir / "generator_receiver.py")],
                cwd=self.project_dir,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:
            self._log_handle.close()
            self._log_handle = None
            raise
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        if self.owns_running_receiver:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        return self.snapshot()

    def send_test(self) -> dict[str, Any]:
        client = self._client_factory(MAX_IP, MAX_PORT)
        client.send_message("/hyponoia/test", 1)
        return {
            "sent": True,
            "address": "/hyponoia/test",
            "destination": f"{MAX_IP}:{MAX_PORT}",
        }
