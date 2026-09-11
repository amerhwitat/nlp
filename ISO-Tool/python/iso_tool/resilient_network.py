"""Connectivity monitoring and retry scheduling for ISO-Tool acquisition jobs."""
from __future__ import annotations

from dataclasses import dataclass
import socket
import time
from typing import Callable


@dataclass
class NetworkState:
    online: bool
    checked_at: float
    detail: str


class ConnectivityMonitor:
    def __init__(self, interval: float = 15.0, host: str = "github.com", port: int = 443, timeout: float = 3.0):
        self.interval = max(1.0, interval)
        self.host, self.port, self.timeout = host, port, timeout
        self.last = NetworkState(False, 0.0, "not checked")

    def check(self) -> NetworkState:
        try:
            with socket.create_connection((self.host, self.port), self.timeout):
                self.last = NetworkState(True, time.time(), f"reachable: {self.host}:{self.port}")
        except OSError as exc:
            self.last = NetworkState(False, time.time(), f"offline: {type(exc).__name__}: {exc}")
        return self.last

    def wait_until_online(self, log: Callable[[str], None] | None = None, stop: Callable[[], bool] | None = None) -> NetworkState:
        while True:
            state = self.check()
            if state.online:
                if log: log(f"[network] online — {state.detail}")
                return state
            if log: log(f"[network] connection unavailable — retrying in {self.interval:g}s")
            if stop and stop():
                raise RuntimeError("network retry cancelled")
            time.sleep(self.interval)


def retry_when_online(operation: Callable[[], object], monitor: ConnectivityMonitor, log=None, max_attempts: int | None = None, stop=None):
    """Retry acquisition after connectivity returns; None means retry indefinitely."""
    attempts = 0
    while True:
        if stop and stop():
            raise RuntimeError("retry cancelled")
        try:
            return operation()
        except (OSError, TimeoutError, ConnectionError) as exc:
            attempts += 1
            if max_attempts is not None and attempts >= max_attempts:
                raise
            if log: log(f"[network] operation failed ({type(exc).__name__}); waiting for connectivity")
            monitor.wait_until_online(log=log, stop=stop)
        except Exception:
            # Non-network errors are not retried blindly.
            raise
