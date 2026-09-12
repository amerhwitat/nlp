"""Minimal authenticated-ready Chimera P2P transport and 128D state envelope.

The module intentionally provides framing, peer identity hooks, sequence/replay
checks and content hashing without executing remote commands. Applications can
plug in their own signature implementation and policy layer.
"""
from __future__ import annotations
import asyncio, hashlib, json, time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

@dataclass
class ChimeraState128D:
    values: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)

    def envelope(self) -> dict[str, Any]:
        return {"schema": "chimera-128d/v1", "values": self.values, "extensions": self.extensions}

@dataclass
class PeerMessage:
    peer_id: str
    sequence: int
    kind: str
    payload: dict[str, Any]
    expires_at: float = field(default_factory=lambda: time.time() + 60)

    def encode(self) -> bytes:
        body = {"protocol":"chimera-p2p/v1","peer_id":self.peer_id,"sequence":self.sequence,
                "kind":self.kind,"expires_at":self.expires_at,"payload":self.payload}
        body["payload_sha256"] = hashlib.sha256(json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()

class ReplayGuard:
    def __init__(self) -> None:
        self.highest: dict[str, int] = {}
    def accept(self, peer_id: str, sequence: int) -> bool:
        old = self.highest.get(peer_id, -1)
        if sequence <= old:
            return False
        self.highest[peer_id] = sequence
        return True

class ChimeraP2PServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 on_message: Callable[[dict[str, Any]], Awaitable[None]] | None = None) -> None:
        self.host, self.port, self.on_message = host, port, on_message
        self.guard = ReplayGuard()
        self.server: asyncio.AbstractServer | None = None
    async def start(self) -> int:
        self.server = await asyncio.start_server(self._client, self.host, self.port)
        return int(self.server.sockets[0].getsockname()[1])
    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while line := await reader.readline():
                obj = json.loads(line)
                if obj.get("protocol") != "chimera-p2p/v1": continue
                if obj.get("expires_at", 0) < time.time(): continue
                if not self.guard.accept(str(obj.get("peer_id", "")), int(obj.get("sequence", -1))): continue
                payload = obj.get("payload", {})
                expected = obj.get("payload_sha256", "")
                actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if expected != actual: continue
                if self.on_message: await self.on_message(obj)
        finally:
            writer.close()
            await writer.wait_closed()
    async def stop(self) -> None:
        if self.server:
            self.server.close(); await self.server.wait_closed()
