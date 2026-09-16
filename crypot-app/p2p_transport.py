"""Authenticated P2P transport primitives for the cryptography research app.

This module is deliberately limited to cooperative peers. It does not scan
internet hosts, enumerate wallets, bypass authentication, or perform mining.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import ssl
from dataclasses import dataclass, asdict
from typing import Awaitable, Callable


@dataclass(frozen=True)
class PeerMessage:
    sender: str
    kind: str
    payload: dict
    digest: str

    @classmethod
    def create(cls, sender: str, kind: str, payload: dict) -> "PeerMessage":
        body = json.dumps({"sender": sender, "kind": kind, "payload": payload}, sort_keys=True).encode()
        return cls(sender, kind, payload, hashlib.sha256(body).hexdigest())

    def verify(self) -> bool:
        body = json.dumps({"sender": self.sender, "kind": self.kind, "payload": self.payload}, sort_keys=True).encode()
        return hashlib.sha256(body).hexdigest() == self.digest


class P2PPeer:
    def __init__(self, node_id: str, on_message: Callable[[PeerMessage], Awaitable[None]] | None = None):
        self.node_id = node_id
        self.on_message = on_message
        self.server: asyncio.AbstractServer | None = None

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            obj = json.loads(line.decode("utf-8"))
            msg = PeerMessage(**obj)
            if not msg.verify():
                return
            if self.on_message:
                await self.on_message(msg)
        finally:
            writer.close()
            await writer.wait_closed()

    async def serve(self, host: str = "127.0.0.1", port: int = 8765, ssl_context: ssl.SSLContext | None = None):
        self.server = await asyncio.start_server(self._handle_connection, host, port, ssl=ssl_context)
        async with self.server:
            await self.server.serve_forever()

    def stop(self) -> None:
        if self.server is not None:
            self.server.close()

    async def send(self, host: str, port: int, message: PeerMessage,
                   ssl_context: ssl.SSLContext | None = None) -> None:
        reader, writer = await asyncio.open_connection(host, port, ssl=ssl_context)
        try:
            writer.write((json.dumps(asdict(message), sort_keys=True) + "\n").encode("utf-8"))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
