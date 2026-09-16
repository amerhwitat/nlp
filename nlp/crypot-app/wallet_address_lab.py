#!/usr/bin/env python3
"""Safe wallet-address research utility.

Accepts a private key explicitly supplied by the operator and derives an
Ethereum address locally. It does not generate/search private keys, query
balances, or create/sign/broadcast transactions.
"""
from __future__ import annotations
import argparse
import hashlib


def keccak256(data: bytes) -> bytes:
    # Python's stdlib sha3_256 is standardized SHA-3, not legacy Keccak-256.
    # Prefer eth-hash when installed for Ethereum-compatible derivation.
    try:
        from eth_hash.auto import keccak
    except ImportError as exc:
        raise RuntimeError("Install eth-hash for Ethereum Keccak-256 support") from exc
    return keccak(data)


def ethereum_address(private_key_hex: str) -> str:
    from ecdsa import SECP256k1, SigningKey
    raw = private_key_hex.removeprefix("0x")
    if len(raw) != 64:
        raise ValueError("Private key must be exactly 32 bytes / 64 hex characters")
    key_bytes = bytes.fromhex(raw)
    if int.from_bytes(key_bytes, "big") == 0 or int.from_bytes(key_bytes, "big") >= SECP256k1.order:
        raise ValueError("Private key is outside the secp256k1 valid range")
    signing_key = SigningKey.from_string(key_bytes, curve=SECP256k1)
    point = signing_key.verifying_key.pubkey.point
    uncompressed = b"\x04" + int(point.x()).to_bytes(32, "big") + int(point.y()).to_bytes(32, "big")
    address = keccak256(uncompressed[1:])[-20:]
    return "0x" + address.hex()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_key", help="Known test/owned private key; never place it in source control")
    args = parser.parse_args()
    print(ethereum_address(args.private_key))


if __name__ == "__main__":
    main()
