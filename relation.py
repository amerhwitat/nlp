# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 18:28:08 2026

@author: PC1
"""

import hashlib
import ecdsa
import base58


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def ripemd160(data: bytes) -> bytes:
    h = hashlib.new('ripemd160')
    h.update(data)
    return h.digest()


def private_key_hex_to_public_key(priv_hex: str, compressed=True) -> bytes:
    priv_bytes = bytes.fromhex(priv_hex)

    sk = ecdsa.SigningKey.from_string(priv_bytes, curve=ecdsa.SECP256k1)
    vk = sk.verifying_key
    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()

    if compressed:
        prefix = b'\x02' if y % 2 == 0 else b'\x03'
        return prefix + x.to_bytes(32, 'big')
    else:
        return b'\x04' + x.to_bytes(32, 'big') + y.to_bytes(32, 'big')


def public_key_to_address(pubkey: bytes, testnet=False) -> str:
    pubkey_hash = ripemd160(sha256(pubkey))

    version = b'\x6f' if testnet else b'\x00'
    payload = version + pubkey_hash

    checksum = sha256(sha256(payload))[:4]
    address_bytes = payload + checksum

    return base58.b58encode(address_bytes).decode()


def hex_private_key_to_btc_address(priv_hex: str, compressed=True, testnet=False) -> str:
    pubkey = private_key_hex_to_public_key(priv_hex, compressed)
    return public_key_to_address(pubkey, testnet)


# ------------------- Example -------------------

if __name__ == "__main__":
    priv_hex = "1e99423a4ed27608a15a2616cf4c0b5b04f1f4b8fba6b7c5b4d1f8f5a0a12345"

    addr = hex_private_key_to_btc_address(priv_hex, compressed=True)
    print("Bitcoin Address:", addr)
