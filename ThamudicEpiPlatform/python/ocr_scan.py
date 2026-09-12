from __future__ import annotations
import argparse
import requests

BASE = 'http://127.0.0.1:8010/api'

def scan(path: str, engine: str = 'auto') -> dict:
    with open(path, 'rb') as fh:
        response = requests.post(f'{BASE}/ocr/scan', params={'engine': engine}, files={'file': (path, fh, 'application/octet-stream')}, timeout=120)
    response.raise_for_status()
    return response.json()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Scan an ancient-language image through ThamudicEpiPlatform')
    p.add_argument('image'); p.add_argument('--engine', default='auto', choices=['auto','kraken','tesseract'])
    print(scan(p.parse_args().image, p.parse_args().engine))
