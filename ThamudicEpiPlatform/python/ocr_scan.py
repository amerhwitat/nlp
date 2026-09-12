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
    parser = argparse.ArgumentParser(description='Scan an ancient-language image through ThamudicEpiPlatform')
    parser.add_argument('image')
    parser.add_argument('--engine', default='auto', choices=['auto','kraken','tesseract'])
    args = parser.parse_args()
    print(scan(args.image, args.engine))
