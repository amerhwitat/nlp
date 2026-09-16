# Crypot App — Cryptography Research Lab

> Safety note: this package intentionally does **not** implement private-key brute forcing, random wallet hunting, credential recovery, automated sweeping of funds, or API-driven searching for wallets that are not owned by the operator. Those workflows can be used to gain unauthorized access to cryptocurrency assets.

This directory provides safe research equivalents for the requested components:

1. `crypto_research_lab.py` — Tkinter/ttk GUI for SHA-256, SHA-512, SHA3-256 and SHA3-512 hashing.
2. `wallet_address_lab.py` — deterministic address/public-key research from a **user-supplied private key** with explicit network selection; no key generation or sweeping.
3. `address_target_analyzer.py` — validates, normalizes, deduplicates and summarizes public Bitcoin target addresses supplied by the user. It never derives keys or searches private-key space.
4. `target_addresses.example.txt` — safe input template. Replace it locally with addresses you are authorized to analyze; do not commit secrets.

## Installation

```bash
python -m pip install -r requirements.txt
```

Optional Ethereum/Polygon research dependencies are deliberately not required for the GUI. If you add an RPC client, use a provider and account that you control and keep credentials in environment variables rather than source files.

## Run

```bash
python crypto_research_lab.py
python wallet_address_lab.py
python address_target_analyzer.py target_addresses.txt
```

## Security design

- No random 256-bit private-key generation loop.
- No sequential private-key enumeration.
- No balance-triggered transaction construction.
- No hardcoded recipient/private-key secrets.
- No automatic transfer/sweep behavior.
- No embedded third-party target-address corpus.
- Public address analysis is allowed only as data validation/research.

The tools are intended for cryptography education, address-format research, auditing of addresses you control, and reproducible experiments using known test vectors.
