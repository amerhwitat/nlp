# Crypto Research App (`crypot-app`)

Unified Python research application with a single Tkinter GUI for the local
cryptography, public-address research, cooperative networking, client/server,
and RNN/LLM modules.

## GUI

Launch the full interface with:

```bash
python app.py
# or
python app.py --gui
# or
python crypot_gui.py
```

The GUI provides active tabs for:

- **Dashboard** — module readiness and safety boundary.
- **Crypto Lab** — SHA-256, SHA-512, SHA3-256 and SHA3-512 hashing.
- **Address Research** — offline public Bitcoin/EVM address-shape validation.
- **P2P Network** — cooperative asyncio peer listener, stop control, JSON
  payloads, content-digest verification, and live messages.
- **Client / Server** — local HTTP research server plus client job submission
  for `hash`, `address_validate`, and `model_inference` operations.
- **RNN / LLM** — 16-dimensional recurrent-state analysis and the optional
  injected LLM adapter.
- **Logs / Export** — live timestamped activity and text-log export.

## Architecture

```text
crypot-app/
├── app.py
├── crypot_gui.py
├── gui_support.py
├── crypto_research_lab.py
├── p2p_transport.py
├── client_server.py
├── rnn_llm_module.py
├── tests/
│   └── test_gui_support.py
├── requirements.txt
└── README.md
```

The networking layer is intentionally cooperative: peers connect to explicitly
configured endpoints and exchange content-verified messages. The application
does not perform Internet-wide scanning, credential attacks, private-key
brute forcing, wallet draining, or automatic transfer of discovered funds.

The RNN/LLM layer is an integration facade. A host application may inject its
own model callable; no external model service is contacted by default.

## CLI compatibility

The original command-line functions remain available:

```bash
python app.py "hello" --hash
python app.py "network research" --analyze
python crypto_research_lab.py
```

## Testing

The GUI's pure helper layer has tests under `tests/`. Run them with:

```bash
python -m pytest crypot-app/tests
```

For production networking, place the P2P transport behind TLS and an
application-level authentication/authorization layer. Do not transmit private
keys or wallet seed material through the network interface.
