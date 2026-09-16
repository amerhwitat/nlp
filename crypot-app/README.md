# Crypto Research App (`crypot-app`)

Unified Python research application combining:

- Standard cryptographic hashing and Tkinter GUI.
- Public Bitcoin/EVM address validation and analysis.
- Cooperative P2P messaging over asyncio.
- Local client/server job submission.
- RNN-style state tracking and an optional LLM adapter.

## Architecture

```text
crypot-app/
├── app.py
├── crypto_research_lab.py
├── wallet_address_lab.py
├── address_target_analyzer.py
├── p2p_transport.py
├── client_server.py
├── rnn_llm_module.py
├── target_addresses.example.txt
├── requirements.txt
└── README.md
```

The networking layer is intentionally cooperative: peers connect to explicitly
configured endpoints and exchange signed-by-content messages. The application
does not perform Internet-wide scanning, credential attacks, private-key
brute forcing, wallet draining, or automatic transfer of discovered funds.

The RNN/LLM layer is an integration facade. A host application may inject its
own model callable; no external model service is contacted by default.

## Examples

```bash
python app.py "hello" --hash
python app.py "network research" --analyze
python crypto_research_lab.py
```

For production networking, place the P2P transport behind TLS and an
application-level authentication/authorization layer. Do not transmit private
keys or wallet seed material through the network interface.
