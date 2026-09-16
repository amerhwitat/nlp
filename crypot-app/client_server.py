"""Small authenticated-friendly client/server layer for local research jobs."""
from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen


class ResearchHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/job":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON")
            return
        # Only accepts declarative, non-wallet-sensitive research operations.
        allowed = {"hash", "address_validate", "model_inference"}
        if body.get("operation") not in allowed:
            self.send_error(400, "unsupported operation")
            return
        result = {"accepted": True, "operation": body["operation"], "payload": body.get("payload", {})}
        encoded = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt, *args):
        return


def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    ThreadingHTTPServer((host, port), ResearchHandler).serve_forever()


def submit(base_url: str, operation: str, payload: dict) -> dict:
    data = json.dumps({"operation": operation, "payload": payload}).encode()
    request = Request(base_url.rstrip("/") + "/api/job", data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read())
