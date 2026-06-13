"""End-to-end probe for the Signal Foundry MCP trigger.

Drives a real MCP client → n8n SF MCP trigger → backend `/refresh`
round-trip and prints the SSE response stream.

Requires SF_MCP_URL and SF_MCP_TOKEN in the environment (typically
sourced from the workspace `.env`).

Usage:
    . .env && export SF_MCP_URL SF_MCP_TOKEN
    uv run python experiment-lab/scripts/sf_mcp_probe.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = os.environ["SF_MCP_URL"]
TOKEN = os.environ["SF_MCP_TOKEN"]
HEADERS_SSE = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "text/event-stream",
}
HEADERS_RPC = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}

session_id: str | None = None
sse_events: list[str] = []
sse_done = threading.Event()


def sse_reader() -> None:
    """Read SSE stream in background; capture sessionId then later events."""
    global session_id
    req = urllib.request.Request(BASE, headers=HEADERS_SSE)
    with urllib.request.urlopen(req, timeout=30) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            sse_events.append(line)
            print(f"SSE | {line[:200]}")
            if session_id is None and line.startswith("data:"):
                m = re.search(r"sessionId=([0-9a-f-]+)", line)
                if m:
                    session_id = m.group(1)
            if session_id and len(sse_events) > 30:
                break
    sse_done.set()


def rpc(method: str, params: dict | None = None, rpc_id: int = 1) -> None:
    body: dict[str, object] = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if params is not None:
        body["params"] = params
    data = json.dumps(body).encode()
    url = f"{BASE}?sessionId={session_id}"
    req = urllib.request.Request(url, data=data, headers=HEADERS_RPC, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            print(f"POST {method} → {resp.status} {resp.reason}")
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", "replace")
        print(f"POST {method} → {exc.code} {exc.reason} body={body_text[:300]}")


def notify(method: str) -> None:
    """Fire-and-forget JSON-RPC notification (no id, no response expected)."""
    data = json.dumps({"jsonrpc": "2.0", "method": method}).encode()
    req = urllib.request.Request(
        f"{BASE}?sessionId={session_id}",
        data=data,
        headers=HEADERS_RPC,
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10).read()
        print(f"POST {method} → ok (notification)")
    except Exception as exc:
        print(f"POST {method} → {exc}")


def main() -> int:
    t = threading.Thread(target=sse_reader, daemon=True)
    t.start()
    for _ in range(50):
        if session_id:
            break
        time.sleep(0.1)
    if not session_id:
        print("FAIL: no sessionId from SSE handshake")
        return 1
    print(f"\n>> sessionId={session_id}\n")

    rpc(
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "sf-mcp-probe", "version": "0.1.0"},
        },
        rpc_id=1,
    )
    time.sleep(0.5)
    notify("notifications/initialized")
    time.sleep(0.3)
    rpc("tools/list", rpc_id=2)
    time.sleep(1.0)

    # The minimal invocation that should "just work" after the workflow patch
    # that gave $fromAI defaults to mode / internal_stream_ref / apify_dataset_id.
    rpc(
        "tools/call",
        {
            "name": "sf_refresh_session",
            "arguments": {
                "session_id": "demo_miners",
                "scenario": "reputation_monitor",
            },
        },
        rpc_id=3,
    )

    sse_done.wait(timeout=10)
    print(f"\n>> SSE event count: {len(sse_events)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
