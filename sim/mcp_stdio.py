"""A minimal, dependency-free MCP server over stdio (JSON-RPC 2.0, one message per line).

Supports what Claude Code needs for tools: initialize, notifications, ping,
tools/list and tools/call. Tool handlers are plain Python functions that
return a JSON-serialisable value (sent back as text) or raise ToolError.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable


class ToolError(Exception):
    pass


class Server:
    def __init__(self, name: str, version: str = "1.0.0", instructions: str = "") -> None:
        self.name = name
        self.version = version
        self.instructions = instructions
        self.tools: dict[str, dict] = {}

    def tool(self, name: str, description: str, properties: dict | None = None,
             required: list[str] | None = None) -> Callable:
        def deco(fn: Callable) -> Callable:
            self.tools[name] = {
                "fn": fn,
                "spec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {"type": "object", "properties": properties or {},
                                    "required": required or [], "additionalProperties": False},
                },
            }
            return fn
        return deco

    # -- protocol ---------------------------------------------------------
    def _reply(self, mid: Any, result: Any = None, error: dict | None = None) -> None:
        msg: dict = {"jsonrpc": "2.0", "id": mid}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    def _handle(self, req: dict) -> None:
        method = req.get("method")
        mid = req.get("id")
        params = req.get("params") or {}
        if mid is None:  # notification
            return
        if method == "initialize":
            self._reply(mid, {
                "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.name, "version": self.version},
                "instructions": self.instructions,
            })
        elif method == "ping":
            self._reply(mid, {})
        elif method == "tools/list":
            self._reply(mid, {"tools": [t["spec"] for t in self.tools.values()]})
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            tool = self.tools.get(name)
            if not tool:
                self._reply(mid, error={"code": -32602, "message": f"unknown tool {name}"})
                return
            try:
                out = tool["fn"](**args)
                text = out if isinstance(out, str) else json.dumps(out, indent=1, default=str)
                self._reply(mid, {"content": [{"type": "text", "text": text}], "isError": False})
            except ToolError as e:
                self._reply(mid, {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
            except TypeError as e:
                self._reply(mid, {"content": [{"type": "text", "text": f"bad arguments: {e}"}], "isError": True})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                self._reply(mid, {"content": [{"type": "text", "text": f"internal error: {e}"}], "isError": True})
        elif method in ("resources/list", "prompts/list"):
            self._reply(mid, {"resources": []} if method == "resources/list" else {"prompts": []})
        else:
            self._reply(mid, error={"code": -32601, "message": f"method not found: {method}"})

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(req, list):
                for r in req:
                    self._handle(r)
            else:
                self._handle(req)
