#!/usr/bin/env python3
"""Scripted OpenAI-compatible stand-in for the vLLM server (CPU crash-safety tests only).

Serves GET /v1/models and POST /v1/chat/completions on 127.0.0.1:$MOCK_PORT (default 1234). Every
completion is a `python` tool call that plays a random valid action (or a short batch), plus visible
"World model:" / "Plan:" lines, so the whole Duck loop (sandbox, action execution, memory parsing,
history trimming, AGENTFIX/anim/sheet patches) runs end to end without a GPU. A few responses are
deliberately degenerate (no tool call, sandbox exception, RESET) to exercise the error paths.
Each request is logged (size, images, estimated prompt tokens, sampling params, kind = game / live / load) to
$MOCK_LOG.

Failure injection (matched against TAAF_VLLM_MAX_NUM_SEQS of the profile that launched this server):
  MOCK_FAIL_LIVE_SEQS=16             HTTP 500 on the notebook's live-check table lookups
  MOCK_FAIL_LOAD_SEQS=16             HTTP 500 on the live check's overload phase ("Load probe" requests)
  MOCK_CORRUPT_WARM=16               wrong answers on the cache-hit path of the live check
  MOCK_CRASH_AFTER_GAME_REQUESTS=N   the server process dies after N game requests (a mid-run vLLM crash), unless
                                     MOCK_NO_CRASH=1 (set by the mock start_server for the watchdog's replacement)
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("MOCK_PORT", "1234"))
MODEL = os.environ.get("MOCK_MODEL", "Qwen/Qwen3.8-Flash-Next-NVFP4")
LATENCY = float(os.environ.get("MOCK_LATENCY_S", "0.3"))
LOG = os.environ.get("MOCK_LOG", "/tmp/mock_llm_requests.jsonl")
_lock = threading.Lock()
_n = 0
_lookups = 0
_games = 0

CODE_RANDOM = r'''
import random
va = [a for a in valid_actions if a != "RESET"]
if not va:
    va = list(valid_actions)
picks = []
for _ in range(random.choice([1, 1, 2, 3])):
    a = random.choice(va)
    if a == "MOUSE":
        picks.append({"action": "MOUSE", "row": random.randrange(64), "col": random.randrange(64)})
    else:
        picks.append(a)
r = action(picks)
print("did", picks, "->", {k: r.get(k) for k in ("level", "board_changed", "game_over")} if isinstance(r, dict) else r)
f = current_frame
print(f.level, f.step, len(f.ascii))
try:
    lt = last_transition
    fr = getattr(lt, "frames", None)
    print("frames", None if fr is None else len(fr), getattr(lt, "frame_count", None))
except Exception as e:
    print("lt err", e)
'''

CODE_INSPECT = r'''
seg = current_frame.segmentation
print(type(seg).__name__)
print(str(seg)[:300])
print(valid_actions)
'''

CODE_ERROR = "x = undefined_name + 1\n"
CODE_RESET = 'action(["RESET"])\nprint("reset")\n'


def _count_images(messages):
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image_url")
    return n


def _chars(messages):
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    total += len(p.get("text", ""))
        for tc in m.get("tool_calls") or []:
            total += len(json.dumps(tc))
        total += len(str(m.get("reasoning_content") or ""))
    return total


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        elif self.path.rstrip("/").endswith("/health"):
            self._send(200, {})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global _n
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length) or b"{}")
        msgs = req.get("messages", [])
        with _lock:
            _n += 1
            n = _n
        chars = _chars(msgs)
        head = json.dumps(msgs)[:400]
        kind = "live" if "Reference table" in head else "load" if "Load probe" in head else "game"
        seqs = os.environ.get("TAAF_VLLM_MAX_NUM_SEQS", "")
        for key, where in (("MOCK_FAIL_LIVE_SEQS", "live"), ("MOCK_FAIL_LOAD_SEQS", "load")):
            if kind == where and seqs in [s for s in os.environ.get(key, "").split(",") if s]:
                self._send(500, {"error": f"mock: simulated {where}-check failure"})  # e.g. a crash under load
                return
        if kind == "game":
            global _games
            with _lock:
                _games += 1
                games = _games
            crash_after = int(os.environ.get("MOCK_CRASH_AFTER_GAME_REQUESTS", "0") or 0)
            if crash_after and games > crash_after and os.environ.get("MOCK_NO_CRASH") != "1":
                print(f"MOCK_CRASH after {crash_after} game requests", flush=True)
                os._exit(3)   # a dead vLLM: the port closes, the watchdog sees failed health checks
        rec = {
            "n": n, "t": time.time(), "kind": kind, "pid": os.getpid(),
            "prefix": os.environ.get("TAAF_VLLM_ENABLE_PREFIX_CACHING"),
            "msgs": len(msgs), "images": _count_images(msgs), "chars": chars,
            "est_tokens": chars // 3, "max_tokens": req.get("max_tokens"), "temperature": req.get("temperature"),
            "top_p": req.get("top_p"), "top_k": req.get("top_k"), "seed": req.get("seed"),
            "extra": {k: req.get(k) for k in ("chat_template_kwargs", "skip_special_tokens", "priority", "min_tokens")
                      if k in req},
            "tools": [t.get("function", {}).get("name") for t in req.get("tools") or []],
            "last_user_tail": "",
        }
        for m in reversed(msgs):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
                rec["last_user_tail"] = str(c)[-6000:]
                break
        if n <= 3 or n % 50 == 0:
            rec["system_head"] = str(msgs[0].get("content"))[:400] if msgs else ""
        with _lock, open(LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        time.sleep(LATENCY * random.uniform(0.5, 1.5))
        blob = json.dumps(msgs)
        lookup = re.search(r"the 5-digit code on line (L\d{3})", blob)
        if lookup and "Reference table" in blob:  # the notebook's live serving check: answer the table lookup
            label = lookup.group(1)
            found = re.search(label + r": (\d{5})", blob)
            answer = found.group(1) if found else "00000"
            global _lookups
            _lookups += 1
            if os.environ.get("MOCK_CORRUPT_WARM") == os.environ.get("TAAF_VLLM_MAX_NUM_SEQS", "?") and _lookups > 1:
                answer = "12345"  # simulate corrupted cached state on the cache-hit path
            self._send(200, {"id": f"chatcmpl-{n}", "object": "chat.completion", "created": int(time.time()),
                             "model": MODEL, "choices": [{"index": 0, "finish_reason": "stop",
                                                          "message": {"role": "assistant", "content": answer}}],
                             "usage": {"prompt_tokens": len(blob) // 3, "completion_tokens": 3}})
            return
        if kind == "load":
            self._send(200, {"id": f"chatcmpl-{n}", "object": "chat.completion", "created": int(time.time()),
                             "model": MODEL, "choices": [{"index": 0, "finish_reason": "length",
                                                          "message": {"role": "assistant", "content": "L0000"}}],
                             "usage": {"prompt_tokens": chars // 3, "completion_tokens": 256}})
            return
        roll = random.random()
        content = (f"World model: mock world model #{n}. The board is a grid.\n"
                   f"Goal: unknown yet.\nAction model: arrows move things.\n"
                   f"Plan: try a random action and observe.\nOpen questions: what is the goal?\n")
        if roll < 0.05:
            msg = {"role": "assistant", "content": content + "I will think more.", "reasoning_content": "hmm"}
            finish = "stop"
        else:
            code = CODE_RANDOM
            if roll < 0.10:
                code = CODE_ERROR
            elif roll < 0.18:
                code = CODE_INSPECT
            elif roll < 0.20:
                code = CODE_RESET
            msg = {"role": "assistant", "content": content, "reasoning_content": "thinking about the grid",
                   "tool_calls": [{"id": f"call_{n}", "type": "function",
                                   "function": {"name": "python", "arguments": json.dumps({"code": code})}}]}
            finish = "tool_calls"
        self._send(200, {
            "id": f"chatcmpl-{n}", "object": "chat.completion", "created": int(time.time()), "model": MODEL,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": chars // 3, "completion_tokens": 200, "total_tokens": chars // 3 + 200},
        })


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
