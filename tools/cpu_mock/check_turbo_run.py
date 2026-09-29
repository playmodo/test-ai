#!/usr/bin/env python3
"""Check an executed TURBO notebook (CPU mock run) for the expected markers.

    python tools/cpu_mock/check_turbo_run.py EXECUTED.ipynb MOCK_REQUEST_LOG.jsonl [--expect-profile NAME]
        [--temperature T] [--expect-load] [--expect-restart-no-prefix]

Verifies: every cell ran without an error output; the serving chain printed a READY profile (and the expected one);
the TURBO patch report has no *_error / *_missing keys; wave-fit ran; the audit + scorer ran; and, from the mock
request log, that game requests carried the TURBO sampling temperature, the game-clock line and the system-prompt
edits, and never the deleted text. --expect-load: the prefix profile's overload phase ran and passed.
--expect-restart-no-prefix: the mock vLLM died mid-run, the watchdog relaunched it with prefix caching off, and
game requests were served by the replacement.
"""
from __future__ import annotations

import argparse
import json
import re
import sys


def cell_text(cell: dict) -> str:
    parts = []
    for out in cell.get("outputs", []):
        if "text" in out:
            parts.append("".join(out["text"]))
        elif out.get("output_type") == "error":
            parts.append("ERROR: " + out.get("ename", "") + ": " + out.get("evalue", ""))
        elif "data" in out:
            parts.append("".join(out["data"].get("text/plain", "")))
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("request_log")
    ap.add_argument("--expect-profile")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--expect-load", action="store_true")
    ap.add_argument("--expect-restart-no-prefix", action="store_true")
    ap.add_argument("--watchdog-events", default="/kaggle/working/vllm-watchdog-events.jsonl")
    a = ap.parse_args()
    nb = json.load(open(a.notebook))
    text = "\n".join(cell_text(c) for c in nb["cells"] if c.get("cell_type") == "code")
    failures = []

    def need(cond, msg):
        (print("ok   ", msg) if cond else failures.append(msg))

    need("ERROR:" not in text, "no cell raised")
    ready = re.findall(r"TURBO_SERVING_READY (\S+)", text)
    need(bool(ready), f"serving chain ready: {ready}")
    if a.expect_profile:
        need(ready and ready[-1] == a.expect_profile, f"expected profile {a.expect_profile}")
    patch = re.search(r"TURBO_PATCH (\{.*\})", text)
    need(patch is not None and "_error" not in patch.group(1) and "_missing" not in patch.group(1),
         f"TURBO patch report clean: {patch.group(1)[:300] if patch else None}")
    need("TURBO_WAVEFIT" in text, "wave-fit applied: " + (re.search(r"TURBO_WAVEFIT .*", text) or [""])[0])
    need("PUBLIC25_AUDIT" in text, "audit + frozen scorer ran")
    need("AGENTFIX LIVE" in text and "'F6_action7': False" in text, "AGENTFIX live, F6 off (UNDO naming instead)")
    need("'T5_undo': True" in text, "ACTION7 shown as UNDO")
    need("'F11_dedup': True" in text, "AGENTFIX F11 dedup live")

    requests = [json.loads(line) for line in open(a.request_log)]
    game = [r for r in requests if r.get("kind", "game") == "game"
            and "5-digit code on line" not in r.get("last_user_tail", "")]
    if a.expect_load:
        load = [r for r in requests if r.get("kind") == "load"]
        need(bool(re.search(r"TURBO_LIVE_LOAD profile=\S+ ok=True", text)) and len(load) >= 8,
             f"overload phase ran and passed ({len(load)} load requests, "
             f"sampling {sorted({(r.get('temperature'), r.get('top_p'), r.get('top_k')) for r in load})})")
        need("TURBO_LIVE_CHECK post_load" in text and "post_load profile" in text, "post-load cold lookup ran")
    if a.expect_restart_no_prefix:
        need("TURBO_WATCHDOG_RESTART_POLICY prefix_off_on_restart" in text, "restart policy installed")
        need("TURBO_WATCHDOG_RESTART_NO_PREFIX" in text, "watchdog restart switched prefix caching off")
        events = [json.loads(line) for line in open(a.watchdog_events)] if a.watchdog_events else []
        need(any(e.get("event") == "restart_complete" for e in events),
             f"watchdog restart_complete ({[e.get('event') for e in events if e.get('event') != 'health_failed']})")
        pids = sorted({r.get("pid") for r in game})
        after = [r for r in game if r.get("prefix") == "0"]
        need(len(pids) >= 2 and len(after) > 5, f"game requests served by the replacement: pids={pids} "
                                                 f"prefix-off requests={len(after)}")
    need(len(game) > 20, f"game requests seen: {len(game)}")
    temps = {r.get("temperature") for r in game}
    need(temps == {a.temperature}, f"temperature {temps}")
    need(any("Game clock:" in r.get("last_user_tail", "") for r in game), "game-clock line reaches the prompt")
    heads = [r.get("system_head", "") for r in game if r.get("system_head")]
    need(bool(heads) and all("puzzle" not in h for h in heads), "system prompt opener edited ('puzzle' gone)")
    score = re.search(r"mean score:\s+([\d.]+)", text)
    print("mean score (mock policy, meaningless):", score.group(1) if score else None)
    for failure in failures:
        print("FAIL ", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
