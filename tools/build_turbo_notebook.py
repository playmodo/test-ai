#!/usr/bin/env python3
"""Build the TURBO competition notebook from the scored base notebook.

    python tools/build_turbo_notebook.py [--base notebooks/original/taaf-flashnext-sheetu12b-0922.ipynb]
                                         [--out notebooks/arc3-flashnext-turbo.ipynb]

Every edit is an exact, asserted string replacement on the base notebook's cells (the build fails loudly if the
base changes), plus one inserted cell that embeds notebooks/src/turbo_patch.py verbatim. Outputs are cleared.
See docs/ARC3_TURBO_REPORT.md for the evidence behind each change.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "notebooks/original/taaf-flashnext-sheetu12b-0922.ipynb"
OUT = REPO / "notebooks/arc3-flashnext-turbo.ipynb"
PATCH_SRC = REPO / "notebooks/src/turbo_patch.py"


def _replace_once(src: str, old: str, new: str, where: str) -> str:
    count = src.count(old)
    if count != 1:
        raise SystemExit(f"[{where}] expected exactly 1 occurrence, found {count}: {old[:100]!r}")
    return src.replace(old, new)


def _cut(src: str, start: str, end: str, where: str) -> tuple[str, str, str]:
    """Split src into (before, [start .. end), after) with both markers asserted unique."""
    for marker in (start, end):
        if src.count(marker) != 1:
            raise SystemExit(f"[{where}] marker not unique: {marker[:80]!r}")
    i, j = src.index(start), src.index(end)
    if j < i:
        raise SystemExit(f"[{where}] markers out of order")
    return src[:i], src[i:j], src[j:]


def _code_cell(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def _set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


# ------------------------------------------------------------------------------------------------ cell 0 (markdown)
INTRO_MD = """## ARC3 Flash-Next TURBO (fork of taaf-flashnext-sheetu12b-0922)

Same model (`RadixArk/Qwen3.8-Flash-Next-NVFP4`), same attached datasets, same Duck harness and the same AGENTFIX /
animation-frame / contact-sheet / x12-image patches as the base notebook. The changes target its measured bottleneck:
**~87 % of every model call was spent queued in vLLM** (the MTP-3 weights left a 5 GiB KV pool = ~105k tokens for 28
games with ~22k-token prompts), so each game got only ~40 model calls in 2 h and its score was still climbing linearly
when time ran out.

1. **Serving profile chain with fallback** (cells 3 and 9): MTP off (frees ~7.5 GiB of weights) and a 2-2.4x larger
   bf16 KV pool, prefix caching (mamba align mode), `max_num_seqs` matched to the pool. Each profile must start and
   pass a live request check; otherwise it is torn down and the next one is tried, ending with the base notebook's
   proven MTP-3 profile. `TAAF_VLLM_MAX_NUM_BATCHED_TOKENS` is no longer overridden (the override broke the
   watchdog's restart path, so a vLLM crash stayed fatal).
2. **Per-game time fitted to the real game count and the time left after setup** (no wave is cut by the deadline).
3. **Agent fixes** (cell 11, `turbo_patch`): runtime-state history capped (the stock harness re-serialized the whole
   history on every action -> sandbox timeouts late in long games), missing sandbox builtins (`class`, `KeyError`,
   `object`, ...), analyzer exceptions retried instead of permanently crashing the game, temperature 1.0 with MTP off
   (model-card thinking setting), 120 s yield with a 3-tool-call turn cap, ACTION7 shown and executable as `UNDO`,
   the stale "The game is over." line corrected, a neutral per-turn game clock.
4. **System prompt**: Son Pham & Mark Barney's Kaggle-measured arm B (delete four false/distracting claims) + arm C
   (six "mechanics may be" bullets) - measured on this exact bundle on Kaggle hardware - and the action-minimising
   line deleted (cleared levels already run at 0.7-0.8x the human baseline; reach is what is missing).
5. AGENTFIX **F11** (strip repeated boilerplate from older turns in the outgoing request) switched on; end-of-run
   cleanup can no longer raise.

Credit: Tufa Labs (Duck harness), Keith Tyser (Flash-Next NVFP4 serving), Scott Le Grand (AGENTFIX / anim / sheet),
Son Pham & Mark Barney (prompt arms, serving measurements), Thuitanium (MTP-off serving benchmark).
"""

# ------------------------------------------------------------------------------------------------ cell 3
C3_PROFILE_START = "# Apply the measured vLLM winner before any serving setup command runs."
C3_PROFILE_END = "# Pin arc_agi's cached level_reset_only before its client is built (RESET keeps the level)."
C3_PROFILE_NEW = '''# TURBO serving profiles, tried in order by the setup cell until one starts AND passes a live request check.
# Why: with MTP-3 the 5 GiB KV pool held ~105k tokens = ~4.5 of the games' ~22k-token prompts, so ~87 % of every
# model call was spent queued in vLLM (09-22 validation: ~42 calls per game in 2 h, 342 preemptions). MTP off frees
# ~7.5 GiB of weights (81.8 -> 74.0 GiB) that buys a 2-2.4x larger bf16 KV pool (fp8 KV is rejected by this
# runtime's QSA layer). Prefix caching (mamba 'align' mode, auto-selected) only works without MTP on this vLLM build.
# max_num_seqs = what the pool holds at the median ~23k-token request (800-token blocks, 52.9 blocks/GiB with MTP
# off, +9 blocks per request with prefix caching, +5 without); the 28 game lanes keep the batch full.
# Each fallback step moves toward the proven configuration on the axis that may have failed.
# TAAF_VLLM_MAX_NUM_BATCHED_TOKENS is deliberately NOT set: overriding it (even to the default 8192) makes the
# watchdog's restart path raise (serving_setup rejects the no-chunk argv it rebuilds), so a vLLM crash stayed fatal.
_TURBO_BASE_ENV = {
    "TAAF_VLLM_KV_CACHE_DTYPE": "auto",
    "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32",
    "TAAF_VLLM_OMP_THREADS": "1",
}
TURBO_PROFILES = [
    {   # 74.0 GiB weights + 12 GiB KV (~2 GiB headroom vs ~0.25 GiB for the proven MTP-3 profile); ~16.7 requests fit
        "name": "mtp0-kv12-prefix-s16", "live_check": "prefix",
        "env": {**_TURBO_BASE_ENV, "TAAF_VLLM_MTP_TOKENS": "0", "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": str(12 * 1024**3),
                "TAAF_VLLM_MAX_NUM_SEQS": "16", "TAAF_VLLM_ENABLE_PREFIX_CACHING": "1"},
    },
    {   # prefix caching off (the other untested axis), 10 GiB; ~15.6 requests fit
        "name": "mtp0-kv10-s14", "live_check": "basic",
        "env": {**_TURBO_BASE_ENV, "TAAF_VLLM_MTP_TOKENS": "0", "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": str(10 * 1024**3),
                "TAAF_VLLM_MAX_NUM_SEQS": "14", "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0"},
    },
    {   # the MTP-off / 7 GiB pool already run on Kaggle through this serving_setup (Thuitanium m0-s20); ~10.9 fit
        "name": "mtp0-kv7-s11", "live_check": "basic",
        "env": {**_TURBO_BASE_ENV, "TAAF_VLLM_MTP_TOKENS": "0", "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": str(7 * 1024**3),
                "TAAF_VLLM_MAX_NUM_SEQS": "11", "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0"},
    },
    {   # last resort: the base notebook's proven MTP-3 / 5 GiB profile (stock c8 shape: ~3.8 requests fit)
        "name": "base-mtp3-kv5-s8", "live_check": None,
        "env": {**_TURBO_BASE_ENV, "TAAF_VLLM_MTP_TOKENS": "3", "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": str(5 * 1024**3),
                "TAAF_VLLM_MAX_NUM_SEQS": "8", "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0"},
    },
]
# Optional override for experiments: TURBO_PROFILE_START=<index> skips the first profiles.
TURBO_PROFILES = TURBO_PROFILES[max(0, min(len(TURBO_PROFILES) - 1, int(os.environ.get("TURBO_PROFILE_START", "0") or 0))):]
print("TURBO_PROFILES " + json.dumps([profile["name"] for profile in TURBO_PROFILES]), flush=True)
'''

# ------------------------------------------------------------------------------------------------ cell 9
C9_OLD_SETUP = '''# Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.
env = _command_env()
for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
    print(f"taaf.kaggle: setup command: {command}", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
    # Re-read in case the command persisted new env keys.
    env = _command_env()
    os.environ.update(env)
'''
C9_NEW_SETUP = '''# Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.
# TURBO: try the serving profiles in order. A profile whose setup fails, or whose server fails the live request
# check, is torn down (every vLLM / PLE-offload process killed, GPU + host RAM + port released, runtime /tmp freed,
# its log archived) and the next one is tried; only if every profile fails does the notebook raise.
import glob as _glob
import random as _random
import re as _re
import shutil as _shutil
import signal as _signal
import socket as _socket
import threading as _threading
import urllib.request as _urlreq

_TURBO_TMP = Path("/tmp")
_TURBO_RUNTIME_ROOTS = [_TURBO_TMP / "qwen38-flash-next-vllm-runtime", _TURBO_TMP / "qwen38-flash-next-vllm-tmp"]
_TURBO_CACHE_ROOTS = [_TURBO_TMP / "qwen38-flash-next-vllm-cache", _TURBO_TMP / "qwen38-flash-next-vllm-compile-cache"]
_TURBO_SERVER_ENV_MARKERS = (b"VLLM_PLE_CPU_OFFLOAD=1\\x00", b"VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8=1\\x00")


def _turbo_apply_profile(profile: dict) -> None:
    for key in [k for k in os.environ if k.startswith("TAAF_VLLM_")]:
        os.environ.pop(key, None)
    os.environ.update(profile["env"])


def _turbo_gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return max(int(value) for value in out.stdout.split())
    except Exception:
        return None


def _turbo_mem_available_gib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    except Exception:
        return None
    return None


def _turbo_port_open() -> bool:
    try:
        with _socket.create_connection(("127.0.0.1", 1234), timeout=1.0):
            return True
    except OSError:
        return False


def _turbo_serving_pids() -> list:
    """vLLM API/engine/worker processes, the PLE-offload worker and torch_shm_manager (same markers as the
    bundle's serving_teardown.py), plus anything still holding the GPU. Never this kernel or its parent."""
    skip = {os.getpid(), os.getppid()}
    found = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in skip:
            continue
        try:
            comm = (proc / "comm").read_text(errors="replace").strip().lower()
            cmdline = (proc / "cmdline").read_bytes().replace(b"\\x00", b" ").decode(errors="replace").lower()
        except Exception:
            continue
        hit = ("vllm::" in comm or "vllm.entrypoints" in cmdline or "qwen38-flash-next-vllm" in cmdline
               or comm == "torch_shm_manager"
               or any(marker in f"{comm} {cmdline}" for marker in ("ple_offload", "pleworker", "ple_worker")))
        if not hit and ("python" in comm or "spawn_main" in cmdline):
            try:
                environ = (proc / "environ").read_bytes()
                hit = all(marker in environ for marker in _TURBO_SERVER_ENV_MARKERS)
            except Exception:
                hit = False
        if hit:
            found.add(int(proc.name))
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        found.update(int(v) for v in out.stdout.split() if v.strip().isdigit() and int(v) not in skip)
    except Exception:
        pass
    return sorted(found)


def _turbo_kill_serving() -> None:
    for sig, grace in ((_signal.SIGTERM, 15.0), (_signal.SIGKILL, 10.0)):
        pids = _turbo_serving_pids()
        if not pids:
            break
        print(f"TURBO_KILL sig={int(sig)} pids={pids}", flush=True)
        for pid in pids:
            try:
                os.kill(pid, sig)
            except Exception:
                pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and any(Path(f"/proc/{pid}").exists() for pid in pids):
            time.sleep(0.5)
    for path in _glob.glob("/dev/shm/torch_*"):
        try:
            os.unlink(path)
        except Exception:
            pass


def _turbo_teardown() -> None:
    for command in json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
        try:
            subprocess.run(command, shell=True, check=False, cwd=WORKING_DIR, env=_command_env(), timeout=180)
        except Exception as exc:
            print(f"TURBO_TEARDOWN_ERROR {exc!r}", flush=True)


def _turbo_release(attempt: int, timeout_s: float = 300.0) -> None:
    _turbo_kill_serving()
    # serving_setup's own gates need >= 64 GiB MemAvailable (PLE CPU offload), port 1234 free and 36.5 GB free /tmp
    min_mem_gib = float(os.environ.get("TURBO_RELEASE_MIN_MEM_GIB", "64"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        used, avail = _turbo_gpu_used_mib(), _turbo_mem_available_gib()
        if (used is None or used < 2048) and (avail is None or avail >= min_mem_gib) and not _turbo_port_open():
            break
        time.sleep(5)
        _turbo_kill_serving()
    log_path = WORKING_DIR / "vllm-openai-server.log"
    if log_path.exists():   # start_server unlinks it; keep the failed attempt's log for diagnosis
        try:
            _shutil.copyfile(log_path, WORKING_DIR / f"vllm-openai-server.attempt{attempt}.log")
        except Exception:
            pass
    for root in _TURBO_RUNTIME_ROOTS:
        _shutil.rmtree(root, ignore_errors=True)
    try:
        if _shutil.disk_usage(_TURBO_TMP).free < 40 * 1024**3:
            for root in _TURBO_CACHE_ROOTS:
                _shutil.rmtree(root, ignore_errors=True)
    except Exception:
        pass
    print(f"TURBO_RELEASED gpu_used_mib={_turbo_gpu_used_mib()} mem_available_gib={_turbo_mem_available_gib()} "
          f"port_open={_turbo_port_open()}", flush=True)


def _turbo_chat(payload: dict, timeout: float) -> dict:
    base = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
    request = _urlreq.Request(base + "/chat/completions", data=json.dumps(payload).encode(), method="POST",
                              headers={"Content-Type": "application/json",
                                       "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "x")})
    with _urlreq.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _turbo_prefix_hits():
    try:
        with _urlreq.urlopen("http://127.0.0.1:1234/metrics", timeout=20) as response:
            text = response.read().decode(errors="replace")
        values = [float(m.group(1)) for m in _re.finditer(r"^vllm:prefix_cache_hits(?:_total)?(?:\\{[^}]*\\})?\\s+([0-9.eE+]+)", text, _re.M)]
        return sum(values) if values else None
    except Exception:
        return None


def _turbo_live_check(profile: dict) -> bool:
    """Real requests shaped like Duck turns: a long shared prefix + an image. Sequential first (cold, then a
    cache-hit on the same prefix), then a concurrent burst; every answer is a lookup that depends on the whole
    prefix, so corrupted cached state (the realistic prefix-caching failure) gives wrong codes, not just errors."""
    mode = profile.get("live_check")
    if not mode:
        return True
    model = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-Flash-Next-NVFP4")
    rng = _random.Random(20260929)
    codes = [rng.randrange(10000, 100000) for _ in range(900)]
    table = "\\n".join(f"L{i:03d}: {code}" for i, code in enumerate(codes))
    image_part = None
    try:
        import base64 as _b64, io as _io
        from PIL import Image as _Image
        image = _Image.new("RGB", (768, 768), (20, 20, 20))
        for i in range(0, 768, 96):
            image.paste((200, 30 + i // 4, 90), (i, i, i + 48, i + 48))
        buffer = _io.BytesIO()
        image.save(buffer, format="PNG")
        image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64.b64encode(buffer.getvalue()).decode()}}
    except Exception as exc:
        print(f"TURBO_LIVE_CHECK no image ({exc!r})", flush=True)

    def ask(line: int):
        content = [{"type": "text", "text": "Reference table:\\n" + table
                    + f"\\nReply with only the 5-digit code on line L{line:03d}."}]
        if image_part is not None:
            content.append(image_part)
        payload = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 400,
                   "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            message = (_turbo_chat(payload, timeout=420).get("choices") or [{}])[0].get("message") or {}
            text = f"{message.get('content') or ''} {message.get('reasoning_content') or message.get('reasoning') or ''}"
            return True, str(codes[line]) in text
        except Exception as exc:
            print(f"TURBO_LIVE_CHECK request L{line:03d} failed: {exc!r}"[:300], flush=True)
            return False, False

    started = time.monotonic()
    hits_before = _turbo_prefix_hits()
    cold_ok, cold_right = ask(137)
    warm_ok, warm_right = ask(642)       # same prefix: the cache-hit path when prefix caching is on
    burst = []
    threads = [_threading.Thread(target=lambda k=k: burst.append(ask(k))) for k in (11, 333, 505, 777)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(480)
    time.sleep(15)
    healthy = False
    try:
        base = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
        with _urlreq.urlopen(base + "/models", timeout=30) as response:
            healthy = response.status == 200
    except Exception as exc:
        print(f"TURBO_LIVE_CHECK health failed: {exc!r}"[:300], flush=True)
    hits_after = _turbo_prefix_hits()
    all_ok = healthy and cold_ok and warm_ok and len(burst) == 4 and all(ok for ok, _ in burst)
    burst_right = sum(1 for _, right in burst if right)
    # the model reading one line of a table is the control: only when it can (cold answer right) do wrong
    # cached / concurrent answers indicate corrupted state
    corrupted = mode == "prefix" and cold_right and (not warm_right or burst_right < 3)
    ok = all_ok and not corrupted
    print(f"TURBO_LIVE_CHECK profile={profile['name']} ok={ok} seconds={time.monotonic() - started:.1f} healthy={healthy} "
          f"cold={cold_ok}/{cold_right} warm={warm_ok}/{warm_right} burst={len(burst)}/{burst_right} "
          f"prefix_hits={hits_before}->{hits_after}", flush=True)
    return ok


def _turbo_serving_proof() -> None:
    log_path = WORKING_DIR / "vllm-openai-server.log"
    patterns = ("Model loading took", "GPU KV cache size", "Maximum concurrency", "prefix", "mamba_cache_mode",
                "Mamba cache mode", "speculative", "Chunked prefill is enabled", "attention block size")
    try:
        seen = set()
        for line in log_path.read_text(errors="replace").splitlines():
            if any(p in line for p in patterns):
                key = _re.sub(r"\\d{2}:\\d{2}:\\d{2}", "", line)[-220:]
                if key not in seen:
                    seen.add(key)
                    print("TURBO_SERVING_LOG " + line[-300:], flush=True)
                if len(seen) >= 40:
                    break
    except Exception as exc:
        print(f"TURBO_SERVING_LOG unavailable: {exc!r}", flush=True)


TURBO_SERVING = None
_turbo_setup_commands = json.loads((BUNDLE_DIR / "setup_commands.json").read_text())
for _turbo_index, _turbo_profile in enumerate(TURBO_PROFILES):
    _turbo_apply_profile(_turbo_profile)
    _turbo_t0 = time.monotonic()
    print(f"TURBO_SERVING_TRY {_turbo_index} {_turbo_profile['name']} {json.dumps(_turbo_profile['env'], sort_keys=True)}", flush=True)
    env = _command_env()
    _turbo_ok = True
    for command in _turbo_setup_commands:
        print(f"taaf.kaggle: setup command: {command}", flush=True)
        _turbo_rc = subprocess.run(command, shell=True, check=False, cwd=WORKING_DIR, env=env).returncode
        # Re-read in case the command persisted new env keys.
        env = _command_env()
        os.environ.update(env)
        if _turbo_rc != 0:
            print(f"TURBO_SERVING_SETUP_FAILED profile={_turbo_profile['name']} rc={_turbo_rc}", flush=True)
            _turbo_ok = False
            break
    if _turbo_ok:
        _turbo_serving_proof()
        if not _turbo_live_check(_turbo_profile):
            _turbo_ok = False
            _turbo_teardown()
    if _turbo_ok:
        TURBO_SERVING = _turbo_profile
        print(f"TURBO_SERVING_READY {_turbo_profile['name']} seconds={time.monotonic() - _turbo_t0:.0f}", flush=True)
        break
    if _turbo_index + 1 == len(TURBO_PROFILES):
        raise RuntimeError("Every TURBO serving profile failed to start.")
    _turbo_release(_turbo_index)
'''

# ------------------------------------------------------------------------------------------------ cell 10
C10_EDITS = [
    ("_os.environ.setdefault('AGENTFIX_DEDUP', '0')",
     "_os.environ.setdefault('AGENTFIX_DEDUP', '1')   # TURBO: strip repeated boilerplate from older user turns (outgoing copy only)"),
]

# ------------------------------------------------------------------------------------------------ new cell 11
TURBO_CELL_HEAD = '''# TURBO runtime patches (notebooks/src/turbo_patch.py in the source repo, embedded verbatim and executed in its own
# module namespace so it cannot clobber AGENTFIX globals). Installed after AGENTFIX so it wraps AGENTFIX's wrappers.
import types as _turbo_types
_TURBO_SRC = {src!r}
_turbo_mod = _turbo_types.ModuleType("turbo_patch")
exec(compile(_TURBO_SRC, "turbo_patch.py", "exec"), _turbo_mod.__dict__)
import inference.agent.tool_agent as _ta, inference.agent.action_names as _an, inference.framework.solver as _solv
import inference.agent.runtime_state as _rs, inference.agent.python_tool_sandbox as _sb
TURBO_REPORT = _turbo_mod.install(_ta, _an, _solv, _rs, _sb)
print("TURBO_PATCH", TURBO_REPORT, flush=True)
if not TRUE_SUBMISSION:   # a broken anchor must surface in the commit run, never first in the competition rerun
    _turbo_errors = {{k: v for k, v in TURBO_REPORT.items() if k.endswith("_error") or k.endswith("_missing")}}
    assert not _turbo_errors, _turbo_errors
'''

# ------------------------------------------------------------------------------------------------ cell 16
C16_EDITS = [
    (   # validation smoke subset (non-submission branch only) + audit against the chosen games
        "    print(f'PUBLIC25_SELECTION games={len(bm.games)} passes=1', flush=True)\n",
        "    print(f'PUBLIC25_SELECTION games={len(bm.games)} passes=1', flush=True)\n"
        "    # TURBO: optional smoke subset for cheap commit runs, e.g. TURBO_SMOKE_GAMES=ft09-0d8bbf25,ls20-9607627b\n"
        "    _turbo_smoke = [g.strip() for g in os.environ.get('TURBO_SMOKE_GAMES', '').split(',') if g.strip()]\n"
        "    if _turbo_smoke:\n"
        "        bm.games = [offline_by_id[g] for g in _turbo_smoke]\n"
        "        print(f'TURBO_SMOKE games={_turbo_smoke}', flush=True)\n",
    ),
    (   # wave-fit per-game budget from the real game count and the time actually left
        "# Start recovery only after setup readiness and all run gates pass.\n",
        "# TURBO wave-fit: every game gets the same slice of the time actually left, so no wave is cut by soft_end\n"
        "# (games not started before soft_end score 0). The benchmark deep-copies bm.solver, so set it before bm.run.\n"
        "import math as _math\n"
        "_turbo_slots = max(1, int(bm.solver.concurrency or 1))\n"
        "_turbo_waves = max(1, _math.ceil(len(bm.games) * int(bm.n_passes or 1) / _turbo_slots))\n"
        "_turbo_reserve = float(os.environ.get('TURBO_BUDGET_RESERVE_S', '180'))\n"
        "_turbo_left = (soft_end - datetime.now()).total_seconds() - _turbo_reserve\n"
        "_turbo_cap = float(os.environ.get('TURBO_MAX_RUNTIME_S_PER_GAME', '7920'))\n"
        "bm.solver.max_runtime_s_per_game = max(600.0, min(_turbo_cap, _turbo_left / _turbo_waves))\n"
        "print(f'TURBO_WAVEFIT games={len(bm.games)} slots={_turbo_slots} waves={_turbo_waves} left_s={_turbo_left:.0f} '\n"
        "      f'per_game_s={bm.solver.max_runtime_s_per_game:.0f} serving={(TURBO_SERVING or {}).get(\"name\")}', flush=True)\n"
        "# ...and each game's budget is recomputed when it actually starts (turbo_patch T6), from the time then left and\n"
        "# the games not yet started, so a late wave is never cut and time freed by early finishes is reused.\n"
        "try:\n"
        "    print('TURBO_BUDGET_CONFIG', _turbo_mod.configure_budget(\n"
        "        total_games=len(bm.games) * int(bm.n_passes or 1), lanes=_turbo_slots, reserve_s=_turbo_reserve,\n"
        "        cap_s=_turbo_cap), flush=True)\n"
        "except Exception as _turbo_exc:\n"
        "    print(f'TURBO_BUDGET_CONFIG_ERROR {_turbo_exc!r}', flush=True)\n"
        "\n"
        "# Start recovery only after setup readiness and all run gates pass.\n",
    ),
    (   # audit the games actually chosen (smoke subset or the full public 25)
        "        if len(public_runs) != 25 or public_run_ids != list(PUBLIC_GAME_IDS):\n",
        "        if public_run_ids != [game.env_name for game in bm.games]:\n",
    ),
    (   # cleanup can never raise after the games are done
        "finally:\n"
        "    try:\n"
        "        vllm_watchdog.stop_background(timeout_seconds=15.0)\n"
        "    finally:\n"
        "        for command in json.loads((BUNDLE_DIR / \"teardown_commands.json\").read_text()):\n"
        "            print(f\"taaf.kaggle: teardown command: {command}\", flush=True)\n"
        "            subprocess.run(\n"
        "                command,\n"
        "                shell=True,\n"
        "                check=False,\n"
        "                cwd=WORKING_DIR,\n"
        "                env=_command_env(),\n"
        "                timeout=30.0,\n"
        "            )\n",
        "finally:\n"
        "    try:\n"
        "        vllm_watchdog.stop_background(timeout_seconds=15.0)\n"
        "    except Exception as _turbo_exc:   # TURBO: cleanup must never turn a finished run into an error\n"
        "        print(f'TURBO_WATCHDOG_STOP_ERROR {_turbo_exc!r}', flush=True)\n"
        "    finally:\n"
        "        for command in json.loads((BUNDLE_DIR / \"teardown_commands.json\").read_text()):\n"
        "            print(f\"taaf.kaggle: teardown command: {command}\", flush=True)\n"
        "            try:\n"
        "                subprocess.run(\n"
        "                    command,\n"
        "                    shell=True,\n"
        "                    check=False,\n"
        "                    cwd=WORKING_DIR,\n"
        "                    env=_command_env(),\n"
        "                    timeout=30.0,\n"
        "                )\n"
        "            except Exception as _turbo_exc:\n"
        "                print(f'TURBO_TEARDOWN_ERROR {_turbo_exc!r}', flush=True)\n",
    ),
]


def build(base: Path, out: Path) -> Path:
    nb = json.loads(base.read_text())
    cells = nb["cells"]
    src = lambda i: "".join(cells[i]["source"])  # noqa: E731
    assert src(10).startswith("# ----") and "AGENTFIX" in src(10), "unexpected base notebook layout"

    # cell 0: new intro, keep the original "About this fork" text below it
    cells[0]["source"] = (INTRO_MD + "\n---\n\n" + src(0)).splitlines(keepends=True)

    # cell 3: profile chain; the dead LOCAL_ANALYZER_CONTEXT_WINDOW=16384 line goes with the old profile block
    before, _old, after = _cut(src(3), C3_PROFILE_START, C3_PROFILE_END, "C3")
    assert "16384" in _old and "16384" not in before + after
    _set_source(cells[3], before + C3_PROFILE_NEW + after)

    # cell 9: fallback chain
    _set_source(cells[9], _replace_once(src(9), C9_OLD_SETUP, C9_NEW_SETUP, "C9"))

    # cell 10: AGENTFIX switches
    c10 = src(10)
    for old, new in C10_EDITS:
        c10 = _replace_once(c10, old, new, "C10")
    _set_source(cells[10], c10)

    # cell 16: smoke subset, wave-fit, audit, cleanup
    c16 = src(16)
    for old, new in C16_EDITS:
        c16 = _replace_once(c16, old, new, "C16")
    _set_source(cells[16], c16)

    # new cell 11: TURBO patches (after AGENTFIX, before the benchmark is loaded)
    cells.insert(11, _code_cell(TURBO_CELL_HEAD.format(src=PATCH_SRC.read_text())))

    for cell in cells:  # outputs of the base run are not ours
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    nb.setdefault("metadata", {}).pop("papermill", None)
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            code = "".join(cell["source"])
            try:  # top-level await (cell 16) is a notebook feature
                compile(code, f"<cell {index}>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            except SyntaxError as exc:
                raise SystemExit(f"cell {index} does not parse: {exc}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(build(args.base, args.out))


if __name__ == "__main__":
    main()
