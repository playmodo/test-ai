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
   pass a live request check (for prefix caching also a production-shaped overload that forces preemption and cache
   eviction); otherwise it is torn down and the next one is tried, ending with the base notebook's exact MTP-3 argv. If
   vLLM crashes mid-run under prefix caching, the watchdog restarts it with prefix caching off. `TAAF_VLLM_MAX_NUM_BATCHED_TOKENS` is no longer overridden (the override broke the
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
    {   # last resort: exactly the 09-22 run's vLLM argv (MTP-3, 5 GiB, max_num_seqs 16, default batched tokens)
        "name": "base-mtp3-kv5-s16", "live_check": None,
        "env": {**_TURBO_BASE_ENV, "TAAF_VLLM_MTP_TOKENS": "3", "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": str(5 * 1024**3),
                "TAAF_VLLM_MAX_NUM_SEQS": "16", "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0"},
    },
]
# Optional override for experiments: TURBO_PROFILE_START=<index> skips the first profiles.
TURBO_PROFILES = TURBO_PROFILES[max(0, min(len(TURBO_PROFILES) - 1, int(os.environ.get("TURBO_PROFILE_START", "0") or 0))):]
print("TURBO_PROFILES " + json.dumps([profile["name"] for profile in TURBO_PROFILES]), flush=True)


# TURBO: in a real competition rerun, poll the gateway while the (possibly multi-attempt) serving setup runs, so the
# Arcade does not see the setup as inactivity (organizers: the run ends after 15 min without interaction; the one
# scored public startup longer than 15 min, Son Pham's, kept a gateway keepalive). Never raises, never prints.
def _turbo_gateway_keepalive() -> None:
    import urllib.request as _urlreq_keepalive
    while True:
        try:
            base = os.environ.get("ARC_BASE_URL", "http://gateway:8001/")
            request = _urlreq_keepalive.Request(base.rstrip("/") + "/api/games",
                                                headers={"X-API-Key": os.environ.get("ARC_API_KEY", "test-key-123")})
            with _urlreq_keepalive.urlopen(request, timeout=10) as response:
                response.read(1 << 20)
        except Exception:
            pass
        time.sleep(90)


if TRUE_SUBMISSION and os.environ.get("TURBO_GATEWAY_KEEPALIVE", "1") != "0":
    import threading as _threading_keepalive
    _threading_keepalive.Thread(target=_turbo_gateway_keepalive, name="turbo-gateway-keepalive", daemon=True).start()
    print("TURBO_GATEWAY_KEEPALIVE started", flush=True)
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
# check, is torn down (every vLLM / PLE-offload process of that attempt killed, GPU + host RAM + port released,
# runtime /tmp freed, its log archived), the pre-chain environment is restored and the next profile is tried.
# Only if every profile fails does the notebook raise.
import glob as _glob
import random as _random
import re as _re
import shutil as _shutil
import signal as _signal
import socket as _socket
import threading as _threading
import urllib.error as _urlerr
import urllib.request as _urlreq

_TURBO_TMP = Path("/tmp")
_TURBO_RUNTIME_ROOTS = [_TURBO_TMP / "qwen38-flash-next-vllm-runtime", _TURBO_TMP / "qwen38-flash-next-vllm-tmp"]
_TURBO_CACHE_ROOTS = [_TURBO_TMP / "qwen38-flash-next-vllm-cache", _TURBO_TMP / "qwen38-flash-next-vllm-compile-cache"]
_TURBO_SERVER_ENV_MARKERS = (b"VLLM_PLE_CPU_OFFLOAD=1\\x00", b"VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8=1\\x00")
_TURBO_SETUP_TIMEOUT_S = float(os.environ.get("TURBO_SETUP_ATTEMPT_TIMEOUT_S", "2100"))
_TURBO_SETUP_DEADLINE_S = float(os.environ.get("TURBO_SETUP_DEADLINE_S", "2700"))   # after this, only the proven profile
# OOM = the fatal forms, or any other "OOM" once the harmless FlashInfer autotuner line of the healthy 09-22 log
# ("... memory allocation failed with OOM on device 0 ...", a skipped tactic) is removed. A false positive only routes
# to the smaller, Kaggle-proven kv7 profile; a missed OOM would skip it.
_TURBO_OOM_RE = _re.compile(r"OutOfMemoryError|out of memory|cudaErrorMemoryAllocation|CUBLAS_STATUS_ALLOC_FAILED|"
                            r"No available memory for the cache blocks|less than desired GPU memory utilization|\\boom\\b",
                            _re.I)


def _turbo_is_oom(text: str) -> bool:
    kept = [line for line in text.splitlines() if "memory allocation failed with OOM" not in line]
    return bool(_TURBO_OOM_RE.search("\\n".join(kept)))
_TURBO_ENV_SNAPSHOT = dict(os.environ)
_TURBO_SETUP_ENV_SNAPSHOT = SETUP_ENV_PATH.read_text()


def _turbo_apply_profile(profile: dict) -> None:
    # back to the pre-chain environment (a failed attempt may have persisted runtime paths it no longer owns),
    # then exactly this profile's TAAF_VLLM_* keys
    for key in [k for k in os.environ if k not in _TURBO_ENV_SNAPSHOT]:
        os.environ.pop(key, None)
    for key, value in _TURBO_ENV_SNAPSHOT.items():
        if os.environ.get(key) != value:
            os.environ[key] = value
    for key in [k for k in os.environ if k.startswith("TAAF_VLLM_")]:
        os.environ.pop(key, None)
    os.environ.update(profile["env"])
    SETUP_ENV_PATH.write_text(_TURBO_SETUP_ENV_SNAPSHOT)


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


def _turbo_ancestors() -> set:
    pids, pid = set(), os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat[stat.rfind(")") + 2:].split()[1])
        except Exception:
            break
    return pids


def _turbo_serving_pids(live_only: bool = False) -> list:
    """Processes of the failed attempt: vLLM API/engine/worker processes, the PLE-offload worker and
    torch_shm_manager (the markers the bundle's serving_teardown.py uses), plus every process in the saved vLLM
    session. Never this kernel or any of its ancestors; with live_only, never a zombie."""
    skip = _turbo_ancestors()
    session_ids = set()
    try:
        identity = json.loads((WORKING_DIR / "vllm-server-identity.json").read_text())
        session_ids = {int(identity[k]) for k in ("sid", "pgid") if str(identity.get(k, "")).isdigit()} - {0, 1}
    except Exception:
        pass
    found = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in skip:
            continue
        try:
            stat = (proc / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            comm = (proc / "comm").read_text(errors="replace").strip().lower()
            cmdline = (proc / "cmdline").read_bytes().replace(b"\\x00", b" ").decode(errors="replace").lower()
        except Exception:
            continue
        if live_only and fields[0] == "Z":
            continue
        hit = (int(fields[2]) in session_ids or int(fields[3]) in session_ids
               or "vllm::" in comm or "vllm.entrypoints" in cmdline or "qwen38-flash-next-vllm" in cmdline
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


def _turbo_failure_text(attempt: int) -> str:
    text = ""   # only this attempt's records (_turbo_release moved them aside under attempt-numbered names)
    for path in (WORKING_DIR / f"vllm-setup-failure.attempt{attempt}.json",
                 WORKING_DIR / f"vllm-openai-server.attempt{attempt}.log"):
        try:
            text += path.read_text(errors="replace")[-200000:]
        except Exception:
            pass
    return text


def _turbo_release(attempt: int, timeout_s: float = 300.0) -> None:
    _turbo_kill_serving()
    # move the failed attempt's server log and failure record aside: kept for diagnosis, and the next attempt's
    # fallback decision can then only ever read its own (an attempt that fails before start_server writes no log)
    for name, archived in (("vllm-openai-server.log", f"vllm-openai-server.attempt{attempt}.log"),
                           ("vllm-setup-failure.json", f"vllm-setup-failure.attempt{attempt}.json")):
        try:
            (WORKING_DIR / archived).unlink(missing_ok=True)   # never a stale file from an earlier session's run
            if (WORKING_DIR / name).exists():
                os.replace(WORKING_DIR / name, WORKING_DIR / archived)
        except Exception:
            pass
    # serving_setup's own gates need >= 64 GiB MemAvailable (PLE CPU offload), port 1234 free, 36.5 GB free /tmp, and
    # start_server refuses to launch while any live process of the saved server session survives
    min_mem_gib = float(os.environ.get("TURBO_RELEASE_MIN_MEM_GIB", "64"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        used, avail = _turbo_gpu_used_mib(), _turbo_mem_available_gib()
        if ((used is None or used < 2048) and (avail is None or avail >= min_mem_gib) and not _turbo_port_open()
                and not _turbo_serving_pids(live_only=True)):
            break
        time.sleep(5)
        _turbo_kill_serving()
    survivors = _turbo_serving_pids(live_only=True)
    if survivors:   # the identity stays: a live owned process must stay visible to start_server / teardown
        print(f"TURBO_RELEASE_SURVIVORS attempt={attempt} pids={survivors}", flush=True)
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


def _turbo_metric(name: str):
    try:
        with _urlreq.urlopen("http://127.0.0.1:1234/metrics", timeout=20) as response:
            text = response.read().decode(errors="replace")
        pattern = "^" + _re.escape(name) + r"(?:_total)?(?:\\{[^}]*\\})?\\s+([0-9.eE+]+)"
        values = [float(m.group(1)) for m in _re.finditer(pattern, text, _re.M)]
        return sum(values) if values else None
    except Exception:
        return None


def _turbo_prefix_hits():
    return _turbo_metric("vllm:prefix_cache_hits")


def _turbo_healthy() -> bool:
    try:
        base = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
        with _urlreq.urlopen(base + "/models", timeout=30) as response:
            return response.status == 200
    except Exception as exc:
        print(f"TURBO_LIVE_CHECK health failed: {exc!r}"[:300], flush=True)
        return False


def _turbo_load_phase(profile: dict, model: str, image_part, tokens_per_line: float) -> bool:
    """Production-shaped overload for the prefix-caching profile: max_num_seqs+4 concurrent requests with distinct
    ~26k-token prompts (+ the image), thinking on and the harness's sampling (T=1.0, top_p 0.95, top_k 20). Together
    they exceed the KV pool, so the scheduler must preempt running requests and evict cached prefix blocks: the paths
    the short check never reaches. Every request must return HTTP 200 (a 400 = this probe's own size estimate was
    wrong, logged, not held against the server) and the server must stay healthy."""
    count = int(profile["env"].get("TAAF_VLLM_MAX_NUM_SEQS", "16")) + 4
    target = min(float(os.environ.get("TURBO_LOAD_PROMPT_TOKENS", "26000")), 29000.0)
    lines = max(100, int(target / (tokens_per_line + 1.0)))   # +1: one more digit per line than the check's table
    limit_s = float(os.environ.get("TURBO_LOAD_TIMEOUT_S", "300"))
    results = []

    def one(k: int) -> None:
        rng = _random.Random(7000 + k)
        text = (f"Load probe {k:02d}.\\n" + "\\n".join(f"L{i:04d}: {rng.randrange(10000, 100000)}" for i in range(lines))
                + "\\nWhich line holds the largest code? Answer with its label.")
        content = [{"type": "text", "text": text}] + ([image_part] if image_part is not None else [])
        payload = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 256,
                   "min_tokens": 64, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                   "chat_template_kwargs": {"enable_thinking": True}}
        try:
            usage = _turbo_chat(payload, timeout=limit_s).get("usage") or {}
            results.append(("ok", int(usage.get("prompt_tokens") or 0)))
        except _urlerr.HTTPError as exc:
            results.append((f"http{exc.code}", 0))
            if sum(1 for status, _ in results if status != "ok") <= 3:
                print(f"TURBO_LIVE_LOAD request {k} HTTP {exc.code}: {exc.read()[:200]!r}", flush=True)
        except Exception as exc:
            results.append(("error", 0))
            if sum(1 for status, _ in results if status != "ok") <= 3:
                print(f"TURBO_LIVE_LOAD request {k} failed: {exc!r}"[:300], flush=True)

    started = time.monotonic()
    preempted_before = _turbo_metric("vllm:num_preemptions")
    threads = [_threading.Thread(target=one, args=(k,), daemon=True) for k in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(1.0, limit_s + 30.0 - (time.monotonic() - started)))
    healthy = _turbo_healthy()
    preempted_after = _turbo_metric("vllm:num_preemptions")
    statuses = [status for status, _ in results]
    ok = healthy and len(results) == count and all(status in ("ok", "http400") for status in statuses)
    tokens = [n for _, n in results if n]
    print(f"TURBO_LIVE_LOAD profile={profile['name']} ok={ok} seconds={time.monotonic() - started:.1f} "
          f"healthy={healthy} done={len(results)}/{count} statuses={sorted(set(statuses))} "
          f"prompt_tokens={min(tokens) if tokens else None}-{max(tokens) if tokens else None} "
          f"preemptions={preempted_before}->{preempted_after}", flush=True)
    return ok


def _turbo_live_check(profile: dict) -> bool:
    """Real requests shaped like Duck turns: a ~9k-token shared prefix + an image. Sequential first (cold, then the
    same question again = the prefix-cache hit path), then a concurrent burst. HTTP 200 on every request plus a
    healthy server afterwards is required; with prefix caching on, a server that answers the table lookup right
    cold but wrong from cache (corrupted cached state, the realistic failure) is rejected too."""
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

    prompt_tokens = []

    def ask(line: int):
        content = [{"type": "text", "text": "Reference table:\\n" + table
                    + f"\\nReply with only the 5-digit code on line L{line:03d}."}]
        if image_part is not None:
            content.append(image_part)
        payload = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 400,
                   "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            response = _turbo_chat(payload, timeout=150)
            prompt_tokens.append(int((response.get("usage") or {}).get("prompt_tokens") or 0))
            message = (response.get("choices") or [{}])[0].get("message") or {}
            text = f"{message.get('content') or ''} {message.get('reasoning_content') or message.get('reasoning') or ''}"
            return True, str(codes[line]) in text
        except Exception as exc:
            print(f"TURBO_LIVE_CHECK request L{line:03d} failed: {exc!r}"[:300], flush=True)
            return False, False

    started = time.monotonic()
    hits_before = _turbo_prefix_hits()
    cold_ok, cold_right = ask(137)
    warm_ok, warm_right = ask(137)       # identical prompt: the cache-hit path when prefix caching is on
    burst = []
    threads = [_threading.Thread(target=lambda k=k: burst.append(ask(k)), daemon=True) for k in (137, 333, 505, 137)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(1.0, 330.0 - (time.monotonic() - started)))
    time.sleep(10)
    healthy = _turbo_healthy()
    hits_after = _turbo_prefix_hits()
    burst_ok = len(burst) == 4 and all(ok for ok, _ in burst)
    burst_right = sum(1 for _, right in burst if right)
    # the cold lookup is the control: only when the model gets it right do wrong cached answers mean bad state
    corrupted = mode == "prefix" and cold_right and (not warm_right or burst_right < 2)
    ok = healthy and cold_ok and warm_ok and burst_ok and not corrupted
    print(f"TURBO_LIVE_CHECK profile={profile['name']} ok={ok} seconds={time.monotonic() - started:.1f} healthy={healthy} "
          f"cold={cold_ok}/{cold_right} warm={warm_ok}/{warm_right} burst={len(burst)}/{burst_right} "
          f"prefix_hits={hits_before}->{hits_after}", flush=True)
    if ok and mode == "prefix" and os.environ.get("TURBO_LIVE_LOAD", "1") != "0":
        measured = [n for n in prompt_tokens if n > 0]
        tokens_per_line = (min(measured) / len(codes)) if measured else 12.5
        ok = _turbo_load_phase(profile, model, image_part, max(6.0, min(20.0, tokens_per_line)))
        if ok:   # the table's cached blocks were evicted by the overload: the cold lookup must still come back right
            post_ok, post_right = ask(137)
            ok = post_ok and (post_right or not cold_right) and _turbo_healthy()
            print(f"TURBO_LIVE_CHECK post_load profile={profile['name']} ok={ok} answer={post_ok}/{post_right}",
                  flush=True)
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


def _turbo_run_setup(command: str, env: dict, timeout_s: float) -> tuple:
    process = subprocess.Popen(command, shell=True, cwd=WORKING_DIR, env=env, start_new_session=True)
    try:
        return process.wait(timeout=timeout_s), False
    except subprocess.TimeoutExpired:
        print(f"TURBO_SETUP_TIMEOUT after {timeout_s:.0f}s", flush=True)
        try:
            os.killpg(process.pid, _signal.SIGKILL)
        except Exception:
            pass
        try:
            process.wait(timeout=30)
        except Exception:
            pass
        return -9, True


TURBO_SERVING = None
_turbo_setup_commands = json.loads((BUNDLE_DIR / "setup_commands.json").read_text())
_turbo_index = 0
while _turbo_index < len(TURBO_PROFILES):
    _turbo_profile = TURBO_PROFILES[_turbo_index]
    _turbo_apply_profile(_turbo_profile)
    _turbo_t0 = time.monotonic()
    _turbo_elapsed = time.time() - NOTEBOOK_START_EPOCH
    # a non-final attempt may not run past the setup deadline (floor 5 min); the proven profile keeps the full limit
    _turbo_limit = (_TURBO_SETUP_TIMEOUT_S if _turbo_index + 1 == len(TURBO_PROFILES)
                    else min(_TURBO_SETUP_TIMEOUT_S, max(300.0, _TURBO_SETUP_DEADLINE_S - _turbo_elapsed)))
    print(f"TURBO_SERVING_TRY {_turbo_index} {_turbo_profile['name']} elapsed_s={_turbo_elapsed:.0f} "
          f"limit_s={_turbo_limit:.0f} {json.dumps(_turbo_profile['env'], sort_keys=True)}", flush=True)
    env = _command_env()
    _turbo_stage = "ready"
    _turbo_timed_out = False
    for command in _turbo_setup_commands:
        print(f"taaf.kaggle: setup command: {command}", flush=True)
        _turbo_rc, _turbo_timed_out = _turbo_run_setup(command, env,
                                                       max(60.0, _turbo_limit - (time.monotonic() - _turbo_t0)))
        # Re-read in case the command persisted new env keys.
        env = _command_env()
        os.environ.update(env)
        if _turbo_rc != 0:
            print(f"TURBO_SERVING_SETUP_FAILED profile={_turbo_profile['name']} rc={_turbo_rc}", flush=True)
            _turbo_stage = "setup"
            break
    if _turbo_stage == "ready":
        _turbo_serving_proof()
        if not _turbo_live_check(_turbo_profile):
            _turbo_stage = "live_check"
            _turbo_teardown()
    if _turbo_stage == "ready":
        TURBO_SERVING = _turbo_profile
        print(f"TURBO_SERVING_READY {_turbo_profile['name']} seconds={time.monotonic() - _turbo_t0:.0f} "
              f"elapsed_s={time.time() - NOTEBOOK_START_EPOCH:.0f}", flush=True)
        break
    if _turbo_index + 1 == len(TURBO_PROFILES):
        raise RuntimeError("Every TURBO serving profile failed to start.")
    _turbo_release(_turbo_index)
    # Next profile: a prefix-caching failure -> the next (prefix-off) profile; an OOM -> the next (smaller) one;
    # any other failure of a prefix-off MTP-0 profile, a setup hang (more likely MTP-0-wide than prefix-specific) or
    # the setup deadline passed -> straight to the proven one.
    _turbo_oom = _turbo_is_oom(_turbo_failure_text(_turbo_index))
    _turbo_prefix = _turbo_profile["env"].get("TAAF_VLLM_ENABLE_PREFIX_CACHING") == "1"
    _turbo_late = time.time() - NOTEBOOK_START_EPOCH > _TURBO_SETUP_DEADLINE_S
    if _turbo_late or _turbo_timed_out or not (_turbo_prefix or _turbo_oom):
        _turbo_next = len(TURBO_PROFILES) - 1
    else:
        _turbo_next = _turbo_index + 1
    print(f"TURBO_FALLBACK from={_turbo_profile['name']} stage={_turbo_stage} oom={_turbo_oom} late={_turbo_late} "
          f"timed_out={_turbo_timed_out} to={TURBO_PROFILES[_turbo_next]['name']}", flush=True)
    _turbo_index = _turbo_next
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
    (   # the watchdog can only help if it can rebuild the running server's argv: check that contract up front
        "vllm_watchdog_setup = vllm_watchdog.load_setup(BUNDLE_DIR / 'serving_setup.py')\n",
        "vllm_watchdog_setup = vllm_watchdog.load_setup(BUNDLE_DIR / 'serving_setup.py')\n"
        "# TURBO: the watchdog restarts vLLM by rebuilding the saved argv from this environment; prove it can (the base\n"
        "# notebook's TAAF_VLLM_MAX_NUM_BATCHED_TOKENS override silently broke this, so a vLLM crash stayed fatal).\n"
        "try:\n"
        "    assert 'TAAF_VLLM_MAX_NUM_BATCHED_TOKENS' not in os.environ, 'batched-token override breaks restarts'\n"
        "    vllm_watchdog._model_dir_from_identity(vllm_watchdog_setup, vllm_watchdog._identity(vllm_watchdog_setup))\n"
        "    print('TURBO_WATCHDOG_CONTRACT ok', flush=True)\n"
        "except Exception as _turbo_exc:\n"
        "    print(f'TURBO_WATCHDOG_CONTRACT BROKEN {_turbo_exc!r}', flush=True)\n"
        "    if not TRUE_SUBMISSION:\n"
        "        raise\n"
        "# TURBO: if vLLM dies mid-run under the prefix-caching profile, the watchdog relaunches it WITHOUT prefix caching\n"
        "# (same KV pool; the one axis no Kaggle run has used) instead of relaunching the same argv into the same crash.\n"
        "# restart_owned_server checks the saved argv against this environment before it calls start_server, so the\n"
        "# switch happens inside start_server; the new identity then matches the new environment for later restarts.\n"
        "if (os.environ.get('TAAF_VLLM_ENABLE_PREFIX_CACHING') == '1' and hasattr(vllm_watchdog_setup, 'start_server')\n"
        "        and os.environ.get('TURBO_RESTART_NO_PREFIX', '1') != '0'):\n"
        "    _turbo_start_server = vllm_watchdog_setup.start_server\n"
        "\n"
        "    def _turbo_start_server_no_prefix(model_dir, env, *args, **kwargs):\n"
        "        if os.environ.get('TAAF_VLLM_ENABLE_PREFIX_CACHING') != '1':\n"
        "            return _turbo_start_server(model_dir, env, *args, **kwargs)\n"
        "        os.environ['TAAF_VLLM_ENABLE_PREFIX_CACHING'] = '0'\n"
        "        print('TURBO_WATCHDOG_RESTART_NO_PREFIX', flush=True)\n"
        "        try:\n"
        "            return _turbo_start_server(model_dir, env, *args, **kwargs)\n"
        "        except BaseException:\n"
        "            try:   # nothing was launched with the new argv: keep the environment matching the saved identity\n"
        "                vllm_watchdog._model_dir_from_identity(vllm_watchdog_setup, vllm_watchdog._identity(vllm_watchdog_setup))\n"
        "            except Exception:\n"
        "                os.environ['TAAF_VLLM_ENABLE_PREFIX_CACHING'] = '1'\n"
        "            raise\n"
        "\n"
        "    vllm_watchdog_setup.start_server = _turbo_start_server_no_prefix\n"
        "    print('TURBO_WATCHDOG_RESTART_POLICY prefix_off_on_restart', flush=True)\n",
    ),
    (   # a failed summary write must not turn a finished competition run into an error
        "    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)\n"
        "    bm._save_json()\n",
        "    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)\n"
        "    try:\n"
        "        bm._save_json()\n"
        "    except Exception as _turbo_exc:   # TURBO\n"
        "        print(f'TURBO_SAVE_JSON_ERROR {_turbo_exc!r}', flush=True)\n"
        "        if not TRUE_SUBMISSION:\n"
        "            raise\n",
    ),
    (   # validation audit: a game ended by the TURBO analyzer guard is reported, not fatal to the commit run
        "            if run.state not in {'won', 'gave_up', 'cancelled'}\n",
        "            if run.state not in {'won', 'gave_up', 'cancelled', 'crashed'}\n",
    ),
    (
        "        if crashed:\n"
        "            raise RuntimeError(f'Public runs crashed: {crashed}.')\n",
        "        if crashed:   # TURBO: report loudly; the commit version must stay submittable\n"
        "            print(f'TURBO_WARNING public runs crashed: {crashed} '\n"
        "                  f'{[(run.game_id, run.solver_note) for run in public_runs if run.state == \"crashed\"]}', flush=True)\n",
    ),
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
    (   # the frozen scorer only counts a pass whose runs ALL finalized as won/gave_up/cancelled: with a crashed game
        # it raises ValueError; keep the commit version submittable (the placeholder parquet is already written)
        "        score_summary = evaluate_runs([WORKING_DIR])\n"
        "        score_path = save_score_file(\n"
        "            score_summary,\n"
        "            run_dirs=[WORKING_DIR],\n"
        "            output_path=WORKING_DIR / \"score.json\",\n"
        "        )\n"
        "        if Path(score_path) != WORKING_DIR / 'score.json' or not Path(score_path).is_file():\n"
        "            raise RuntimeError(f'Frozen scorer did not write score.json: {score_path}.')\n"
        "        print(\n"
        "            f'PUBLIC25_AUDIT runs=25 actions={total_actions} score_path={score_path}',\n"
        "            flush=True,\n"
        "        )\n",
        "        try:\n"
        "            score_summary = evaluate_runs([WORKING_DIR])\n"
        "        except ValueError as _turbo_exc:   # TURBO\n"
        "            if not crashed:\n"
        "                raise\n"
        "            score_summary = None\n"
        "            print(f'TURBO_SCORE_SKIPPED crashed={crashed} {_turbo_exc!r}', flush=True)\n"
        "        if score_summary is not None:\n"
        "            score_path = save_score_file(\n"
        "                score_summary,\n"
        "                run_dirs=[WORKING_DIR],\n"
        "                output_path=WORKING_DIR / \"score.json\",\n"
        "            )\n"
        "            if Path(score_path) != WORKING_DIR / 'score.json' or not Path(score_path).is_file():\n"
        "                raise RuntimeError(f'Frozen scorer did not write score.json: {score_path}.')\n"
        "            print(\n"
        "                f'PUBLIC25_AUDIT runs={len(public_runs)} actions={total_actions} score_path={score_path}',\n"
        "                flush=True,\n"
        "            )\n",
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
