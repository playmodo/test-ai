#!/usr/bin/env python3
"""CPU stand-in for the bundle's serving_setup.py (installed as <mock bundle>/serving_setup.py).

It imports the REAL serving_setup.py (copied next to it as real_serving_setup.py) so the notebook's
TAAF_VLLM_* serving knobs are validated by the real `resolve_vllm_tuning()` and the exact vLLM command
line the real script would launch is printed. Instead of vLLM it starts tools/cpu_mock/mock_llm_server.py
on 127.0.0.1:1234, writes a server identity whose argv is the real `server_command()` (so the watchdog's
restart contract can be checked), and persists the analyzer environment through the real
`persist_analyzer_environment()` (so LOCAL_ANALYZER_CONTEXT_WINDOW etc. are persisted exactly as on Kaggle).
The module also re-exports what vllm_server_watchdog.py reads (BASE_URL, SERVED_MODEL_NAME, WORKING_DIR, ...).

Failure injection (env, matched against the profile's max_num_seqs):
  MOCK_FAIL_SETUP_SEQS=16,14   raise before starting (a generic startup failure)
  MOCK_FAIL_OOM_SEQS=16        raise with a CUDA out-of-memory failure record
  MOCK_FAIL_ORPHAN_SEQS=16     leave a vLLM-looking orphan process holding port 1234, then raise
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("real_serving_setup", _HERE / "real_serving_setup.py")
real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(real)

for _name in dir(real):
    if _name.isupper() or _name in {"request_json", "request_bytes", "alive", "port_open", "tail_log",
                                    "proc_record", "stop_owned_server", "read_json", "write_json",
                                    "server_command", "resolve_vllm_tuning"}:
        globals().setdefault(_name, getattr(real, _name))

MOCK_SERVER = Path(os.environ.get("MOCK_LLM_SERVER", str(_HERE / "mock_llm_server.py")))
WORKDIR = Path(os.environ["TAAF_KAGGLE_WORKING_DIR"])
PID_FILE = WORKDIR / "mock-llm-server.pid"
MODEL_DIR = Path(os.environ.get("MOCK_MODEL_DIR", "/kaggle/input/models/mock-model"))


def _listed(env_key: str, seqs: int) -> bool:
    return str(seqs) in [s for s in os.environ.get(env_key, "").split(",") if s]


def _wait_ready(timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            real.request_json(f"{real.BASE_URL}/models", timeout=2)
            return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("mock LLM server did not start (port 1234 busy?)")


def main() -> None:
    t0 = time.monotonic()
    tuning = real.resolve_vllm_tuning()
    print("VLLM_SETUP_MODE mock", flush=True)
    print("MOCK_VLLM_TUNING " + json.dumps(tuning, sort_keys=True), flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "chat_template.jinja").write_text("{# mock #}\n")
    argv = real.server_command(MODEL_DIR, tuning=tuning)
    print("VLLM_START_COMMAND " + json.dumps(argv[1:]), flush=True)
    seqs = int(tuning["max_num_seqs"])
    log = open(WORKDIR / "vllm-openai-server.log", "a")
    log.write(f"INFO mock server for max_num_seqs={seqs}\n")
    log.flush()
    if _listed("MOCK_FAIL_ORPHAN_SEQS", seqs):
        # a detached "vLLM worker" that survives the failed setup and keeps port 1234
        subprocess.Popen([sys.executable, str(MOCK_SERVER), "vllm.entrypoints.cli.main", "serve", "orphan"],
                         stdout=log, stderr=log, start_new_session=True, env=os.environ.copy())
        time.sleep(1.0)
        raise RuntimeError("mock: setup failed and left an orphan server behind")
    if _listed("MOCK_FAIL_OOM_SEQS", seqs):
        real.write_json(real.FAILURE_PATH, {"error_type": "RuntimeError",
                                            "error": "CUDA error: out of memory (mock)", "server_log_tail": ""})
        raise RuntimeError("mock: CUDA error: out of memory")
    if _listed("MOCK_FAIL_SETUP_SEQS", seqs):
        raise RuntimeError(f"mock: simulated setup failure for max_num_seqs={seqs}")
    proc = subprocess.Popen([sys.executable, str(MOCK_SERVER)], stdout=log, stderr=log,
                            start_new_session=True, env=os.environ.copy())
    PID_FILE.write_text(str(proc.pid))
    _wait_ready()
    identity = {"pid": proc.pid, "start_ticks": real.start_ticks(proc.pid), "sid": os.getsid(proc.pid),
                "pgid": os.getpgid(proc.pid), "argv": argv, "workers": [], "mock": True}
    real.write_json(real.SERVER_IDENTITY, identity)
    env = {k: os.environ.get(k, "") for k in ("PYTHONPATH", "PATH", "LD_LIBRARY_PATH")}
    env.update({"CUDA_HOME": "/usr/local/cuda", "CUDACXX": "/usr/local/cuda/bin/nvcc"})
    real.persist_analyzer_environment(env)
    print("VLLM_SETUP_COMPLETE " + json.dumps({"mode": "mock", "pid": proc.pid,
                                               "ready_seconds": time.monotonic() - t0}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print("VLLM_SETUP_FAILED", json.dumps({"error": str(exc)}), flush=True)
        raise
