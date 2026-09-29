#!/usr/bin/env python3
"""CPU stand-in for the bundle's serving_setup.py (installed as <mock bundle>/serving_setup.py).

It imports the REAL serving_setup.py (copied next to it as real_serving_setup.py) so the notebook's
TAAF_VLLM_* serving knobs are validated by the real `resolve_vllm_tuning()` and the exact vLLM command
line the real script would launch is printed. Instead of vLLM it starts tools/cpu_mock/mock_llm_server.py
on 127.0.0.1:1234 and persists the analyzer environment through the real `persist_analyzer_environment()`
(so LOCAL_ANALYZER_CONTEXT_WINDOW etc. are persisted exactly as on Kaggle). The module also re-exports what
vllm_server_watchdog.py reads (BASE_URL, SERVED_MODEL_NAME, WORKING_DIR, request_json, ...).
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
                                    "proc_record", "stop_owned_server", "read_json", "write_json"}:
        globals().setdefault(_name, getattr(real, _name))

MOCK_SERVER = Path(os.environ.get("MOCK_LLM_SERVER", str(_HERE / "mock_llm_server.py")))
PID_FILE = Path(os.environ["TAAF_KAGGLE_WORKING_DIR"]) / "mock-llm-server.pid"


def main() -> None:
    t0 = time.monotonic()
    tuning = real.resolve_vllm_tuning()
    print("VLLM_SETUP_MODE mock", flush=True)
    print("MOCK_VLLM_TUNING " + json.dumps(tuning, sort_keys=True), flush=True)
    cmd = real.server_command(Path("/kaggle/input/models/mock-model"), tuning=tuning)
    print("VLLM_START_COMMAND " + json.dumps(cmd[1:]), flush=True)
    log = open(Path(os.environ["TAAF_KAGGLE_WORKING_DIR"]) / "vllm-openai-server.log", "a")
    proc = subprocess.Popen([sys.executable, str(MOCK_SERVER)], stdout=log, stderr=log,
                            start_new_session=True, env=os.environ.copy())
    PID_FILE.write_text(str(proc.pid))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            real.request_json(f"{real.BASE_URL}/models", timeout=2)
            break
        except Exception:
            time.sleep(0.2)
    else:
        raise RuntimeError("mock LLM server did not start")
    env = {k: os.environ.get(k, "") for k in ("PYTHONPATH", "PATH", "LD_LIBRARY_PATH")}
    env.update({"CUDA_HOME": "/usr/local/cuda", "CUDACXX": "/usr/local/cuda/bin/nvcc"})
    real.persist_analyzer_environment(env)
    print("VLLM_SETUP_COMPLETE " + json.dumps({"mode": "mock", "pid": proc.pid,
                                               "ready_seconds": time.monotonic() - t0}), flush=True)


if __name__ == "__main__":
    main()
