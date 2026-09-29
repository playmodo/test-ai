#!/usr/bin/env python3
"""CPU stand-in for serving_teardown.py: stops the mock LLM server."""
import os
import signal
from pathlib import Path

pid_file = Path(os.environ.get("TAAF_KAGGLE_WORKING_DIR", "/kaggle/working")) / "mock-llm-server.pid"
if pid_file.exists():
    try:
        os.killpg(int(pid_file.read_text()), signal.SIGTERM)
    except Exception as exc:  # already gone
        print("mock teardown:", exc)
print("VLLM_SERVER_TEARDOWN mock", flush=True)
