#!/usr/bin/env python3
"""Execute a competition notebook end to end on CPU against the mock LLM (crash-safety harness).

    python tools/cpu_mock/run_notebook_cpu.py NOTEBOOK.ipynb --out executed.ipynb [--runtime-s 120] [--env K=V ...]

Requires the fake /kaggle tree (build_fake_kaggle.py) and a Jupyter kernel with arc-agi 0.9.8 /
arcengine 0.9.3 (kernel name via --kernel). The run cell's own offline branch plays the 25 public
games; AGENTFIX_VALIDATION_RUNTIME_S caps the run so it ends quickly. Exits non-zero if any cell raises.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import nbformat
from nbclient import NotebookClient


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", default="arcvenv")
    ap.add_argument("--runtime-s", type=float, default=120.0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    os.environ["AGENTFIX_VALIDATION_RUNTIME_S"] = str(a.runtime_s)
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    nb = nbformat.read(a.notebook, as_version=4)
    client = NotebookClient(nb, kernel_name=a.kernel, timeout=a.timeout, allow_errors=False,
                            resources={"metadata": {"path": "/kaggle/working"}})
    t0 = time.time()
    rc = 0
    try:
        client.execute()
    except Exception as exc:  # CellExecutionError carries the traceback
        print("NOTEBOOK FAILED:", str(exc)[:6000])
        rc = 1
    finally:
        nbformat.write(nb, a.out)
    print(f"elapsed {time.time() - t0:.1f}s -> {a.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
