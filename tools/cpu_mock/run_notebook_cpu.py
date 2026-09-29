#!/usr/bin/env python3
"""Execute a competition notebook end to end on CPU against the mock LLM (crash-safety harness).

    python tools/cpu_mock/run_notebook_cpu.py NOTEBOOK.ipynb --out executed.ipynb [--runtime-s 120] [--env K=V ...]
    python tools/cpu_mock/run_notebook_cpu.py NOTEBOOK.ipynb --out executed.ipynb --competition \
        --taaf-src <duck-harness>/tufa-arc-agi-framework/src --env-files <environment_files> [--competition-run-s 600]

Requires the fake /kaggle tree (build_fake_kaggle.py) and a Jupyter kernel with arc-agi 0.9.8 /
arcengine 0.9.3 (kernel name via --kernel).

Default mode = the notebook's own offline validation branch on the 25 public games; AGENTFIX_VALIDATION_RUNTIME_S
caps it. --competition rehearses the REAL submission branch (KAGGLE_IS_COMPETITION_RERUN=1): a local
competition-mode Arcade gateway (taaf.competition_arcade, 110 cloned public games, one scorecard) is started on
127.0.0.1:8001, and a cell is injected before the run cell that shrinks the notebook's 9 h budget so the whole
110-game schedule (waves, per-game budgets, soft end, teardown) plays out in --competition-run-s seconds.
Exits non-zero if any cell raises.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

SHRINK_CELL = """# MOCK ONLY (tools/cpu_mock/run_notebook_cpu.py --competition): shrink the 9 h budget for a CPU rehearsal.
import time as _mock_time
target.max_runtime_s = (_mock_time.time() - NOTEBOOK_START_EPOCH) + 600.0 + float(os.environ["MOCK_COMPETITION_RUN_S"])
print("MOCK_COMPETITION target.max_runtime_s", round(target.max_runtime_s), flush=True)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", default="arcvenv")
    ap.add_argument("--runtime-s", type=float, default=120.0)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--competition", action="store_true")
    ap.add_argument("--competition-run-s", type=float, default=600.0)
    ap.add_argument("--taaf-src")
    ap.add_argument("--env-files")
    a = ap.parse_args()
    os.environ["AGENTFIX_VALIDATION_RUNTIME_S"] = str(a.runtime_s)
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    nb = nbformat.read(a.notebook, as_version=4)
    server = None
    if a.competition:
        sys.path.insert(0, a.taaf_src)
        import taaf.competition_arcade as arcade

        server = arcade.CompetitionArcadeServer(game_ids=arcade.official_game_ids(), total_runs=110, port=8001,
                                                environments_dir=a.env_files).start()
        print("mock competition gateway:", server.base_url, len(server.exposed_game_ids), "games")
        os.environ.update({"KAGGLE_IS_COMPETITION_RERUN": "1", "ARC_BASE_URL": server.base_url + "/",
                           "ARC_API_KEY": arcade.DEFAULT_API_KEY,
                           "MOCK_COMPETITION_RUN_S": str(a.competition_run_s)})
        index = next(i for i, c in enumerate(nb.cells)
                     if c.cell_type == "code" and "def _competition_games()" in c.source)
        nb.cells.insert(index, nbformat.v4.new_code_cell(SHRINK_CELL))
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
        if server is not None:
            server.stop()
    print(f"elapsed {time.time() - t0:.1f}s -> {a.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
