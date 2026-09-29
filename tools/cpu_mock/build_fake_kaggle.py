#!/usr/bin/env python3
"""Build a fake /kaggle tree so the competition notebook runs unmodified on CPU against a mock LLM.

    python tools/cpu_mock/build_fake_kaggle.py --duck-src <tufalabs/duck-harness> \
        --bundle-ref <dir with pristine serving_setup.py, vllm_server_watchdog.py, *.pkl> \
        --env-files <environment_files dir with the 25 public games> [--root /kaggle]

Layout produced (same mount paths the notebook probes):
  /kaggle/input/datasets/keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1/   mock source bundle
  /kaggle/input/datasets/keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1/ (empty)
  /kaggle/input/competitions/arc-prize-2026-arc-agi-3/{arc_agi_3_wheels,environment_files}
  /kaggle/working/                                                           (emptied)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duck-src", type=Path, required=True)
    ap.add_argument("--bundle-ref", type=Path, required=True)
    ap.add_argument("--env-files", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("/kaggle"))
    a = ap.parse_args()
    root = a.root
    bundle = root / "input/datasets/keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "src").mkdir(parents=True)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", ".git")
    shutil.copytree(a.duck_src / "ARC3-Inference", bundle / "src/ARC3-Inference", ignore=ign)
    shutil.copytree(a.duck_src / "tufa-arc-agi-framework", bundle / "src/tufa-arc-agi-framework", ignore=ign)
    for name in ("taaf-kaggle-bundle.json", "vllm_server_watchdog.py", "benchmark_initial.pkl",
                 "deploy_target.pkl", "SOURCE_IDENTITY.json"):
        shutil.copy(a.bundle_ref / name, bundle / name)
    shutil.copy(a.bundle_ref / "serving_setup.py", bundle / "real_serving_setup.py")
    shutil.copy(HERE / "mock_serving_setup.py", bundle / "serving_setup.py")
    shutil.copy(HERE / "mock_serving_teardown.py", bundle / "serving_teardown.py")
    shutil.copy(HERE / "mock_llm_server.py", bundle / "mock_llm_server.py")
    (bundle / "setup_commands.json").write_text(json.dumps(['"$PYTHON" "$TAAF_KAGGLE_BUNDLE_DIR/serving_setup.py"']))
    (bundle / "teardown_commands.json").write_text(json.dumps(['"$PYTHON" "$TAAF_KAGGLE_BUNDLE_DIR/serving_teardown.py"']))
    (root / "input/datasets/keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1").mkdir(parents=True, exist_ok=True)
    comp = root / "input/competitions/arc-prize-2026-arc-agi-3"
    (comp / "arc_agi_3_wheels").mkdir(parents=True, exist_ok=True)
    if (comp / "environment_files").exists():
        shutil.rmtree(comp / "environment_files")
    shutil.copytree(a.env_files, comp / "environment_files")
    work = root / "working"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    print("fake kaggle tree ready at", root)


if __name__ == "__main__":
    main()
