# CPU mock harness for the ARC-AGI-3 Duck notebook

Runs the competition notebook end to end on CPU (no GPU, no vLLM) so notebook-level patches can be
checked for crashes before a Kaggle submission.

1. Python 3.12 venv with `arc-agi==0.9.8 arcengine==0.9.3 requests pillow matplotlib numpy scipy imageio
   imageio-ffmpeg python-dotenv pandas pyarrow ipykernel nbclient nbformat`, registered as kernel `arcvenv`.
2. `environment_files/` for the 25 public games (`<game>/<ver>/{<game>.py,metadata.json}`).
3. `python build_fake_kaggle.py --duck-src <Tufalabs/duck-harness clone> --bundle-ref <dir with the bundle's
   serving_setup.py, vllm_server_watchdog.py, benchmark_initial.pkl, deploy_target.pkl, SOURCE_IDENTITY.json,
   taaf-kaggle-bundle.json> --env-files <environment_files>`
4. Offline validation branch (25 public games):
   `python run_notebook_cpu.py NOTEBOOK.ipynb --out executed.ipynb --runtime-s 150 --env MOCK_LOG=/tmp/req.jsonl --env TURBO_RELEASE_MIN_MEM_GIB=0`
5. Real-submission branch rehearsal (`KAGGLE_IS_COMPETITION_RERUN=1`, local competition-mode gateway with 110 cloned
   games and one scorecard, 9 h budget shrunk to `--competition-run-s`):
   `python run_notebook_cpu.py NOTEBOOK.ipynb --out executed.ipynb --competition --competition-run-s 720
   --taaf-src <duck-harness>/tufa-arc-agi-framework/src --env-files /kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files
   --env TURBO_BUDGET_FLOOR_S=60 --env TURBO_BUDGET_RESERVE_S=30 --env TURBO_RELEASE_MIN_MEM_GIB=0`
6. `python check_turbo_run.py executed.ipynb /tmp/req.jsonl [--expect-profile NAME] [--temperature 0.6]`

`mock_serving_setup.py` imports the real `serving_setup.py`, so the TAAF_VLLM_* knobs set by the notebook are
validated by the real `resolve_vllm_tuning()`, the exact vLLM command line is printed, a server identity with the real
`server_command()` argv is written (the watchdog restart contract is checked against it) and the analyzer env is
persisted by the real `persist_analyzer_environment()`. `mock_llm_server.py` answers game turns with random-action
python tool calls (plus a few degenerate replies), answers the notebook's live serving check, and logs every
request's size / images / sampling params / user-prompt tail.

Failure injection (matched against the profile's `max_num_seqs`): `MOCK_FAIL_SETUP_SEQS`, `MOCK_FAIL_OOM_SEQS`,
`MOCK_FAIL_ORPHAN_SEQS` (leaves a vLLM-looking process holding port 1234), `MOCK_FAIL_LIVE_SEQS` (HTTP 500 on the live
check), `MOCK_CORRUPT_WARM` (wrong answers on the cache-hit path of the live check).
