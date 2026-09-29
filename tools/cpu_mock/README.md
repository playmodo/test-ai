# CPU mock harness for the ARC-AGI-3 Duck notebook

Runs the competition notebook end to end on CPU (no GPU, no vLLM) so notebook-level patches can be
checked for crashes before a Kaggle submission.

1. Python 3.12 venv with `arc-agi==0.9.8 arcengine==0.9.3 requests pillow matplotlib numpy scipy imageio
   imageio-ffmpeg python-dotenv pandas pyarrow ipykernel nbclient nbformat`, registered as kernel `arcvenv`.
2. `environment_files/` for the 25 public games (`<game>/<ver>/{<game>.py,metadata.json}`).
3. `python build_fake_kaggle.py --duck-src <Tufalabs/duck-harness clone> --bundle-ref <dir with the bundle's
   serving_setup.py, vllm_server_watchdog.py, benchmark_initial.pkl, deploy_target.pkl, SOURCE_IDENTITY.json,
   taaf-kaggle-bundle.json> --env-files <environment_files>`
4. `python run_notebook_cpu.py NOTEBOOK.ipynb --out executed.ipynb --runtime-s 90`

`mock_serving_setup.py` imports the real `serving_setup.py`, so the TAAF_VLLM_* knobs set by the notebook are
validated by the real `resolve_vllm_tuning()` and the exact vLLM command line is printed; the analyzer env is
persisted by the real `persist_analyzer_environment()`. `mock_llm_server.py` answers with random-action python
tool calls (plus a few degenerate replies) and logs every request's size / images / sampling params.
