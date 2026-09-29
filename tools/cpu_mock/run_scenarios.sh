#!/bin/bash
# Sequential CPU mock scenarios for the TURBO notebook (port 1234 and /kaggle/working are shared).
#   A  mock vLLM dies mid-run under the prefix profile -> watchdog relaunches it with prefix caching off
#   B  the prefix profile fails the overload phase -> next (prefix-off) profile
#   C  OOM on profile 0, generic failure on profile 1 -> proven MTP-3 profile (OOM read from each attempt's own records)
#   D  setup hang on profile 0 -> attempt limit fires -> straight to the proven MTP-3 profile
#   E  failed setup leaves an orphan holding port 1234 -> killed, next profile starts
#   F  corrupted cache-hit answers -> prefix profile rejected -> next profile
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${TURBO_SCENARIO_PY:-/home/user/arcvenv/bin/python}
OUT=${TURBO_SCENARIO_OUT:-/tmp/turbo_scenarios}
NB=$REPO/notebooks/arc3-flashnext-turbo.ipynb
mkdir -p $OUT

clean() {
  pkill -f mock_llm_server.py 2>/dev/null; sleep 1
  rm -f /kaggle/working/vllm-* /kaggle/working/mock-llm-server.pid
}

run() {  # name runtime checker-args... -- env...
  local name=$1 runtime=$2; shift 2
  local check=() envs=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do check+=("$1"); shift; done
  [ $# -gt 0 ] && shift
  for kv in "$@"; do envs+=(--env "$kv"); done
  clean
  rm -f $OUT/$name.jsonl
  echo "=== $name (runtime $runtime) ${envs[*]:-}"
  $PY $REPO/tools/cpu_mock/run_notebook_cpu.py $NB --out $OUT/$name.ipynb --runtime-s $runtime \
      --env MOCK_LOG=$OUT/$name.jsonl --env TURBO_RELEASE_MIN_MEM_GIB=0 "${envs[@]}" > $OUT/$name.run.log 2>&1
  echo "rc=$? $(tail -1 $OUT/$name.run.log)"
  $PY $REPO/tools/cpu_mock/check_turbo_run.py $OUT/$name.ipynb $OUT/$name.jsonl "${check[@]}" 2>&1 | tail -25
  $PY - "$OUT/$name.ipynb" <<'PYEOF'
import json, re, sys
nb = json.load(open(sys.argv[1]))
keys = ("SERVING_TRY", "FALLBACK", "SERVING_READY", "LIVE_CHECK", "LIVE_LOAD", "RELEASED", "RELEASE_SURVIVORS",
        "SETUP_TIMEOUT", "WATCHDOG", "SCORE_SKIPPED", "WARNING")
for cell in nb["cells"]:
    for out in cell.get("outputs", []):
        for line in "".join(out.get("text", "")).splitlines():
            if any(line.startswith("TURBO_" + k) for k in keys):
                print(line[:260])
PYEOF
  cp /kaggle/working/vllm-watchdog-events.jsonl $OUT/$name.watchdog.jsonl 2>/dev/null
}

case "${1:-all}" in
  A|all) run A_crash_restart 240 --expect-profile mtp0-kv12-prefix-s16 --expect-load --expect-restart-no-prefix \
           -- MOCK_CRASH_AFTER_GAME_REQUESTS=80 ;;&
  B|all) run B_load_fail 90 --expect-profile mtp0-kv10-s14 -- MOCK_FAIL_LOAD_SEQS=16 ;;&
  C|all) run C_oom_then_generic 90 --expect-profile base-mtp3-kv5-s16 --temperature 0.6 \
           -- MOCK_FAIL_OOM_SEQS=16/0 MOCK_FAIL_SETUP_SEQS=14/0 ;;&
  D|all) run D_hang 90 --expect-profile base-mtp3-kv5-s16 --temperature 0.6 \
           -- MOCK_HANG_SEQS=16/0 TURBO_SETUP_ATTEMPT_TIMEOUT_S=45 ;;&
  E|all) run E_orphan 90 --expect-profile mtp0-kv10-s14 -- MOCK_FAIL_ORPHAN_SEQS=16/0 ;;&
  F|all) run F_corrupt_cache 90 --expect-profile mtp0-kv10-s14 -- MOCK_CORRUPT_WARM=16 ;;&
esac
clean
echo done
