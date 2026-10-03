#!/bin/bash
# Set RUN_SUFFIX (e.g. RUN_SUFFIX=_rerun1) to write new runs next to the released ones; without it, stored episodes are skipped.
# The undefended agent in our harness, several independent draws (run names undefended_draw<k>_benign, _direct, _injected).
# usage: scripts/run_undefended_draws.sh ["1 2 3 4 5"] [benign|direct|injected|all]   (injected = the AgentDojo suites; all = benign and direct)
cd "$(dirname "$0")/.."
DRAWS=${1:-1 2 3 4 5}; WHAT=${2:-all}; DOJO=banking,slack,travel,workspace; ADYN=dailylife,github,shopping; mkdir -p data/runs/logs
H="uv run python src/agent_harness/run_episodes.py"
run() { for T in 1 2 3; do "$@" && return 0; done; return 1; }
job() { K=$1; W=$2; RID=undefended${RUN_SUFFIX:-}_draw${K}_$W; L=data/runs/logs/$RID.log; D=""; [ $W = direct ] && D="--direct_attack"
  [ $W = injected ] && { run $H --suites $DOJO --do_attack --attack_type important_instructions --run_id $RID >> $L 2>&1; echo "$RID done"; return; }
  run $H --suites $DOJO $D --run_id $RID >> $L 2>&1
  run env PYTHONPATH=data/benchmarks/AgentDyn/src $H --suites $ADYN $D --run_id $RID >> $L 2>&1; echo "$RID done"; }
for K in $DRAWS; do for W in $([ $WHAT = all ] && echo "benign direct" || echo $WHAT); do job $K $W & done; done
wait; echo "all done $(date)"
