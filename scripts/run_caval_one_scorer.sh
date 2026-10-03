#!/bin/bash
# Set RUN_SUFFIX (e.g. RUN_SUFFIX=_rerun1) to write new runs next to the released ones; without it, stored episodes are skipped.
# One CAVAL scorer online on the seven suites (GPT-4o-mini agent; needs OPENAI_API_KEY). Rerun-safe: stored episodes are skipped.
# usage: [ALPHA=0.30] [ENFORCE=isolate|stop] [PATHS=learned|rules|both] [SUITES=all|banking,slack,...] scripts/run_caval_one_scorer.sh <scorer 1-5> [benign|direct|injected|all]
# Run names: caval_deployed (alpha 0.30, recovery, learned scorer), caval_stop_alpha030, caval_stop_alpha025, caval_recovery_alpha025,
# rules_only and rules_and_caval (component analysis), each with _scorer<k> and _benign/_direct for the benign and direct episodes.
cd "$(dirname "$0")/.."
S=$1; WHAT=${2:-all}; ALPHA=${ALPHA:-0.30}; ENFORCE=${ENFORCE:-isolate}; PATHS=${PATHS:-learned}; SUITES=${SUITES:-all}
A=$(python3 -c "print(f'{int(round($ALPHA*100)):03d}')")
case $PATHS in
  learned) PFLAG=--no_gate; if [ $ENFORCE = isolate ] && [ $A = 030 ]; then NAME=caval_deployed; elif [ $ENFORCE = isolate ]; then NAME=caval_recovery_alpha$A; else NAME=caval_stop_alpha$A; fi;;
  rules)   PFLAG=--no_score; NAME=rules_only;;
  both)    PFLAG="";         NAME=rules_and_caval;;
  *) echo "PATHS must be learned|rules|both"; exit 1;;
esac
RID=${NAME}${RUN_SUFFIX:-}_scorer$S; L=data/runs/logs/$RID.log; mkdir -p data/runs/logs
C="--caval $PFLAG --alpha_esc $ALPHA --enforce $ENFORCE --checkpoint results/checkpoints/caval_scorer_seed$S.pt"
DOJO=banking,slack,travel,workspace; ADYN=dailylife,github,shopping
if [ $SUITES != all ]; then DOJO=$(echo $SUITES | tr , "\n" | grep -E "banking|slack|travel|workspace" | paste -sd,); ADYN=$(echo $SUITES | tr , "\n" | grep -E "dailylife|github|shopping" | paste -sd,); fi
run() { case "$*" in *"--suites  "*|*"--suites --"*) return 0;; esac; for T in 1 2 3; do "$@" && return 0; echo "retry $T: $*"; done; return 1; }
H="uv run python src/agent_harness/run_episodes.py"; HD="env PYTHONPATH=data/benchmarks/AgentDyn/src $H"
[ $WHAT = benign ]   || [ $WHAT = all ] && { run $H --suites $DOJO $C --run_id ${RID}_benign >> $L 2>&1; run $HD --suites $ADYN $C --run_id ${RID}_benign >> $L 2>&1; }
[ $WHAT = direct ]   || [ $WHAT = all ] && { run $H --suites $DOJO --direct_attack $C --run_id ${RID}_direct >> $L 2>&1; run $HD --suites $ADYN --direct_attack $C --run_id ${RID}_direct >> $L 2>&1; }
[ $WHAT = injected ] || [ $WHAT = all ] && { run $H --suites $DOJO --do_attack --attack_type important_instructions $C --run_id $RID >> $L 2>&1
                                             run $HD --suites $ADYN --do_attack --attack_type important_instructions $C --run_id $RID >> $L 2>&1; }
echo "$RID $WHAT done"
