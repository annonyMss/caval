#!/bin/bash
# Set RUN_SUFFIX (e.g. RUN_SUFFIX=_rerun1) to write new runs next to the released ones; without it, stored episodes are skipped.
# AgentDojo's important_instructions attack with less or wrong knowledge of the user and model names (Figure 5), on
# Banking, Travel and Slack, for CAVAL (scorer 1, deployed configuration) or the undefended agent.
# usage: scripts/run_adversary_knowledge_variants.sh <no_names|wrong_user_name> <caval|undefended> [suites=banking,travel,slack]
# Run names caval_deployed_<variant> and undefended_<variant>.
cd "$(dirname "$0")/.."
V=$1; WHO=$2; SUITES=${3:-banking,travel,slack}; mkdir -p data/runs/logs
H="uv run python src/agent_harness/run_episodes.py --suites $SUITES --do_attack --attack_type important_instructions_$V"
case $WHO in
  caval)      ARGS="--caval --no_gate --enforce isolate --alpha_esc 0.30 --checkpoint results/checkpoints/caval_scorer_seed1.pt"; RID=caval_deployed_$V${RUN_SUFFIX:-};;
  undefended) ARGS=""; RID=undefended_$V${RUN_SUFFIX:-};;
  *) echo "second argument must be caval|undefended"; exit 1;;
esac
for T in 1 2 3; do $H $ARGS --run_id $RID >> data/runs/logs/$RID.log 2>&1 && break; echo "retry $T"; done
echo "$RID done"
