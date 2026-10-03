#!/bin/bash
# Set RUN_SUFFIX (e.g. RUN_SUFFIX=_rerun1) to write new runs next to the released ones; without it, stored episodes are skipped.
# Injected attacks whose text addresses the verifier itself (Figure 5), for CAVAL (scorer 1, deployed configuration), the
# undefended agent, or DRIFT. Run names caval_deployed_verifier_directed, undefended_verifier_directed, drift_verifier_directed.
# usage: scripts/run_verifier_directed_attack.sh <caval|undefended|drift> [suites=banking,travel,slack,workspace]
cd "$(dirname "$0")/.."
WHO=$1; SUITES=${2:-banking,travel,slack,workspace}; mkdir -p data/runs/logs
H="uv run python src/agent_harness/run_episodes.py --suites $SUITES --do_attack --attack_type important_instructions --adaptive_attack"
case $WHO in
  caval)      ARGS="--caval --no_gate --enforce isolate --alpha_esc 0.30 --checkpoint results/checkpoints/caval_scorer_seed1.pt"; RID=caval_deployed_verifier_directed${RUN_SUFFIX:-};;
  undefended) ARGS="";                                                       RID=undefended_verifier_directed${RUN_SUFFIX:-};;
  drift)      ARGS="--build_constraints --injection_isolation --dynamic_validation"; RID=drift_verifier_directed${RUN_SUFFIX:-};;
  *) echo "first argument must be caval|undefended|drift"; exit 1;;
esac
for T in 1 2 3; do $H $ARGS --run_id $RID >> data/runs/logs/$RID.log 2>&1 && break; echo "retry $T"; done
echo "$RID done"
