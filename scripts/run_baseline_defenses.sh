#!/bin/bash
# Baseline defenses (tool filter, Progent, CaMeL) on the AgentDojo suites, run in AgentDyn's copy of the AgentDojo harness with the
# same layout as the runs AgentDyn released, so `paper_tables.py agentdojo_defenses` reads them.
# usage: scripts/run_baseline_defenses.sh <defense> [suites=banking,slack,travel,workspace] [extra fork args, e.g. --user-task user_task_0 --injection-task injection_task_0 for a smoke test]
#   defense in tool_filter progent camel spotlighting_with_delimiting transformers_pi_detector repeat_user_prompt
# Needs OPENAI_API_KEY in the environment. Output: data/benchmarks/AgentDyn/runs/gpt-4o-mini-2024-07-18-<defense>/<suite>/...
cd "$(dirname "$0")/.."
D=$1; SUITES=${2:-banking,slack,travel,workspace}; shift $(( $# > 1 ? 2 : $# )); EXTRA="$@"
TAG=$(echo "$SUITES" | tr , _)   # one log per suite set, so parallel per-suite launches do not interleave
SARGS=""; for s in ${SUITES//,/ }; do SARGS="$SARGS -s $s"; done
for MODE in "--attack important_instructions" ""; do
  # The shipped TaskResults model leaves FunctionCall as an unresolved forward reference, which breaks loading of
  # stored episodes (skip-if-stored path); rebuild both before the CLI runs, as the fork's CaMeL code does.
  PYTHONPATH=data/benchmarks/AgentDyn/src uv run python -c "
from agentdojo.functions_runtime import FunctionCall; FunctionCall.model_rebuild()
from agentdojo.benchmark import TaskResults; TaskResults.model_rebuild()
from agentdojo.scripts.benchmark import main; main()" \
    --model GPT_4O_MINI_2024_07_18 --benchmark-version v1.2 --logdir data/benchmarks/AgentDyn/runs \
    --defense $D $SARGS $MODE $EXTRA >> data/runs/logs/baseline_${D}_${TAG}.log 2>&1 || echo "baseline run failed: $D $MODE (see data/runs/logs/baseline_${D}_${TAG}.log)"
done
echo "baseline $D done: $(find data/benchmarks/AgentDyn/runs/gpt-4o-mini-2024-07-18-$D -name '*.json' 2>/dev/null | wc -l) episodes"
