#!/bin/bash
# Token count for a defense run in AgentDyn's copy of the benchmark (tool_filter | progent | camel), Banking benign tasks.
# The harness records no token counts, so every OpenAI SDK completion call in the process is counted at the SDK class and
# the totals are written at exit. Fresh log folder (data/benchmarks/AgentDyn/runs_tokens) so the tasks run again.
# usage: scripts/count_baseline_tokens.sh <defense>   -> data/runs/baseline_tokens/<defense>.json  {calls, prompt, completion}
cd "$(dirname "$0")/.."
D=$1; mkdir -p data/runs/baseline_tokens; OUT=data/runs/baseline_tokens/$D.json
PYTHONPATH=data/benchmarks/AgentDyn/src uv run python -c "
import atexit, json
from openai.resources.chat.completions import Completions
totals = {'calls': 0, 'prompt': 0, 'completion': 0}
original_create = Completions.create
def counted_create(self, *args, **kwargs):
    response = original_create(self, *args, **kwargs)
    usage = getattr(response, 'usage', None)
    if usage is not None:
        totals['calls'] += 1; totals['prompt'] += usage.prompt_tokens or 0; totals['completion'] += usage.completion_tokens or 0
    return response
Completions.create = counted_create
atexit.register(lambda: open('$OUT', 'w').write(json.dumps(totals)))
from agentdojo.functions_runtime import FunctionCall; FunctionCall.model_rebuild()
from agentdojo.benchmark import TaskResults; TaskResults.model_rebuild()
from agentdojo.scripts.benchmark import main; main()" \
  --model GPT_4O_MINI_2024_07_18 --benchmark-version v1.2 --logdir data/benchmarks/AgentDyn/runs_tokens --defense $D -s banking \
  >> data/runs/logs/baseline_tokens_$D.log 2>&1
echo "baseline tokens $D done: $(cat $OUT 2>/dev/null)"
