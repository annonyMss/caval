#!/bin/bash
# The five CAVAL scorers online, every (scorer, episode type) pair in parallel; same options as run_caval_one_scorer.sh.
# usage: [ALPHA=0.30] [ENFORCE=isolate|stop] [PATHS=learned|rules|both] [SUITES=...] scripts/run_caval_five_scorers.sh ["1 2 3 4 5"] [benign|direct|injected|all]
cd "$(dirname "$0")/.."
SCORERS=${1:-1 2 3 4 5}; WHAT=${2:-all}
[ "$WHAT" = all ] && TYPES="benign direct injected" || TYPES=$WHAT
for S in $SCORERS; do for W in $TYPES; do scripts/run_caval_one_scorer.sh $S $W > /dev/null 2>&1 & done; done
echo "launched $(jobs -p | wc -l) jobs: scorers [$SCORERS] x [$TYPES]"; wait; echo "all done $(date)"
