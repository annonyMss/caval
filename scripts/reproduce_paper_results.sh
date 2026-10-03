#!/usr/bin/env bash
# Regenerate the tables and figures of the paper from the released episode logs, offline score files and scorers
# (no model calls, no training; CPU is enough, a few minutes). Every output goes to results/.
set -e; cd "$(dirname "$0")/.."
PY=${PY:-uv run python}; TB="$PY src/analysis/paper_tables.py"; C=data/corpus
# online comparison on the matched episodes (Table 2 seven-suite block, Table 7, the budget and full-coverage text)
$TB online_comparison --arms stop=caval_stop_alpha030,recovery=caval_deployed --undefended undefended --shared --out seven_suite_comparison
$TB online_comparison --arms stop=caval_stop_alpha030,recovery=caval_deployed,rules_only=rules_only,rules_and_caval=rules_and_caval --undefended undefended --shared --out components
$TB online_comparison --arms stop=caval_stop_alpha030,recovery=caval_deployed --undefended undefended --out full_coverage
$TB online_comparison --arms stop=caval_stop_alpha025,recovery=caval_recovery_alpha025 --out budget_alpha025
# five-scorer summaries (Table 5 recovery, the online benign stops in Figure 4)
$TB five_scorer_summary caval_recovery_alpha025 banking,slack,travel caval_stop_alpha025 --out recovery
$TB five_scorer_summary caval_stop_alpha025 all --out online_stops_alpha025
$TB five_scorer_summary caval_stop_alpha030 all --out online_stops_alpha030
# defense comparisons per benchmark (Table 2, Figure 3, Table 11)
$TB agentdojo_defenses --caval caval_deployed --undefended undefended --out agentdojo_defenses
$TB agentdyn_defenses --caval caval_deployed --out agentdyn_defenses
# adaptive attacks (Figure 5)
$TB verifier_directed_attack --caval-runs caval_deployed_scorer1,caval_deployed_verifier_directed --undef-adaptive undefended_verifier_directed --out verifier_directed_attack
$TB adversary_knowledge --caval "caval_deployed_{v}" --plain caval_deployed_scorer1 --out adversary_knowledge
$TB score_aware_attack caval_deployed --out score_aware_attack
# offline results (Tables 3, 4, 6 and 8) and verifier latency
$TB unseen_wording --ckpt-prefix results/checkpoints/caval_scorer_seed --out unseen_wording
$TB offline_models
$TB calibration_rules --scores results/offline/ten_seed_scores.npz --index $C/action_labels.pkl --out calibration_rules
$TB overhead
$TB latency data/runs/gpt-4o-mini-2024-07-18/no_attack/caval_deployed_scorer{1,2,3,4,5}_benign --out latency
# figures
$PY src/analysis/paper_figures.py paper
$PY src/analysis/paper_figures.py trajectory
