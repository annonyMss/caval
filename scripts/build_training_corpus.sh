#!/usr/bin/env bash
# Rebuild the training corpus from the shipped episode logs (CPU, minutes; the MiniLM encoder downloads once).
#   data/corpus/trace_graphs.pkl     1,994 trace graphs, seven suites (Table 10)
#   data/corpus/content_embeddings.npz frozen content embeddings
#   data/corpus/action_labels.pkl          one row per proposed action, labels from the compromise step t*
#   ..._unseen_wording.*                         the same plus the other AgentDojo injection wordings (Table 6, unseen wording)
set -e; cd "$(dirname "$0")/.."
G=data/corpus; PY="uv run python"
$PY src/corpus/build_trace_graphs.py --output $G/trace_graphs.pkl
$PY src/caval/trace_graph.py content --graphs $G/trace_graphs.pkl --output $G/content_embeddings.npz
$PY src/corpus/label_compromise_step.py --graphs $G/trace_graphs.pkl --output $G/action_labels.pkl
$PY src/corpus/build_trace_graphs.py --extra-attack-modes attack_ignore_previous attack_system_message attack_tool_knowledge \
    --vocab-from $G/trace_graphs.pkl --output $G/trace_graphs_unseen_wording.pkl
$PY src/caval/trace_graph.py content --graphs $G/trace_graphs_unseen_wording.pkl --output $G/content_embeddings_unseen_wording.npz
$PY src/corpus/label_compromise_step.py --graphs $G/trace_graphs_unseen_wording.pkl --output $G/action_labels_unseen_wording.pkl
