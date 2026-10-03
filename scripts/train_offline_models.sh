#!/usr/bin/env bash
# Offline training behind Tables 3, 4, 6 and 8 and the deployed scorers (GPU recommended).
# Retraining reproduces the reported values up to GPU nondeterminism; the shipped results/offline files are the record.
set -e; cd "$(dirname "$0")/.."
T="uv run python src/experiments/train_scorer.py"; G=data/corpus; IDX=$G/action_labels.pkl; R=results/offline; S="1 2 3 4 5 6 7 8 9 10"
mkdir -p $R results/checkpoints
# five architectures, ten seeds (Tables 3, 4, 8a; Figure 4)
$T prefix --seeds $S --models rgcn gcn seq_lstm mlp_pool gatv2 --aux-weight 0 --index $IDX --output $R/ten_seed_models.json --dump-scores $R/ten_seed_scores.npz
# R-GCN ablations (Table 8b)
for K in temporal call_return data_flow; do
  $T prefix --seeds $S --models rgcn --drop-kinds $K --aux-weight 0 --index $IDX --output $R/ablation_no_$K.json
done
$T prefix --seeds $S --models rgcn gatv2 --no-content --aux-weight 0 --index $IDX --output $R/ablation_no_content.json
$T prefix --seeds $S --models rgcn gatv2 --aux-weight 1 --index $IDX --output $R/ablation_utility_objective.json
for N in 1 3; do
  V=$G/variants/overlap$N; mkdir -p $V
  uv run python src/corpus/build_trace_graphs.py --df-min-overlap $N --output $V/trace_graphs.pkl
  uv run python src/caval/trace_graph.py content --graphs $V/trace_graphs.pkl --output $V/content_emb.npz
  uv run python src/corpus/label_compromise_step.py --graphs $V/trace_graphs.pkl --output $V/action_labels.pkl
  $T prefix --seeds $S --models rgcn --graphs $V/trace_graphs.pkl --index $V/action_labels.pkl --content-emb $V/content_emb.npz \
     --aux-weight 0 --output $R/ablation_dataflow_overlap$N.json
done
# attack objectives held out of training and calibration (Table 6)
$T prefix --seeds $S --models rgcn gatv2 --aux-weight 0 --split heldout_attack --index $IDX \
   --output $R/heldout_objectives.json --dump-scores $R/heldout_objectives_scores.npz
# the five deployed scorers, each with its benign calibration maxima (used online)
for K in 1 2 3 4 5; do $T freeze --seed $K --index $IDX --output results/checkpoints/caval_scorer_seed$K.pt; done
