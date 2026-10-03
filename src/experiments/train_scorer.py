"""Training and offline evaluation of the compromise scorer.

Subcommands:
  prefix  train the five architectures over seeds on the action labels and evaluate the running-maximum
          conformal rule on calibration and test traces (results/offline/*.json, optional score dump)
  freeze  train one deployable R-GCN scorer and store it with its benign calibration maxima (results/checkpoints/)
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.data import Data

from caval.scorer import graph_to_data, train_one, build_model, load_state_compat
from caval.trace_graph import build_tool_vocab_matrix
from torch_geometric.loader import DataLoader as PyGLoader
import sys

# ==== whole-trace (was run_multisuite.py) ====
"""Multi-suite compromised-detection.
  --mode within : pooled across suites, grouped by suite:task -> powered model comparison.
  --mode loso   : leave-one-suite-out -> D4 generalization (tool-identity embedding is what
                  carries the held-out suite's unseen tools).
AUC-first (F1-at-argmax is jumpy on small folds). Models: gcn/gatv2 (structure) vs seq_lstm
(order) vs mlp_pool (bag-of-nodes).
"""


def _metrics(res):
    logits = res["test"]["label_logits"]
    y = res["test"]["label_y"]
    p = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    pred = (p >= 0.5).astype(int)
    pos, neg = y == 1, y == 0
    return dict(
        auc=float(roc_auc_score(y, p)) if pos.any() and neg.any() else float("nan"),
        recall=float(pred[pos].mean()) if pos.any() else float("nan"),
        fpr=float(pred[neg].mean()) if neg.any() else float("nan"),
        npos=int(pos.sum()), nneg=int(neg.sum()),
    )


def _gs(pool, groups, frac, seed):
    g = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    a, b = next(g.split(pool, groups=groups[pool]))
    return pool[a], pool[b]


def within_split(groups, seed, v=0.15, c=0.175, t=0.175):
    idx = np.arange(len(groups))
    rest, te = _gs(idx, groups, t, seed)
    rest, ca = _gs(rest, groups, c / (1 - t), seed)
    tr, va = _gs(rest, groups, v / (1 - t - c), seed)
    return tr, va, ca, te


def attack_aware_groups(records, trace_ids, base_groups):
    """heldout_attack split: any trace carrying an attack task is regrouped as
    {suite}:ATK:{attack_task} so no injection/direct objective spans
    train/cal/test; pure-benign traces keep suite:user_task. Injected-but-safe
    traces are also regrouped (strict: payload text never spans the split)."""
    out = []
    for t, g in zip(trace_ids, base_groups):
        r = records[t]
        atk = r.get("injection_task_id") or (
            r["user_task_id"] if r.get("attack_kind") == "direct" else None)
        out.append(f'{r["suite"]}:ATK:{atk}' if atk else g)
    return np.array(out)


def build_prefix_data(pay, rows, tool_emb, cmats, drop_kinds=()):
    """Full-graph Data once per trace, prefixes as slices (x rows are views)."""
    full = {}
    for i in sorted({r["trace"] for r in rows}):
        full[i] = graph_to_data(
            pay["graphs"][i], pay["records"][i], len(pay["tool_vocab"]),
            drop_leaky=True, tool_emb=tool_emb,
            content_emb=None if cmats is None else cmats[f"g{i}"],
            drop_kinds=tuple(drop_kinds))
    data = []
    for r in rows:
        f, t = full[r["trace"]], r["t"]
        m = (f.edge_index[0] <= t) & (f.edge_index[1] <= t)
        data.append(Data(x=f.x[: t + 1], edge_index=f.edge_index[:, m],
                         edge_type=f.edge_type[m],
                         y_label=torch.tensor(r["y"], dtype=torch.long),
                         y_utility=f.y_utility))
    return data


def conformal_report(rows_split, p, alpha):
    """rows_split: index rows for one split, p: matching P(compromised) scores."""
    traces = defaultdict(list)
    for r, s in zip(rows_split, p):
        traces[r["trace"]].append((r["t"], s, r["y_trace"], r["t_star"]))
    return traces


def eval_traces(cal, test, alpha):
    m_ben = sorted(max(s for _, s, _, _ in v) for v in cal.values()
                   if v[0][2] == 0)
    n = len(m_ben)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    qhat = m_ben[k - 1] if k <= n else float("inf")

    far, prev, delay, miss = [], [], [], 0
    for v in test.values():
        v = sorted(v)
        y, t_star = v[0][2], v[0][3]
        alarm = next((t for t, s, _, _ in v if s > qhat), None)
        if y == 0:
            far.append(int(alarm is not None))
        else:
            if alarm is None:
                miss += 1
                prev.append(0)
            else:
                prev.append(int(alarm <= t_star))
                if alarm > t_star:
                    steps = [t for t, _, _, _ in v]  # delay in proposed actions
                    delay.append(steps.index(alarm) - steps.index(t_star))
    return dict(alpha=alpha, qhat=float(qhat) if np.isfinite(qhat) else None,
                n_cal_benign=n,
                prevention=float(np.mean(prev)) if prev else float("nan"),
                per_trace_far=float(np.mean(far)) if far else float("nan"),
                mean_delay=float(np.mean(delay)) if delay else 0.0,
                missed=miss, n_test_pos=len(prev), n_test_neg=len(far))


def main_prefix():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="data/corpus/trace_graphs.pkl")
    ap.add_argument("--index", default="data/corpus/action_labels.pkl")
    ap.add_argument("--content-emb", default="data/corpus/content_embeddings.npz")
    ap.add_argument("--models", nargs="+",
                    default=["rgcn", "gcn", "seq_lstm", "mlp_pool", "gatv2"])
    ap.add_argument("--split", choices=["within", "heldout_attack"], default="within",
                    help="heldout_attack: group attack-carrying traces by their attack task so no "
                         "injected or direct objective spans train, calibration and test (Table 6)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--no-content", action="store_true", help="drop the content embeddings (ablation)")
    ap.add_argument("--drop-kinds", nargs="*", default=[],
                    help="edge kinds to drop (ablation), e.g. data_flow")
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="weight of the auxiliary task-utility loss; 0 is the paper's model, 1 the ablation")
    ap.add_argument("--dump-scores", default=None,
                    help="npz path: per (model, seed) validation, calibration and test row indices and scores")
    ap.add_argument("--output", default="results/offline/ten_seed_models.json")
    args = ap.parse_args()

    pay = pickle.loads(Path(args.graphs).read_bytes())
    rows = pickle.loads(Path(args.index).read_bytes())["rows"]
    tool_emb = build_tool_vocab_matrix(pay["tool_vocab"])[0]
    cmats = None if args.no_content else np.load(args.content_emb)
    emb = 384 + (0 if cmats is None else 384)

    data = build_prefix_data(pay, rows, tool_emb, cmats, drop_kinds=args.drop_kinds)
    trace_ids = sorted({r["trace"] for r in rows})
    tgroup = {r["trace"]: r["group"] for r in rows}
    trace_groups = np.array([tgroup[t] for t in trace_ids])
    if args.split == "heldout_attack":
        trace_groups = attack_aware_groups(pay["records"], trace_ids, trace_groups)
        atk = sorted({g for g in trace_groups if ":ATK:" in g})
        n_atk = int(sum(":ATK:" in g for g in trace_groups))
        print(f"heldout_attack: {len(atk)} attack groups / {n_atk} of "
              f"{len(trace_groups)} traces regrouped")
        for s in sorted({g.split(":")[0] for g in atk}):
            print(f"  {s}: {sum(g.startswith(s + ':') for g in atk)} attack tasks")
    rows_of = defaultdict(list)
    for j, r in enumerate(rows):
        rows_of[r["trace"]].append(j)
    print(f"{len(data)} prefixes / {len(trace_ids)} traces | emb_dim={emb} "
          f"| in_dim={data[0].x.shape[1]}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    score_store = {}
    for seed in args.seeds:
        tr_t, va_t, ca_t, te_t = within_split(trace_groups, seed)
        if args.split == "heldout_attack":
            seen = {}
            for pname, tidx in zip(("tr", "va", "ca", "te"), (tr_t, va_t, ca_t, te_t)):
                for g in set(trace_groups[tidx]):
                    if ":ATK:" in g:
                        assert seen.setdefault(g, pname) == pname, \
                            f"attack group {g} spans {seen[g]} and {pname}"
            held = sorted({g for g in trace_groups[te_t] if ":ATK:" in g})
            print(f"[s{seed}] held-out attack tasks in test ({len(held)}): "
                  f"{', '.join(held)}")
        sp = {name: np.array([j for t in tidx for j in rows_of[trace_ids[t]]])
              for name, tidx in zip(("tr", "va", "ca", "te"), (tr_t, va_t, ca_t, te_t))}
        for m in args.models:
            res = train_one(data, sp["tr"], sp["va"], sp["ca"], sp["te"],
                            model_name=m, hidden=32, dropout=0.3,
                            epochs=80, batch_size=64, lr=1e-3,
                            weight_decay=5e-4, seed=seed, device=dev, emb_dim=emb,
                            aux_weight=args.aux_weight)
            p_ca = torch.softmax(torch.tensor(res["cal"]["label_logits"]), -1).numpy()[:, 1]
            p_te = torch.softmax(torch.tensor(res["test"]["label_logits"]), -1).numpy()[:, 1]
            # validation trace-level AUC: for hyperparameter SELECTION only (never cal/test)
            p_va = torch.softmax(torch.tensor(res["val"]["label_logits"]), -1).numpy()[:, 1]
            va = conformal_report([rows[j] for j in sp["va"]], p_va, None)
            auc_val_trace = float(roc_auc_score([va[t][0][2] for t in va],
                                                [max(x[1] for x in va[t]) for t in va]))
            if args.dump_scores:
                score_store[f"{m}_s{seed}_va_idx"] = sp["va"]
                score_store[f"{m}_s{seed}_va_p"] = p_va
                score_store[f"{m}_s{seed}_ca_idx"] = sp["ca"]
                score_store[f"{m}_s{seed}_ca_p"] = p_ca
                score_store[f"{m}_s{seed}_ca_logit"] = np.asarray(res["cal"]["label_logits"])
                score_store[f"{m}_s{seed}_te_idx"] = sp["te"]
                score_store[f"{m}_s{seed}_te_p"] = p_te
                score_store[f"{m}_s{seed}_te_logit"] = np.asarray(res["test"]["label_logits"])
            cal = conformal_report([rows[j] for j in sp["ca"]], p_ca, None)
            test = conformal_report([rows[j] for j in sp["te"]], p_te, None)

            y_pref = np.array([rows[j]["y"] for j in sp["te"]])
            auc_pref = float(roc_auc_score(y_pref, p_te))
            m_te = {tr: max(s for _, s, _, _ in v) for tr, v in test.items()}
            y_tr = {tr: v[0][2] for tr, v in test.items()}
            auc_trace = float(roc_auc_score([y_tr[t] for t in m_te],
                                            [m_te[t] for t in m_te]))
            row = dict(model=m, seed=seed, auc_prefix=auc_pref, auc_trace=auc_trace,
                       auc_val_trace=auc_val_trace,
                       cfg=dict(hidden=32, dropout=0.3, lr=1e-3, layers=2, readout="mean"),
                       conformal=[eval_traces(cal, test, a) for a in ALPHAS])
            out.append(row)
            cf = row["conformal"][1]  # alpha = 0.10
            print(f"[{m:9s} s{seed}] AUCpref={auc_pref:.3f} AUCtrace={auc_trace:.3f} | "
                  f"a=0.10: prev={cf['prevention']:.3f} FAR={cf['per_trace_far']:.3f} "
                  f"delay={cf['mean_delay']:.1f} miss={cf['missed']}")

    print("\n=== mean over seeds ===")
    for m in args.models:
        rr = [x for x in out if x["model"] == m]
        for ai, a in enumerate(ALPHAS):
            pv = np.nanmean([x["conformal"][ai]["prevention"] for x in rr])
            fr = np.nanmean([x["conformal"][ai]["per_trace_far"] for x in rr])
            dl = np.nanmean([x["conformal"][ai]["mean_delay"] for x in rr])
            print(f"{m:9s} a={a:.2f}: prevention={pv:.3f} per-trace-FAR={fr:.3f} "
                  f"delay={dl:.1f} | AUCtrace={np.nanmean([x['auc_trace'] for x in rr]):.3f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"-> {args.output}")
    if args.dump_scores:
        Path(args.dump_scores).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dump_scores, **score_store)
        print(f"-> {args.dump_scores} ({len(score_store)} arrays)")


# ==== deployable scorer ====
"""Persist a deployable CAVAL scorer: train the R-GCN at a fixed seed on the same split as `prefix`, calibrate on
the benign calibration traces, and save the weights, the sorted benign calibration maxima (cal_M) and everything
needed to rebuild the model and its features online. Verification: the checkpoint reloaded from disk must reproduce
the just-trained numbers exactly; the comparison with results/offline/ten_seed_models.json is a sanity band only,
because GPU nondeterminism makes independent trainings of the same data and seed differ slightly."""

SEED = 1
ALPHA = 0.10
MODEL_NAME = "rgcn"


def main_freeze():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="data/corpus/trace_graphs.pkl")
    ap.add_argument("--index", default="data/corpus/action_labels.pkl")
    ap.add_argument("--content-emb", default="data/corpus/content_embeddings.npz")
    ap.add_argument("--output", default="results/checkpoints/caval_scorer_seed1.pt")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    global SEED
    SEED = args.seed

    pay = pickle.loads(Path(args.graphs).read_bytes())
    rows = pickle.loads(Path(args.index).read_bytes())["rows"]
    tool_emb = build_tool_vocab_matrix(pay["tool_vocab"])[0]
    cmats = np.load(args.content_emb)
    emb_dim = tool_emb.shape[1] + 384

    data = build_prefix_data(pay, rows, tool_emb, cmats)
    trace_ids = sorted({r["trace"] for r in rows})
    tgroup = {r["trace"]: r["group"] for r in rows}
    trace_groups = np.array([tgroup[t] for t in trace_ids])
    rows_of = defaultdict(list)
    for j, r in enumerate(rows):
        rows_of[r["trace"]].append(j)

    tr_t, va_t, ca_t, te_t = within_split(trace_groups, SEED)
    sp = {}
    for name, tidx in zip(("tr", "va", "ca", "te"), (tr_t, va_t, ca_t, te_t)):
        sp[name] = np.array([j for t in tidx for j in rows_of[trace_ids[t]]])

    comp = {k: (len({rows[j]["trace"] for j in v}), int(sum(rows[j]["y"] for j in v))) for k, v in sp.items()}
    print("split (traces, positive prefixes):", comp)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training {MODEL_NAME} seed={SEED} on {dev} ...")
    # aux_weight=0: the paper's model (the utility objective is the ablation in Table 8b)
    res = train_one(data, sp["tr"], sp["va"], sp["ca"], sp["te"], model_name=MODEL_NAME,
                    hidden=32, dropout=0.3, epochs=80, batch_size=64, lr=1e-3,
                    weight_decay=5e-4, seed=SEED, device=dev, emb_dim=emb_dim,
                    aux_weight=0)

    p_ca = torch.softmax(torch.tensor(res["cal"]["label_logits"]), -1).numpy()[:, 1]
    p_te = torch.softmax(torch.tensor(res["test"]["label_logits"]), -1).numpy()[:, 1]
    cal = conformal_report([rows[j] for j in sp["ca"]], p_ca, None)
    test = conformal_report([rows[j] for j in sp["te"]], p_te, None)
    report = eval_traces(cal, test, ALPHA)
    print(f"in-memory (just-trained) @alpha={ALPHA}: qhat={report['qhat']:.4f} "
          f"prevention={report['prevention']:.3f} FAR={report['per_trace_far']:.3f} "
          f"delay={report['mean_delay']:.2f} missed={report['missed']}")

    in_dim = data[0].x.shape[1]
    struct_dim = in_dim - emb_dim
    # cal_M: the sorted benign calibration maxima. Storing the vector lets deployment derive the threshold at any
    # alpha via k = ceil((n+1)(1-alpha)) (Algorithm 1) without retraining; qhat is kept as a consistency check.
    cal_M = sorted(max(s for _, s, _, _ in v) for v in cal.values() if v[0][2] == 0)
    ckpt = dict(
        model_name=MODEL_NAME, seed=SEED, alpha=ALPHA,
        model_state=res["best_state"],
        in_dim=in_dim, struct_dim=struct_dim, emb_dim=emb_dim, proj_dim=32,
        hidden=32, dropout=0.3,
        qhat=report["qhat"],
        cal_M=[float(m) for m in cal_M],
        tool_vocab=pay["tool_vocab"],
        n_cal_benign=report["n_cal_benign"],
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)
    print(f"-> {args.output}")

    # --- verification: reload from disk, rerun eval_traces; hard gate = exact
    # round-trip vs the just-trained numbers, soft check = sanity band vs the
    # firmed guardfix artifact (see module docstring) ---
    print("\n=== verifying round-trip from disk ===")
    loaded = torch.load(args.output, weights_only=False)
    model = build_model(loaded["model_name"], in_dim=loaded["in_dim"], hidden=loaded["hidden"],
                        dropout=loaded["dropout"], emb_dim=loaded["emb_dim"],
                        proj_dim=loaded["proj_dim"]).to(dev)
    load_state_compat(model, loaded["model_state"])
    model.eval()

    def predict(idx):
        loader = PyGLoader([data[j] for j in idx], batch_size=64, shuffle=False)
        out = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(dev)
                logits, _ = model(batch)
                out.append(torch.softmax(logits, -1).cpu().numpy()[:, 1])
        return np.concatenate(out)

    p_ca2 = predict(sp["ca"])
    p_te2 = predict(sp["te"])
    cal2 = conformal_report([rows[j] for j in sp["ca"]], p_ca2, None)
    test2 = conformal_report([rows[j] for j in sp["te"]], p_te2, None)
    report2 = eval_traces(cal2, test2, ALPHA)
    print(f"reloaded-from-disk @alpha={ALPHA}: qhat={report2['qhat']:.4f} "
          f"prevention={report2['prevention']:.3f} FAR={report2['per_trace_far']:.3f} "
          f"delay={report2['mean_delay']:.2f} missed={report2['missed']}")

    # HARD gate: the reload must reproduce the just-trained numbers EXACTLY --
    # this is the actual persistence verification (weights, qhat, vocab all
    # round-trip). This is the only pass/fail check.
    roundtrip_ok = (abs(report2["prevention"] - report["prevention"]) < 1e-9 and
                    abs(report2["per_trace_far"] - report["per_trace_far"]) < 1e-9 and
                    abs(report2["qhat"] - report["qhat"]) < 1e-9 and
                    report2["missed"] == report["missed"])
    print(f"\nROUND-TRIP EXACT: {roundtrip_ok}")
    if not roundtrip_ok:
        print("PERSISTENCE MISMATCH -- checkpoint on disk does not reproduce the "
              "trained model. Do not deploy; investigate.")
        return

    # Sanity band against the ten-seed results (same model and seed). An exact match is not expected because of GPU
    # nondeterminism; this only catches a wrong dataset, split or pipeline. Run `prefix` first so the file exists.
    FIRMED = "results/offline/ten_seed_models.json"
    target = json.loads(Path(FIRMED).read_text())
    tgt = next(r for r in target if r["model"] == MODEL_NAME and r["seed"] == SEED)
    tgt_c = next(c for c in tgt["conformal"] if abs(c["alpha"] - ALPHA) < 1e-9)
    print(f"\nfirmed artifact ({FIRMED}) same-seed row: qhat={tgt_c['qhat']:.4f} "
          f"prevention={tgt_c['prevention']:.3f} FAR={tgt_c['per_trace_far']:.3f} "
          f"delay={tgt_c['mean_delay']:.2f} missed={tgt_c['missed']}")
    d_prev = abs(report2["prevention"] - tgt_c["prevention"])
    d_far = abs(report2["per_trace_far"] - tgt_c["per_trace_far"])
    print(f"deltas: prevention {d_prev:.3f}, FAR {d_far:.3f} "
          f"(GPU-nondeterminism-scale differences are expected)")
    if d_prev > 0.15 or d_far > 0.10:
        print("WARNING: deltas exceed the nondeterminism band -- check that the "
              "dataset/index/artifact versions actually correspond before deploying.")
    else:
        print("OK: within the expected nondeterminism band. Checkpoint verified for deployment.")


if __name__ == "__main__":
    CMDS = {"prefix": main_prefix, "freeze": main_freeze}
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        sys.exit(f"usage: train.py {{{'|'.join(CMDS)}}} [args]")
    cmd = sys.argv.pop(1)   # every subcommand uses argparse on the remaining argv
    CMDS[cmd]()
