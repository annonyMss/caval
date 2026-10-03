"""Figures of the CAVAL paper, written as PDF to results/figures/.

  paper_figures.py paper        Figures 3, 4 and 5 (defense comparison, calibration, adaptive attacks), from the read-outs in results/
  paper_figures.py trajectory   Figure 6 (PCA of the deployed scorer's prefix representations), from the checkpoint and the corpus
Run the read-outs in paper_tables.py first (scripts/reproduce_paper_results.sh); this file only plots.
"""
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader as PyGLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.paper_tables import _dump_traces, _lam_alpha, _stat_traces, _trace_metrics
from caval.trace_graph import build_tool_vocab_matrix
from caval.scorer import build_model
from experiments.train_scorer import build_prefix_data, within_split

FIGS = Path("results/figures")
FIGS.mkdir(parents=True, exist_ok=True)


def set_paper_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Nimbus Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        # 5.5in text width, Times 10pt body, footnotesize 9pt. Figures are
        # authored at FINAL size (5.5in wide -> width=\linewidth scales by 1.0),
        # so these ARE the printed sizes: 9pt labels, 8pt ticks/legend.
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'legend.title_fontsize': 9,
        'figure.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        # Embed TrueType (Type-42), not bitmap Type-3 -- camera-ready requirement
        # at most security venues.
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        # Horizontal + vertical grid in every axis plot (co-author requirement);
        # drawn behind the data. Network renders use axis('off'), unaffected.
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        'axes.axisbelow': True,
        # Open X-Y coordinate axes, not a closed box (co-author request):
        # only the left/bottom spines carry the coordinate system.
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "purple": "#CC79A7",
         "vermillion": "#D55E00", "sky": "#56B4E9", "grey": "#7F7F7F"}


def _bar_labels(ax, bars, fmt="{:.1f}", dy=0.8):
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=7)


def cmd_trajectory():
    """Figure 6: PCA of the pooled prefix representations of the deployed scorer (seed 1). Rebuilds the split the scorer
    was trained on, checks that the rebuilt calibration maxima equal the checkpoint's, and marks one compromised test
    trace per outcome at alpha 0.30 (on time, late, missed). Usage: paper_figures.py trajectory"""
    set_paper_style()
    # IEEE figure conventions: Times-metric serif, 9 pt axis labels, 8 pt ticks and legend, no in-plot title
    plt.rcParams.update({"font.serif": ["Liberation Serif", "Tinos", "Nimbus Roman", "DejaVu Serif"],
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
    SUITE_NAME = {"banking": "Banking", "slack": "Slack", "travel": "Travel", "workspace": "Workspace",
                  "dailylife": "DailyLife", "github": "GitHub", "shopping": "Shopping"}   # as the paper writes them
    CKPT, INDEX, ALPHA = "results/checkpoints/caval_scorer_seed1.pt", "data/corpus/action_labels.pkl", 0.30
    ck = torch.load(CKPT, weights_only=False)
    pay = pickle.loads(Path("data/corpus/trace_graphs.pkl").read_bytes())
    rows = pickle.loads(Path(INDEX).read_bytes())["rows"]
    tool_emb = build_tool_vocab_matrix(pay["tool_vocab"])[0]
    data = build_prefix_data(pay, rows, tool_emb, np.load("data/corpus/content_embeddings.npz"))
    trace_ids = sorted({r["trace"] for r in rows}); tgroup = {r["trace"]: r["group"] for r in rows}
    rows_of = defaultdict(list)
    for j, r in enumerate(rows): rows_of[r["trace"]].append(j)
    _, _, ca_t, te_t = within_split(np.array([tgroup[t] for t in trace_ids]), ck["seed"])
    ca = [j for t in ca_t for j in rows_of[trace_ids[t]]]; te = [j for t in te_t for j in rows_of[trace_ids[t]]]

    model = build_model(ck["model_name"], in_dim=ck["in_dim"], hidden=ck["hidden"], dropout=ck["dropout"],
                        emb_dim=ck["emb_dim"], proj_dim=ck["proj_dim"])
    model.load_state_dict(ck["model_state"]); model.eval()
    captured = {}
    model.base.heads.register_forward_pre_hook(lambda m, inp: captured.__setitem__("h", inp[0].detach().numpy()))
    def run(idx):                                   # pooled representation and compromise score per prefix
        H, P = [], []
        with torch.no_grad():
            for batch in PyGLoader([data[j] for j in idx], batch_size=64, shuffle=False):
                logits, _ = model(batch)
                H.append(captured["h"]); P.append(torch.softmax(logits, -1)[:, 1].numpy())
        return np.concatenate(H), np.concatenate(P)
    H_ca, p_ca = run(ca); H_te, p_te = run(te)

    # deployed threshold from the checkpoint's benign calibration maxima, and a consistency check against the scores
    cal_M = np.array(ck["cal_M"]); n = len(cal_M); lam = cal_M[int(np.ceil((n + 1) * (1 - ALPHA))) - 1]
    cal_max = defaultdict(float)
    for j, s in zip(ca, p_ca):
        if not rows[j]["y_trace"]: cal_max[rows[j]["trace"]] = max(cal_max[rows[j]["trace"]], s)
    assert np.allclose(sorted(cal_max.values()), cal_M, atol=1e-4), "rebuilt scores do not match the checkpoint"
    by_trace = defaultdict(list)
    for j, s in zip(te, p_te): by_trace[rows[j]["trace"]].append((rows[j]["t"], s, rows[j]["y_trace"], rows[j]["t_star"]))
    outcome = {}
    for tr, v in by_trace.items():
        v.sort(); y, tstar = v[0][2], v[0][3]
        if not y or tstar is None: continue
        alarm = next((t for t, s, _, _ in v if s > lam), None)
        outcome[tr] = ("missed" if alarm is None else "on time" if alarm <= tstar else "late", tstar)
    print(f"alpha {ALPHA}: lambda {lam:.3f}; outcomes of {len(outcome)} compromised test traces:",
          dict(Counter(o for o, _ in outcome.values())))

    H = np.vstack([H_ca, H_te]); meta = [rows[j] for j in ca + te]
    pca = PCA(n_components=2, random_state=0); H2 = pca.fit_transform(H)
    chosen, seen = [], set()
    for o in ("on time", "late", "missed"):
        pool = sorted(tr for tr, (oo, ts) in outcome.items() if oo == o and (o != "on time" or ts >= 3))
        fresh = [tr for tr in pool if pay["records"][tr]["suite"] not in seen] or pool
        if fresh:
            tr = fresh[len(fresh) // 2]; chosen.append((tr, o)); seen.add(pay["records"][tr]["suite"])
    print("examples:", [(tr, pay["records"][tr]["suite"], o, outcome[tr][1]) for tr, o in chosen])

    y = np.array([m["y"] for m in meta])
    fig, ax = plt.subplots(figsize=(3.5, 3.0))                  # IEEE single column
    ax.scatter(H2[y == 0, 0], H2[y == 0, 1], s=4, color="#d9d9d9", linewidths=0, label="Benign so far")
    ax.scatter(H2[y == 1, 0], H2[y == 1, 1], s=4, color="#595959", linewidths=0, label="At or after $t^{*}$")
    for (tr, o), col in zip(chosen, (OKABE["blue"], OKABE["orange"], OKABE["vermillion"])):
        ix = sorted((i for i, m in enumerate(meta) if m["trace"] == tr), key=lambda i: meta[i]["t"])
        ax.plot(H2[ix, 0], H2[ix, 1], color=col, lw=1.2, marker="o", ms=2.5, label=f"{SUITE_NAME.get(pay['records'][tr]['suite'], pay['records'][tr]['suite'])}, {o}")
        ax.scatter(*H2[ix[0]], color=col, marker="s", s=22, edgecolors="black", linewidths=0.5, zorder=5)
        star = [i for i in ix if meta[i]["t"] == outcome[tr][1]]
        if star: ax.scatter(*H2[star[0]], color=col, marker="*", s=70, edgecolors="black", linewidths=0.5, zorder=6)
    ax.set_xlabel(f"First principal component ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    ax.set_ylabel(f"Second principal component ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    ax.legend(loc="best", frameon=False, markerscale=1.5, handlelength=1.5)
    fig.savefig(FIGS / "embed_trajectory.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"-> {FIGS / 'embed_trajectory.pdf'}")


def cmd_paper():
    """Figures 3, 4 and 5 (defense comparison, calibration, adaptive attacks) from the read-outs in results/. Usage: paper_figures.py paper"""
    set_paper_style()
    # a TrueType Times-metric face, so pdf.fonttype 42 embeds a real TrueType program (no Type 3, no CFF-in-TrueType)
    plt.rcParams["font.serif"] = ["Liberation Serif", "Tinos", "Nimbus Roman", "DejaVu Serif"]
    def save(fig, name):
        fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight"); plt.close(fig); print(f"-> {FIGS / name}.pdf")

    # ---- Figure: defense comparison, direct (left) and injected (right), bars per defense by benchmark ----
    dojo = json.load(open("results/agentdojo_defenses.json"))["pooled"]
    adyn = json.load(open("results/agentdyn_defenses.json"))["pooled"]
    names = ["Undefended", "Tool filter", "Progent", "CaMeL", "DRIFT", "CAVAL"]
    dojo_keys = ["undefended (our harness)", "tool_filter (fork harness)", "progent (fork harness)",
                 "camel (fork harness)", "DRIFT (our harness)", "CAVAL (our harness)"]
    adyn_keys = ["undefended", "tool_filter", "progent", "camel", "drift", "CAVAL (ours, same tasks)"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
    x = np.arange(len(names)); w = 0.38
    for ax, metric, title in zip(axes, ("asr_direct", "asr_injected"), ("Direct attacks", "Injected attacks")):
        a = [dojo[k][metric] for k in dojo_keys]; b = [adyn[k][metric] for k in adyn_keys]
        ba = ax.bar(x - w / 2, a, w, color=OKABE["blue"], label="AgentDojo", linewidth=0)
        bb = ax.bar(x + w / 2, b, w, color=OKABE["orange"], label="AgentDyn", linewidth=0)
        if metric == "asr_direct":                      # the message of the figure; injected values are in the tables
            _bar_labels(ax, ba); _bar_labels(ax, bb)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_title(title); ax.set_ylim(0, 108)
    axes[0].set_ylabel("attack success (%)"); axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.5); save(fig, "defense_comparison")

    # ---- Figure: calibration, false-alarm rate vs budget (left), prevention and online stops vs budget (right) ----
    rows = pickle.load(open("data/corpus/action_labels.pkl", "rb"))["rows"]
    Z = np.load("results/offline/ten_seed_scores.npz")
    alphas = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    seeds = sorted({int(k.split("_s")[1].split("_")[0]) for k in Z.files if k.startswith("rgcn_s") and k.endswith("_te_p")})
    far = np.zeros((len(seeds), len(alphas))); prev_ = np.zeros_like(far)
    for i, s in enumerate(seeds):
        ca = _stat_traces(_dump_traces(Z, rows, "rgcn", s, "ca"), "max"); te = _stat_traces(_dump_traces(Z, rows, "rgcn", s, "te"), "max")
        ben = [max(x[1] for x in v) for v in ca.values() if v[0][2] == 0]
        for j, a in enumerate(alphas):
            m = _trace_metrics(te, _lam_alpha(ben, a)); far[i, j], prev_[i, j] = m["far"], m["prev"]
    online = {0.25: json.load(open("results/online_stops_alpha025.json"))["benign stop rate"]["candidate"],
              0.30: json.load(open("results/online_stops_alpha030.json"))["benign stop rate"]["candidate"]}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
    ax = axes[0]
    ax.plot([0, 0.5], [0, 50], color=OKABE["grey"], linestyle="--", linewidth=1, label="budget")
    ax.errorbar(alphas, 100 * far.mean(0), yerr=100 * far.std(0), color=OKABE["blue"], marker="o", markersize=3.5,
                linewidth=1.5, capsize=2, label="observed, all suites")
    ax.set_xlabel(r"false-alarm budget $\alpha$"); ax.set_ylabel("per-trace false-alarm rate (%)"); ax.set_xlim(0, 0.52); ax.set_ylim(0, 52)
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    ax.plot(alphas, 100 * prev_.mean(0), color=OKABE["blue"], marker="o", markersize=3.5, linewidth=1.5, label="prevention, offline")
    ax.fill_between(alphas, 100 * (prev_.mean(0) - prev_.std(0)), 100 * (prev_.mean(0) + prev_.std(0)), color=OKABE["blue"], alpha=0.15, linewidth=0)
    ax.plot(list(online), list(online.values()), color=OKABE["orange"], marker="s", markersize=5, linestyle="none", label="benign stops, online")
    for a in (0.25, 0.30):
        ax.axvline(a, color=OKABE["grey"], linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"false-alarm budget $\alpha$"); ax.set_ylabel("prevention, benign stops (%)"); ax.set_xlim(0, 0.52); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout(w_pad=1.5); save(fig, "calibration")
    Path("results/calibration_curve.json").write_text(json.dumps(dict(
        alpha=alphas, far_mean=list(far.mean(0)), far_sd=list(far.std(0)), prevention_mean=list(prev_.mean(0)),
        prevention_sd=list(prev_.std(0)), online_benign_stops=online), indent=1))
    print("-> results/calibration_curve.json")

    # ---- Figure: stress test, attack success per attacker variant (left), score-access rewrite by round (right) ----
    # deployed configuration, one scorer, all four variants on the Banking, Slack and Travel pairs (389) so one population
    lad = json.load(open("results/adversary_knowledge.json"))
    rw = json.load(open("results/score_aware_attack.json"))
    plain_ref = lad["no_names"]["refs"]
    adp = json.load(open("results/verifier_directed_attack.json"))
    def three(lbl, which):                                                    # Banking, Travel, Slack only, the panel population
        rows = [adp[lbl]["per_suite"][su] for su in ("banking", "travel", "slack") if su in adp[lbl]["per_suite"]]
        return 100 * sum(r[which] for r in rows) / sum(r["n"] for r in rows)
    T3 = {"plain": {"Undefended": plain_ref["undefended (plain template)"]["pct"], "CAVAL": plain_ref["CAVAL deployed, same scorer, plain template"]["pct"]},
          "verifier-directed": {"CAVAL": three("CAVAL", "adaptive_k"), **({"Undefended": three("Undefended", "adaptive_k")} if "Undefended" in adp else {})}}
    variants = [("plain", T3["plain"]), ("verifier-directed", T3["verifier-directed"]),
                ("no names", {"Undefended": lad["no_names"]["POOLED"]["undef_asr"]["pct"], "CAVAL": lad["no_names"]["POOLED"]["caval_asr"]["pct"]}),
                ("wrong user name", {"Undefended": lad["wrong_user_name"]["POOLED"]["undef_asr"]["pct"], "CAVAL": lad["wrong_user_name"]["POOLED"]["caval_asr"]["pct"]})]
    systems = [("Undefended", OKABE["grey"]), ("CAVAL", OKABE["blue"])]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]; x = np.arange(len(variants)); w = 0.36
    for k, (sysname, col) in enumerate(systems):
        vals = [v.get(sysname, np.nan) for _, v in variants]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, color=col, label=sysname, linewidth=0)
        _bar_labels(ax, bars)
    ax.set_xticks(x); ax.set_xticklabels([v[0] for v in variants], rotation=20, ha="right")
    ax.set_ylabel("attack success (%)"); ax.set_ylim(0, 58); ax.legend(frameon=False, loc="upper right", ncol=1, handlelength=1.2)
    ax.set_title("Attackers who know the defense")
    ax = axes[1]; rounds = np.arange(6)
    for sname, col, ls in (("banking", OKABE["vermillion"], "-"), ("slack", OKABE["green"], "-"), ("travel", OKABE["purple"], "-"), ("POOLED", OKABE["blue"], "--")):
        ax.plot(rounds, [r["pct"] for r in rw[sname]["by_round"]], color=col, linestyle=ls, marker="o", markersize=3, linewidth=1.3,
                label="all three suites" if sname == "POOLED" else sname.capitalize())
    ax.set_xlabel("rewrite round"); ax.set_ylabel("stopped attacks recovered (%)"); ax.set_ylim(0, 25); ax.set_xticks(rounds)
    ax.legend(frameon=False, loc="upper left"); ax.set_title("Score-access rewrite")
    fig.tight_layout(w_pad=1.5); save(fig, "stress")



if __name__ == "__main__":
    CMDS = {"paper": cmd_paper, "trajectory": cmd_trajectory}
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        sys.exit(f"usage: paper_figures.py {'|'.join(CMDS)}")
    CMDS[sys.argv.pop(1)]()
